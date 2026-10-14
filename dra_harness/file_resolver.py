"""
file_resolver.py — Resolves file references to local paths.

The PromptPackage may reference files from multiple sources:
  - Local paths (already on disk)
  - Google Drive URLs (shared links, folder links)
  - Google Drive file IDs (direct references)

The FileResolver normalizes all of these to local paths in a staging
directory, so the rest of the pipeline (file_processor, adapters,
corpus_tools) works uniformly regardless of where files originated.

Architecture:
    PromptPackage.file_paths
        │
        ├── "/local/path/file.pdf"    → pass through (already local)
        ├── "gdrive://folder_id"       → download folder contents
        ├── "gdrive://file_id"         → download single file
        ├── "https://drive.google.com/..." → parse URL, download
        └── "https://docs.google.com/..." → export as PDF/XLSX/DOCX
        │
        ▼
    staging_dir/
    ├── file.pdf
    ├── report.docx
    └── data.xlsx

Google Drive authentication:
    The resolver uses a service account or OAuth credentials.
    It supports two authentication methods:
      1. Service account key (JSON file) — for server/automated use
      2. OAuth client credentials — for interactive/developer use

    Set one of:
      GOOGLE_SERVICE_ACCOUNT_KEY=/path/to/service-account.json
      GOOGLE_CLIENT_SECRETS=/path/to/client_secrets.json

Dependencies:
    pip install google-api-python-client google-auth google-auth-oauthlib
"""

from __future__ import annotations

import os
import re
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("dra.file_resolver")


# ─── Google Drive URL parsing ────────────────────────────────────────────

# Patterns for various GDrive URL formats.
# Each pattern is written to be robust against all known URL variants:
#
#   File URLs:
#     https://drive.google.com/file/d/{ID}/view
#     https://drive.google.com/file/d/{ID}/view?usp=sharing
#
#   Open / direct download URLs:
#     https://drive.google.com/open?id={ID}
#     https://drive.google.com/uc?id={ID}
#
#   Folder URLs (all account-scoped variants):
#     https://drive.google.com/drive/folders/{ID}
#     https://drive.google.com/drive/folders/{ID}?usp=sharing
#     https://drive.google.com/drive/folders/{ID}?usp=drive_link
#     https://drive.google.com/drive/u/0/folders/{ID}       ← user-scoped
#     https://drive.google.com/drive/u/1/folders/{ID}       ← second account
#     https://drive.google.com/drive/u/0/folders/{ID}?usp=drive_link
#
#   Google Workspace (Docs / Sheets / Slides):
#     https://docs.google.com/document/d/{ID}/edit
#     https://docs.google.com/document/d/{ID}/edit?usp=sharing
#     https://docs.google.com/spreadsheets/d/{ID}/edit
#     https://docs.google.com/spreadsheets/d/{ID}/edit#gid=0
#     https://docs.google.com/presentation/d/{ID}/edit
#
#   Custom scheme:
#     gdrive://{ID}

GDRIVE_FILE_PATTERN = re.compile(
    r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)'
)
GDRIVE_OPEN_PATTERN = re.compile(
    r'drive\.google\.com/(?:open|uc)\?(?:.*&)?id=([a-zA-Z0-9_-]+)'
)
GDRIVE_FOLDER_PATTERN = re.compile(
    r'drive\.google\.com/drive(?:/u/\d+)?/folders/([a-zA-Z0-9_-]+)'
)
GDOCS_PATTERN = re.compile(
    r'docs\.google\.com/(document|spreadsheets|presentation)/d/([a-zA-Z0-9_-]+)'
)
GDRIVE_SCHEME_PATTERN = re.compile(
    r'^gdrive://([a-zA-Z0-9_-]+)$'
)

# Export MIME types for Google Workspace files
WORKSPACE_EXPORT_TYPES = {
    "application/vnd.google-apps.document": {
        "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "ext": ".docx",
    },
    "application/vnd.google-apps.spreadsheet": {
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "ext": ".xlsx",
    },
    "application/vnd.google-apps.presentation": {
        "mime": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "ext": ".pptx",
    },
    "application/vnd.google-apps.drawing": {
        "mime": "application/pdf",
        "ext": ".pdf",
    },
}


def parse_gdrive_reference(ref: str) -> Optional[dict]:
    """
    Parse a file reference to determine if it's a GDrive link.

    Returns None for local paths, or a dict with:
      - type: "file" | "folder" | "workspace_doc"
      - id: the GDrive file/folder ID
      - doc_type: "document" | "spreadsheets" | "presentation" (for workspace docs)

    Examples:
        parse_gdrive_reference("/local/file.pdf") → None
        parse_gdrive_reference("https://drive.google.com/file/d/abc123/view")
            → {"type": "file", "id": "abc123"}
        parse_gdrive_reference("https://docs.google.com/spreadsheets/d/xyz789/edit")
            → {"type": "workspace_doc", "id": "xyz789", "doc_type": "spreadsheets"}
        parse_gdrive_reference("gdrive://abc123")
            → {"type": "file", "id": "abc123"}
    """
    ref = ref.strip()

    # Local path — not a GDrive reference
    if os.path.exists(ref) or ref.startswith("/") or ref.startswith("./"):
        return None

    # gdrive:// scheme
    m = GDRIVE_SCHEME_PATTERN.match(ref)
    if m:
        return {"type": "file", "id": m.group(1)}

    # Google Docs/Sheets/Slides URL
    m = GDOCS_PATTERN.search(ref)
    if m:
        return {
            "type": "workspace_doc",
            "id": m.group(2),
            "doc_type": m.group(1),
        }

    # Standard file URL
    m = GDRIVE_FILE_PATTERN.search(ref)
    if m:
        return {"type": "file", "id": m.group(1)}

    # Open URL (?id=)
    m = GDRIVE_OPEN_PATTERN.search(ref)
    if m:
        return {"type": "file", "id": m.group(1)}

    # Folder URL
    m = GDRIVE_FOLDER_PATTERN.search(ref)
    if m:
        return {"type": "folder", "id": m.group(1)}

    # Query param id= (fallback)
    parsed = urlparse(ref)
    if "google.com" in parsed.netloc:
        params = parse_qs(parsed.query)
        if "id" in params:
            return {"type": "file", "id": params["id"][0]}

    return None


# ─── Google Drive client ─────────────────────────────────────────────────

class GDriveClient:
    """
    Thin wrapper around the Google Drive API for file downloading.

    Handles authentication, file metadata retrieval, content download,
    and Google Workspace file export (Docs→DOCX, Sheets→XLSX, etc.).

    Authentication priority:
      1. Explicit credentials passed to constructor
      2. GOOGLE_SERVICE_ACCOUNT_KEY env var (path to JSON key file)
      3. GOOGLE_APPLICATION_CREDENTIALS env var (standard GCP approach)
      4. GOOGLE_CLIENT_SECRETS env var (OAuth flow for interactive use)
    """

    def __init__(self, credentials=None):
        self._service = None
        self._credentials = credentials

    def _get_service(self):
        """Lazy-initialize the Drive API service."""
        if self._service is not None:
            return self._service

        try:
            from googleapiclient.discovery import build
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError(
                "Google Drive integration requires:\n"
                "  pip install google-api-python-client google-auth google-auth-oauthlib\n"
            )

        creds = self._credentials

        if creds is None:
            # Try service account key
            key_path = os.environ.get(
                "GOOGLE_SERVICE_ACCOUNT_KEY",
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
            )
            if key_path and os.path.exists(key_path):
                creds = service_account.Credentials.from_service_account_file(
                    key_path,
                    scopes=["https://www.googleapis.com/auth/drive.readonly"],
                )
                logger.info("Authenticated via service account: %s", key_path)
            else:
                # Try OAuth client secrets (interactive)
                client_secrets = os.environ.get("GOOGLE_CLIENT_SECRETS", "")
                if client_secrets and os.path.exists(client_secrets):
                    creds = self._oauth_flow(client_secrets)
                else:
                    raise RuntimeError(
                        "No Google credentials found. Set one of:\n"
                        "  GOOGLE_SERVICE_ACCOUNT_KEY=/path/to/key.json\n"
                        "  GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json\n"
                        "  GOOGLE_CLIENT_SECRETS=/path/to/client_secrets.json\n"
                    )

        self._service = build("drive", "v3", credentials=creds)
        return self._service

    def _oauth_flow(self, client_secrets_path: str):
        """Run OAuth flow for interactive authentication."""
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(
            client_secrets_path,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        creds = flow.run_local_server(port=0)
        logger.info("Authenticated via OAuth flow")
        return creds

    def get_file_metadata(self, file_id: str) -> dict:
        """Get metadata for a file (name, mimeType, size)."""
        service = self._get_service()
        return service.files().get(
            fileId=file_id,
            fields="id,name,mimeType,size",
            supportsAllDrives=True,          # shared-drive files: without it the API answers "not found"
        ).execute()

    def download_file(self, file_id: str, dest_path: str, name_prefix: str = "") -> str:
        """
        Download a file from GDrive to a local path.

        For Google Workspace files (Docs, Sheets, Slides), exports
        to the appropriate Office format (DOCX, XLSX, PPTX).

        Returns the actual path written (may differ from dest_path
        if the extension was adjusted for export).
        """
        from googleapiclient.http import MediaIoBaseDownload
        import io

        service = self._get_service()
        meta = self.get_file_metadata(file_id)
        mime_type = meta.get("mimeType", "")
        name = meta.get("name", file_id)

        # Google Workspace file → export
        if mime_type in WORKSPACE_EXPORT_TYPES:
            export_info = WORKSPACE_EXPORT_TYPES[mime_type]
            export_mime = export_info["mime"]
            ext = export_info["ext"]

            # Adjust filename
            stem = Path(name).stem
            actual_path = str(Path(dest_path).parent / f"{name_prefix}{stem}{ext}")

            logger.info(
                "Exporting workspace file %s (%s) as %s",
                name, mime_type, ext,
            )

            request = service.files().export_media(
                fileId=file_id,
                mimeType=export_mime,
            )
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            with open(actual_path, "wb") as f:
                f.write(fh.getvalue())

            logger.info("Exported to: %s (%d bytes)", actual_path, len(fh.getvalue()))
            return actual_path

        # Regular file → direct download
        actual_path = str(Path(dest_path).parent / f"{name_prefix}{name}")

        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        with open(actual_path, "wb") as f:
            f.write(fh.getvalue())

        logger.info("Downloaded: %s (%d bytes)", actual_path, len(fh.getvalue()))
        return actual_path

    def list_folder(self, folder_id: str) -> list[dict]:
        """List all files in a GDrive folder (non-recursive)."""
        service = self._get_service()
        results = []
        page_token = None

        while True:
            # Shared drives: without both flags the API silently returns NO files for a shared-drive folder
            # (Oct 2026: the EDGAR task folders listed 0 files here and 10 with the flags).
            response = service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken,files(id,name,mimeType,size)",
                pageToken=page_token,
                pageSize=100,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()

            results.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        logger.info("Listed %d files in folder %s", len(results), folder_id)
        return results


# ─── The File Resolver ───────────────────────────────────────────────────

class FileResolver:
    """
    Resolves file references to local paths.

    Takes a list of mixed references (local paths, GDrive URLs, file IDs)
    and returns a list of local file paths in a staging directory.

    Usage:
        resolver = FileResolver(staging_dir="/tmp/eval_staging")
        local_paths = resolver.resolve([
            "/local/data.csv",
            "https://drive.google.com/file/d/abc123/view",
            "gdrive://xyz789",
            "https://docs.google.com/spreadsheets/d/sheet_id/edit",
        ])
        # local_paths = [
        #   "/local/data.csv",
        #   "/tmp/eval_staging/report.pdf",
        #   "/tmp/eval_staging/analysis.docx",
        #   "/tmp/eval_staging/financials.xlsx",
        # ]
    """

    def __init__(
        self,
        staging_dir: Optional[str] = None,
        gdrive_client: Optional[GDriveClient] = None,
    ):
        self.staging_dir = staging_dir or tempfile.mkdtemp(prefix="dra_staging_")
        self._gdrive: Optional[GDriveClient] = gdrive_client
        os.makedirs(self.staging_dir, exist_ok=True)

    def _get_gdrive(self) -> GDriveClient:
        """Lazy-initialize GDrive client (only when needed)."""
        if self._gdrive is None:
            self._gdrive = GDriveClient()
        return self._gdrive

    def resolve(self, file_refs: list[str]) -> list[str]:
        """
        Resolve a list of file references to local paths.

        Local paths are passed through unchanged.
        GDrive references are downloaded to the staging directory.
        Returns a list of local file paths (same order as input,
        with folder references expanded).
        """
        resolved = []

        for ref in file_refs:
            gdrive_info = parse_gdrive_reference(ref)

            if gdrive_info is None:
                # Local path — pass through
                if os.path.exists(ref):
                    resolved.append(ref)
                else:
                    logger.warning("Local file not found: %s", ref)
                continue

            try:
                if gdrive_info["type"] == "folder":
                    # Download all files in the folder
                    folder_paths = self._resolve_folder(gdrive_info["id"])
                    resolved.extend(folder_paths)
                else:
                    # Download single file
                    path = self._resolve_file(gdrive_info["id"])
                    if path:
                        resolved.append(path)
            except Exception as e:
                logger.error(
                    "Failed to resolve GDrive ref %s: %s", ref, e,
                )

        logger.info(
            "Resolved %d references → %d local files",
            len(file_refs), len(resolved),
        )
        return resolved

    def _resolve_file(self, file_id: str) -> Optional[str]:
        """Download a single file from GDrive."""
        client = self._get_gdrive()
        dest = os.path.join(self.staging_dir, file_id)
        return client.download_file(file_id, dest)

    def _resolve_folder(self, folder_id: str, prefix: str = "") -> list[str]:
        """Download all files from a GDrive folder, recursing into sub-folders."""
        client = self._get_gdrive()
        files = client.list_folder(folder_id)

        paths = []
        for f in files:
            # Sub-folder → recurse with folder name as prefix
            if f.get("mimeType") == "application/vnd.google-apps.folder":
                folder_name = f.get("name", f["id"])
                child_prefix = f"{prefix}{folder_name}__"
                paths.extend(self._resolve_folder(f["id"], prefix=child_prefix))
                continue

            try:
                dest = os.path.join(self.staging_dir, f["id"])
                path = client.download_file(f["id"], dest, name_prefix=prefix)
                paths.append(path)
            except Exception as e:
                logger.error("Failed to download %s: %s", f.get("name"), e)

        return paths

    def cleanup(self):
        """Remove the staging directory and all downloaded files."""
        if os.path.isdir(self.staging_dir):
            shutil.rmtree(self.staging_dir)
            logger.info("Cleaned up staging dir: %s", self.staging_dir)


# ─── Convenience function ────────────────────────────────────────────────

def resolve_files(
    file_refs: list[str],
    staging_dir: Optional[str] = None,
) -> list[str]:
    """
    One-shot file resolution.

    Usage:
        local_paths = resolve_files([
            "/local/file.csv",
            "https://drive.google.com/file/d/abc123/view",
        ])
    """
    resolver = FileResolver(staging_dir=staging_dir)
    return resolver.resolve(file_refs)