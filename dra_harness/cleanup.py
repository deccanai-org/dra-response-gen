"""
cleanup.py — Delete input files once runs are done, to save disk space.

Inputs are downloaded once into the batch's staging folder and symlinked into each run folder
(runner._stage_input_files). Agents sometimes also write real copies of inputs into their run
folder. After a run, and again after the batch, this module removes:

  per run folder   input symlinks; regular files carrying an input's name; files that are
                   byte-identical copies of an input under another name. Deliverables (the run's
                   output_files) are never touched, and nothing outside the run folder is.
  per batch        the downloaded inputs, but ONLY those located inside the batch's staging folder.
                   Local input paths given in the CSV (e.g. a mirror on disk) are passed through by the
                   resolver and are never deleted.

Every function is idempotent and never raises: a cleanup problem is logged, never fatal.
"""

from __future__ import annotations

import os
import hashlib
import logging

logger = logging.getLogger("dra.cleanup")


def _sha256(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _inside(path: str, root: str) -> bool:
    try:
        p, r = os.path.realpath(path), os.path.realpath(root)
        return os.path.commonpath([p, r]) == r and p != r
    except ValueError:                      # different drives on Windows
        return False


def input_fingerprints(input_paths: list) -> tuple[set, dict]:
    """(input basenames, {size: {sha256, ...}}) for the run's inputs that still exist."""
    names, by_size = set(), {}
    for p in input_paths or []:
        names.add(os.path.basename(p))
        try:
            if os.path.isfile(p):
                h = _sha256(p)
                if h:
                    by_size.setdefault(os.path.getsize(p), set()).add(h)
        except OSError:
            continue
    return names, by_size


def clean_run_dir(run_dir: str, input_paths: list, keep: list | None = None,
                  fingerprints: tuple | None = None) -> dict:
    """Remove input files from one run folder. Returns {"removed": n, "bytes": b}."""
    stats = {"removed": 0, "bytes": 0}
    if not run_dir or not os.path.isdir(run_dir):
        return stats
    keep_real = {os.path.realpath(k) for k in (keep or [])}
    names, by_size = fingerprints or input_fingerprints(input_paths)
    for root, _dirs, files in os.walk(run_dir):
        for name in files:
            full = os.path.join(root, name)
            try:
                if os.path.islink(full):
                    target = os.readlink(full)
                    if name in names or os.path.basename(target) in names:
                        os.unlink(full); stats["removed"] += 1
                    continue
                if os.path.realpath(full) in keep_real:
                    continue                                   # a reported deliverable
                size = os.path.getsize(full)
                is_input = name in names or (size in by_size and _sha256(full) in by_size[size])
                if is_input:
                    os.remove(full); stats["removed"] += 1; stats["bytes"] += size
            except OSError as e:
                logger.warning("cleanup: could not remove %s: %s", full, e)
    return stats


def clean_staged_inputs(input_paths: list, staging_dir: str) -> dict:
    """Remove downloaded inputs that live inside the batch's staging folder (never anything else)."""
    stats = {"removed": 0, "bytes": 0, "skipped_outside": 0}
    for p in sorted(set(input_paths or [])):
        try:
            if not os.path.isfile(p) or os.path.islink(p):
                continue
            if not _inside(p, staging_dir):
                stats["skipped_outside"] += 1              # a local input given in the CSV: never delete
                continue
            size = os.path.getsize(p)
            os.remove(p); stats["removed"] += 1; stats["bytes"] += size
        except OSError as e:
            logger.warning("cleanup: could not remove %s: %s", p, e)
    return stats