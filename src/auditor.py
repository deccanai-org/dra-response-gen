# src/auditor.py
"""
Two-call task auditor for the DRA prompt-QC pipeline.

Flow per task:
  0. Parse + gate the existing verifiers (verifier_parser). Uncertain -> flag.
  1. CALL 1 (Opus): audit + emit structured arithmetic claims + preliminary findings.
  2. CODE: recompute the claims (arithmetic_verifier). Authoritative for numbers.
  3. Short-circuit: if call 1 is clean (SOUND, no claims errored, no findings),
     skip call 2 and emit a SOUND result.
  4. CALL 2 (Opus): finalize verdict + corrections, grounded in the code results.
  5. Assemble AuditResult (JSON-serializable) + prose.

Column access is by HEADER NAME via a resolver (never by index).

The Opus judge has no fallback (hard-stop) — see prompt_evaluator.MODEL_FALLBACK_CHAIN.
"""

from __future__ import annotations

import os, re

#: Verifier-id pattern shared across the pipeline — classic (V1, V5a) AND
#: semantic (V_P1_pat_2004, V_P1_v2.0) ids some synthetic pipelines emit.
_VID = r"V(?:\d+[a-z]?|_[A-Za-z0-9][A-Za-z0-9_.]*)"
_VID_LINE = (r"\s*(?P<vid>" + _VID + r")\s*(?:\[[^\]]*\])?"
             r"\s*(?::\s*|\s+-\s+)(?P<text>.*)")
_VID_BARE = r"\b" + _VID + r"\b"

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

from src.prompt_evaluator import _call_llm, DEFAULT_JUDGE_MODEL, _repair_and_parse_json
from src.auditor_templates import (
    AUDITOR_SYSTEM_PROMPT,
    AUDIT_CLAIM_TEMPLATE,
    CORRECTION_TEMPLATE,
)
from src.arithmetic_verifier import (
    ArithmeticClaim,
    verify_claims,
    summarize,
)
from src.verifier_parser import parse_verifiers
from src.verifier_qc import run_verifier_qc, qc_summary
from src.verifier_grammar import derive_expected_values
from src.derive_dag import (claim_graph, graph_health, map_verifiers_to_steps,
                            step_coverage)
from src.verifier_audit import (audit_verifiers, apply_rewrites, apply_splits,
                                format_verifiers_ids)

logger = logging.getLogger("dra.auditor")

# Output token ceilings. These are safety rails against truncation, NOT budgets —
# Opus 4.8 supports up to 128K output, and you pay only for tokens actually
# generated, so a high ceiling you don't hit is free. They are set generously so
# a legitimate audit never truncates, but low enough that a response which
# actually hits the ceiling trips `likely_truncated` and signals an abnormally
# long (rambling / over-claiming) response worth investigating rather than
# silently absorbing. Call 1 (analysis + all claims + findings) is the large one;
# Call 2 (corrections) is smaller.
#: Opus 4.8 allows up to 128,000 output tokens on the Messages API, and
#: _call_llm already streams (required above roughly 16k to avoid a request
#: timeout). Adaptive thinking shares this budget with the visible text, so the
#: figure must cover BOTH — which is why these are far above the size of the JSON
#: alone. Ceiling left under 128k for headroom.
#: https://platform.claude.com/docs/en/about-claude/models/overview
MAX_OUTPUT_CEILING = 96000

CALL1_MAX_TOKENS = 32000
#: Call 2 now re-emits EVERY claim with links, plus judgment steps, plus four
#: full corrected artifacts (solution logic, prompt, sanity check, verifiers),
#: plus changes and findings. On a 60-verifier task that dwarfs the old 16k.
CALL2_BASE_TOKENS = 24000
CALL2_PER_VERIFIER = 350
CALL2_PER_CLAIM = 400


def call2_token_budget(n_verifiers: int, n_claims: int) -> int:
    """Size call 2 from what it actually has to emit, not a fixed guess."""
    return min(MAX_OUTPUT_CEILING,
               CALL2_BASE_TOKENS + CALL2_PER_VERIFIER * max(n_verifiers, 0)
               + CALL2_PER_CLAIM * max(n_claims, 0))

# Verdicts that mean "fit to proceed downstream" (scoring/augmentation)
PROCEEDABLE = {"SOUND", "SALVAGEABLE"}


# ---------------------------------------------------------------------------
# Header resolution (by name, tolerant of case/whitespace variants)
# ---------------------------------------------------------------------------

# canonical field -> list of acceptable header spellings (lowercased, stripped)
_HEADER_ALIASES = {
    "task_id": ["task_id", "task id", "id"],
    "prompt": ["prompt", "prompt text"],
    # Post-augmentation delivery CSVs carry the CORRECTED artifacts under
    # different names than the raw authoring sheet, so both spellings resolve.
    "sanity_check": ["sanity check", "sanity_check", "sc",
                     "corrected_sanity_check", "corrected sanity check"],
    "solution_logic": ["solution logic", "solution_logic", "logic",
                       "golden_solution_logic", "golden solution logic",
                       "corrected_solution_logic", "corrected solution logic"],
    "json": ["json", "json_path", "package_json", "sidecar"],
    "verifiers": ["verifiers", "verifier", "verifiers ",
                  "augmented_verifiers", "augmented verifiers"],
    "golden_deliverable": ["golden_deliverable", "golden deliverable"],
    "drive_link": ["drive link", "drive_link", "drive_url", "drive url", "gdrive", "drive",
                   "input_files_link", "input files link"],
    "prompt_type": ["prompt type", "prompt_type", "type"],
    "final_task_tags": ["final task tags", "final_task_tags", "task tags"],
}


class HeaderError(KeyError):
    pass


def read_task_csv(path: str) -> Tuple[List[str], List[dict]]:
    """Read a task CSV, tolerating a merged banner row above the real header.

    Spreadsheet exports of the delivery sheet carry a merged cell — "To be filled
    by POCs" — spanning several columns on the first row. Read with the default
    header that becomes one named column and fourteen "Unnamed:" ones, which a
    phantom-column filter then strips, leaving a single useless header. This has
    now bitten three separate entry points, so it lives in one place.

    Also drops genuinely empty trailing columns, which Excel exports do leave.
    """
    import pandas as pd

    best = None
    for hdr in (0, 1, 2):
        df = pd.read_csv(path, header=hdr, dtype=str).fillna("")
        cols = [str(c).replace("\ufeff", "").replace("\u200b", "").strip()
                for c in df.columns]
        named = [c for c in cols
                 if c and not c.lower().startswith("unnamed") and not c.startswith(".")]
        if best is None or len(named) > len(best[2]):
            best = (hdr, cols, named, df)
        # a real header row has most of its columns named
        if len(named) > len(cols) * 2 // 3:
            break

    hdr, cols, named, df = best
    df.columns = cols
    keep = [c for c in cols
            if c and not c.lower().startswith("unnamed") and not c.startswith(".")]
    rows = [{k: r.get(k, "") for k in keep} for r in df.to_dict("records")]
    return keep, rows


def build_header_map(headers: List[str]) -> Dict[str, str]:
    """Map canonical field names to the actual header strings present.
    Raises HeaderError listing any REQUIRED field that cannot be resolved."""
    norm = { (h or "").strip().lower(): h for h in headers }
    resolved: Dict[str, str] = {}
    for canon, aliases in _HEADER_ALIASES.items():
        for a in aliases:
            if a in norm:
                resolved[canon] = norm[a]
                break
    required = ["task_id", "prompt", "sanity_check", "solution_logic", "verifiers"]
    missing = [r for r in required if r not in resolved]
    if missing:
        raise HeaderError(
            f"Required column(s) not found by header name: {missing}. "
            f"Present headers: {list(headers)}"
        )
    return resolved


def _parse_v_lines(block: str) -> List[dict]:
    """Parse a 'V<n>[a-z]: text' block into [{'id','text'}], preserving SUFFIXED
    ids (V5a, V5b). Unlike verifier_parser.parse_verifiers, which extracts an
    integer index and would turn 'V5a: ...' into id 'V5' with 'a:' leaking into
    the text, this keys on the full id token so split children survive a re-parse.

    TOLERANT to the two non-canonical formats the SEED verifier column ships in,
    because that column is authored by hand and is not guaranteed canonical:
      * a DASH separator  -- "V1 - text"  (seen on most tasks)
      * LOWERCASE ids     -- "v1: text"   (seen on others)
    Both parsed to ZERO verifiers under the old colon-only, upper-V-only pattern,
    which silently dropped the entire authored seed set and left only the
    augment-added verifiers downstream (the cause of the collapsed verifier DAG /
    crux). The id is normalized to an uppercase 'V' so the rest of the pipeline —
    which keys on 'V\\d+[a-z]?' — matches it. A trailing comma (the seed uses
    "..., \\n" between entries) is stripped.

    Separator rules, chosen to avoid false splits on verifier PROSE:
      * colon  -> optional trailing space ("V4:Gas" and "V4: Gas" both parse)
      * dash   -> REQUIRES surrounding spaces (" - "), so an internal hyphen
                  ("Only L1-L4 rows") or a negative-number clause inside the text
                  can never be mistaken for the id/text separator.
    """
    out = []
    for line in (block or "").splitlines():
        m = re.match(_VID_LINE, line)
        if m:
            vid = m.group("vid")
            if re.match(r"[Vv]\d", vid):
                vid = "V" + vid[1:]
            text = m.group("text").strip().rstrip(",").strip()
            if text:
                out.append({"id": vid, "text": text})
    return out


def get_field(row: dict, header_map: Dict[str, str], canon: str, default: str = "") -> str:
    col = header_map.get(canon)
    if col is None:
        return default
    val = row.get(col, default)
    return "" if val is None else str(val)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    task_id: str
    verdict: str = "UNKNOWN"
    primary_reason: str = ""
    proceedable: bool = False
    decision_inversion: bool = False

    # verifier gate
    verifier_parse_status: str = ""          # CLEAN | UNCERTAIN | EMPTY
    verifier_count: int = 0
    verifier_parse_reasons: List[str] = field(default_factory=list)
    verifier_qc_findings: List[dict] = field(default_factory=list)
    verifier_qc_summary: Dict[str, Any] = field(default_factory=dict)

    # arithmetic
    arithmetic_summary: Dict[str, Any] = field(default_factory=dict)
    claim_verdicts: List[dict] = field(default_factory=list)
    #: What the auditor could actually read. Qualifies every finding computed
    #: over the input files — above all leakage, which is an absence claim.
    input_coverage: dict = field(default_factory=dict)

    # input-file provenance: did the auditor actually receive the model-facing
    # input files, and did the source-checking (C-layer) run?
    input_files_supplied: bool = False
    input_files_names: List[str] = field(default_factory=list)
    provenance_checked: bool = False
    provenance_expected_not_run: bool = False
    provenance_note: str = ""

    # corrections + findings
    corrected_solution_logic: str = ""
    #: Full replacement text, empty when unchanged. The prompt and sanity check
    #: were previously correctable only as diff fragments inside `changes`, so
    #: two of the three artifacts the auditor may correct had no reconstructed
    #: version anywhere.
    corrected_prompt: str = ""
    corrected_sanity_check: str = ""
    #: Call 2 already proposed verifier corrections in `changes` but there was
    #: nowhere to put the result, so a verifier left pinning a figure the
    #: correction changed stayed stale. Verifiers are structured, so unlike the
    #: prose artifacts the actual diff is computable — see verifier_change_audit.
    corrected_verifiers: str = ""
    verifier_change_audit: dict = field(default_factory=dict)
    #: The derivation. Claims re-emitted by call 2 with corrections applied and
    #: `from_claim` links, then recomputed a SECOND time. `claim_verdicts` above
    #: describes the ORIGINAL solution logic, which call 2 may have replaced —
    #: verifying once and correcting afterwards left the verification describing
    #: an artifact that no longer existed.
    corrected_claim_verdicts: List[dict] = field(default_factory=list)
    judgment_steps: List[dict] = field(default_factory=list)
    corrected_arithmetic_summary: dict = field(default_factory=dict)

    # --- finalized verifier set (1b: the property audit now runs HERE, so the
    #     auditor returns a verifier set that is already atomic/split, with
    #     targets frozen from the verifier TEXT and the trajectory derived from
    #     the corrected claims. augment_task consumes this instead of authoring
    #     it, which removes the Call-2 -> augment -> property-audit cycle). All
    #     additive: no existing field changes, so audit_report / run_audit_check
    #     keep working unchanged.
    final_verifiers_text: str = ""            # canonical 'V<n>: ...' after splits
    expected_values: Dict[str, dict] = field(default_factory=dict)
    step_graph: Dict[str, list] = field(default_factory=dict)
    step_nodes: Dict[str, dict] = field(default_factory=dict)
    step_graph_health: dict = field(default_factory=dict)
    verifier_to_step: Dict[str, str] = field(default_factory=dict)
    verifier_mapping_report: dict = field(default_factory=dict)
    step_coverage_report: dict = field(default_factory=dict)
    verifier_audit: dict = field(default_factory=dict)
    verifier_splits_applied: List[dict] = field(default_factory=list)
    verifier_rewrites_applied: List[str] = field(default_factory=list)
    target_grammar_problems: List[dict] = field(default_factory=list)
    #: Deterministic gate over the FINAL derivation. Distinct from `proceedable`,
    #: which was the model's own verdict and which nothing enforced.
    gate: dict = field(default_factory=dict)
    #: Claims call 2 emitted without an operation or without input values. These
    #: are unverifiable for a reason that has nothing to do with the golden.
    malformed_claims: List[dict] = field(default_factory=list)
    #: Outcome of the one-shot repair pass for malformed claims (empty operation
    #: / valueless input). Empty when nothing needed repair.
    malformed_repair_note: str = ""
    findings_note: str = ""
    changes: List[dict] = field(default_factory=list)
    findings: List[dict] = field(default_factory=list)
    leakage_findings: List[dict] = field(default_factory=list)
    temporal_drift_findings: List[dict] = field(default_factory=list)
    missing_inputs: List[dict] = field(default_factory=list)
    trap_assessment: Dict[str, Any] = field(default_factory=dict)

    prose_findings: str = ""

    # provenance / debugging
    model_used: str = ""
    calls_made: int = 0
    skipped_call2: bool = False
    error: str = ""
    raw_response_debug: str = ""
    raw_response_len: int = 0
    likely_truncated: bool = False

    def to_dict(self, include_raw: bool = False) -> dict:
        d = asdict(self)
        if not include_raw:
            # Keep the bulky raw model dump out of the JSON; it's written to a
            # separate .raw.txt file by the CLI. Retain the len/truncation flags.
            d.pop("raw_response_debug", None)
        return d

    def to_json(self, indent: int = 2, include_raw: bool = False) -> str:
        return json.dumps(self.to_dict(include_raw=include_raw),
                          indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

#: gdrive_raw_fetcher writes this marker into a file's text when the middle was
#: cut. The auditor is handed the assembled text, not the fetcher's return value,
#: so truncation is recovered from the text itself — that works no matter who
#: assembled it and needs no signature change to the fetcher.
_TRUNC_MARKER = "chars truncated (NOT summarized)"


def input_coverage(input_files_text: str,
                   input_files_names: Optional[List[str]] = None,
                   skipped_inputs: Optional[List[str]] = None,
                   files_expected: bool = False) -> dict:
    """What the auditor could and could not see.

    Leakage and missing-input findings are only as good as the text they were
    computed over. Without this, a clean leakage result on a task with an
    unreadable screenshot is textually identical to a clean result on a task
    where everything was read.
    """
    truncated: List[str] = []
    for section in (input_files_text or "").split("### File: ")[1:]:
        name = section.split("\n", 1)[0].strip()
        if _TRUNC_MARKER in section:
            truncated.append(name)
    read = list(input_files_names or [])
    skipped = list(skipped_inputs or [])
    # A task that DECLARES input files but yielded none did not have "complete"
    # coverage — it had none at all. Before this, a fetch failure produced
    # files_read=[] and skipped=[], so complete came back True and a golden built
    # on no data looked fully audited. That happened to an entire batch when a
    # /u/1/ account segment in the folder URL made every reference unparseable.
    fetch_failed = bool(files_expected) and not read
    ok = not skipped and not truncated and not fetch_failed
    return {
        "files_read": read,
        "files_skipped": skipped,
        "files_truncated": truncated,
        "files_expected": bool(files_expected),
        "fetch_failed": fetch_failed,
        "complete": ok,
        "note": ("" if ok else
                 ("Input files were declared for this task but NONE were read, so "
                  "every figure in the golden is unverifiable against source. "
                  "Check the drive reference and the service-account share. "
                  if fetch_failed else "") +
                 ("Findings below were computed over PARTIAL input. "
                  if (skipped or truncated) else "")
                 + (f"Unreadable: {skipped}. " if skipped else "")
                 + (f"Truncated (middle omitted): {truncated}. " if truncated else "")
                 + "A clean result here does not mean no defect exists in the "
                   "unseen material."),
    }


#: Only a genuine arithmetic failure blocks.
#:
#: INPUT_ERROR deliberately does NOT. It means "a declared input value was not
#: found verbatim in the source files", and on the first two real tasks EVERY
#: blocker was an INPUT_ERROR and NOT ONE was a defect: the correction pass
#: legitimately collapses a sum of eight raw arrivals into one computed
#: intermediate (170), which appears in no file by construction. Blocking on it
#: gave a false-block rate of 2 out of 2. It is now a warning, and the real fix
#: is upstream — the correction prompt requires a computed value to be its own
#: claim rather than an inline literal.
BLOCKING_CLAIM_STATUSES = ("ARITHMETIC_ERROR",)
#: MISLABELLED_INPUT warns for now. It is a narrower and more concerning case
#: than INPUT_ERROR — a value asserted to come from a file that is not in any
#: file, hiding an unverified computation — but it appeared on 2 of 43 inputs
#: across two tasks, which is too thin to justify blocking. To promote it, move
#: the string to BLOCKING_CLAIM_STATUSES; nothing else needs to change.
#: WRONG_PERIOD (new): the input value exists in source but under a different period than the claim
#: attributes it to (e.g. a Dec-2006 comparative used as the Sep-2007 current value). Warns for now,
#: alongside INPUT_ERROR — surfaced to the SME, not auto-blocking.
#: NOTE on INPUT_ERROR: it was demoted to a warning because the OLD provenance match was loose and
#: false-flagged computed intermediates. The provenance check has since been hardened (boundary-guarded
#: matching; derived/inline inputs already skipped), so INPUT_ERROR now reliably means a fabricated /
#: absent input. Promoting it to BLOCKING_CLAIM_STATUSES is now reasonable — left as a deliberate
#: policy choice for the team rather than changed unilaterally.
WARNING_CLAIM_STATUSES = ("INPUT_ERROR", "MISLABELLED_INPUT", "WRONG_PERIOD")


def arithmetic_gate(verdicts, input_files_supplied: bool = True,
                    derivation_available: bool = True,
                    malformed_claims: Optional[List[dict]] = None) -> dict:
    """Does the FINAL derivation reconcile?

    Runs over the corrected claims, not the original ones. UNVERIFIABLE blocks
    only when input files were actually supplied — otherwise every claim on an
    unaudited-inputs task would be unverifiable and the gate would refuse
    everything.

    A claim that is UNVERIFIABLE *because its own emission is malformed* (empty
    operation, or an input with no value) is NOT a golden-reconciliation failure —
    it is a pipeline/emission fault that says nothing about whether the golden is
    correct. Such claims are excluded from `blocking` and surfaced under
    `malformed_claims` instead, so an empty-operation claim (e.g. a round-up step
    the model left blank) no longer masquerades as a broken golden.
    """
    malformed_ids = {str(m.get("id")) for m in (malformed_claims or [])}
    blocking, warning = [], []
    for v in verdicts or []:
        st = getattr(v, "status", None) or (v.get("status") if isinstance(v, dict) else "")
        if st in WARNING_CLAIM_STATUSES:
            warning.append({"id": getattr(v, "id", None) or v.get("id"),
                            "status": st,
                            "label": getattr(v, "label", None) or v.get("label", ""),
                            "detail": (getattr(v, "detail", None)
                                       or (v.get("detail") if isinstance(v, dict) else ""))})
        elif st in BLOCKING_CLAIM_STATUSES:
            blocking.append({"id": getattr(v, "id", None) or v.get("id"),
                             "status": st,
                             "label": getattr(v, "label", None) or v.get("label", ""),
                             "detail": (getattr(v, "detail", None)
                                        or (v.get("detail") if isinstance(v, dict) else ""))})
        elif st == "UNVERIFIABLE":
            cid = getattr(v, "id", None) or v.get("id")
            entry = {"id": cid, "status": st,
                     "label": getattr(v, "label", None) or v.get("label", "")}
            # a malformed emission (empty operation / valueless input) is a
            # pipeline fault, not a golden fault — do not block on it
            if str(cid) in malformed_ids:
                continue
            (blocking if input_files_supplied else warning).append(entry)
    return {
        # A malformed emission does not reconcile the derivation either, but it
        # is a REPAIRABLE pipeline fault, not a broken golden. It fails the gate
        # (we cannot verify a step with no operation) but is reported distinctly
        # so callers can retry the emission rather than reject the task.
        "passed": (bool(derivation_available) and not blocking
                   and not malformed_claims),
        "blocking": blocking,
        "warnings": warning,
        "malformed_blocking": bool(malformed_claims),
        "n_claims": len(verdicts or []),
        # If call 2 returned no corrected claims there is no derivation to
        # judge. Running the gate over the pre-correction claims instead would
        # repeat the exact bug this phase removed — verifying an artifact that
        # was subsequently replaced — so that case fails outright rather than
        # producing a gate result with a caveat attached to it.
        "derivation_available": derivation_available,
        "malformed_claims": list(malformed_claims or []),
        "reason": (
            "" if (derivation_available and not blocking and not malformed_claims)
            else
            ("the derivation could not be rebuilt: call 2 returned no "
             "corrected claims, so there is nothing to check"
             if not derivation_available else
             (f"call 2 emitted {len(malformed_claims)} claim(s) with no "
              f"operation or no input values ("
              + ", ".join(str(m.get("id")) for m in malformed_claims)
              + ") — the emission is incomplete (a pipeline fault, repairable by "
                "re-emitting the operation), which says nothing about whether the "
                "golden is correct"
              if malformed_claims else
              f"{len(blocking)} claim(s) do not reconcile: "
              + ", ".join(f"{b['id']}({b['status']})" for b in blocking)))),
    }


def audit_verifier_changes(original_text: str, corrected_text: str,
                           changes: List[dict]) -> dict:
    """Compare the corrected verifier block against what call 2 SAID it changed.

    Prose corrections cannot be checked this way — there is no reliable diff
    between two rewritten paragraphs. Verifiers are "V<n>: text" records, so the
    real per-verifier diff IS computable, and three things worth knowing fall out:

      * edits made but never declared in `changes` (silent rewrites)
      * edits declared but not present in the block (claimed, not done)
      * a JUDGMENT_REQUIRED verifier item that WAS resolved in the block, which
        means an authorial decision was taken on the model's own authority

    Reports only. Nothing here rewrites or reverts anything.
    """
    from src.verifier_parser import parse_verifiers

    if not (corrected_text or "").strip():
        return {"corrected": False, "note": "no corrected verifier block emitted"}

    before = {v["id"]: v["text"] for v in _parse_v_lines(original_text)}
    after = {v["id"]: v["text"] for v in _parse_v_lines(corrected_text)}

    changed = sorted(k for k in before.keys() & after.keys()
                     if before[k] != after[k])
    declared: Dict[str, str] = {}
    for ch in changes or []:
        if str(ch.get("artifact", "")).lower() != "verifiers":
            continue
        for vid in re.findall(_VID_BARE, str(ch.get("location", ""))
                              + " " + str(ch.get("old", ""))):
            declared[vid] = str(ch.get("type", ""))

    judgment_resolved = [v for v in changed
                         if declared.get(v) == "JUDGMENT_REQUIRED"]
    return {
        "corrected": True,
        "n_before": len(before), "n_after": len(after),
        "changed": changed,
        "declared": declared,
        "undeclared_edits": [v for v in changed if v not in declared],
        "declared_not_done": sorted(v for v in declared if v not in changed),
        "judgment_resolved_silently": judgment_resolved,
        "dropped": sorted(before.keys() - after.keys()),
        "added": sorted(after.keys() - before.keys()),
        "note": ("; ".join(filter(None, [
            f"{len(changed)} verifier(s) edited" if changed else "",
            f"JUDGMENT items resolved without asking: {judgment_resolved}"
            if judgment_resolved else "",
            f"verifiers dropped: {sorted(before.keys() - after.keys())}"
            if before.keys() - after.keys() else ""]))),
    }


def audit_task(
    row: dict,
    header_map: Dict[str, str],
    input_files_text: str = "",
    #: When given, the PROMPT uses corpus.prompt_view (an index plus excerpts)
    #: while provenance searches corpus.full_text. Those are different needs and
    #: were previously the same truncated string: on a 21.8 MB annual report the
    #: auditor saw 2.7% of the corpus and 6 of 26 findable values came back as
    #: false "not found in source".
    input_corpus=None,
    input_files_names: Optional[List[str]] = None,
    model_name: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 8000,
    skipped_inputs: Optional[List[str]] = None,
    run_verifier_audit: bool = True,
) -> AuditResult:
    """Run the two-call audit on one task row. input_files_text is the extracted
    text of the model-facing input files (e.g. from gdrive_fetcher); pass "" if
    not available — the audit still runs on prompt/logic/SC."""

    task_id = get_field(row, header_map, "task_id") or "(no id)"
    prompt_text = get_field(row, header_map, "prompt")
    sanity_check_text = get_field(row, header_map, "sanity_check")
    solution_logic_text = get_field(row, header_map, "solution_logic")
    verifiers_text = get_field(row, header_map, "verifiers")
    prompt_type = get_field(row, header_map, "prompt_type") or "(unspecified)"

    # --- Prefer a sidecar JSON when the row points to one ------------------------------------------
    # A large task's Verifiers / Solution Logic cell can exceed Excel's 32,767-char cell cap; if the
    # CSV was opened+saved in Excel the cell is truncated and the auditor would silently audit a
    # partial verifier set. If the row carries a `json` column with the task's package path, read the
    # authoritative full text from there instead — it has no cell limit and is round-trip-proof.
    _json_path = get_field(row, header_map, "json")
    if _json_path and os.path.exists(_json_path):
        try:
            _pkg = json.load(open(_json_path, encoding="utf-8"))
            _dr = _pkg.get("draft", _pkg)
            def _fmt_claims(cl):
                out=[]
                for c in cl:
                    ins=", ".join(f"{i.get('name')}={i.get('value')}" for i in (c.get('inputs') or []))
                    out.append(f"{c.get('id')} — {c.get('label','')}: {c.get('operation','')} = "
                               f"{c.get('claimed_result','')}" + (f" [inputs: {ins}]" if ins else "")
                               + (f" (source: {c.get('source','')})" if c.get('source') else ""))
                return "\n".join(out)
            def _fmt_vers(vs):
                out=[]
                for v in vs:
                    crux=" [crux]" if v.get("crux") else ""
                    out.append(f"{v.get('id')}{crux}: {v.get('check','')}")
                return "\n".join(out)
            if _dr.get("prompt"): prompt_text = _dr["prompt"]
            if _dr.get("solution_logic"): solution_logic_text = _fmt_claims(_dr["solution_logic"])
            if _dr.get("verifiers"): verifiers_text = _fmt_vers(_dr["verifiers"])
            if _dr.get("sanity_check"): sanity_check_text = _dr.get("sanity_check","")
            # EMBEDDED SOURCE: if the package carries source_text (embed_source.py) and the caller
            # gave no input_files_text, use it — so a JSON copied to another machine still runs
            # provenance/grounding with no external files or Drive.
            if not input_files_text and _dr.get("source_text"):
                input_files_text = _dr["source_text"]
        except Exception as _e:
            pass   # fall back to CSV cells on any read/parse error

    result = AuditResult(task_id=task_id)
    # search_text is what CODE greps; prompt_text is what the model reads.
    if input_corpus is not None and getattr(input_corpus, "full_text", ""):
        search_text = input_corpus.full_text
        prompt_text_files = input_corpus.prompt_view
        if not input_files_names:
            input_files_names = [f["name"] for f in (input_corpus.files or [])
                                 if f.get("extracted")]
        if not skipped_inputs:
            skipped_inputs = list(input_corpus.skipped or [])
    else:
        search_text = input_files_text
        prompt_text_files = input_files_text
    result.input_files_supplied = bool(search_text and search_text.strip())
    result.input_files_names = input_files_names or []
    if not result.input_files_supplied:
        result.provenance_note = (
            "Input files were NOT supplied to the auditor; the source-checking "
            "(provenance) layer did not run. Arithmetic was checked only for "
            "internal consistency, not against the actual input data. Declared "
            "input values are UNVERIFIED against source."
        )

    # --- Stage 0: verifier gate + QC checklist ---------------------------
    vparse = parse_verifiers(verifiers_text)
    result.verifier_parse_status = vparse.status
    result.verifier_count = vparse.count
    result.verifier_parse_reasons = vparse.reasons

    # Deterministic verifier-QC checklist (R2/R6/R_DIM/S2/...). Runs on the
    # parsed records when available; on the raw split otherwise.
    if vparse.records:
        v_texts = [r.text for r in vparse.records]
    else:
        v_texts = [s.strip() for s in (verifiers_text or "").splitlines() if s.strip()]
    qc_findings = run_verifier_qc(v_texts)
    result.verifier_qc_findings = [f.to_dict() for f in qc_findings]
    result.verifier_qc_summary = qc_summary(qc_findings)

    # --- Stage 1: audit + claims (call 1) --------------------------------
    # Format QC findings for the prompt
    if result.verifier_qc_findings:
        qc_text = "\n".join(
            f"- [{f['code']} {f['severity']}] {f['message']}"
            for f in result.verifier_qc_findings
        )
    else:
        qc_text = "(no deterministic verifier-QC defects found)"

    _cov = input_coverage(
        search_text, input_files_names, skipped_inputs,
        files_expected=bool(get_field(row, header_map, "drive_link").strip()))
    _cov_banner = ("" if _cov["complete"] else
                   "\n\n!! PARTIAL INPUT COVERAGE — " + _cov["note"] +
                   "\nDo not report absence of leakage or of a required input as "
                   "established fact for material you were not given. Say what you "
                   "could not check.\n")
    call1_prompt = AUDIT_CLAIM_TEMPLATE.format(
        task_id=task_id,
        prompt_type=prompt_type,
        prompt_text=prompt_text or "(empty)",
        input_files_text=(prompt_text_files
                          or "(no input files supplied to auditor)") + _cov_banner,
        sanity_check_text=sanity_check_text or "(none)",
        solution_logic_text=solution_logic_text or "(none)",
        verifiers_text=verifiers_text or "(none)",
        verifier_qc_text=qc_text,
    )
    try:
        # Call 1 (analysis scratchpad + structured claims) can be long, especially
        # once input files are in the prompt. Generous ceiling — see constant note.
        call1_tokens = max(max_tokens, CALL1_MAX_TOKENS)
        raw1 = _call_llm(call1_prompt, model_name, max_tokens=call1_tokens,
                         system_prompt=AUDITOR_SYSTEM_PROMPT)
    except Exception as e:
        result.error = f"call 1 failed: {e}"
        result.verdict = "AUDIT_FAILED"
        return result
    result.calls_made = 1

    call1 = _repair_and_parse_json(raw1, task_id)
    if call1 is None:
        result.error = "call 1 returned unparseable JSON"
        result.verdict = "AUDIT_FAILED"
        # Preserve the raw response for diagnosis (truncation vs malformed).
        result.raw_response_debug = raw1
        result.raw_response_len = len(raw1 or "")
        # Heuristic: if it doesn't end with a closing brace, it was likely truncated.
        tail = (raw1 or "").rstrip()
        result.likely_truncated = not tail.endswith("}")
        return result

    result.trap_assessment = call1.get("trap_assessment", {})
    result.input_coverage = input_coverage(
        search_text, input_files_names, skipped_inputs,
        files_expected=bool(get_field(row, header_map, "drive_link").strip()))
    result.leakage_findings = call1.get("leakage_findings", []) or []
    # An absence claim inherits the coverage it was computed over. Stamping each
    # finding means the qualification survives into any downstream consumer that
    # reads findings without reading the coverage block.
    if not result.input_coverage.get("complete", True):
        for f in result.leakage_findings:
            f["coverage_partial"] = True
        if not result.leakage_findings:
            result.leakage_findings.append({
                "location": "(coverage)", "confirmed": False,
                "coverage_partial": True,
                "what": ("No leakage found in the material that could be read, "
                         "but coverage was partial — "
                         + result.input_coverage.get("note", ""))})
    result.temporal_drift_findings = call1.get("temporal_drift_findings", []) or []
    result.missing_inputs = call1.get("missing_inputs", []) or []

    # --- Stage 2: recompute claims in code (authoritative) ---------------
    claims = [ArithmeticClaim.from_dict(c) for c in call1.get("arithmetic_claims", [])]
    verdicts = verify_claims(claims, source_text=(search_text or None))
    result.claim_verdicts = [v.to_dict() for v in verdicts]
    result.arithmetic_summary = summarize(verdicts)

    # provenance_checked is True only if the source-check actually evaluated at
    # least one declared input (found_in_source is not None somewhere).
    any_checked = any(
        p.get("found_in_source") is not None
        for v in verdicts for p in v.input_provenance
    )
    result.provenance_checked = bool(result.input_files_supplied and any_checked)
    if result.provenance_checked:
        result.provenance_note = "Declared input values were checked against the supplied input files."
    # LOUD GUARD: provenance was EXPECTED but did not run. If the task has numeric raw inputs to check
    # yet no source was supplied, a "clean" arithmetic result is HOLLOW — it never verified any figure
    # against source. Do NOT let such a task look SOUND. Flag it and block proceedability.
    _has_checkable_inputs = any(
        (p.get("found_in_source") is None and p.get("source_kind") == "raw")
        for v in verdicts for p in v.input_provenance)
    result.provenance_expected_not_run = bool(_has_checkable_inputs and not result.provenance_checked)
    if result.provenance_expected_not_run:
        result.provenance_note = (
            "PROVENANCE NOT RUN: the task declares raw source-backed inputs but NO input files were "
            "supplied, so NOT ONE figure was checked against source. Arithmetic passed on internal "
            "consistency only. This is NOT a clean result — fabricated or mis-sourced figures would "
            "pass undetected. Supply the source (Drive link / files) and re-audit.")

    # --- Stage 3: short-circuit if call 1 is clean -----------------------
    prelim = (call1.get("preliminary_verdict") or "").upper().replace("-", "_")
    no_arith_errors = not result.arithmetic_summary.get("any_error", False)
    # NOTE: a partial-coverage placeholder is appended to leakage_findings above,
    # so this can no longer report a clean sweep on a task with unread input.
    no_findings = not (result.leakage_findings or result.temporal_drift_findings
                       or result.missing_inputs)
    if prelim == "SOUND" and no_arith_errors and no_findings and not result.provenance_expected_not_run:
        result.verdict = "SOUND"
        result.primary_reason = "No defects found; arithmetic confirmed; no leakage/drift."
        result.proceedable = True
        result.skipped_call2 = True
        result.model_used = model_name
        result.prose_findings = (
            "Task audited SOUND. All load-bearing figures recomputed and confirmed; "
            "cognitive trap valid; no answer leakage or unpinned time-sensitive values."
        )
        # Even a SOUND task needs its verifier set made atomic and its targets
        # frozen before augment; call 2 was skipped, so finalize over the ORIGINAL
        # verifiers and the call-1 claims (which are the unchanged derivation).
        if run_verifier_audit:
            _finalize_verifier_set(result, verifiers_text, model_name)
        return result

    # --- Stage 4: corrections grounded in code results (call 2) ----------
    arithmetic_results_text = _format_arithmetic_for_prompt(verdicts)
    call1_findings = {
        "preliminary_verdict": call1.get("preliminary_verdict"),
        "trap_assessment": result.trap_assessment,
        "leakage_findings": result.leakage_findings,
        "temporal_drift_findings": result.temporal_drift_findings,
        "missing_inputs": result.missing_inputs,
        "preliminary_notes": call1.get("preliminary_notes", ""),
    }
    call2_prompt = CORRECTION_TEMPLATE.format(
        task_id=task_id,
        call1_findings_json=json.dumps(call1_findings, indent=2, ensure_ascii=False),
        arithmetic_results_text=arithmetic_results_text,
        prompt_text=prompt_text or "(empty)",
        sanity_check_text=sanity_check_text or "(none)",
        solution_logic_text=solution_logic_text or "(none)",
    )
    try:
        call2_tokens = max(max_tokens,
                           call2_token_budget(len(vparse.records or []),
                                              len(verdicts or [])))
        raw2 = _call_llm(call2_prompt, model_name, max_tokens=call2_tokens,
                         system_prompt=AUDITOR_SYSTEM_PROMPT)
    except Exception as e:
        # We have call-1 + arithmetic; degrade to a partial result rather than lose it.
        result.error = f"call 2 failed: {e}"
        result.verdict = prelim or "SALVAGEABLE"
        result.primary_reason = "Correction call failed; verdict is preliminary only."
        result.proceedable = result.verdict in PROCEEDABLE
        result.model_used = model_name
        return result
    result.calls_made = 2

    call2 = _repair_and_parse_json(raw2, task_id)
    if call2 is None:
        result.error = "call 2 returned unparseable JSON"
        result.verdict = prelim or "SALVAGEABLE"
        result.primary_reason = "Correction call unparseable; verdict is preliminary only."
        result.proceedable = result.verdict in PROCEEDABLE
        result.model_used = model_name
        return result

    result.verdict = (call2.get("verdict") or prelim or "UNKNOWN").upper().replace("-", "_")
    if result.provenance_expected_not_run:
        result.proceedable = False
        if result.verdict in ("SOUND",):
            result.verdict = "UNGRADEABLE"
        result.primary_reason = ("Provenance not run (no source supplied) — figures unverified against "
                                 "source; " + (result.primary_reason or ""))
    result.primary_reason = call2.get("primary_reason", "")
    result.decision_inversion = bool(call2.get("decision_inversion", False))
    result.corrected_solution_logic = call2.get("corrected_solution_logic", "") or ""
    result.corrected_prompt = call2.get("corrected_prompt", "") or ""
    result.corrected_sanity_check = call2.get("corrected_sanity_check", "") or ""
    result.corrected_verifiers = call2.get("corrected_verifiers", "") or ""
    result.verifier_change_audit = audit_verifier_changes(
        verifiers_text, result.corrected_verifiers, result.changes)
    result.changes = call2.get("changes", []) or []

    # --- SECOND VERIFICATION PASS -------------------------------------------
    # Call 2 has just rewritten the solution logic. The first pass verified the
    # ORIGINAL claims, so without this the recorded verdicts describe an artifact
    # that no longer exists — on one task that meant reporting eight arithmetic
    # errors call 2 had already fixed.
    corrected_raw = call2.get("corrected_claims") or []

    # A claim with no operation, or with inputs carrying no value, cannot be
    # checked — and that is a PIPELINE fault, not evidence the golden is wrong.
    # Observed cause: the model read source_type="file" as "code will look this
    # up", omitted every value, and eight claims came back unverifiable. Naming
    # this separately stops a schema misread being reported as a broken task.
    _NONARITH = {"non_arithmetic", "nonarithmetic", "not_arithmetic", "formula",
                 "source_file", "llm_judgment", "judgment", "external"}
    result.malformed_claims = [
        {"id": c.get("id"), "label": c.get("label", ""),
         "missing": ([] if str(c.get("operation") or "").strip() else ["operation"])
                    + (["input values"]
                       if any((i or {}).get("value") is None
                              for i in (c.get("inputs") or [])) else [])}
        for c in corrected_raw
        # a claim that declares itself non-arithmetic legitimately has no
        # operation — that is not a malformed slip, so do not flag or repair it
        if str(c.get("source_of_verification") or "").lower() not in _NONARITH
        and (not str(c.get("operation") or "").strip()
             or any((i or {}).get("value") is None for i in (c.get("inputs") or [])))]

    # REPAIR PASS. A malformed claim (empty operation, or an input with no value)
    # is a model slip, not a broken golden, and it varies run to run — so instead
    # of letting it fail the gate, ask the model ONCE to fill in just those
    # claims. Bounded (one retry), targeted (only the malformed ids), and it
    # leaves everything else untouched. If the repair still comes back malformed,
    # the gate reports it as a pipeline fault (see arithmetic_gate) rather than a
    # golden-reconciliation failure.
    if corrected_raw and result.malformed_claims:
        repaired, repair_note = _repair_malformed_claims(
            corrected_raw, result.malformed_claims, task_id, model_name,
            max_tokens)
        result.malformed_repair_note = repair_note
        if repaired is not None:
            corrected_raw = repaired
            # recompute which (if any) are STILL malformed after the repair
            result.malformed_claims = [
                {"id": c.get("id"), "label": c.get("label", ""),
                 "missing": ([] if str(c.get("operation") or "").strip()
                             else ["operation"])
                            + (["input values"]
                               if any((i or {}).get("value") is None
                                      for i in (c.get("inputs") or [])) else [])}
                for c in corrected_raw
                if not str(c.get("operation") or "").strip()
                or any((i or {}).get("value") is None
                       for i in (c.get("inputs") or []))]

    if corrected_raw:
        cclaims = [ArithmeticClaim.from_dict(c) for c in corrected_raw]
        cverdicts = verify_claims(cclaims, source_text=(search_text or None))
        result.corrected_claim_verdicts = [v.to_dict() for v in cverdicts]
        result.corrected_arithmetic_summary = summarize(cverdicts)
        final_verdicts = cverdicts
        # carry the grouping label through; it is not the verifier's business
        step_of = {c.get("id"): c.get("solution_step", "") for c in corrected_raw}
        for d in result.corrected_claim_verdicts:
            d["solution_step"] = step_of.get(d["id"], "")
    else:
        # Nothing to re-verify. Say so rather than letting the first pass stand
        # in for a derivation it does not describe.
        result.corrected_claim_verdicts = []
        result.corrected_arithmetic_summary = {}
        final_verdicts = verdicts
        result.findings_note = (
            "call 2 returned no corrected_claims, so the derivation could not be "
            "rebuilt and the gate ran against the pre-correction claims")

    result.judgment_steps = call2.get("judgment_steps") or []

    result.gate = arithmetic_gate(
        final_verdicts, input_files_supplied=result.input_files_supplied,
        derivation_available=bool(corrected_raw),
        malformed_claims=result.malformed_claims)

    result.findings = call2.get("findings", []) or []
    result.prose_findings = call2.get("prose_findings", "")
    # The model's verdict is necessary but no longer sufficient: a golden whose
    # own corrected arithmetic does not reconcile is not proceedable however the
    # model graded it.
    result.proceedable = (result.verdict in PROCEEDABLE
                          and result.gate.get("passed", True))
    result.model_used = model_name

    # --- Stage 5 (1b): finalize the verifier set INSIDE the auditor -----------
    # The property audit needs only the trajectory (corrected claims + judgment
    # steps) and the verifier text + text-derived targets — all Call-2 output, no
    # augment. Running it here means the auditor returns an ALREADY-atomic,
    # already-split verifier set with frozen targets, so augment authors the
    # golden over a final set and can no longer be forced to create verifiers
    # after the fact. This removes the Call-2 -> augment -> property-audit cycle.
    #
    # Gate on the ARITHMETIC reconciling, NOT on the verdict being proceedable: a
    # SALVAGEABLE or even BROKEN task still needs its verifiers made atomic and
    # split so the SME reviews the real verifier structure. Skipping the split on
    # a broken task (an earlier 1b draft did, via `proceedable`) hid the split
    # from exactly the tasks a reviewer most needs to see it on. What the
    # property audit cannot run without is a derivation to map against, which is
    # what the arithmetic gate establishes.
    gate_ok = (result.gate or {}).get("passed", True) if result.gate else True
    if run_verifier_audit and gate_ok:
        _finalize_verifier_set(result, verifiers_text, model_name)

    return result


def _finalize_verifier_set(result: "AuditResult", original_verifiers_text: str,
                           model_name: str) -> None:
    """Derive the trajectory + targets from Call-2 output, run the property
    audit, and apply its splits/rewrites — all on the AuditResult, in place.

    Everything here is a function of Call-2 output only (corrected claims,
    judgment steps, corrected verifier text). No augment dependency, which is the
    whole point: the verifier set is finalized before augment runs.
    """
    # the verifier text to finalize: prefer call 2's corrected block, else original
    vtext = (result.corrected_verifiers or "").strip() or original_verifiers_text
    # Parse with a suffix-aware reader: if the corrected block ever carries a
    # suffixed id (V5a) — e.g. a re-augment of an already-split task — parse_verifiers
    # would collapse it to V5 with "a:" leaking into the text. _parse_v_lines keys
    # on the full "V\d+[a-z]?" id so suffixes survive. (The split that CREATES
    # suffixes runs later in this function; this guards the input side.)
    all_vs = _parse_v_lines(vtext)
    if not all_vs:
        return

    # trajectory from the corrected derivation (pure). On a SOUND task call 2 was
    # skipped, so corrected_claim_verdicts is empty; fall back to the call-1
    # claims, which for a SOUND task ARE the (unchanged) derivation.
    claim_source = (result.corrected_claim_verdicts
                    or result.claim_verdicts or [])
    step_graph, step_nodes = claim_graph(claim_source, result.judgment_steps)
    result.step_graph = step_graph
    result.step_nodes = step_nodes
    result.step_graph_health = graph_health(step_graph, step_nodes)

    # targets frozen from the verifier TEXT (authoritative; augment only adds
    # unit/source metadata later). The task's known trap values (the wrong
    # figures the derivation records on trapped claims) are passed in so the
    # widened compute-result reader can freeze interior "Calculate X as N"
    # targets while REFUSING to ever freeze a trap value — see verifier_grammar.
    _trap_values = sorted({c.get("trap_value") for c in (claim_source or [])
                           if c.get("trap_value") is not None})
    derived, grammar_problems = derive_expected_values(
        format_verifiers_ids(all_vs), trap_values=_trap_values)
    result.target_grammar_problems = grammar_problems
    ev = dict(derived)

    if not step_graph:
        # no trajectory to map against; still hand back the parsed set + targets
        result.final_verifiers_text = format_verifiers_ids(all_vs)
        result.expected_values = ev
        return

    # provisional mapping + coverage the property audit reads as context
    vmap = map_verifiers_to_steps(
        ev, {v["id"]: v.get("text", "") for v in all_vs}, step_nodes)
    frozen_map = dict(vmap.verifier_to_step)
    mapping_report = {
        "n_mapped": len(frozen_map), "n_verifiers": len(all_vs),
        "unmatched": vmap.unmatched, "ambiguous": vmap.ambiguous,
        "near_misses": vmap.near_misses, "detail": vmap.detail}
    coverage = step_coverage(step_nodes, step_graph, frozen_map)

    # the property audit (atomicity / splits / rewrites)
    va = audit_verifiers(
        task_id=result.task_id, verifiers=all_vs, expected_values=ev,
        step_nodes=step_nodes, solution_logic=result.corrected_solution_logic,
        sanity_check=result.corrected_sanity_check,
        mapping_report=mapping_report, coverage=coverage,
        verifier_to_step=frozen_map, model=model_name)
    result.verifier_audit = va.to_dict()
    if not va.error:
        if va.rewrites:
            all_vs = apply_rewrites(all_vs, va.rewrites)
            result.verifier_rewrites_applied = sorted(va.rewrites)
        if va.splits:
            all_vs, ev, split_log = apply_splits(all_vs, va.splits, ev)
            result.verifier_splits_applied = split_log
        valid = {v["id"] for v in all_vs}
        ev = {k: v for k, v in ev.items() if k in valid}
        # re-map after splits: children are new verifiers
        vmap = map_verifiers_to_steps(
            ev, {v["id"]: v.get("text", "") for v in all_vs}, step_nodes)
        frozen_map = dict(vmap.verifier_to_step)
        for vid, sid in va.tests_step.items():
            if vid in valid:
                frozen_map.setdefault(vid, sid)
        mapping_report = {
            "n_mapped": len(frozen_map), "n_verifiers": len(all_vs),
            "unmatched": vmap.unmatched, "ambiguous": vmap.ambiguous,
            "near_misses": vmap.near_misses, "detail": vmap.detail}
        coverage = step_coverage(step_nodes, step_graph, frozen_map)

    result.final_verifiers_text = format_verifiers_ids(all_vs)
    result.expected_values = ev
    result.verifier_to_step = frozen_map
    result.verifier_mapping_report = mapping_report
    result.step_coverage_report = coverage


def _format_arithmetic_for_prompt(verdicts) -> str:
    """Render claim verdicts as a compact text table for call 2."""
    if not verdicts:
        return "(no arithmetic claims were emitted in call 1)"
    lines = []
    for v in verdicts:
        rec = f"{v.recomputed:g}" if v.recomputed is not None else "n/a"
        clm = f"{v.claimed:g}" if v.claimed is not None else "n/a"
        trap = " [MATCHES TRAP VALUE]" if v.matches_trap else ""
        lines.append(
            f"- {v.id} ({v.label}): {v.status} | claimed={clm} recomputed={rec}{trap}\n"
            f"    {v.detail}"
        )
    return "\n".join(lines)

def _repair_malformed_claims(corrected_raw, malformed, task_id, model_name,
                             max_tokens):
    """One-shot targeted repair for malformed claims.

    A malformed claim (empty `operation`, or an input whose `value` is null) is a
    model slip that varies run to run. Rather than let it fail the gate, hand the
    model back JUST those claims and ask it to fill in the missing operation /
    values, keeping every id and every already-present field. Returns
    (repaired_claims, note): repaired_claims is the full corrected_raw list with
    the malformed entries replaced, or None if the repair call failed to parse.

    Bounded to a single call. If the repaired claims are still malformed, the
    caller re-flags them and the gate reports a pipeline fault (not a broken
    golden), so this never loops and never silently ships an unverifiable step.
    """
    bad_ids = [str(m.get("id")) for m in malformed]
    bad_claims = [c for c in corrected_raw if str(c.get("id")) in bad_ids]
    if not bad_claims:
        return None, "no malformed claims to repair"

    prompt = (
        "One or more arithmetic claims you emitted are incomplete and cannot be "
        "verified. For EACH claim below, supply the missing field(s) WITHOUT "
        "changing its id, label, inputs that already have values, or intended "
        "result:\n"
        "- an empty \"operation\" must become the explicit expression that "
        "produces the claimed result from the inputs (e.g. \"ceil(offered_load)\", "
        "\"a / b\"). Whitelisted functions: sqrt abs min max round ceil floor sum "
        "pow log exp median mean. \"^\" means exponent.\n"
        "- an input with \"value\": null must be given its actual number.\n\n"
        "CLAIMS TO REPAIR:\n"
        + json.dumps(bad_claims, indent=2, ensure_ascii=False)
        + "\n\nOutput ONLY a JSON object: "
          "{\"repaired_claims\": [ ...the same claims, now complete... ]}. "
          "Emit every claim id shown above, and nothing else."
    )
    try:
        raw = _call_llm(prompt, model_name,
                        max_tokens=max(4000, min(max_tokens, 16000)),
                        system_prompt=AUDITOR_SYSTEM_PROMPT)
    except Exception as e:                                        # noqa: BLE001
        return None, f"repair call failed: {e}"

    obj = _repair_and_parse_json(raw, task_id)
    if not obj or "repaired_claims" not in obj:
        return None, "repair call returned unparseable JSON"

    fixed = {str(c.get("id")): c for c in (obj.get("repaired_claims") or [])
             if c.get("id")}
    # splice the repaired claims back into the full list, preserving order and
    # leaving the well-formed claims untouched
    out = [fixed.get(str(c.get("id")), c) for c in corrected_raw]
    n = sum(1 for cid in bad_ids if cid in fixed)
    return out, f"repaired {n}/{len(bad_ids)} malformed claim(s): {bad_ids}"