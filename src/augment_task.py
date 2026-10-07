# src/augment_task.py
"""
Task augmenter. Runs on top of the existing two-call auditor.

ORDERING INVARIANT (why this file is shaped the way it is):
  Every automatic change, suggested change, and judgment-required generation is
  produced BEFORE the finalization call (auditor Call 2). The frozen graph
  (DAG -> weights -> crux -> Shapley) authors no text, so it is computed LAST —
  and "last" means after the SME seals the package (see apply_decisions), not
  here. This module runs a PREVIEW freeze so a package is inspectable before SME
  review; apply_decisions re-runs the identical freeze on the sealed artifacts.

Per task:
  0. audit_task(...)         -> verified arithmetic + corrected artifacts +
                                the FINALIZED verifier set. As of the 1b reorder
                                the auditor runs the property audit (splits/
                                rewrites) and freezes targets from verifier text,
                                so the verifier set arrives atomic and final and
                                augment never authors or splits verifiers.
  1. apply corrections       -> corrected package fields (no SME gate; JUDGMENT
                                edits applied now but tagged for later SME review)
  2. AUGMENT Opus call       -> golden deliverable + Sanity-Check anchors, authored
                                OVER the auditor's final verifier set. Augment may
                                only EXTEND the set (add verifiers for uncovered
                                gold values); it never re-splits or re-authors.
                                The augmenter's asserted DAG edges are NOT used.
  3. derive_frozen_graph()   -> verifier set -> step graph -> mapping -> coverage
                                -> DAG -> base weights -> crux -> Shapley.
                                PREVIEW ONLY here; the authoritative freeze runs
                                in apply_decisions after SME resolution.
  4. assemble AugmentResult (JSON-serializable) for CSV + HTML emit

Because the verifier set is finalized in the auditor (before augment), the old
Call-2 -> augment -> property-audit cycle is gone: augment now consumes a final
set instead of producing one.

The three crux metrics are computed later per-response by the scorer; the
freeze fixes the crux set + Shapley weights so scoring is pure matching.

derive_frozen_graph is the single re-derivation routine shared by this module
(preview) and apply_decisions (authoritative seal), so the two can never
diverge: identical code, only the inputs (pre- vs post-SME artifacts) differ.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any

from src.prompt_evaluator import _call_llm, DEFAULT_JUDGE_MODEL, _repair_and_parse_json
from src.auditor import audit_task, get_field, AuditResult
from src.augment_templates import AUGMENT_SYSTEM_PROMPT, AUGMENT_TEMPLATE
from src.verifier_parser import parse_verifiers, format_verifiers, VerifierRecord
from src.verifier_audit import format_verifiers_ids
from src.verifier_weights import compute_weights, classify_dim, cascade_analysis
from src.crux_shapley import select_crux, crux_shapley, ancestors

logger = logging.getLogger("dra.augment")

AUGMENT_MAX_TOKENS = 32000


@dataclass
class AugmentResult:
    task_id: str
    audit_verdict: str = ""
    # why the AUDIT itself failed, when it did (auditor call error / unparseable or truncated response).
    # Previously computed by the auditor and dropped here, so an AUDIT_FAILED run recorded no cause.
    audit_error: str = ""
    audit_raw_response_len: int = 0
    audit_likely_truncated: bool = False
    # Verified arithmetic, carried through instead of discarded. These are the
    # trajectory's only input (src/trajectory.py) and re-deriving them costs an
    # Opus call, so they belong in the persisted package.
    claim_verdicts: List[dict] = field(default_factory=list)
    arithmetic_summary: Dict[str, Any] = field(default_factory=dict)
    proceedable: bool = False

    # corrected/augmented package
    corrected_solution_logic: str = ""
    corrected_prompt: str = ""
    corrected_sanity_check: str = ""
    corrected_verifiers_applied: bool = False
    verifier_change_audit: dict = field(default_factory=dict)
    #: Cycles found in the ASSERTED dag_edges and broken. A dependency graph with
    #: a cycle is not one; before the guard this crashed select_crux.
    dag_cycles_broken: List[list] = field(default_factory=list)
    #: The trajectory graph and the verifier graph derived from it. `dag` IS the
    #: derived graph now — the augmenter's asserted dag_edges is no longer used.
    step_graph: Dict[str, List[str]] = field(default_factory=dict)
    step_graph_health: dict = field(default_factory=dict)
    verifier_to_step: Dict[str, str] = field(default_factory=dict)
    verifier_mapping_report: dict = field(default_factory=dict)
    dag_derived: Dict[str, List[str]] = field(default_factory=dict)
    dag_source: str = "derived"
    #: The pre-correction text of each correctable artifact, so a report can
    #: distinguish "unchanged" from "rewritten to the same thing".
    _originals: dict = field(default_factory=dict)
    _verifiers_before_audit: List[dict] = field(default_factory=list)
    #: Identifies this run. A decisions file carries it back, so decisions made
    #: against a different run are refused rather than applied to text the
    #: reviewer never saw.
    run_hash: str = ""
    sealed: bool = False
    #: Which derivation steps no verifier watches. An unwatched step is real work
    #: a response can skip with nothing objecting. Read alongside
    #: verifier_mapping_report: a step can look unwatched only because the
    #: verifier that watches it failed to map.
    step_coverage: dict = field(default_factory=dict)
    #: Fourth-call output: property verdicts, the frozen verifier->step mapping it
    #: resolved, rewrites applied, and splits PROPOSED (never applied — a split
    #: mints ids and the frozen targets, crux set and scores are keyed by id).
    verifier_audit: dict = field(default_factory=dict)
    verifier_rewrites_applied: List[str] = field(default_factory=list)
    #: Verifiers with no frozen target. They cannot be scored, and select_crux
    #: drops them silently, so they must be counted somewhere visible.
    verifiers_without_target: List[str] = field(default_factory=list)
    #: The subset whose text states a value, so a missing target IS a defect.
    verifiers_value_bearing_without_target: List[str] = field(default_factory=list)
    #: Verifiers testing the final answer — terminal-mapped or decision-kind.
    #: One of the two direct grounds for crux; the other is the anchors.
    final_answer_verifiers: List[str] = field(default_factory=list)
    #: Verifiers whose frozen target came from the model's emission because their
    #: text carries no readable target clause.
    targets_emitted_only: List[str] = field(default_factory=list)
    #: Where the model's emitted target and the verifier's own text named
    #: DIFFERENT quantities. The text wins; this records what was overridden.
    target_disagreements: List[dict] = field(default_factory=list)
    #: Target clauses that could not be read — multi-value, or malformed.
    target_grammar_problems: List[dict] = field(default_factory=list)
    #: [{parent, children, target_went_to, parent_text}] — splits APPLIED. Children
    #: take suffixed ids (V5 -> V5a, V5b) so no existing id is renumbered.
    verifier_splits_applied: List[dict] = field(default_factory=list)
    verifier_rewrites_pending: List[dict] = field(default_factory=list)
    corrected_claim_verdicts: List[dict] = field(default_factory=list)
    judgment_steps: List[dict] = field(default_factory=list)
    gate: dict = field(default_factory=dict)
    augmented_verifiers_text: str = ""          # canonical V<n>: text (existing + added)
    augmented_verifiers_added: List[str] = field(default_factory=list)  # ids augment added
    changes_applied: List[dict] = field(default_factory=list)
    judgment_changes_pending_sme: List[dict] = field(default_factory=list)

    # golden deliverable
    gold_deliverable_format: str = ""
    gold_deliverable_sections: List[dict] = field(default_factory=list)
    gold_deliverable_text: str = ""             # flattened, for CSV cell + scoring

    # structure
    dag: Dict[str, List[str]] = field(default_factory=dict)
    base_weights: Dict[str, float] = field(default_factory=dict)
    amzn_weights: Dict[str, float] = field(default_factory=dict)
    depths: Dict[str, int] = field(default_factory=dict)

    # crux
    crux_ids: List[str] = field(default_factory=list)
    crux_anchors_trap: List[str] = field(default_factory=list)
    crux_anchors_expert: List[str] = field(default_factory=list)
    crux_shapley_weights: Dict[str, float] = field(default_factory=dict)
    expected_values: Dict[str, dict] = field(default_factory=dict)  # frozen scoring targets
    crux_dropped_no_expected: List[str] = field(default_factory=list)  # reachable but no target
    scoreable: bool = True                       # False if any crux verifier is judgment_flagged
    not_scoreable_reason: str = ""               # human-readable why-excluded

    # provenance
    model_used: str = ""
    skipped_inputs: List[str] = field(default_factory=list)  # inputs that could not be read
    input_coverage: dict = field(default_factory=dict)       # what the auditor saw
    error: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _flatten_deliverable(fmt: str, sections: List[dict]) -> str:
    parts = [f"[FORMAT: {fmt}]"]
    for s in sections or []:
        parts.append(f"## {s.get('title','')}\n{s.get('content','')}")
    return "\n\n".join(parts)


def final_answer_verifiers(expected_values: Dict[str, dict],
                           verifier_to_step: Dict[str, str],
                           step_graph: Dict[str, List[str]]) -> List[str]:
    """Verifiers that test the FINAL ANSWER.

    Two grounds, both direct:
      * mapped to a TERMINAL step — one nothing else consumes, so it is an output
        of the derivation rather than working towards one
      * asserting a DECISION — a decision is a final answer by definition

    The second is also insurance against the first: mapping prose verifiers to
    steps is the weakest link in the pipeline, and on one task it placed no
    verifier on a terminal at all while six were decisions.
    """
    consumed = {p for ps in (step_graph or {}).values() for p in ps}
    terminals = {k for k in (step_graph or {}) if k not in consumed}
    out = {v for v, sid in (verifier_to_step or {}).items() if sid in terminals}
    out |= {v for v, ev in (expected_values or {}).items()
            if str((ev or {}).get("kind", "")).lower() == "decision"}
    return sorted(out)


def _verifiers_from_text(augmented_verifiers_text: str) -> List[dict]:
    """Parse a canonical 'V<n>: text' block back into [{'id','text'}].

    The id space (V1, V5a, ...) is the contract the rest of the pipeline keys on,
    so re-derivation reads the verifier set from the same text the report and the
    SME saw, never from a stale in-memory list.

    TOLERANT to the two non-canonical seed formats ("V1 - text" dash, "v1: text"
    lowercase). This routine also parses text that FELL BACK to the raw seed
    column (finalized_vtext = audit.final_verifiers_text or verifiers_text), so it
    must accept the seed's shape or the whole set is dropped and only the
    augment-added verifiers survive — the cause of the collapsed DAG/crux. Kept in
    lockstep with auditor._parse_v_lines; see its docstring for the separator
    rules (colon: optional space; dash: requires surrounding spaces, so an
    internal hyphen in the prose is never mistaken for the separator). Ids are
    normalized to an uppercase 'V'; a trailing inter-entry comma is stripped.
    """
    import re
    vid_line = (r"\s*(?P<vid>V(?:\d+[a-z]?|_[A-Za-z0-9][A-Za-z0-9_.]*))\s*"
                r"(?:\[[^\]]*\])?\s*(?::\s*|\s+-\s+)(?P<text>.*)")
    out = []
    for line in (augmented_verifiers_text or "").splitlines():
        m = re.match(vid_line, line)
        if m:
            vid = m.group("vid")
            if re.match(r"[Vv]\d", vid):
                vid = "V" + vid[1:]
            text = m.group("text").strip().rstrip(",").strip()
            if text:
                out.append({"id": vid, "text": text})
    return out


def derive_frozen_graph(pkg: dict, compute_shapley: bool = False,
                        forced_verifier_to_step=None,
                        seal_crux: bool = False) -> dict:
    """Recompute the whole frozen graph from a package dict, in place.

    This is the ONLY routine that computes the scored structure, and it authors
    no text — it only reads the (possibly SME-edited) artifacts and writes the
    derived fields. That is why it is safe to run last, and why it must run again
    after any edit to the verifier set, the targets, or the claim verdicts. It is
    called twice with the same code: once here as a preview, once in
    apply_decisions as the authoritative seal; only the inputs differ.

    `compute_shapley` gates the crux-only Shapley computation. It is expensive
    (Monte-Carlo over verifier coalitions) AND its value is meaningful only on the
    FINAL sealed set — computing it at preview time produces a number that the SME
    edits then invalidate, and whose run-to-run wobble is pure noise because the
    pre-seal verifier set itself varies. So Shapley is OFF at preview and ON only
    at the seal (apply_decisions passes compute_shapley=True). Preview still emits
    the crux set and base weights; only the Shapley layer waits for the seal.

    Reads from pkg:  augmented_verifiers_text, expected_values,
                     corrected_claim_verdicts, judgment_steps,
                     trap_anchor_ids, expert_anchor_ids.
    Writes to pkg:   step_graph, step_graph_health, verifier_to_step,
                     verifier_mapping_report, step_coverage, dag, dag_derived,
                     dag_source, base_weights, amzn_weights, depths,
                     final_answer_verifiers, crux_ids, crux_shapley_weights, and
                     (on inconsistency) scoreable / not_scoreable_reason.

    The rebuild starts from claim_graph so BOTH the step graph and step_nodes are
    reconstructed from the persisted claim/judgment steps — a reverted split or an
    added verifier changes the id space, and a partial patch would leave stale
    ids. claim_graph is pure, so the rebuild is deterministic.
    """
    from src.derive_dag import (claim_graph, graph_health,
                                 map_verifiers_to_steps, derive_verifier_dag,
                                 step_coverage)

    all_vs = _verifiers_from_text(pkg.get("augmented_verifiers_text", ""))
    vtexts = {v["id"]: v.get("text", "") for v in all_vs}
    valid_ids = {v["id"] for v in all_vs}
    ev = {k: v for k, v in (pkg.get("expected_values") or {}).items()
          if k in valid_ids}
    pkg["expected_values"] = ev

    # 1) step graph + nodes, rebuilt from the persisted trajectory
    step_graph, step_nodes = claim_graph(
        pkg.get("corrected_claim_verdicts", []), pkg.get("judgment_steps", []))
    pkg["step_graph"] = step_graph
    pkg["step_graph_health"] = graph_health(step_graph, step_nodes)
    if not step_graph:
        pkg["scoreable"] = False
        pkg["not_scoreable_reason"] = (
            "no trajectory: no corrected claims, so no dependency graph can be "
            "derived")
        return pkg

    # 2) verifier -> step mapping, then coverage (the inverse question)
    vmap = map_verifiers_to_steps(ev, vtexts, step_nodes)
    frozen_map = dict(vmap.verifier_to_step)
    # Seal-only: overlay any caller-forced verifier->step bindings. Used by
    # apply_decisions for gap verifiers added over a judgment step, which carry no
    # numeric value and so cannot be reached by value-based mapping — but the step
    # they were authored for is known. Only binds verifiers that exist and steps
    # that exist; never overrides an automatic mapping.
    if forced_verifier_to_step:
        for vid, sid in forced_verifier_to_step.items():
            if vid in valid_ids and sid in step_nodes and vid not in frozen_map:
                frozen_map[vid] = sid
    pkg["verifier_to_step"] = frozen_map
    pkg["verifier_mapping_report"] = {
        "n_mapped": len(frozen_map), "n_verifiers": len(all_vs),
        "unmatched": vmap.unmatched, "ambiguous": vmap.ambiguous,
        "near_misses": vmap.near_misses, "detail": vmap.detail}
    pkg["step_coverage"] = step_coverage(step_nodes, step_graph, frozen_map)

    # 3) verifier DAG derived from step ancestry
    dag = derive_verifier_dag(step_graph, frozen_map,
                              all_verifier_ids=[v["id"] for v in all_vs])
    pkg["dag"] = dag
    pkg["dag_derived"] = dag
    pkg["dag_source"] = "derived"
    if not dag:
        pkg["scoreable"] = False
        pkg["not_scoreable_reason"] = (
            "no verifier could be placed on a derivation step, so no dependency "
            "graph could be derived")
        return pkg

    # 4) base weights (depth+1, normalized to 100 -> stored as fraction)
    dw, aw, depths = compute_weights(all_vs, dag)
    pkg["base_weights"] = {k: v / 100.0 for k, v in dw.items()}
    pkg["amzn_weights"] = aw
    pkg["depths"] = depths

    # 5) deterministic crux from anchors + final-answer verifiers
    trap_anchors = [a for a in (pkg.get("trap_anchor_ids") or []) if a in valid_ids]
    expert_anchors = [a for a in (pkg.get("expert_anchor_ids") or []) if a in valid_ids]
    fav = final_answer_verifiers(ev, frozen_map, step_graph)
    pkg["final_answer_verifiers"] = fav
    # SEAL-ONLY crux methodology: at seal (seal_crux=True) the crux is widened by
    # immediate parents and the dropped/deduped verifiers are excluded, computed
    # against the FINAL post-decision DAG. During augmentation this stays the
    # direct set (the preview crux), untouched.
    _excluded = pkg.get("_crux_excluded_ids") or [] if seal_crux else []
    sel = select_crux(all_vs, dag,
                      trap_anchor_ids=trap_anchors or None,
                      expert_anchor_ids=expert_anchors or None,
                      expected_value_ids=list(ev.keys()),
                      final_answer_ids=fav,
                      include_immediate_parents=seal_crux,
                      excluded_ids=_excluded)
    pkg["crux_ids"] = sel.crux_ids
    pkg["crux_anchors_trap"] = sel.anchors_trap
    pkg["crux_anchors_expert"] = sel.anchors_expert
    pkg["crux_dropped_no_expected"] = sel.dropped_no_expected

    # 6) crux-only Shapley — SEAL ONLY. Skipped at preview (see docstring): the
    # value is meaningful only on the final sealed set, and computing it pre-seal
    # both wastes the Monte-Carlo cost and reports a number the SME's edits will
    # invalidate. Preview leaves crux_shapley_weights empty; the seal fills it.
    if compute_shapley:
        pkg["crux_shapley_weights"] = crux_shapley(
            all_vs, dag, pkg["base_weights"], pkg["crux_ids"])
    else:
        pkg.setdefault("crux_shapley_weights", {})

    # 7) re-assert the scoreability gates on the (possibly re-derived) set.
    #    (a0) any UNRESOLVED judgment question still pending blocks. apply_decisions
    #         now keeps a question in judgment_changes_pending_sme when its answer
    #         could not be applied (a binary-file edit, an unlocatable anchor, a
    #         golden step it could not synthesize) — those are real, unfinished SME
    #         edits. A package with pending questions is NOT scoreable; the seal
    #         must honor the list rather than trust that it was cleared. Without
    #         this, a package can report scoreable=True while still carrying
    #         pending questions, which is the exact inconsistency the pending list
    #         exists to prevent.
    pending_q = pkg.get("judgment_changes_pending_sme") or []
    if pending_q:
        pkg["scoreable"] = False
        pkg["not_scoreable_reason"] = (
            f"{len(pending_q)} unresolved judgment question(s) pending SME edit: "
            + ", ".join(str(q.get("artifact", "?")) for q in pending_q[:6])
            + (" ..." if len(pending_q) > 6 else "")
            + ". These answers could not be auto-applied (binary file, unlocatable "
              "anchor, or unsynthesizable step) and must be made by hand before "
              "the package is scoreable.")
        return pkg
    #    (a) a crux verifier still tied to an unresolved judgment question blocks.
    flagged = [vid for vid in pkg["crux_ids"]
               if (ev.get(vid, {}) or {}).get("source_of_verification")
               == "judgment_flagged"]
    if flagged:
        pkg["scoreable"] = False
        pkg["not_scoreable_reason"] = ("crux verifier(s) judgment_flagged: "
                                       + ", ".join(sorted(flagged)))
        return pkg
    #    (b) a load-bearing derivation step that NO verifier watches is a hole in
    #        the scoring: a response could get that step wrong with nothing
    #        objecting. Previously reported only; now it blocks, so the coverage
    #        gap is fixed (author a verifier) rather than silently shipped.
    ulb = (pkg.get("step_coverage", {}) or {}).get("unwatched_load_bearing", [])
    if ulb:
        pkg["scoreable"] = False
        pkg["not_scoreable_reason"] = (
            f"{len(ulb)} load-bearing step(s) watched by no verifier: "
            + ", ".join(map(str, ulb[:6])) + ("..." if len(ulb) > 6 else "")
            + ". A response could get these wrong with nothing objecting; author "
              "a verifier for each before scoring.")
        return pkg
    #    (c) a split child carrying a live value with no date/source pin is a
    #        temporal_drift defect created after Call 1's screen. It must be
    #        pinned (by the SME at seal) before the package is scoreable.
    temporal_unpinned = []
    for entry in (pkg.get("verifier_splits_applied") or []):
        temporal_unpinned += entry.get("temporal_unpinned_children", []) or []
    temporal_unpinned = [t for t in temporal_unpinned
                         if t in {v["id"] for v in all_vs}]
    # Re-screen against CURRENT text: a child the augmenter flagged may since have
    # been PINNED by the SME at seal (a date or source added). Only children whose
    # current text is still unpinned should block scoring — otherwise a resolved
    # temporal decision would wrongly keep the package unscoreable.
    if temporal_unpinned:
        try:
            from src.verifier_audit import _child_has_unpinned_live_value
            vtext_now = {v["id"]: v.get("text", "") for v in all_vs}
            temporal_unpinned = [
                t for t in temporal_unpinned
                if _child_has_unpinned_live_value(vtext_now.get(t, ""))]
        except Exception:                                       # noqa: BLE001
            pass
    if temporal_unpinned:
        pkg["scoreable"] = False
        pkg["not_scoreable_reason"] = (
            "split verifier(s) carry an unpinned live value (temporal drift): "
            + ", ".join(sorted(temporal_unpinned))
            + ". Pin each to a date or source file before scoring.")
        return pkg
    # All STRUCTURAL gates passed. Clear any stale not-scoreable state carried in
    # from a prior derive (an earlier coverage/temporal failure that later SME
    # decisions resolved). derive_frozen_graph judges structure only; a
    # non-structural block such as the audit verdict is re-applied by the caller
    # (apply_decisions) afterwards, so clearing here does not override it.
    pkg["scoreable"] = True
    pkg["not_scoreable_reason"] = ""
    return pkg


def _run_hash(res) -> str:
    """Stable fingerprint of what a reviewer was shown.

    Covers the artifacts and the decision surface, so re-running the task
    produces a different hash and stale decisions are caught.
    """
    import hashlib

    h = hashlib.sha256()
    for part in (res.task_id, res.corrected_solution_logic, res.corrected_prompt,
                 res.corrected_sanity_check, res.augmented_verifiers_text,
                 json.dumps(res.changes_applied, sort_keys=True, default=str),
                 json.dumps(res.judgment_changes_pending_sme, sort_keys=True,
                            default=str)):
        h.update(str(part or "").encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def _or_original(corrected: str, original: str) -> str:
    """An empty correction means UNCHANGED, not blank.

    Call 2 returns an empty string when it changed nothing, so writing the raw
    value through would erase the artifact. Named once here so the rule is
    testable and cannot drift between the three corrected artifacts.
    """
    return (corrected or "").strip() or (original or "")


def _apply_changes(base_logic: str, corrected_logic: str, changes: List[dict]):
    """Apply corrections directly. Returns (final_logic, applied, pending_sme).

    MECHANICAL changes are applied silently. JUDGMENT_REQUIRED changes are also
    applied now (no SME gate per current instruction) but recorded separately so
    a later SME pass can review them.
    """
    final_logic = corrected_logic.strip() or base_logic
    applied, pending = [], []
    for ch in changes or []:
        (applied if ch.get("type") == "MECHANICAL" else pending).append(ch)
    return final_logic, applied, pending


def _merge_verifiers(existing_records: List[VerifierRecord],
                     augmented: List[dict]) -> (str, List[dict]):
    """Append augmented verifiers after the existing ones, renumbering continuously.
    Returns (canonical_text, full_verifier_dicts)."""
    vs = []
    for r in existing_records:
        vs.append({"id": f"V{r.index}", "text": r.text,
                   "dim": classify_dim(r.text), "type": None, "is_decision": False})
    nxt = (max((r.index for r in existing_records), default=0)) + 1
    for a in augmented or []:
        vs.append({"id": f"V{nxt}", "text": a.get("text", ""),
                   "dim": a.get("dim") or classify_dim(a.get("text", "")),
                   "type": a.get("type"), "is_decision": bool(a.get("is_decision"))})
        nxt += 1
    recs = [VerifierRecord(index=int(v["id"][1:]), text=v["text"]) for v in vs]
    return format_verifiers(recs), vs


def _break_cycles(dag: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], List[list]]:
    """Remove the edges that close a cycle, and report them.

    A dependency graph with a cycle is not a dependency graph. Self-loops were
    already stripped, but a two-node cycle survived and reached select_crux.
    The edge that closes a cycle is dropped rather than the whole task failing,
    because an asserted graph is advisory — but it is reported, never silent.
    """
    found: List[list] = []
    colour: Dict[str, int] = {}

    def visit(n: str, path: List[str]) -> None:
        colour[n] = 1
        for p in list(dag.get(n, [])):
            if colour.get(p, 0) == 1:                 # p is on the current path
                found.append(path[path.index(p):] + [p] if p in path else [n, p])
                dag[n] = [x for x in dag[n] if x != p]
            elif colour.get(p, 0) == 0:
                visit(p, path + [p])
        colour[n] = 2

    for n in list(dag):
        if colour.get(n, 0) == 0:
            visit(n, [n])
    return dag, found


def _remap_dag(dag_edges: Dict[str, list], valid_ids: set) -> Dict[str, List[str]]:
    """UNUSED in the main path since the graph became derived. Kept because
    load_augment and the older per-task artifacts still contain asserted graphs
    that a reader may want to clean."""
    """Keep only edges among known verifier ids; ensure every id is a key."""
    dag = {vid: [] for vid in valid_ids}
    for child, parents in (dag_edges or {}).items():
        if child not in valid_ids:
            continue
        dag[child] = [p for p in (parents or []) if p in valid_ids and p != child]
    return dag


def augment_task(
    row: dict,
    header_map: Dict[str, str],
    input_files_text: str = "",
    input_files_names: Optional[List[str]] = None,
    model_name: str = DEFAULT_JUDGE_MODEL,
    run_verifier_audit: bool = True,
    audit: Optional[AuditResult] = None,
    skipped_inputs: Optional[List[str]] = None,
    input_corpus=None,
) -> AugmentResult:
    task_id = get_field(row, header_map, "task_id") or "(no id)"
    prompt_text = get_field(row, header_map, "prompt")
    sanity_check_text = get_field(row, header_map, "sanity_check")
    verifiers_text = get_field(row, header_map, "verifiers")

    res = AugmentResult(task_id=task_id, model_used=model_name)
    res.skipped_inputs = list(skipped_inputs or [])

    # --- Stage 0/1: audit (reuse existing auditor) + apply corrections ---
    if audit is None:
        audit = audit_task(row, header_map, input_files_text=input_files_text,
                           input_files_names=input_files_names,
                           input_corpus=input_corpus,
                           model_name=model_name,
                           run_verifier_audit=run_verifier_audit,
                           # the auditor's leakage verdict is an absence claim,
                           # so it must know which files it never saw
                           skipped_inputs=skipped_inputs)
    res.audit_verdict = audit.verdict
    res.audit_error = getattr(audit, "error", "") or ""
    res.audit_raw_response_len = int(getattr(audit, "raw_response_len", 0) or 0)
    res.audit_likely_truncated = bool(getattr(audit, "likely_truncated", False))
    res.claim_verdicts = list(audit.claim_verdicts or [])
    res.input_coverage = dict(audit.input_coverage or {})
    res.arithmetic_summary = dict(audit.arithmetic_summary or {})
    res.proceedable = audit.proceedable
    res.corrected_claim_verdicts = list(audit.corrected_claim_verdicts or [])
    res.judgment_steps = list(audit.judgment_steps or [])
    res.gate = dict(audit.gate or {})

    # HONOUR THE GATE. It was previously computed and recorded but never acted
    # on, so a golden whose own arithmetic did not reconcile still produced
    # verifiers, targets and a crux set, and shipped to an SME to hand-score.
    if res.gate and not res.gate.get("passed", True):
        res.scoreable = False
        res.not_scoreable_reason = (
            "arithmetic gate: " + res.gate.get("reason", "derivation does not "
                                               "reconcile"))
        res.error = res.not_scoreable_reason
        return res

    base_logic = get_field(row, header_map, "solution_logic")
    final_logic, applied, pending = _apply_changes(
        base_logic, audit.corrected_solution_logic, audit.changes)
    res.corrected_solution_logic = final_logic
    # Same fall-back-to-original rule as the solution logic: an empty correction
    # means unchanged, not blank.
    res.verifier_change_audit = dict(audit.verifier_change_audit or {})

    # keep the inputs so the report can say "unchanged" as a fact rather than
    # rendering an untouched artifact as though it had been rewritten
    res._originals = {
        "solution_logic": base_logic,
        "sanity_check": get_field(row, header_map, "sanity_check"),
        "prompt": get_field(row, header_map, "prompt"),
        "verifiers": verifiers_text,
    }
    res.corrected_prompt = _or_original(
        audit.corrected_prompt, get_field(row, header_map, "prompt"))
    res.corrected_sanity_check = _or_original(
        audit.corrected_sanity_check, get_field(row, header_map, "sanity_check"))
    res.changes_applied = applied
    res.judgment_changes_pending_sme = pending

    # --- Stage 2: single AUGMENT call (golden deliverable + DAG + anchors) ---
    from src.auditor import _format_arithmetic_for_prompt  # reuse formatter
    arithmetic_results_text = ""
    if audit.claim_verdicts:
        # rebuild a compact text view from stored verdict dicts
        lines = []
        for v in audit.claim_verdicts:
            lines.append(f"- {v.get('id')} ({v.get('label')}): {v.get('status')} "
                         f"| claimed={v.get('claimed')} recomputed={v.get('recomputed')}")
        arithmetic_results_text = "\n".join(lines)
    else:
        arithmetic_results_text = "(no arithmetic claims)"

    # The verifier set is ALREADY finalized by the auditor (1b): splits applied,
    # targets frozen from text. Augment authors the golden over that final set and
    # may only EXTEND it (add verifiers for uncovered gold values); it does not
    # re-author or split. Feed it the finalized block, not the raw originals.
    finalized_vtext = (audit.final_verifiers_text or "").strip() or verifiers_text
    prompt = AUGMENT_TEMPLATE.format(
        task_id=task_id,
        prompt_text=prompt_text or "(empty)",
        solution_logic_text=final_logic or "(none)",
        sanity_check_text=sanity_check_text or "(none)",
        verifiers_text=finalized_vtext or "(none)",
        arithmetic_results_text=arithmetic_results_text,
    )
    try:
        raw = _call_llm(prompt, model_name, max_tokens=AUGMENT_MAX_TOKENS,
                        system_prompt=AUGMENT_SYSTEM_PROMPT)
    except Exception as e:
        res.error = f"augment call failed: {e}"
        return res

    aug = _repair_and_parse_json(raw, task_id)
    if aug is None:
        res.error = "augment call returned unparseable JSON"
        return res

    # golden deliverable
    gd = aug.get("gold_deliverable", {}) or {}
    res.gold_deliverable_format = gd.get("format", "")
    res.gold_deliverable_sections = gd.get("sections", []) or []
    res.gold_deliverable_text = _flatten_deliverable(
        res.gold_deliverable_format, res.gold_deliverable_sections)
    res.notes = aug.get("notes", "")

    # --- Stage 3: verifier set is ALREADY FINAL (from the auditor, 1b) -------
    # The auditor ran the property audit and froze the targets from text, so the
    # verifier set arrives atomic and split. Augment may only EXTEND it with new
    # verifiers for gold values nothing yet covers; it never re-splits or
    # re-authors. This is what removes the Call-2 -> augment -> property-audit
    # cycle: augment consumes a final set instead of producing one.
    # The finalized set carries SUFFIXED ids (V3a, V3b from splits). parse_verifiers
    # extracts an INTEGER index, so it would turn "V3a: ..." into id="V3",
    # text="a: ..." — mangling every split child (observed: canonical showed
    # "V3: a: ..." and the report could not find V3a). _verifiers_from_text keys on
    # the full "V\d+[a-z]?" id, so suffixed children survive intact.
    base_vs = _verifiers_from_text(
        audit.final_verifiers_text or verifiers_text)
    # merge in any augment-added verifiers (extension only; ids that already
    # exist are ignored so augment can never overwrite a finalized verifier)
    existing = {v["id"] for v in base_vs}
    added = []
    for av in (aug.get("augmented_verifiers", []) or []):
        vid = str(av.get("id") or "").strip()
        txt = str(av.get("text") or "").strip()
        if txt and vid and vid not in existing:
            added.append({"id": vid, "text": txt})
            existing.add(vid)
    all_vs = base_vs + added
    res.augmented_verifiers_added = [v["id"] for v in added]
    res.corrected_verifiers_applied = bool(audit.final_verifiers_text)
    res.augmented_verifiers_text = format_verifiers_ids(all_vs)
    valid_ids = {v["id"] for v in all_vs}

    # carry over the auditor's property-audit + split log so the report and the
    # temporal/coverage gates see them
    res.verifier_audit = dict(audit.verifier_audit or {})
    res.verifier_splits_applied = list(audit.verifier_splits_applied or [])
    res.verifier_rewrites_pending = list(getattr(audit, "verifier_rewrites_pending", []) or [])
    res.verifier_rewrites_applied = list(audit.verifier_rewrites_applied or [])
    res.target_grammar_problems = list(audit.target_grammar_problems or [])

    # --- frozen scoring targets ---------------------------------------------
    # The auditor already froze targets from the verifier TEXT. Augment's separate
    # expected_values emission is used ONLY for metadata (unit,
    # source_of_verification) and for any augment-ADDED verifier, whose target has
    # no auditor entry yet. The text-derived value+band always wins over the
    # model's emission (they used to desynchronise — see the long note that used
    # to live here; the fix, deriving from text, now lives in the auditor).
    from src.verifier_grammar import derive_expected_values

    emitted = {k: v for k, v in (aug.get("expected_values") or {}).items()
               if k in valid_ids}
    merged = dict(audit.expected_values or {})           # auditor's frozen targets
    # derive targets for augment-ADDED verifiers only (auditor never saw them)
    if added:
        derived_added, _ = derive_expected_values(
            format_verifiers_ids(added))
        for vid, dv in derived_added.items():
            base = dict(emitted.get(vid) or {})
            base.update(dv)
            merged[vid] = base
    # fold augment metadata (unit / source_of_verification) onto existing targets
    # without letting it move the value or band
    for vid, em in emitted.items():
        if vid in merged and isinstance(em, dict):
            for meta in ("unit", "source_of_verification"):
                if meta in em and meta not in merged[vid]:
                    merged[vid][meta] = em[meta]
        elif vid not in merged:
            merged[vid] = em
            res.targets_emitted_only.append(vid)
    res.expected_values = {k: v for k, v in merged.items() if k in valid_ids}
    res.crux_anchors_trap = [a for a in aug.get("trap_anchor_ids", [])
                             if a in {v["id"] for v in all_vs}]
    res.crux_anchors_expert = [a for a in aug.get("expert_anchor_ids", [])
                               if a in {v["id"] for v in all_vs}]

    # --- Stage 4: PREVIEW freeze (DAG -> weights -> crux -> Shapley) ---------
    # Authors no text; recomputes the scored structure from the final verifier
    # set. This is a preview only: apply_decisions re-runs the identical routine
    # on the SME-sealed artifacts, and that run is authoritative.
    pkg = res.to_dict()
    pkg["trap_anchor_ids"] = res.crux_anchors_trap
    pkg["expert_anchor_ids"] = res.crux_anchors_expert
    pkg = derive_frozen_graph(pkg)
    # copy the derived fields back onto the dataclass
    for k in ("step_graph", "step_graph_health", "verifier_to_step",
              "verifier_mapping_report", "step_coverage", "dag", "dag_derived",
              "dag_source", "base_weights", "amzn_weights", "depths",
              "final_answer_verifiers", "crux_ids", "crux_anchors_trap",
              "crux_anchors_expert", "crux_dropped_no_expected",
              "crux_shapley_weights", "expected_values"):
        if k in pkg:
            setattr(res, k, pkg[k])
    if not pkg.get("dag"):
        res.scoreable = False
        res.not_scoreable_reason = (
            pkg.get("not_scoreable_reason")
            or "no dependency graph could be derived")
        res.error = res.not_scoreable_reason
        return res

    # --- Scoreability gate: a crux verifier tied to an unresolved JUDGMENT_REQUIRED
    # question (source_of_verification == "judgment_flagged") means the task's
    # answer is contested and must NOT be scored until an SME resolves it (handoff
    # §3). Surface it as a first-class column so list_clean_tasks can filter
    # automatically instead of relying on a hand-maintained --exclude-tasks list.
    flagged = [vid for vid in res.crux_ids
               if (res.expected_values.get(vid, {}) or {}).get(
                   "source_of_verification") == "judgment_flagged"]
    # An UNRESOLVED judgment change is the same blocker as a judgment_flagged
    # crux target, and for a sharper reason: the model resolves an open question
    # differently on different runs. On one task the FX ambiguity was resolved as
    # the live rate (savings 2.20 Cr) and on the next as the file rate (0.918 Cr)
    # — same task, same code, two goldens with different terminal answers and
    # therefore different frozen targets. Nothing downstream can be trusted until
    # a human pins it.
    # Always recorded, whatever the verdict: a targetless verifier is invisible
    # otherwise, since select_crux simply drops it.
    res.verifiers_without_target = [v["id"] for v in all_vs
                                    if v["id"] not in res.expected_values]
    # Only a verifier whose TEXT STATES A VALUE needs a frozen target. A presence
    # or decision check stated in prose legitimately has none and is graded against
    # its own text — counting those as defects would fail a package for doing the
    # right thing.
    from src.verifier_audit import _VALUE_IN_TEXT
    res.verifiers_value_bearing_without_target = [
        v["id"] for v in all_vs
        if v["id"] not in res.expected_values
        and _VALUE_IN_TEXT.search(v.get("text", "") or "")]
    pending = res.judgment_changes_pending_sme or []
    if not res.proceedable:
        res.scoreable = False
        res.not_scoreable_reason = (
            f"audit verdict {res.audit_verdict or '?'} is not proceedable"
            + (f" — {res.gate.get('reason')}" if not res.gate.get("passed", True)
               else ""))
    elif pending:
        res.scoreable = False
        res.not_scoreable_reason = (
            f"{len(pending)} unresolved judgment question(s): "
            + "; ".join(str(c.get("location") or c.get("artifact"))[:40]
                        for c in pending[:4])
            + ". Left open, the next run may resolve them differently and change "
              "the golden's answer.")
    elif flagged:
        res.scoreable = False
        res.not_scoreable_reason = ("crux verifier(s) judgment_flagged: "
                                    + ", ".join(sorted(flagged)))
    elif res.error:
        res.scoreable = False
        res.not_scoreable_reason = f"augment error: {res.error}"
    elif not res.crux_ids:
        res.scoreable = False
        res.not_scoreable_reason = "no crux verifiers selected"
    elif (len(((res.verifier_change_audit or {}).get("undeclared_edits")) or [])
          >= max(3, int(0.8 * ((res.verifier_change_audit or {}).get("n_before")
                               or 10 ** 6)))):
        # The verifier block was rewritten wholesale without declaring any of it.
        # On one task all 14 were silently rewritten, reassigning which quantity
        # each id carried — detected here, acted on nowhere, and the package was
        # reported SALVAGEABLE with every numeric target one position out.
        ue = (res.verifier_change_audit or {}).get("undeclared_edits") or []
        res.scoreable = False
        res.not_scoreable_reason = (
            f"{len(ue)} of "
            f"{(res.verifier_change_audit or {}).get('n_before')} verifiers were "
            f"rewritten without being declared, so the verifier set cannot be "
            f"reconciled with the authored one: {', '.join(sorted(ue)[:8])}"
            + ("..." if len(ue) > 8 else ""))
    elif (res.verifiers_value_bearing_without_target
          and len(res.verifiers_value_bearing_without_target) > len(all_vs) // 3):
        # A verifier with no frozen target cannot be scored at all. select_crux
        # drops those silently, so crux_ids stays non-empty and every check above
        # passes — which reported scoreable=True on a package where 20 of 31
        # verifiers had nothing to compare a response against.
        res.scoreable = False
        res.not_scoreable_reason = (
            f"{len(res.verifiers_without_target)} of {len(all_vs)} verifiers have "
            f"no frozen target and cannot be scored: "
            + ", ".join(res.verifiers_without_target[:8])
            + ("..." if len(res.verifiers_without_target) > 8 else ""))

    res.run_hash = _run_hash(res)
    return res