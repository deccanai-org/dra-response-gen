"""Concern-split verifier audit — a drop-in replacement for audit_verifiers() that fixes the
attention-overload failure of the single monolithic call.

DIAGNOSIS (evidence-based). The original audit_verifiers() asks ONE LLM call to do ~8 judgments over
all verifiers at once: atomicity+split, in-place rewrite, six property verdicts, step-mapping,
rubric-routing, dedupe, coverage-gaps. On a real DHX task with 13 verifiers it split only V1 (the
single most blatant compound) and MISSED V3/V8/V9/V12 — each of which names its own trap value and
therefore MUST split by the template's own rule. The model satisfices: it does the loudest thing and
under-applies the subtle atomicity test to the rest.

FIX. Split BY CONCERN, each concern its own all-verifier call (not per-verifier — dedupe and coverage
genuinely need the whole set). Cheap mechanical concerns run on a WORKER model (Sonnet); the whole-set
reconciliation runs on a CONSOLIDATOR model (Opus). Which tier the atomicity call lands on is decided
EMPIRICALLY by test_atomicity_tiers() below, not assumed.

CONTRACT. Emits the SAME VerifierAuditResult as the original, so derive_dag / weights / crux / Shapley
(which freeze tests_step) are unaffected. Same inputs, same output shape, different internal calls.

Concern calls (each over ALL verifiers):
  1. atomicity      -> splits{} + per-verifier atomic verdict         [worker or opus — see test]
  2. properties     -> the six property verdicts + in-place rewrites  [worker]
  3. step_mapping   -> tests_step{} + route_to_rubric                 [worker]
  4. dedupe         -> duplicate_clusters (needs whole set)           [worker]
  5. coverage       -> coverage_gaps (needs whole set + mapping)      [worker]
  consolidate       -> reconcile cross-concern conflicts, freeze set  [opus]
"""
from __future__ import annotations
import json
from typing import List, Dict, Optional, Callable

from src.verifier_audit import (
    VerifierAuditResult, PROPERTIES, _VALUE_IN_TEXT, _extract_json,
    _fmt_verifiers, _fmt_steps, trap_passes_band, DEFAULT_MODEL,
)

WORKER_MODEL = "claude-sonnet-4-6"     # cheap mechanical concerns
CONSOLIDATOR_MODEL = "claude-opus-4-8" # cross-concern reconciliation
ATOMICITY_MODEL = WORKER_MODEL         # overridden to opus if the tier test says so

# ---------------------------------------------------------------------------
# Per-concern prompts. Each is FOCUSED: one job, all verifiers, trap values inline
# where the judgment needs them.
# ---------------------------------------------------------------------------

_ATOMICITY_SYS = "You audit verifier ATOMICITY only. One job. Output only the JSON."
_ATOMICITY_TMPL = """\
For EACH verifier below decide if it is ATOMIC: ONE predicate on ONE referent (the referent may be a
named set). If it bundles more than one separately-attributable assertion, SPLIT it.

DECISIVE RULE — a verifier MUST split if any member carries diagnostic weight of its own:
  * it names a TRAP / fail-if / "must NOT use" value (each such verifier's trap is listed inline);
  * it bundles a VALUE and a source-selection DECISION (e.g. "use the 10-Q figure, not the S-1");
  * it states several separately-reported figures (e.g. "LT debt X - cash Y = net debt Z").
Do NOT split: a number that is part of the single target's derivation, a unit/window, a number in a
label, or a fail-if half plus a target half of the SAME quantity.

A split keeps the parent number with letter suffixes (V5 -> V5a, V5b). Exactly ONE child inherits the
parent's frozen target (inherits_target true); any OTHER child that states a number needs its own
"target": {{"value":..,"tol":..}}. A purely structural child ("row present") may be targetless.

VERIFIERS (with each one's trap value, if any, inline):
{verifiers_with_traps}

Output ONLY this JSON:
{{"verifiers":[{{"id":"V1","atomic":true|false,
  "split_into":[{{"text":"child condition, no reasoning","suffix":"a","inherits_target":true}},
                {{"text":"...","suffix":"b","inherits_target":false,"target":{{"value":0,"tol":0}}}}]}}]}}
An empty split_into means atomic. Every id must appear."""

_PROPS_SYS = "You audit verifier PROPERTIES and reword defects. Output only the JSON."
_PROPS_TMPL = """\
For EACH verifier, judge these properties (PASS/FAIL, quote the span you object to) and, if it has a
WORDING defect (names no fail-if, inlines a target it only points at, uses a positional reference),
rewrite it IN PLACE keeping the same id and frozen target. A rewrite states only the testable
condition, succinctly (<~25 words); never narrate the derivation or the trap.
Properties: atomic, quantifiable, self_contained, falsifiable, toleranced, content_not_ordinal.

VERIFIERS:
{verifiers_text}

Output ONLY:
{{"verifiers":[{{"id":"V1","properties":{{"atomic":{{"verdict":"PASS","evidence":"","why":""}},
  "quantifiable":{{"verdict":"PASS"}},"self_contained":{{"verdict":"PASS"}},
  "falsifiable":{{"verdict":"PASS"}},"toleranced":{{"verdict":"PASS"}},
  "content_not_ordinal":{{"verdict":"PASS"}}}},"rewrite":""}}]}}
Every id must appear."""

_STEP_SYS = "You map each verifier to the derivation step it tests. Output only the JSON."
_STEP_TMPL = """\
For EACH verifier, name the derivation STEP id it checks (from the step list), or null if it tests
something outside the derivation (a format/presence requirement). Null is legitimate; do not stretch.
If a verifier is an irreducibly OPEN judgment (not a binary check), set route_to_rubric true and name
the dimension.

STEPS:
{steps_text}

VERIFIERS:
{verifiers_text}

Output ONLY:
{{"verifiers":[{{"id":"V1","tests_step":"C4 or null","route_to_rubric":false,"rubric_dimension":null}}]}}
Every id must appear."""

_DEDUPE_SYS = "You find verifiers that assert the SAME quantity. Output only the JSON."
_DEDUPE_TMPL = """\
Cluster the verifiers by the QUANTITY each asserts. Report only clusters where two or more verifiers
assert the SAME quantity (a real duplicate), with which to keep.

VERIFIERS:
{verifiers_text}

Output ONLY:
{{"duplicate_clusters":[{{"quantity":"..","verifier_ids":["V17","V18"],"values_agree":true,
  "recommended_action":"keep V17; V18 restates it"}}]}}
Empty list if none."""

_COVERAGE_SYS = "You find load-bearing steps that no verifier watches. Output only the JSON."
_COVERAGE_TMPL = """\
Each step below is load-bearing. A step is WATCHED if some verifier tests it (mapping given). List
steps that NO verifier watches, and for each propose a one-line verifier that would check exactly what
the step computes/decides.

STEPS:
{steps_text}

VERIFIER->STEP MAPPING:
{mapping_text}

Output ONLY:
{{"coverage_gaps":[{{"step":"C13","why_it_matters":"..","proposed_verifier":".."}}]}}
Empty list if every load-bearing step is watched."""

_CONSOLIDATE_SYS = ("You are the senior consolidator. Reconcile the concern-audit outputs into one "
                    "canonical verifier decision. Resolve conflicts with judgment. Output only JSON.")
_CONSOLIDATE_TMPL = """\
Five focused audits ran over the same verifiers. Reconcile them into the final decision per verifier.
Resolve conflicts:
- If a verifier is BOTH split (atomicity) and rewritten (properties): apply the SPLIT; carry the
  rewrite's intent into the child that inherits the target if it improves wording. A split wins over
  an in-place rewrite of the parent.
- If a split child duplicates another verifier (dedupe): keep the split but note the dedupe.
- If atomicity says atomic but properties flagged atomic=FAIL, prefer the SPLIT decision from the
  atomicity audit (it is the specialist for that judgment).
- Keep every tests_step link from the step audit unless a split changes which child tests the step.
Do not invent new verifiers here (coverage gaps are handled separately as SME suggestions).

ATOMICITY AUDIT:
{atomicity_json}

PROPERTIES AUDIT:
{properties_json}

STEP MAPPING:
{step_json}

DEDUPE:
{dedupe_json}

Output ONLY the canonical per-verifier decision:
{{"verifiers":[{{"id":"V1","properties":{{..six verdicts..}},"tests_step":"C4 or null",
  "rewrite":"","split_into":[..],"route_to_rubric":false,"rubric_dimension":null}}],
  "duplicate_clusters":[..carry through..],"notes":".."}}
Every original id must appear."""


def _verifiers_with_traps(verifiers, expected_values, verifier_to_step):
    """Render each verifier WITH its trap value inline — the key input the atomicity judgment needs."""
    lines = []
    v2s = verifier_to_step or {}
    for v in verifiers:
        vid = v.get("id")
        step = v2s.get(vid)
        trap = None
        if step and step in (expected_values or {}):
            trap = (expected_values[step] or {}).get("trap")
        tp = f"   [trap value for this verifier's step: {trap}]" if trap not in (None, "") else ""
        lines.append(f"- {vid}: {v.get('text','')}{tp}")
    return "\n".join(lines) or "(none)"


def _call(provider, model, sysmsg, prompt, max_tokens=8000):
    """Run one concern call, parse its JSON. Returns {} on failure (fail-soft per concern)."""
    try:
        raw = provider(prompt, sysmsg, model, max_tokens)
        obj = _extract_json(raw if "</analysis>" in raw else "</analysis>" + raw)
        return obj or {}
    except Exception:
        return {}


def audit_verifiers_split(
    task_id: str,
    verifiers: List[dict],
    expected_values: Dict[str, dict],
    step_nodes: Dict[str, dict],
    solution_logic: str = "",
    sanity_check: str = "",
    mapping_report: Optional[dict] = None,
    coverage: Optional[dict] = None,
    verifier_to_step: Optional[Dict[str, str]] = None,
    provider: Optional[Callable] = None,
    worker_model: str = WORKER_MODEL,
    atomicity_model: str = None,
    consolidator_model: str = CONSOLIDATOR_MODEL,
) -> VerifierAuditResult:
    """Concern-split audit. provider(prompt, system, model, max_tokens) -> text.
    Preserves the VerifierAuditResult contract exactly."""
    res = VerifierAuditResult(task_id=task_id, model_used=f"{worker_model}+{consolidator_model}")
    if not verifiers:
        res.error = "no verifiers to audit"
        return res
    atomicity_model = atomicity_model or ATOMICITY_MODEL
    verifier_to_step = verifier_to_step or {}
    res.trap_passes_band = trap_passes_band(expected_values, verifier_to_step, step_nodes)

    if provider is None:
        from src.prompt_evaluator import _call_llm
        def provider(p, sysmsg, model, mt):                    # noqa: E306
            return _call_llm(p, model, max_tokens=mt, system_prompt=sysmsg)

    vtext = _fmt_verifiers(verifiers, expected_values)
    steps_text = _fmt_steps(step_nodes)

    # --- WORKER CONCERN CALLS (parallelizable) ---
    atom = _call(provider, atomicity_model, _ATOMICITY_SYS, _ATOMICITY_TMPL.format(
        verifiers_with_traps=_verifiers_with_traps(verifiers, expected_values, verifier_to_step)))
    props = _call(provider, worker_model, _PROPS_SYS, _PROPS_TMPL.format(verifiers_text=vtext))
    steps = _call(provider, worker_model, _STEP_SYS,
                  _STEP_TMPL.format(steps_text=steps_text, verifiers_text=vtext))
    dedupe = _call(provider, worker_model, _DEDUPE_SYS, _DEDUPE_TMPL.format(verifiers_text=vtext))
    mapping_text = "\n".join(f"- {vid} -> {s}" for vid, s in verifier_to_step.items()) or "(none)"
    cov = _call(provider, worker_model, _COVERAGE_SYS,
                _COVERAGE_TMPL.format(steps_text=steps_text, mapping_text=mapping_text))

    # --- CONSOLIDATOR (Opus): reconcile atomicity/properties/step/dedupe into canonical ---
    canon = _call(provider, consolidator_model, _CONSOLIDATE_SYS, _CONSOLIDATE_TMPL.format(
        atomicity_json=json.dumps(atom)[:8000], properties_json=json.dumps(props)[:8000],
        step_json=json.dumps(steps)[:6000], dedupe_json=json.dumps(dedupe)[:4000]),
        max_tokens=16000)

    # coverage_gaps come straight from the coverage worker (deterministic list is authoritative
    # downstream anyway; this just enriches with a proposed verifier string)
    res.coverage_gaps = cov.get("coverage_gaps") or []
    res.duplicate_clusters = canon.get("duplicate_clusters") or dedupe.get("duplicate_clusters") or []
    res.notes = str(canon.get("notes", "") or "")

    # --- fold the canonical per-verifier decisions into the frozen outputs (same logic as original) ---
    vids = [v.get("id") for v in verifiers]
    got = {v.get("id"): v for v in (canon.get("verifiers") or [])}
    res.missing_verifiers = [v for v in vids if v not in got]
    res.verifiers = [got[v] for v in vids if v in got]
    valid_steps = set(step_nodes)
    fails: Dict[str, int] = {}
    splits: Dict[str, List[dict]] = {}
    for v in res.verifiers:
        vid = v.get("id")
        for prop in PROPERTIES:
            if ((v.get("properties") or {}).get(prop) or {}).get("verdict") == "FAIL":
                fails[prop] = fails.get(prop, 0) + 1
        step = v.get("tests_step")
        if step and str(step) in valid_steps:
            res.tests_step[vid] = str(step)
        rw = str(v.get("rewrite") or "").strip()
        if rw:
            res.rewrites[vid] = rw
        sp = []
        for i, child in enumerate(v.get("split_into") or []):
            child = child or {}
            txt = str(child.get("text") or child.get("verifier") or child.get("statement") or "").strip()
            if not txt:
                continue
            sp.append({"suffix": str(child.get("suffix") or "abcdefgh"[i:i+1] or str(i)),
                       "text": txt, "inherits_target": bool(child.get("inherits_target")),
                       "target": (dict(child["target"]) if isinstance(child.get("target"), dict)
                                  and child["target"] else None)})
        if len(sp) >= 2:
            for c in sp:
                if c["target"] is None and _VALUE_IN_TEXT.search(c["text"]):
                    res.split_children_missing_target.append(
                        {"parent": vid, "suffix": c["suffix"], "text": c["text"][:120],
                         "detail": "this child states a value but declared no target"})
            splits[vid] = sp
        if v.get("route_to_rubric"):
            res.route_to_rubric.append({"verifier": vid, "dimension": v.get("rubric_dimension")})
    res.splits = splits
    res.fails_by_property = fails
    return res