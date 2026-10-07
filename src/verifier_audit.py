# src/verifier_audit.py
"""
Fourth call: are the verifiers themselves well formed, and which step does each
prose verifier test?

WHY A CALL AND NOT CODE
    Atomicity, mutual exclusivity, self-containment and falsifiability are
    semantic. A regex attempt at them measured 59% precision on a hand-labelled
    sample of 25 — too low to act on. Matching a prose verifier to its judgment
    step is the same kind of judgment: "reject adding permanent front end
    staffing" and "Option A add front-end staff or Option B pool" mean the same
    thing and share almost no vocabulary (name agreement 0.18).

WHY THE OUTPUT IS FROZEN
    The derived verifier DAG must be repeatable — same trajectory in, same graph
    out. So this call runs ONCE, its resolved mappings are frozen into the
    artifact, and derive_verifier_dag reads the frozen mapping. Determinism is
    preserved by freezing the non-deterministic step, not by avoiding it.

WHAT IT MAY AND MAY NOT CHANGE
    May: rewrite a verifier IN PLACE, keeping its id.
    May: resolve which step a verifier tests.
    May: split a verifier. Children take SUFFIXED ids — V5 becomes V5a and V5b —
    so the parent's lineage stays readable and no existing id is renumbered. Since
    the dependency graph is DERIVED from the trajectory rather than asserted, a
    split needs no edge surgery: the children are mapped to steps and the graph,
    weights, crux set and Shapley weights all recompute from them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from src.arithmetic_verifier import parse_number

DEFAULT_MODEL = "claude-opus-5"
BASE_TOKENS = 16000
PER_VERIFIER_TOKENS = 550
#: Opus 4.8 allows 128k output tokens on the Messages API and _call_llm streams,
#: so the old 32k cap was four times too conservative — it truncated on the three
#: real tasks carrying 83, 60 and 32 verifiers. Adaptive thinking shares this
#: budget with the visible text, hence the generous headroom.
#: https://platform.claude.com/docs/en/about-claude/models/overview
TOKEN_CAP = 96000

PROPERTIES = ("atomic", "quantifiable", "self_contained", "falsifiable",
              "toleranced", "content_not_ordinal")

#: A number large enough to be a scoring target rather than a label or a count.
#: Used to catch a split child that states a value but declared no target — the
#: single most common way a split is wasted: on one task a clean three-way split
#: came back with one target, leaving two thirds of the new verifiers dead.
_VALUE_IN_TEXT = re.compile(
    r"\d[\d,]*\.?\d*\s*(?:%|mln|cr\b|crore|lakh|days?|/kg|/hr|units?|per\s)"
    r"|[$\u20b9\u20ac\u00a3]\s?\d",
    re.IGNORECASE)

#: A verifier that leans on a LIVE, unpinned value — "today's market rate", "the
#: current index", "the latest price" — decays as the world moves and is a
#: temporal_drift defect. Call 1 screens the ORIGINAL verifiers for this, but a
#: split mints NEW verifier text after Call 1 has finished, so a live value that
#: first appears in a split child would otherwise never be screened. This catches
#: the unpinned phrasing; a value already tied to a date or a named source file
#: (e.g. "the Jan-2026 survey rate") does not match, because the pin is present.
_TEMPORAL_UNPINNED = re.compile(
    r"\b(?:today'?s|current(?:ly)?|latest|prevailing|live|now|present|"
    r"as[- ]of[- ]today|market)\b[^.]{0,40}\b(?:rate|price|index|value|"
    r"level|wage|cost|yield|fx|exchange)\b",
    re.IGNORECASE)
#: If the child ALSO names a pin (a date, a year, or 'file'/'source'/'survey'/
#: 'annexure'/'vintage'), the live phrasing is anchored and is not drift.
_TEMPORAL_PINNED = re.compile(
    r"\b(?:20\d{2}|q[1-4]\s*20\d{2}|dated|vintage|as[- ]of\s+\w+\s+\d|"
    r"file|source\s+file|annexure|survey|input\s+file)\b",
    re.IGNORECASE)


def _child_has_unpinned_live_value(text: str) -> bool:
    """A split child states a live value with no date/source pin.

    Modular temporal re-screen: apply_splits calls this on every child so a live
    value introduced by a split is caught at creation, closing the gap where a
    verifier born after Call 1 escaped the only temporal screen.
    """
    t = text or ""
    return bool(_TEMPORAL_UNPINNED.search(t)) and not _TEMPORAL_PINNED.search(t)

SYSTEM = """\
You are auditing whether each verifier in a benchmark is well FORMED. You are not \
grading a response, not re-deriving the answer, and not judging whether the \
verifier's target value is correct — the target is already frozen and verified.

Two failure directions, and the second is worse. Passing a badly formed verifier \
means a scoring criterion nobody can apply consistently. But flagging a sound one \
costs a human's attention and, if the rewrite lands, changes what the benchmark \
measures. So when a verifier is defensible under a reasonable reading, PASS it.

Quote the span you are objecting to. A verdict of FAIL with no quoted text is not \
reviewable and will be treated as noise.

You may rewrite a verifier in place, and you may split one that tests two things. \
A split keeps the parent's number with a letter suffix, so V5 becomes V5a and V5b \
and no other verifier is renumbered. Say which child carries the parent's frozen \
target; a child that carries none is not eligible for the crux set, which is the \
correct outcome for a structural check.\
"""

PROPERTY_SPEC = """\
=== ATOMIC ===
One verifier, one assertion. ONE ASSERTION = ONE PREDICATE ON ONE REFERENT, WHERE
THE REFERENT MAY BE A NAMED SET.
  atomic:     "the bridge contains exactly these 7 rows, in order: (1)... (7)..."
              — one predicate ("contains exactly, in order") on an enumerated set
  NOT atomic: "bridge has 7 rows AND row 3 is X AND row 4 is Y" — three assertions
Do NOT flag: a number that is part of the single target's derivation ("= 162,532.41
(4.0% of FY2025 labour)"); a unit or window ("240 tickets in 4 hours"); a number
inside a label ("Phase 2", "2-wheeler", "FY2025"); or a fail-if half plus a target
half of the SAME quantity.

THE SET CASE — DECIDE IT WITH THIS TEST, NOT BY FEEL.
"the referent may be a named set" is ambiguous on its own: "allocation: PC = 2.565,
Filtration = 1.215, IC = 1.620" reads as one predicate over a three-member set AND
as three assertions, and both are defensible. That ambiguity made the same verifier
pass as atomic on one run and split three ways on the next. Resolve it by asking:

    IS A PARTIAL FAILURE WORTH ATTRIBUTING?

  * NO -> atomic. "contains exactly these 7 rows, in order" — a response either
    produced the right bridge or it did not; knowing which row is wrong buys the
    grader nothing, because the bridge is the unit of correctness.
  * YES -> NOT atomic, split it. If any single member carries diagnostic weight of
    its own, bundling makes a response that gets one member wrong and the rest
    right unscoreable in either direction.

A member carries diagnostic weight when it is any of:
  * the TRAP element — the one a shallow solver gets wrong. You are given each
    step's trap value; if a member matches or relates to one, the set MUST split.
  * a DECISION, or an input to one, where members can disagree (two segments
    flagged and one not).
  * separately reported in the deliverable, so a grader could observe one member
    without the others.

Worked example, from a real task: overhead allocated across three segments where
the whole trap is that the obsolete driver wrongly pushes ONE segment below the
threshold. That segment's figure is the discriminating fact. Bundled, a response
that gets it wrong and the other two right cannot be graded correctly. SPLIT.

Counter-example: a bridge table's seven rows, where the task is to reconcile the
bridge. No single row is the trap and none is reported alone. ATOMIC.

=== QUANTIFIABLE ===
The target is drawn from a stated, finite, checkable set. NOT "must be a number":
  numeric   — a number with a band
  decision  — a member of a stated set: {GO, NO-GO}, {ACCEPT, REJECT}
  presence  — {present, absent}: "names the brand-dilution risk", "cites the URL"
The genuinely bad case is the open judgment with no finite answer set — "is the
analysis rigorous?" — which belongs in the 0-4 subjective rubric, not here. The
trap is the seductive middle: "the recommendation is well-justified" feels
checkable but states no set. Decompose it into presence checks or send it to the
rubric; never leave it as a pseudo-binary verifier.

=== SELF-CONTAINED ===
Could a competent grader holding the response, but who has NOT seen the golden,
score this from the verifier's own text? If not it points at the standard instead
of being it.
  "bridge is exactly 7 rows"     -> name the 7 rows
  "the correct discount rate"    -> state the rate
  "all required sections"        -> enumerate them
  "matches the golden"           -> inline the actual target

=== FALSIFIABLE ===
The verifier can be FAILED by a wrong answer. The question is not whether wrong
numbers are listed — it is whether anything a wrong solver could do would slip
past.

For a numeric verifier with a tight band, THE BAND IS THE FAILURE CONDITION and
that is sufficient. PASS it. Do not ask for a clause naming particular wrong
values: "annual savings = ₹2.0-2.4 Cr" already rejects ₹22 Cr, ₹1.5 Cr and every
other wrong figure, so appending "FAIL if ₹22 Cr" adds nothing and merely fits the
one error someone happened to notice. Enumerating wrong values is not rigour.

FAIL only where a wrong solver could SATISFY the verifier anyway:

  (a) Wrong route, right number. The verifier checks a value but not how it was
      obtained, and the value is reachable by a method the task forbids. A bridge
      that reconciles on wrong inputs passes "the bridge reconciles". Fix by
      naming the required source or method, not by listing wrong totals:
        weak:   "Exchange rate used = ₹92.60/USD"
        strong: "Exchange rate = ₹92.60/USD, fetched live for 19 Apr 2026 and
                cited; the file's rate must not be used as the source."
      The distinction is SOURCE, not arithmetic — both could print 92.60.

  (b) A large failing space with no boundary. Decisions, presence and string
      verifiers have no band, so "= ACCEPT" leaves open what a hedged answer,
      an absent recommendation or a conditional counts as:
        weak:   "Final decision = ACCEPT dual sourcing"
        strong: "Final decision = ACCEPT dual sourcing, stated unconditionally.
                A hedged or conditional recommendation, or no explicit decision
                line, fails."

  (c) The verifier asserts a relationship rather than a realised value, so it
      holds regardless of inputs: "the bridge reconciles", "totals are
      internally consistent".

You are shown each step's TRAP VALUE where one exists. Use it ONLY to judge
whether the trap could pass the verifier as written — not as a value to quote.
Whether the trap falls inside the frozen band is checked in code and reported to
you separately; do not duplicate that check.

When you fail one, the rewrite adds a source or method constraint, or a boundary
for a non-numeric answer. It does not add a list of wrong numbers.

=== HOW TO WRITE A VERIFIER: THE TARGET CLAUSE ===
Everything you write is free prose EXCEPT the clause that states what a correct
response must show. That clause has a fixed shape so it can be read by code as
well as by a grader. Measured on a real 29-verifier task, only 17 could be read
mechanically; the 12 failures were almost all punctuation drift, not missing
information.

WRITE THE TARGET CLAUSE LAST, IN ONE OF THESE FOUR FORMS:

  numeric    ... = <number> <unit> (+/- <band>)
             The band comes IMMEDIATELY after the value. Put any derivation
             BEFORE the "=", never between the value and the band.
               yes: Precision Components corrected operating margin (2.535/38)
                    = 6.67% (+/- 0.05pp)
               no:  ... = 6.67% (2.535/38) (+/- 0.05pp)     <- band unreachable
               no:  ... corrected to $1.215 mln (+/- 0.001) <- use "=", not "to"

  decision   ... = <TOKEN>
             One token from a set the verifier states. The token must follow the
             "=", because a decision word inside prose is not a target: a FORMAT
             verifier reading "each with its own explicit FLAG / DO NOT FLAG
             conclusion" was mis-read as a decision verifier.
               yes: Decision for Precision Components = FLAG
               no:  States an explicit decision to FLAG Precision Components

  string     ... = "<exact string>"
               yes: Workbook filename = "ABC_Q2FY26_MarginReview.xlsx"

  presence   no "=" clause at all. Presence checks assert that something is
             stated, cited or included, and have no value to compare.
               yes: Cites the Finance Policy Memo dated 23 March 2026
               yes: Workbook contains three dynamic, formula-driven tabs

  Then, if the verifier names a failing case, add it as its own final clause:
               FAIL IF <condition>
             e.g. "... = 8.25% (+/- 0.05pp). FAIL IF reported as 7.13%, which is
             what the obsolete employee-count basis produces."

ONE VALUE PER VERIFIER. A clause asserting several values at once ("confirms the
allocations: PC $2.565, Filtration $1.215, IC $1.620") has no single target and
cannot be read mechanically — and by the atomicity test above it should have been
split anyway. If it genuinely must stay whole, write it as a presence check.

This constrains only the tail. The description before it is yours: say whatever
the verifier needs to say, including for a coverage gap you are filling.

=== TOLERANCED ===
A numeric verifier states its band IN ITS OWN TEXT, proportioned to the quantity
and no wider than the task's determinism requires. The frozen target carries a
`tol` machine-side, but a grader reading only the verifier must be able to see
what counts as close enough.

Three things are routinely confused:
  a BAND bounds distance from a stated target: "= 1,628,354.48 +/- 500",
    "within 0.5%", "in the range 8580 to 8780", "₹127-132"
  a THRESHOLD says which side of a line to be on: ">= 95%", "< 18% hurdle". A
    threshold is NOT a band.
  an UNDECLARED BAND is the defect: "~₹5.09", "about 17.2 hours",
    "approximately 50.8%". The word "about" is not a band. State the number and
    its band.
You are shown each verifier's frozen tol. FAIL when the text carries no band and
tol is non-zero (the band exists but only the machine can see it), when the text
says "~" or "about" instead of a band, or when the text's range and the frozen
tol disagree materially. PASS exact integer counts, given input constants,
decisions and strings — say so rather than flagging them.

=== CONTENT_NOT_ORDINAL ===
Pin by semantic key ("the pre-opening-costs row"), not by position ("row 3").
Position is legitimate ONLY when the prompt fixes the layout — say so if it does.

=== MUTUAL EXCLUSIVITY (set-level) ===
No two verifiers assert the same fact. Duplicates double-weight whatever they
duplicate and can ship a self-contradicting golden — two verifiers once asserted
the same CAGR at 22.78% and 17.11%, invisible until they disagreed. Duplicates are
defects EVEN WHEN THEY AGREE, because they drift apart later.
"""

TEMPLATE = """\
{property_spec}

===========================================================================
TASK {task_id}
===========================================================================

## CORRECTED SOLUTION LOGIC
{solution_logic}

## SANITY CHECK (what makes a fact decisive)
{sanity_check}

## DERIVATION STEPS
(each step's id, what it computes, its value, and — where one exists — the TRAP
VALUE a shallow solver would get instead. A verifier on a step with a trap value
must name that value as its failing case.)
{steps_text}

## VERIFIERS, WITH THEIR FROZEN TARGETS
{verifiers_text}

## WHAT DETERMINISTIC CHECKS ALREADY FOUND
These are not conclusions — they are where to look. Value-based matching placed
some verifiers onto steps and could not place others.

Unplaced, with the closest step it could find:
{near_misses_text}

Unplaced with no candidate at all:
{unmatched_text}

Steps watched by more than one verifier (possible duplication):
{double_watched_text}

Verifiers whose band ACCEPTS the trap value (checked in code — these are
falsifiability failures already established, so fail them and widen the fix to a
source or method constraint rather than restating the number):
{trap_passes_text}

Derivation steps NO verifier watches:
{unwatched_text}

===========================================================================
WHAT TO PRODUCE
===========================================================================

Work in an <analysis> block: first cluster the verifiers by the QUANTITY each
asserts, so duplication is visible; then walk them one at a time.

Then emit a SINGLE JSON object after </analysis>:

{{
  "task_id": "{task_id}",
  "verifiers": [
    {{
      "id": "V1",
      "asserted_quantity": "canonical name of the ONE thing this asserts",
      "kind": "numeric|decision|presence|string|not_a_binary_verifier",
      "properties": {{
        "atomic":              {{"verdict": "PASS|FAIL", "evidence": "quoted span or ''", "why": "one line"}},
        "quantifiable":        {{"verdict": "PASS|FAIL", "evidence": "", "why": ""}},
        "self_contained":      {{"verdict": "PASS|FAIL", "evidence": "", "why": ""}},
        "falsifiable":         {{"verdict": "PASS|FAIL", "evidence": "", "why": ""}},
        "toleranced":          {{"verdict": "PASS|FAIL|NOT_APPLICABLE", "evidence": "", "why": ""}},
        "content_not_ordinal": {{"verdict": "PASS|FAIL|NOT_APPLICABLE", "evidence": "", "why": ""}}
      }},
      "tests_step": "C4 or J2 or null",
      "tests_step_why": "why this verifier tests that step, or why it tests no step",
      "rewrite": "the verifier rewritten in place to satisfy every property, or '' if unchanged",
      "split_into": [],
      "_split_into_example": [{{"text": "child verifier statement", "suffix": "a", "inherits_target": true}}, {{"text": "second child statement", "suffix": "b", "inherits_target": false, "target": {{"value": 12.5, "tol": 0.1}}}}],
      "route_to_rubric": false,
      "rubric_dimension": null
    }}
  ],
  "duplicate_clusters": [
    {{"quantity": "what is double-asserted", "verifier_ids": ["V17", "V18"],
      "values_agree": true, "recommended_action": "keep V17; V18 restates it"}}
  ],
  "coverage_gaps": [
    {{"step": "C13", "why_it_matters": "", "proposed_verifier": ""}}
  ],
  "notes": ""
}}

Rules:
- EVERY verifier id above must appear in "verifiers", in order. None may be omitted.
- "tests_step" is the id of the derivation step this verifier checks, or null when
  it tests something outside the derivation — a format requirement, or what the
  report must STATE rather than compute. Null is a legitimate answer; do not
  stretch for a match. This field is FROZEN and a dependency graph is derived from
  it, so a wrong link silently reweights the whole scoring scheme.
- "rewrite" keeps the SAME id and must preserve the verifier's intent and its
  frozen target value. Use it for wording defects: naming a fail-if condition,
  inlining a target the verifier only pointed at, replacing a positional
  reference with a semantic one.
  STATE ONLY THE TESTABLE CONDITION, precisely and comprehensively, but
  SUCCINCTLY. A verifier is a check, not an explanation: give the condition and,
  if needed, a short "fails if X" clause — do NOT narrate the derivation, the
  trap, or WHY a wrong answer is wrong. Aim for one clause; a rewrite over ~25
  words is almost always carrying reasoning that belongs in the solution logic,
  not the verifier. E.g. prefer "Required front-desk staffing = 4 FTE (fails if
  3)." over "...= 4 FTE (ceil of the 3.333 Erlang offered load); FAIL IF reported
  as 3, which is what omitting the round-up to a stable capacity produces."
- "split_into" replaces a non-atomic verifier with two or more children. For
  each child give: "text" (the child verifier's full statement, stated with the
  same succinctness rule as a rewrite — condition only, no reasoning), a short
  "suffix" (a, b, c), and set "inherits_target" true on exactly the ONE child the
  parent's frozen value belongs to. An empty list means no split. Put the child's
  statement under the key "text" — not "verifier" or "statement".
- IF ANOTHER CHILD ALSO ASSERTS A NUMBER, give it its own "target". A child that
  asserts a value with no target cannot be scored: there is nothing to compare
  against. Observed: "Lead time applied correctly (40 days vs 10 days)" split into
  40-days and 10-days children, and the second was left targetless. Only a purely
  structural child ("the row is present") may have no target.
- If a verifier is an irreducibly open judgment, set kind to
  "not_a_binary_verifier", route_to_rubric true, and name the dimension.
- Output ONLY the <analysis> block then the JSON. No prose after the JSON.
"""


@dataclass
class VerifierAuditResult:
    task_id: str
    model_used: str = ""
    verifiers: List[dict] = field(default_factory=list)
    duplicate_clusters: List[dict] = field(default_factory=list)
    coverage_gaps: List[dict] = field(default_factory=list)
    notes: str = ""
    # frozen outputs the deterministic layer consumes
    tests_step: Dict[str, str] = field(default_factory=dict)
    rewrites: Dict[str, str] = field(default_factory=dict)
    splits: Dict[str, List[dict]] = field(default_factory=dict)
    route_to_rubric: List[dict] = field(default_factory=list)
    #: Deterministic falsifiability failure: the trap value fits inside the band.
    trap_passes_band: List[dict] = field(default_factory=list)
    #: Split children whose text states a value but which carry no target.
    split_children_missing_target: List[dict] = field(default_factory=list)
    missing_verifiers: List[str] = field(default_factory=list)
    fails_by_property: Dict[str, int] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_json(text: str) -> Optional[dict]:
    tail = text.split("</analysis>")[-1]
    start = tail.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(tail[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = tail[start:i + 1]
                for attempt in (blob, re.sub(r",(\s*[}\]])", r"\1", blob)):
                    try:
                        return json.loads(attempt)
                    except json.JSONDecodeError:
                        continue
                return None
    return None


def _fmt_verifiers(verifiers: List[dict], expected: Dict[str, dict]) -> str:
    out = []
    for v in verifiers:
        vid = v.get("id", "")
        ev = (expected or {}).get(vid) or {}
        if ev:
            tgt = (f"  [frozen target: value={ev.get('value')} tol={ev.get('tol')} "
                   f"unit={ev.get('unit','')} kind={ev.get('kind','')}]")
        else:
            # a verifier with no frozen target cannot be scored at all; say so
            # loudly rather than leaving the call to infer it
            tgt = "  [NO FROZEN TARGET — nothing to score this against]"
        out.append(f"- {vid}: {v.get('text','')}{tgt}")
    return "\n".join(out) or "(none)"


def _fmt_steps(nodes: Dict[str, dict]) -> str:
    out = []
    for sid, n in nodes.items():
        if n.get("kind") == "arithmetic":
            trap = n.get("trap_value")
            # the wrong answer is what a falsifiable verifier on this step must name
            tail = (f"   TRAP VALUE (what a shallow solver gets): {trap}"
                    if trap is not None else "")
            out.append(f"- {sid} [computed] {n.get('label','')} = {n.get('value')}"
                       + tail)
        else:
            out.append(f"- {sid} [judgement] {n.get('label','')} "
                       f"-> {n.get('ruling','')}")
    return "\n".join(out) or "(none)"


def trap_passes_band(expected_values: Dict[str, dict],
                     verifier_to_step: Dict[str, str],
                     step_nodes: Dict[str, dict]) -> List[dict]:
    """Verifiers whose band is wide enough to ACCEPT the trap value.

    This is the one falsifiability failure that needs no judgment: if the trap
    value sits inside the frozen band, a solver who fell for the trap passes the
    verifier, and the verifier tests nothing on the axis it exists for. Checking
    it in code keeps the model from being asked to enumerate wrong numbers — which
    is both redundant against the band and a straight invitation to overfit the
    one error someone happened to see.
    """
    from src.arithmetic_verifier import parse_number

    out = []
    for vid, sid in (verifier_to_step or {}).items():
        ev = (expected_values or {}).get(vid) or {}
        node = (step_nodes or {}).get(sid) or {}
        target = parse_number(ev.get("value"))
        trap = parse_number(node.get("trap_value"))
        if target is None or trap is None:
            continue
        tol = abs(parse_number(ev.get("tol")) or 0.0)
        if abs(trap - target) <= tol:
            out.append({
                "verifier": vid, "step": sid, "target": target, "tol": tol,
                "trap_value": trap,
                "detail": (f"the trap value {trap:g} is inside the band "
                           f"{target:g} +/- {tol:g}, so a solver who fell for the "
                           f"trap passes this verifier"),
            })
    return out


def audit_verifiers(
    task_id: str,
    verifiers: List[dict],
    expected_values: Dict[str, dict],
    step_nodes: Dict[str, dict],
    solution_logic: str = "",
    sanity_check: str = "",
    mapping_report: Optional[dict] = None,
    coverage: Optional[dict] = None,
    verifier_to_step: Optional[Dict[str, str]] = None,
    model: str = DEFAULT_MODEL,
    provider=None,
) -> VerifierAuditResult:
    """Run the property audit. `provider(prompt, system, max_tokens) -> text`."""
    res = VerifierAuditResult(task_id=task_id, model_used=model)
    if not verifiers:
        res.error = "no verifiers to audit"
        return res

    mapping_report = mapping_report or {}
    coverage = coverage or {}
    res.trap_passes_band = trap_passes_band(
        expected_values, verifier_to_step or {}, step_nodes)

    def _lines(items, fmt):
        return "\n".join(fmt(i) for i in items) or "(none)"

    prompt = TEMPLATE.format(
        property_spec=PROPERTY_SPEC, task_id=task_id,
        solution_logic=solution_logic or "(none)",
        sanity_check=sanity_check or "(none)",
        steps_text=_fmt_steps(step_nodes),
        verifiers_text=_fmt_verifiers(verifiers, expected_values),
        near_misses_text=_lines(
            mapping_report.get("near_misses") or [],
            lambda n: (f"- {n['verifier']} ~ {n['candidate']} "
                       f"(agreement {n['agreement']}): {n['candidate_text'][:90]}")),
        unmatched_text=_lines(
            mapping_report.get("unmatched") or [],
            lambda u: f"- {u['verifier']}: {u.get('reason','')}"),
        trap_passes_text=_lines(
            res.trap_passes_band,
            lambda t: f"- {t['verifier']} (step {t['step']}): {t['detail']}"),
        double_watched_text=_lines(
            list((coverage.get("double_watched") or {}).items()),
            lambda kv: f"- step {kv[0]} watched by {kv[1]}"),
        unwatched_text=_lines(
            [u for u in (coverage.get("unwatched") or [])
             if u.get("load_bearing") or u.get("is_terminal")],
            lambda u: (f"- {u['step']} ({'terminal' if u.get('is_terminal') else 'load-bearing'})"
                       f" {u.get('label','')[:80]}")),
    )
    budget = min(TOKEN_CAP, BASE_TOKENS + PER_VERIFIER_TOKENS * len(verifiers))

    if provider is None:
        from src.prompt_evaluator import _call_llm

        def provider(p, sysmsg, mt):                            # noqa: E306
            return _call_llm(p, model, max_tokens=mt, system_prompt=sysmsg)

    try:
        raw = provider(prompt, SYSTEM, budget)
    except Exception as e:                                       # noqa: BLE001
        res.error = f"call failed: {e}"
        return res
    obj = _extract_json(raw)
    if obj is None:
        res.error = "unparseable JSON from the verifier audit"
        return res

    vids = [v.get("id") for v in verifiers]
    got = {v.get("id"): v for v in (obj.get("verifiers") or [])}
    res.missing_verifiers = [v for v in vids if v not in got]
    res.verifiers = [got[v] for v in vids if v in got]
    res.duplicate_clusters = obj.get("duplicate_clusters") or []
    res.coverage_gaps = obj.get("coverage_gaps") or []
    res.notes = str(obj.get("notes", "") or "")

    valid_steps = set(step_nodes)
    fails: Dict[str, int] = {}
    splits: Dict[str, List[dict]] = {}
    for v in res.verifiers:
        vid = v.get("id")
        for prop in PROPERTIES:
            if ((v.get("properties") or {}).get(prop) or {}).get("verdict") == "FAIL":
                fails[prop] = fails.get(prop, 0) + 1
        step = v.get("tests_step")
        # a link to a step that does not exist is dropped, not trusted
        if step and str(step) in valid_steps:
            res.tests_step[vid] = str(step)
        rw = str(v.get("rewrite") or "").strip()
        if rw:
            res.rewrites[vid] = rw
        sp = []
        for i, child in enumerate(v.get("split_into") or []):
            # The child's text has been emitted under several field names across
            # runs — "text", "verifier", "statement" — because the schema example
            # showed an empty list and never named the field. Accept any of them
            # so a correctly-proposed split is not silently dropped (observed: a
            # clean V5 "4 FTE AND wage" split came back under "verifier" and was
            # discarded, leaving the non-atomic verifier un-split). The template
            # now names "text" explicitly; this stays tolerant for older/variant
            # emissions.
            child = child or {}
            txt = str(child.get("text") or child.get("verifier")
                      or child.get("statement") or "").strip()
            if not txt:
                continue
            sp.append({"suffix": str(child.get("suffix") or
                                     "abcdefgh"[i:i + 1] or str(i)),
                       "text": txt,
                       # inheritance is a fallback for older emissions only
                       "inherits_target": bool(child.get("inherits_target")),
                       "target": (dict(child["target"])
                                  if isinstance(child.get("target"), dict)
                                  and child["target"] else None)})
        if len(sp) >= 2:                     # a "split" into one child is a rewrite
            # A child whose text states a value but carries no target cannot be
            # scored. This is the single most common way a split is wasted, so it
            # is reported rather than discovered later in the crux count.
            for c in sp:
                if c["target"] is None and _VALUE_IN_TEXT.search(c["text"]):
                    res.split_children_missing_target.append(
                        {"parent": vid, "suffix": c["suffix"],
                         "text": c["text"][:120],
                         "detail": ("this child states a value but declared no "
                                    "target, so nothing can score it")})
            splits[vid] = sp
        if v.get("route_to_rubric"):
            res.route_to_rubric.append(
                {"verifier": vid, "dimension": v.get("rubric_dimension")})
    res.splits = splits
    res.fails_by_property = fails
    return res


# ---------------------------------------------------------------------------------------------------
# which rewrites may be applied without the SME
# ---------------------------------------------------------------------------------------------------
_BAND_RE = re.compile(r"±|\+/-|\+-\s*\d|\bwithin\s+[\d.,]+|\btoleran\w*|\bband\b", re.I)
_VID_RE = re.compile(r"\bV\d+[a-z]?\b")
_NUM_RE = re.compile(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?")


def _nums(text: str):
    out = []
    for m in _NUM_RE.finditer(_VID_RE.sub(" ", str(text or ""))):
        raw = m.group(0).replace(",", "")
        try:
            out.append((float(raw), len(raw.split(".")[1]) if "." in raw else 0, m.group(0)))
        except ValueError:
            pass
    return out


def _traceable(n: float, decimals: int, pool) -> bool:
    """n matches a trusted number, allowing the rounding its own precision implies (6.8 vs 6.8123)
    and percent/fraction form (6.8 vs 0.068)."""
    tol = 0.5 * 10 ** (-decimals)
    for t in pool:
        for cand in (t, t * 100, t / 100):
            if abs(n - cand) <= max(tol, 1e-9) or (cand and abs(n - cand) <= 0.005 * abs(cand)):
                return True
    return False


def classify_rewrites(verifiers: List[dict], rewrites: Dict[str, str], audited: Optional[List[dict]] = None,
                      trusted_text: str = "", expected_values: Optional[Dict[str, dict]] = None):
    """Split proposed rewrites into (auto: {id: text}, pending: [dict]).

    PENDING — needs the SME:
      * tolerance: the rewrite adds or changes a tolerance band, or the audit failed the verifier on
        'toleranced'. Choosing a band is an authorial decision.
      * untraceable: the rewrite contains a number found nowhere in the original verifier, the expected
        values or the corrected solution logic (CCL, 5 Oct: a rewrite wrote 'operating income 23,574'
        where the pack prints 3,574).
    AUTO — everything else (self-containment, falsifiability, wording) is applied without a card."""
    orig = {v.get("id"): v.get("text", "") for v in verifiers}
    failed_props = {}
    for v in audited or []:
        failed_props[v.get("id")] = [k for k, x in (v.get("properties") or {}).items()
                                     if (x or {}).get("verdict") == "FAIL"]
    pool = {n for n, _, _ in _nums(trusted_text)}
    for ev in (expected_values or {}).values():
        for k in ("value", "target", "expected", "golden"):
            try:
                pool.add(float(str((ev or {}).get(k)).replace(",", "")))
            except (TypeError, ValueError):
                pass
    auto, pending = {}, []
    for vid, new in (rewrites or {}).items():
        old = orig.get(vid, "")
        fails = failed_props.get(vid, [])
        band_added = bool(_BAND_RE.search(new or "")) and not _BAND_RE.search(old or "")
        if "toleranced" in fails or band_added:
            pending.append({"id": vid, "original": old, "proposed": new, "reason": "tolerance",
                            "properties_failed": fails})
            continue
        local = pool | {n for n, _, _ in _nums(old)}
        bad = [raw for n, d, raw in _nums(new) if not _traceable(n, d, local)]
        if bad:
            pending.append({"id": vid, "original": old, "proposed": new, "reason": "untraceable",
                            "untraceable_numbers": bad, "properties_failed": fails})
            continue
        auto[vid] = new
    return auto, pending


def apply_rewrites(verifiers: List[dict], rewrites: Dict[str, str]) -> List[dict]:
    """Apply in-place rewrites. IDs are preserved, so nothing downstream shifts.

    Splits are deliberately NOT handled here — see the module docstring.

    A "fails if <n>" clause is KEPT verbatim. An earlier version stripped numeric
    fail-if clauses to guard against a fabricated trap value (the model once wrote
    "fails if 1" where the real trap is 3). That was wrong twice over: a regex
    cannot tell a fabricated number (1) from a correct one (3) — they are
    identical text — so it also destroyed correct clauses ("= 4 FTE (fails if 3)"
    -> "= 4 FTE ("), and it left broken syntax. A fail-if clause is legitimate and
    useful; the real defect (an implausible trap VALUE) lives in the claims stage
    and is a model-judgment problem, not a text-surgery one. So no stripping.
    """
    out = []
    for v in verifiers:
        nv = dict(v)
        rw = (rewrites or {}).get(v.get("id"))
        if rw:
            nv["text"] = rw
            nv["rewritten"] = True
        out.append(nv)
    return out


def _vnum(vid: str) -> int:
    m = re.match(r"V(\d+)", vid or "")
    return int(m.group(1)) if m else 10 ** 6


def apply_splits(verifiers: List[dict], splits: Dict[str, List[dict]],
                 expected_values: Dict[str, dict]):
    """Replace each split parent with suffixed children, in place in the ordering.

    V5 becomes V5a, V5b. No other id moves, so a reader can still see where a
    verifier came from and no unrelated verifier is renumbered. The frozen target
    follows the child marked inherits_target; children without one simply have no
    target, which makes them ineligible for the crux set — correct for a
    structural check.

    Returns (verifiers, expected_values, log).
    """
    out, ev, log = [], dict(expected_values or {}), []
    for v in verifiers:
        vid = v.get("id")
        children = (splits or {}).get(vid)
        if not children:
            out.append(dict(v))
            continue
        parent_target = ev.pop(vid, None)
        heir = next((c for c in children if c.get("inherits_target")), None)
        made, targetless, temporal_unpinned = [], [], []
        for c in children:
            cid = f"{vid}{c['suffix']}"
            out.append({**v, "id": cid, "text": c["text"], "split_from": vid})
            if parent_target is not None and c is heir:
                ev[cid] = parent_target
            elif isinstance(c.get("target"), dict) and c["target"]:
                ev[cid] = dict(c["target"])
            else:
                targetless.append(cid)
            # temporal re-screen: a child born here with a live, unpinned value
            # never passed through Call 1, so flag it for the SME to pin at seal.
            if _child_has_unpinned_live_value(c["text"]):
                temporal_unpinned.append(cid)
            made.append(cid)
        log.append({"parent": vid, "children": made,
                    "target_went_to": (f"{vid}{heir['suffix']}" if heir and
                                       parent_target is not None else None),
                    "children_with_own_target": [f"{vid}{c['suffix']}" for c in children
                                                 if c is not heir and c.get("target")],
                    # a child with no target cannot be scored; structural children
                    # are legitimately targetless, a numeric one is not
                    "targetless_children": targetless,
                    # a child carrying a live value with no date/source pin — a
                    # temporal_drift defect created after Call 1's screen, so it
                    # is surfaced here for the SME to pin before sealing
                    "temporal_unpinned_children": temporal_unpinned,
                    "parent_text": v.get("text", "")[:120]})
    out.sort(key=lambda x: (_vnum(x["id"]), x["id"]))
    return out, ev, log


def format_verifiers_ids(verifiers: List[dict]) -> str:
    """Canonical text keyed by the id string, so suffixed ids survive.

    verifier_parser.format_verifiers rebuilds ids from an integer index and would
    turn V5a back into V5.
    """
    return "\n".join(f"{v['id']}: {v.get('text','')}"
                      for v in sorted(verifiers,
                                      key=lambda x: (_vnum(x["id"]), x["id"])))