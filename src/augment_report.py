# src/augment_report.py
"""
Per-task HTML emitters for the augmenter:
  write_augment_report(res, path)   -> {task_id}_augment.html  (corrections, DAG,
                                        crux set + Shapley weights, verifiers)
  write_golden_report(res, path)    -> {task_id}_golden.html    (golden deliverable)

Both are self-contained, dependency-free HTML for SME eyeballing.
"""

from __future__ import annotations
import html
import re

#: Verifier-id pattern — classic (V1, V5a) and semantic (V_P1_pat_2004) ids.
_VIDR = r"V(?:\d+[a-z]?|_[A-Za-z0-9][A-Za-z0-9_.]*)"
_VIDR_BARE = r"\b" + _VIDR + r"\b"
from typing import Dict


_PROPS = ("atomic", "quantifiable", "self_contained", "falsifiable",
          "toleranced", "content_not_ordinal")
_PROP_LABEL = {"atomic": "Atomic", "quantifiable": "Quantifiable",
               "self_contained": "Self-contained", "falsifiable": "Falsifiable",
               "toleranced": "Toleranced", "content_not_ordinal": "By content"}


DECISION_JS = """
<script>
const TASK_ID  = "%(task_id)s";
const RUN_HASH = "%(run_hash)s";
const ITEMS    = %(items_json)s;   // [{id, kind, needs_answer}]
const decisions = {};
const answers   = {};

function decide(id, choice) {
  decisions[id] = choice;
  const card = document.getElementById('it_' + id);
  const st   = document.getElementById('st_' + id);
  card.classList.remove('accepted','rejected','other','undecided');
  card.classList.add(choice === 'accept' ? 'accepted'
                    : choice === 'reject' ? 'rejected' : 'other');
  // fill the clicked button and dim its siblings, so the click is unmistakable
  card.querySelectorAll('.btns button').forEach(function(b) {
    b.classList.remove('on-accept','on-reject','on-other','dim');
    if (b.dataset.choice === choice) { b.classList.add('on-' + choice); }
    else { b.classList.add('dim'); }
  });
  st.className = 'pill ' + (choice === 'accept' ? 'ok'
                          : choice === 'reject' ? 'bad' : 'warn');
  st.textContent = choice === 'accept' ? '\u2713 accepted'
                 : choice === 'reject' ? '\u2717 rejected'
                 : '\u25c6 something else';
  // Reject and Something Else REQUIRE a reason: a bare rejection tells the
  // pipeline to drop a change without recording why, which is unauditable.
  const box = document.getElementById('why_' + id);
  if (box) { box.style.display = (choice === 'accept') ? 'none' : 'block'; }
  tally();
}

function note(id, el) { answers[id] = el.value; tally(); }

function tally() {
  let done = 0, blocked = [];
  ITEMS.forEach(function(it) {
    const c = decisions[it.id];
    const a = (answers[it.id] || '').trim();
    if (!c) return;
    if (c !== 'accept' && !a) { blocked.push(it.id + ' needs a reason'); return; }
    if (it.needs_answer && !a) { blocked.push(it.id + ' needs an answer'); return; }
    done++;
  });
  const el = document.getElementById('tally');
  el.textContent = done + ' of ' + ITEMS.length + ' complete'
                 + (blocked.length ? '  \u2014  ' + blocked.slice(0,3).join('; ') : '')
                 + (done === ITEMS.length ? '  \u2014 ready to save' : '  \u2014 unsaved');
  el.className = 'pill ' + (done === ITEMS.length ? 'ok' : 'warn');
}

function saveDecisions() {
  const missing = ITEMS.filter(function(it) {
    const c = decisions[it.id]; const a = (answers[it.id] || '').trim();
    return !c || (c !== 'accept' && !a) || (it.needs_answer && !a);
  }).map(function(it) { return it.id; });
  if (missing.length) {
    var NL = String.fromCharCode(10);
    if (!confirm('Incomplete: ' + missing.join(', ') + NL + NL
                 + 'Save anyway? apply_decisions.py will refuse to apply '
                 + 'an incomplete file.')) return;
  }
  const payload = {task_id: TASK_ID, run_hash: RUN_HASH,
                   saved_at: new Date().toISOString(),
                   decisions: decisions, answers: answers, incomplete: missing};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'decisions_' + TASK_ID + '_'
             + new Date().toISOString().slice(0,10) + '.json';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  document.getElementById('tally').textContent = 'saved \u2713';
  document.getElementById('tally').className = 'pill ok';
}

window.addEventListener('beforeunload', function(e) {
  if (Object.keys(decisions).length &&
      document.getElementById('tally').textContent.indexOf('saved') === -1) {
    e.preventDefault(); e.returnValue = '';
  }
});
window.addEventListener('load', tally);
</script>
"""

DECISION_CSS = """
.card{border:1px solid #e7e2d8;border-radius:8px;padding:12px;margin:10px 0;background:#fff}
.card.accepted{border-left:6px solid #1a7a4a;background:#eef8f2}
.card.rejected{border-left:6px solid #b4413c;background:#fbeceb}
.card.other{border-left:6px solid #9a7400;background:#fdf6e3}
.card.undecided{border-left:6px solid #d6d0c4}
.card h4{margin:0 0 6px 0;font-size:15px}
.btns{margin-top:8px}
.btns button{font-size:13px;padding:5px 14px;margin-right:6px;border-radius:6px;
  border:1px solid #d6d0c4;background:#f6f3ec;cursor:pointer;font-weight:600;
  color:#4a463d}
.btns button:hover{background:#ece7dc}
/* the clicked button fills in. Previously only the CARD changed, with a tint of
   #f6fbf8 on a #fbfaf7 page — invisible — so a click looked like it did nothing. */
.btns button.on-accept{background:#1a7a4a;border-color:#166b41;color:#fff}
.btns button.on-reject{background:#b4413c;border-color:#9c3833;color:#fff}
.btns button.on-other{background:#9a7400;border-color:#836300;color:#fff}
.btns button.dim{opacity:.45}
.why{display:none;margin-top:8px}
.why textarea{width:100%;min-height:52px;font-family:inherit;font-size:13px;
  padding:6px;border:1px solid #d6d0c4;border-radius:6px}
.ask{margin-top:8px}
.ask textarea{width:100%;min-height:64px;font-family:inherit;font-size:13px;
  padding:6px;border:1px solid #9a7400;border-radius:6px;background:#fffdf5}
.sticky{position:sticky;top:0;background:#fbfaf7;padding:8px 0;z-index:5;
  border-bottom:1px solid #e7e2d8}
"""


def _step_index(res: dict) -> dict:
    """Map every step id (claim or judgment) to its content, so a gap on that step
    can be turned into an authored verifier suggestion."""
    idx = {}
    for c in res.get("corrected_claim_verdicts", []):
        idx[str(c.get("id"))] = {
            "kind": "claim", "label": c.get("label") or "",
            "value": c.get("recomputed"),
            "sov": c.get("source_of_verification")}
    for j in res.get("judgment_steps", []):
        idx[str(j.get("id"))] = {
            "kind": "judgment",
            "label": j.get("question") or j.get("label") or "",
            "value": j.get("expected") if j.get("expected") is not None
            else j.get("answer")}
    return idx


def _author_gap_verifier(step_id: str, step: dict) -> str:
    """Author a suggested verifier for a load-bearing step that no verifier
    watches, so the SME can Accept/Reject/Something-else rather than write one from
    scratch. Deterministic and content-derived — it phrases a check of exactly
    what the step computes or decides. A claim with a numeric value becomes an
    equality check on that value; a judgment becomes a decision check. This is a
    SUGGESTION (marked as such), not an applied change."""
    label = str(step.get("label") or "").strip() or step_id
    if step.get("kind") == "claim" and step.get("value") is not None:
        v = step["value"]
        try:
            fv = float(v)
            vs = str(int(fv)) if fv == int(fv) else f"{fv:.4g}"
        except (TypeError, ValueError):
            vs = str(v)
        return (f"The response computes {label} = {vs} "
                f"(within tolerance). FAIL IF absent or a different value.")
    if step.get("kind") == "judgment":
        return (f"The response explicitly resolves: {label} — with the "
                f"conclusion the solution logic reaches. FAIL IF the step is "
                f"skipped or decided the other way.")
    # claim without a numeric value (string/decision claim)
    return (f"The response states {label} consistent with the solution logic. "
            f"FAIL IF absent or contradictory.")


def _decision_items(res: dict):
    """Everything a human must rule on, as uniform cards.

    Four kinds, because they apply differently downstream:
      change   — a MECHANICAL edit already applied; reject to revert it
      question — an unresolved JUDGMENT_REQUIRED item; needs a written answer, and
                 until it has one the task is not scoreable, because the next run
                 may resolve it the other way and change the golden's answer
      rewrite  — a verifier reworded in place by the property audit
      split    — a verifier divided into children
      gap      — a proposed new verifier for an unwatched step
      dedupe   — two+ verifiers assert the same quantity; the audit recommends
                 keeping one and dropping the rest. Detected but, before this,
                 only DISPLAYED — so a redundant (non-MECE) verifier still shipped.
                 Now an actionable item: accept to drop the restated verifier(s).
    """
    items = []
    # verifier id -> text, from the canonical set; used by several item kinds
    vtext_map = {}
    for _line in (res.get("augmented_verifiers_text") or "").splitlines():
        _m = re.match(r"\s*(" + _VIDR + r")\s*(?:\[[^\]]*\])?\s*:\s*(.*)", _line)
        if _m:
            vtext_map[_m.group(1)] = _m.group(2).strip()
    # Verifiers the property audit REWROTE supersede any earlier Call-2 change to
    # the same verifier text — otherwise the SME sees two decisions for one
    # verifier (a "change" and a "rewrite"), each proposing its own wording. The
    # rewrite is later and wins, so the change item for that verifier is dropped.
    va = res.get("verifier_audit") or {}
    rewritten_vids = {v.get("id") for v in (va.get("verifiers") or [])
                      if (v.get("rewrite") or "").strip()}

    def _change_targets_rewritten_verifier(c):
        if str(c.get("artifact", "")).lower() != "verifiers":
            return False
        import re as _re
        loc = str(c.get("location", "")) + " " + str(c.get("old", ""))
        return any(vid in _re.findall(_VIDR_BARE, loc)
                   for vid in rewritten_vids)

    for i, c in enumerate(res.get("changes_applied") or []):
        if _change_targets_rewritten_verifier(c):
            continue  # superseded by a property-audit rewrite below
        items.append({"id": f"chg{i}", "kind": "change", "needs_answer": False,
                      "title": f"{c.get('artifact')} — {c.get('location')}",
                      "body": (f"<b>was:</b> {_esc(c.get('old'))}<br>"
                               f"<b>now:</b> {_esc(c.get('new'))}<br>"
                               f"<i>{_esc(c.get('rationale'))}</i>"),
                      "meta": "mechanical, already applied"})
    for i, c in enumerate(res.get("judgment_changes_pending_sme") or []):
        items.append({"id": f"q{i}", "kind": "question", "needs_answer": True,
                      "title": f"{c.get('artifact')} — {c.get('location')}",
                      "body": (f"<b>{_esc(c.get('sme_question'))}</b><br>"
                               f"<i>{_esc(c.get('rationale'))}</i>"),
                      "meta": "NOT applied — artifact keeps its original wording"})
    for i, v in enumerate(va.get("verifiers") or []):
        rw = (v.get("rewrite") or "").strip()
        if not rw:
            continue
        fails = [k for k, x in (v.get("properties") or {}).items()
                 if (x or {}).get("verdict") == "FAIL"]
        items.append({"id": f"rw{i}", "kind": "rewrite", "needs_answer": False,
                      "title": f"{v.get('id')} reworded ({', '.join(fails)})",
                      "body": f"<b>now:</b> {_esc(rw)}",
                      "meta": "applied in place, id unchanged"})
    # the children's text and their targets, because "V4 split into V4a, V4b, V4c"
    # with only the parent's old text shown gives no way to judge the split
    vtext, ev = {}, res.get("expected_values") or {}
    for line in (res.get("augmented_verifiers_text") or "").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            vtext[k.strip()] = v.strip()
    # cross-reference: a split child that duplicates an EXISTING verifier would
    # double-count if kept (e.g. V5 -> V5a[4 FTE] + V5b[wage], but V4 already
    # checks 4 FTE, so V5a duplicates V4). Surface that inside the split decision
    # so the SME resolves it in one place instead of connecting the split and the
    # dedupe decisions themselves.
    dup_of = {}
    for cl in (va.get("duplicate_clusters") or []):
        vids = cl.get("verifier_ids") or []
        keep = cl.get("keep") or (vids[0] if vids else None)
        for vid in vids:
            if vid != keep:
                dup_of[vid] = keep
    for i, x in enumerate(res.get("verifier_splits_applied") or []):
        kids = x.get("children") or []
        rows = []
        redundant_kids = []
        for k in kids:
            e = ev.get(k)
            tgt = (f"<span class='pill ok'>target {e.get('value')}"
                   f"{' ±' + str(e.get('tol')) if e.get('tol') else ''}</span>"
                   if e else "<span class='pill bad'>NO TARGET — unscoreable</span>")
            dup_note = ""
            if k in dup_of:
                redundant_kids.append(k)
                dup_note = (f"<br><span class='pill bad'>duplicates "
                            f"{_esc(dup_of[k])} — dropped to avoid double-counting"
                            f"</span>")
            rows.append(f"<li><span class='mono'>{_esc(k)}</span> {tgt}{dup_note}"
                        f"<br>{_esc(vtext.get(k, '(text not found)'))}</li>")
        tl = x.get("targetless_children") or []
        meta = (f"target to {x.get('target_went_to')}"
                + (f" — {len(tl)} child(ren) with no target: {', '.join(tl)}"
                   if tl else ""))
        if redundant_kids:
            meta += (f" — accepting drops duplicate child(ren) "
                     f"{', '.join(redundant_kids)}")
        items.append({"id": f"sp{i}", "kind": "split", "needs_answer": False,
                      "parent": x.get("parent"), "children": kids,
                      "redundant_children": redundant_kids,
                      "title": (f"{x.get('parent')} split into "
                                f"{', '.join(kids)}"),
                      "body": (f"<b>was:</b> {_esc(x.get('parent_text'))}"
                               f"<br><b>becomes:</b><ul>{''.join(rows)}</ul>"),
                      "meta": meta})
    # FIX 3 — a value-bearing verifier that maps to NO derivation step. The
    # mapping only links a numeric verifier to a step whose COMPUTED value matches
    # the verifier's stated value; so a verifier asserting a number the derivation
    # never produces (the planted "wait = 38" where the golden computes 42) lands
    # in `unmatched`. That is a value error hiding as a coverage miss: the verifier
    # is either factually wrong or the golden is missing the step it checks. The
    # property audit passes such a verifier — it is structurally clean — so this
    # deterministic signal is the only thing that catches it. Surface it.
    #
    # BUT the mapping uses ZERO tolerance for an exact match, so a verifier that
    # merely ROUNDS (target 3.333 vs a step's 3.33333) also lands in `unmatched`
    # though it is correct. Before flagging, re-check each unmatched numeric target
    # against every computed step value with a small RELATIVE tolerance: if it is
    # within rounding distance of some step, it is a rounding artifact, not a value
    # error, and must not be flagged. Only a genuinely-off value (38 vs 42, 26.6 vs
    # 29.4) is surfaced.
    step_vals = []
    for _sid, _node in (res.get("step_nodes") or {}).items():
        if isinstance(_node, dict) and isinstance(_node.get("value"), (int, float)):
            step_vals.append(float(_node["value"]))
    # step_nodes isn't always carried on the package; fall back to claim verdicts
    if not step_vals:
        for _c in (res.get("corrected_claim_verdicts") or []):
            rv = _c.get("recomputed")
            if isinstance(rv, (int, float)):
                step_vals.append(float(rv))

    def _is_rounding_of_some_step(target) -> bool:
        try:
            t = float(target)
        except (TypeError, ValueError):
            return False
        for sv in step_vals:
            denom = max(abs(sv), abs(t), 1.0)
            if abs(sv - t) <= 0.005 * denom:   # within 0.5% -> a rounding, not an error
                return True
        return False

    mapping = res.get("verifier_mapping_report") or {}
    for i, u in enumerate(mapping.get("unmatched") or []):
        if str(u.get("kind", "numeric")).lower() != "numeric":
            continue  # a non-numeric verifier not attaching is a weaker signal
        if _is_rounding_of_some_step(u.get("target")):
            continue  # correct value, just rounded — the mapping's 0-tol missed it
        vid = u.get("verifier")
        items.append({"id": f"unmatched{i}", "kind": "value_mismatch",
                      "needs_answer": True,
                      "title": (f"{vid} asserts a value the derivation never "
                                f"produces"),
                      "body": (f"<b>{_esc(vtext_map.get(vid, vid))}</b><br>"
                               f"Stated target <b>{_esc(u.get('target'))}</b> "
                               f"matches no computed step. Either the verifier's "
                               f"number is wrong (correct it to the derivation's "
                               f"value), or the golden is missing the step this "
                               f"verifier checks (add the claim). A structurally "
                               f"clean but factually wrong verifier passes every "
                               f"other check — this is the only one that catches "
                               f"it."),
                      "meta": "NOT scoreable until resolved"})

    # FIX 4 — coverage gaps: the DETERMINISTIC unwatched_load_bearing list is
    # authoritative (it is computed from the mapping, not the model's judgment).
    # The property audit's own coverage_gaps list is used only to enrich a step
    # with a proposed-verifier string when it happens to name that step; the two
    # disagreed on the toy task ([C2,C3,J1,J2] deterministic vs [J1] model), and
    # trusting the model's shorter list silently dropped the real lambda hole.
    # SUGGESTED VERIFIERS — ONE merged display list (previously two overlapping lists: "has no
    # verifier" was a subset of "coverage gaps", forcing the reader to compute the set relation).
    # Each step appears ONCE, tagged by CONFIDENCE. IDS ARE PRESERVED (gap{i} for the deterministic
    # entries, gapx{j} for advisory) because the id is the contract with apply_decisions — only the
    # DISPLAY is merged, not the decision-item identity.
    #   definite  — deterministically unwatched (provably zero verifiers on this load-bearing step)
    #   suggested — model-flagged only (may already be watched; advisory)
    cov = res.get("step_coverage") or {}
    step_idx = _step_index(res)
    model_gaps = {str(g.get("step")): g for g in (va.get("coverage_gaps") or [])}
    det_list = [str(s) for s in (cov.get("unwatched_load_bearing") or [])]
    det_steps = set(det_list)
    def _sv_card(cid, step, confidence):
        mg = model_gaps.get(str(step), {})
        proposed = mg.get("proposed_verifier") or _author_gap_verifier(
            str(step), step_idx.get(str(step), {}))
        authored = not mg.get("proposed_verifier")
        why = (mg.get("why_it_matters")
               or ("a load-bearing step no verifier watches: a response could get it wrong "
                   "with nothing objecting"))
        if confidence == "definite":
            badge = ("<span style='background:#fde2e0;color:#b4413c;padding:1px 7px;"
                     "border-radius:5px;font-size:12px;font-weight:600'>definite — unwatched</span>")
        else:
            badge = ("<span style='background:#eee7d6;color:#9a7400;padding:1px 7px;"
                     "border-radius:5px;font-size:12px'>suggested — advisory</span>")
        return {"id": cid, "kind": "gap", "needs_answer": False,
                "step": str(step), "confidence": confidence,
                "proposed_verifier": proposed, "proposed_is_authored": authored,
                "title": f"suggested verifier for step {step}",
                "body": (f"{badge}<br>{_esc(why)}<br><b>proposed"
                         + (" (auto-authored)" if authored else "")
                         + f":</b> {_esc(proposed)}"),
                "meta": "NOT added — accept to add it"}
    # definite first (id gap{i}, matching apply_decisions), then advisory (id gapx{j})
    for i, step in enumerate(det_list):
        items.append(_sv_card(f"gap{i}", step, "definite"))
    for j, step in enumerate(s2 for s2 in model_gaps if s2 not in det_steps):
        items.append(_sv_card(f"gapx{j}", step, "suggested"))
    # Duplicate clusters: two+ verifiers assert the same quantity. The audit
    # recommends which to keep; make it an actionable decision so a redundant
    # verifier is actually dropped (MECE) rather than merely noted and shipped.
    # Duplicate clusters: two+ verifiers assert the same quantity. The audit
    # recommends which to keep; make it an actionable decision so a redundant
    # verifier is actually dropped (MECE) rather than merely noted and shipped.
    for i, c in enumerate(va.get("duplicate_clusters") or []):
        ids = c.get("verifier_ids") or []
        items.append({"id": f"dup{i}", "kind": "dedupe", "needs_answer": False,
                      "title": (f"{', '.join(ids)} assert the same quantity"
                                + ("" if c.get("values_agree", True)
                                   else " (VALUES DISAGREE)")),
                      "body": (f"<b>quantity:</b> {_esc(c.get('quantity'))}"
                               f"<br><b>recommended:</b> "
                               f"{_esc(c.get('recommended_action'))}"),
                      "meta": ("NOT applied — accept to drop the restated "
                               "verifier(s) and keep one")})
    # Temporal drift: a split child (or verifier) that states a LIVE value with no
    # date/source pin. First-class defect, so it gets a first-class decision —
    # otherwise it is only visible in the raw split log, and the SME cannot see
    # that a verifier will decay as the world moves.
    seen_temporal = []
    for x in (res.get("verifier_splits_applied") or []):
        seen_temporal += x.get("temporal_unpinned_children") or []
    for i, vid in enumerate(dict.fromkeys(seen_temporal)):  # dedupe, keep order
        items.append({"id": f"tmp{i}", "kind": "temporal", "needs_answer": True,
                      "title": f"{vid} states an unpinned live value (temporal drift)",
                      "body": (f"<b>{_esc(vtext_map.get(vid, vid))}</b><br>"
                               f"This verifier depends on a value that changes over "
                               f"time (e.g. \u201ctoday\u2019s market rate\u201d) with "
                               f"no date or source pin, so the golden decays as the "
                               f"world moves. Pin it to a dated source, or confirm "
                               f"the intended fixed value."),
                      "meta": "NOT scoreable until pinned — answer with the pin"})
    # SME verdict override — shown only when the model's verdict is
    # non-proceedable. Once the SME resolves the defects that drove the verdict,
    # this is where they re-grade the task so it can become scoreable. Answering
    # is REQUIRED to seal a scoreable package from a BROKEN augment.
    if str(res.get("audit_verdict", "")).upper() in (
            "BROKEN", "UNGRADEABLE", "NON_DETERMINISTIC"):
        items.append({
            "id": "verdict", "kind": "verdict_override", "needs_answer": True,
            "title": f"Re-grade the {_esc(res.get('audit_verdict'))} verdict",
            "body": (f"The model graded this task "
                     f"<b>{_esc(res.get('audit_verdict'))}</b> (not scoreable). If "
                     f"your decisions above resolve the defects that caused it, "
                     f"re-grade it to a proceedable verdict so the sealed task can "
                     f"be scored. Answer with the new verdict and a one-line "
                     f"justification, e.g. <span class=mono>SALVAGEABLE: V1/V2 "
                     f"corrected, gaps closed</span>. Accepted values: "
                     f"<span class=mono>SOUND</span> or "
                     f"<span class=mono>SALVAGEABLE</span>. Leave rejected to keep "
                     f"the task not-scoreable."),
            "meta": "answer 'SOUND: ...' or 'SALVAGEABLE: ...' to make scoreable"})
    return items


def _decision_section(res: dict) -> str:
    import json as _json
    items = _decision_items(res)
    if not items:
        return "<p><i>Nothing to decide.</i></p>"
    cards = []
    for it in items:
        ask = ""
        if it["needs_answer"]:
            ask = (f"<div class='ask'><b>Your answer (required):</b><br>"
                   f"<textarea id='why_{it['id']}' "
                   f"oninput=\"note('{it['id']}', this)\" "
                   f"placeholder='State the decision. This is what gets applied.'>"
                   f"</textarea></div>")
        else:
            ask = (f"<div class='why' id='why_{it['id']}'>"
                   f"<b>Reason (required for reject / something else):</b><br>"
                   f"<textarea oninput=\"note('{it['id']}', this)\"></textarea></div>")
        cards.append(
            f"<div class='card undecided' id='it_{it['id']}'>"
            f"<h4>{_esc(it['title'])} "
            f"<span class='pill warn'>{it['kind']}</span> "
            f"<span id='st_{it['id']}' class='pill warn'>undecided</span></h4>"
            f"<div>{it['body']}</div>"
            f"<div class='mono' style='color:#7a7568;font-size:12px'>{_esc(it['meta'])}</div>"
            f"<div class='btns'>"
            f"<button data-choice='accept' "
            f"onclick=\"decide('{it['id']}','accept')\">Accept</button>"
            f"<button data-choice='reject' "
            f"onclick=\"decide('{it['id']}','reject')\">Reject</button>"
            f"<button data-choice='other' "
            f"onclick=\"decide('{it['id']}','other')\">Something else</button>"
            f"</div>{ask}</div>")
    slim = _json.dumps([{"id": i["id"], "kind": i["kind"],
                         "needs_answer": i["needs_answer"]} for i in items])
    js = DECISION_JS % {"task_id": res.get("task_id", ""),
                        "run_hash": res.get("run_hash", ""),
                        "items_json": slim}
    return (f"<div class='sticky'><span id='tally' class='pill warn'></span> "
            f"&nbsp;<button onclick='saveDecisions()'>Save decisions</button>"
            f"&nbsp;<span class='mono' style='font-size:12px'>run "
            f"{_esc(res.get('run_hash'))}</span></div>"
            + "".join(cards) + js)


def _trajectory(res: dict) -> str:
    """The golden trajectory: the ordered derivation, with what each step feeds.

    Built since call 2 gained corrected_claims with from_claim links and
    judgment_steps with consumes, but never rendered — so the artifact the whole
    design turns on was invisible in the report.
    """
    steps = res.get("corrected_claim_verdicts") or []
    judg = res.get("judgment_steps") or []
    graph = res.get("step_graph") or {}
    health = res.get("step_graph_health") or {}
    v2s = res.get("verifier_to_step") or {}
    if not steps and not judg:
        return "<p><i>No trajectory: call 2 emitted no corrected claims.</i></p>"

    watched = {}
    for vid, sid in v2s.items():
        watched.setdefault(sid, []).append(vid)
    consumed = {p for ps in graph.values() for p in ps}
    feeds = {}
    for child, parents in graph.items():
        for par in parents:
            feeds.setdefault(par, []).append(child)

    rows = []
    for c in steps:
        sid = c.get("id")
        parents = [p.get("from_claim") for p in (c.get("input_provenance") or [])
                   if p.get("from_claim")]
        st = c.get("status", "")
        pill = ("ok" if st == "CONFIRMED"
                else "bad" if st in ("ARITHMETIC_ERROR", "UNVERIFIABLE")
                else "warn")
        rows.append(
            f"<tr><td class='mono'>{_esc(sid)}</td>"
            f"<td>{_esc(c.get('label'))}</td>"
            f"<td class='mono'>{_esc(c.get('operation'))}</td>"
            f"<td class='mono'>{_esc(c.get('recomputed'))}</td>"
            f"<td class='mono'>{_esc(', '.join(dict.fromkeys(parents)) or '—')}</td>"
            f"<td class='mono'>{_esc(', '.join(feeds.get(sid, [])) or 'TERMINAL')}</td>"
            f"<td class='mono'>{_esc(', '.join(watched.get(sid, [])) or '—')}</td>"
            f"<td><span class='pill {pill}'>{_esc(st)}</span></td></tr>")
    for j in judg:
        jid = j.get("id")
        rows.append(
            f"<tr><td class='mono'>{_esc(jid)}</td>"
            f"<td><b>{_esc(j.get('question'))}</b><br>"
            f"<i>{_esc(j.get('ruling'))}</i></td>"
            f"<td class='mono'>judgement</td><td class='mono'>—</td>"
            f"<td class='mono'>{_esc(', '.join(j.get('consumes') or []) or '—')}</td>"
            f"<td class='mono'>{_esc(', '.join(feeds.get(jid, [])) or 'TERMINAL')}</td>"
            f"<td class='mono'>{_esc(', '.join(watched.get(jid, [])) or '—')}</td>"
            f"<td><span class='pill warn'>judgement</span></td></tr>")

    hdr = ("<tr><th>step</th><th>what it establishes</th><th>operation</th>"
           "<th>value</th><th>from</th><th>feeds</th><th>watched by</th>"
           "<th>status</th></tr>")
    note = (f"<p>{health.get('n_nodes', 0)} steps, {health.get('n_edges', 0)} "
            f"dependencies, {health.get('connected', 0)} connected to a terminal"
            + (f", <span class='pill bad'>{len(health.get('cycles') or [])} "
               f"cycle(s)</span>" if health.get("cycles") else "")
            + (f". Terminals: <span class='mono'>"
               f"{_esc(', '.join(health.get('terminals') or []))}</span>"
               if health.get("terminals") else "")
            + ". A step with no verifier in <i>watched by</i> is work a response "
              "can get wrong with nothing objecting.</p>")
    return note + f"<table>{hdr}{''.join(rows)}</table>"


def _property_table(res: dict) -> str:
    """Per-verifier property verdicts, plus duplication and coverage gaps.

    The property audit ran from the first real task onward but nothing rendered
    it, so the atomicity failures that caused a split were invisible — a reader
    saw the split children with no stated reason.
    """
    va = res.get("verifier_audit") or {}
    if va.get("error"):
        return f"<p class='pill bad'>property audit failed: {_esc(va['error'])}</p>"
    vs = va.get("verifiers") or []
    if not vs:
        return "<p><i>Property audit did not run.</i></p>"

    head = ("<tr><th>ID</th>"
            + "".join(f"<th>{_PROP_LABEL[p]}</th>" for p in _PROPS)
            + "<th>Tests step</th><th>Finding</th></tr>")
    body = []
    for v in vs:
        cells = []
        why = []
        for p in _PROPS:
            d = (v.get("properties") or {}).get(p) or {}
            verdict = d.get("verdict", "")
            if verdict == "FAIL":
                cells.append("<td><span class='pill bad'>FAIL</span></td>")
                why.append(f"<b>{_PROP_LABEL[p]}</b>: {_esc(d.get('why',''))}")
            elif verdict == "NOT_APPLICABLE":
                cells.append("<td class='mono'>n/a</td>")
            elif verdict == "PASS":
                cells.append("<td class='mono'>ok</td>")
            else:
                cells.append("<td class='mono'>?</td>")
        step = v.get("tests_step") or "—"
        note = "<br>".join(why)
        if v.get("route_to_rubric"):
            note += (f"<br><span class='pill warn'>route to rubric: "
                     f"{_esc(v.get('rubric_dimension'))}</span>")
        body.append(f"<tr><td class='mono'>{_esc(v.get('id'))}</td>"
                    + "".join(cells)
                    + f"<td class='mono'>{_esc(step)}</td><td>{note}</td></tr>")

    extra = []
    for c in va.get("duplicate_clusters") or []:
        extra.append(f"<li><b>{_esc(', '.join(c.get('verifier_ids') or []))}</b> "
                     f"both assert {_esc(c.get('quantity'))} "
                     f"(values agree: {c.get('values_agree')}) — "
                     f"{_esc(c.get('recommended_action'))}</li>")
    _detset = {str(s) for s in ((res.get("step_coverage") or {}).get("unwatched_load_bearing") or [])}
    for g in va.get("coverage_gaps") or []:
        tag = "definite — unwatched" if str(g.get("step")) in _detset else "suggested — advisory"
        extra.append(f"<li>suggested verifier for step <b>{_esc(g.get('step'))}</b> "
                     f"<i>({tag})</i>: {_esc(g.get('why_it_matters'))}<br>"
                     f"<i>proposed:</i> {_esc(g.get('proposed_verifier'))}</li>")
    for t in res.get("target_disagreements") or []:
        extra.append(f"<li><span class='pill bad'>target overridden</span> "
                     f"<b>{_esc(t.get('verifier'))}</b>: the model emitted "
                     f"{_esc(t.get('emitted'))} but the verifier's own text says "
                     f"{_esc(t.get('from_text'))}. {_esc(t.get('detail'))}</li>")
    for t in res.get("target_grammar_problems") or []:
        extra.append(f"<li><span class='pill warn'>target unreadable</span> "
                     f"<b>{_esc(t.get('verifier'))}</b> ({_esc(t.get('form'))}): "
                     f"{_esc(t.get('detail'))}</li>")
    if res.get("targets_emitted_only"):
        extra.append(f"<li><span class='pill warn'>no target clause</span> "
                     f"{_esc(', '.join(res['targets_emitted_only']))} — target "
                     f"taken from the model's emission, not from the text</li>")
    for m in va.get("split_children_missing_target") or []:
        extra.append(f"<li><span class='pill bad'>unscoreable child</span> "
                     f"<b>{_esc(m.get('parent'))}{_esc(m.get('suffix'))}</b>: "
                     f"{_esc(m.get('detail'))}<br>{_esc(m.get('text'))}</li>")
    for t in va.get("trap_passes_band") or []:
        extra.append(f"<li><span class='pill bad'>not falsifiable</span> "
                     f"<b>{_esc(t.get('verifier'))}</b>: {_esc(t.get('detail'))}</li>")
    sp = res.get("verifier_splits_applied") or []
    for x in sp:
        tl = x.get("targetless_children") or []
        extra.append(
            f"<li>split <b>{_esc(x.get('parent'))}</b> → "
            f"{_esc(', '.join(x.get('children') or []))}; target to "
            f"{_esc(x.get('target_went_to'))}"
            + (f" — <span class='pill bad'>no target: "
               f"{_esc(', '.join(tl))}</span>" if tl else "") + "</li>")

    return (f"<table>{head}{''.join(body)}</table>"
            + (f"<ul>{''.join(extra)}</ul>" if extra else ""))


def _corrected_blocks(res: dict) -> str:
    """Render every artifact the auditor may correct, not just the solution logic.

    All four were threaded onto the result but only the solution logic was ever
    rendered, so a corrected sanity check or prompt was invisible in the report a
    reviewer reads. An artifact identical to the input is labelled unchanged
    rather than shown as if it had been rewritten.
    """
    orig = res.get("_originals") or {}
    out = []
    for key, label in (("corrected_solution_logic", "Solution logic"),
                       ("corrected_sanity_check", "Sanity check"),
                       ("corrected_prompt", "Prompt"),
                       ("corrected_verifiers", "Verifiers (in-place edits)")):
        txt = res.get(key) or ""
        was = orig.get(key.replace("corrected_", ""), None)
        if not txt.strip():
            note = "<span class='pill ok'>no in-place edits</span>"
            body = ""
        elif was is not None and txt.strip() == str(was).strip():
            note = "<span class='pill ok'>unchanged</span>"
            body = f"<pre>{_esc(txt)}</pre>"
        else:
            note = "<span class='pill warn'>corrected</span>"
            body = f"<pre>{_esc(txt)}</pre>"
        out.append(f"<h3>{label} &nbsp; {note}</h3>{body}")
    return "\n".join(out)


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


_CSS = """
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:24px auto;
padding:0 20px;color:#1d1b16;line-height:1.55;background:#fbfaf7}
h1{font-size:26px} h2{font-size:19px;margin-top:28px;border-bottom:1px solid #e7e2d8;padding-bottom:4px}
code,.mono{font-family:ui-monospace,Menlo,monospace;font-size:13px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border:1px solid #e7e2d8;padding:6px 9px;text-align:left;vertical-align:top}
th{background:#f2eee6;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.tag{display:inline-block;background:#eceaf6;border-radius:5px;padding:1px 7px;margin:1px;font-size:12px;font-family:monospace}
.crux{background:#e7f3ec;border-left:3px solid #2e7d5b;padding:2px 6px}
.pill{border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600}
.ok{background:#e4f3ea;color:#1a7a4a}.warn{background:#fbf3da;color:#9a7400}.bad{background:#f6e4e2;color:#b4413c}
pre{background:#f6f3ec;border:1px solid #e7e2d8;border-radius:8px;padding:12px;white-space:pre-wrap;font-size:13px}
.bar{height:7px;background:#175FFF;border-radius:4px;display:inline-block;vertical-align:middle}

</style>
"""


def _css_with(extra: str) -> str:
    """Insert extra rules INSIDE the shared style block.

    _CSS carries its own </style>, so concatenating after it dropped the decision
    rules into the head as raw text — present in the file, styling nothing, which
    is why every button stayed white however it was clicked.
    """
    assert "</style>" in _CSS
    return _CSS.replace("</style>", extra + "\n</style>")


def _synthesize_reason(res: dict) -> list:
    """Assemble a human-readable list of the defects the pipeline DETERMINISTICALLY
    detected, for use when the model returns a verdict with no primary_reason. Only
    reads signals the pipeline already computed — it invents nothing. Turns an
    'unexplained BROKEN' into 'here is what we detected', which is what the SME
    needs to act. Each entry names the verifier/step and the defect class."""
    out = []
    mapping = res.get("verifier_mapping_report") or {}
    # value-mismatch: a numeric verifier whose value matches no computed step
    for u in mapping.get("unmatched", []):
        if str(u.get("kind", "numeric")).lower() == "numeric":
            tgt = u.get("target")
            out.append(f"{u.get('verifier')} asserts a value"
                       + (f" ({tgt})" if tgt is not None else "")
                       + " that no derivation step produces (value error, or a "
                         "missing step)")
    # non-atomic verifiers that were split
    for s in res.get("verifier_splits_applied", []):
        kids = s.get("children") or []
        out.append(f"{s.get('parent')} was non-atomic and split into "
                   f"{', '.join(kids)}")
        for child in (s.get("temporal_unpinned_children") or []):
            out.append(f"{child} states an unpinned live value (temporal drift)")
    # duplicate (non-MECE) verifiers
    for c in (res.get("verifier_audit") or {}).get("duplicate_clusters", []):
        ids = c.get("verifier_ids") or []
        if len(ids) > 1:
            out.append(f"{', '.join(ids)} assert the same quantity (not MECE)")
    # decision inversion flagged by the auditor
    if res.get("decision_inversion"):
        out.append("a decision/recommendation appears inverted vs the solution "
                   "logic (decision inversion)")
    # non-blocking claim warnings (mislabelled inputs etc.)
    for c in res.get("corrected_claim_verdicts", []):
        st = c.get("status")
        if st and st not in ("CONFIRMED", "NOT_ARITHMETIC"):
            out.append(f"claim {c.get('id')} is {st} "
                       f"({str(c.get('label', ''))[:40]})")
    # load-bearing steps no verifier watches
    cov = (res.get("step_coverage") or {}).get("unwatched_load_bearing") or []
    if cov:
        out.append(f"{len(cov)} load-bearing step(s) have no verifier watching "
                   f"them ({', '.join(map(str, cov[:6]))})")
    return out


def _verdict_reason_block(res: dict, verdict: str) -> str:
    """Show WHY the auditor reached a non-SOUND verdict, and — critically — when a
    non-proceedable verdict (BROKEN / UNGRADEABLE) carries NO model reason,
    SYNTHESIZE one from the defects the pipeline deterministically detected. The
    model is asked for a one-line primary_reason but sometimes returns the verdict
    with an empty reason, empty findings, and empty prose — leaving the SME a bare
    'BROKEN'. Rather than only flagging that as unexplained, we now list what the
    pipeline itself caught (value mismatches, non-atomic splits, temporal drift,
    coverage gaps), so the verdict is actionable even when the model is silent."""
    reason = (res.get("primary_reason") or "").strip()
    prose = (res.get("prose_findings") or "").strip()
    findings = res.get("findings") or []
    non_proceedable = verdict in ("BROKEN", "UNGRADEABLE", "NON_DETERMINISTIC")
    if reason:
        cls = "bad" if non_proceedable else "warn"
        return f"<p class='pill {cls}'>Reason: {_esc(reason)}</p>"
    if non_proceedable and not prose and not findings:
        detected = _synthesize_reason(res)
        if detected:
            items = "".join(f"<li>{_esc(x)}</li>" for x in detected)
            return ("<div class='pill bad' style='display:block;text-align:left'>"
                    "&#9888; The model returned <b>" + _esc(verdict) + "</b> with "
                    "no stated reason. From the pipeline's own detections, the "
                    "verdict is likely driven by:<ul style='margin:6px 0 0'>"
                    + items + "</ul><span style='font-weight:400'>(synthesized "
                    "from deterministic signals, not the model's words — verify "
                    "against the Decisions section below.)</span></div>")
        return ("<p class='pill bad'>&#9888; UNEXPLAINED VERDICT — the model "
                "graded this <b>" + _esc(verdict) + "</b> but returned no "
                "primary_reason, and the pipeline detected no specific defect "
                "signals either. Re-run this task or inspect the corrected "
                "claims manually.</p>")
    return ""


def write_augment_report(res: dict, out_path: str) -> str:
    tid = _esc(res.get("task_id"))
    verdict = _esc(res.get("audit_verdict"))
    vcls = "ok" if verdict in ("SOUND", "SALVAGEABLE") else "bad"

    crux = set(res.get("crux_ids", []))
    shap = res.get("crux_shapley_weights", {})
    dag = res.get("dag", {})
    base = res.get("base_weights", {})
    expected = res.get("expected_values", {})

    # verifier table
    rows = ""
    for vid in dag.keys():
        is_crux = vid in crux
        w = shap.get(vid, 0.0) * 100 if is_crux else 0.0
        deps = ", ".join(dag.get(vid, [])) or "root"
        sov = (expected.get(vid, {}) or {}).get("source_of_verification", "") if is_crux else ""
        rows += (f"<tr><td class='mono'>{_esc(vid)}</td>"
                 f"<td>{'<span class=crux>CRUX</span>' if is_crux else ''}</td>"
                 f"<td class='mono'>{deps}</td>"
                 f"<td class='mono'>{base.get(vid,0)*100:.1f}%</td>"
                 f"<td class='mono'>{w:.1f}%"
                 + (f" <span class='bar' style='width:{max(2,w):.0f}px'></span>" if is_crux else "")
                 + "</td>"
                 f"<td class='mono'>{_esc(sov)}</td></tr>")

    changes = res.get("changes_applied", []) + res.get("judgment_changes_pending_sme", [])
    chg_rows = ""
    for c in changes:
        pend = "warn" if c.get("type") != "MECHANICAL" else "ok"
        chg_rows += (f"<tr><td>{_esc(c.get('artifact'))}</td><td>{_esc(c.get('location'))}</td>"
                     f"<td><span class='pill {pend}'>{_esc(c.get('type'))}</span></td>"
                     f"<td>{_esc(c.get('old'))[:120]}</td><td>{_esc(c.get('new'))[:120]}</td>"
                     f"<td>{_esc(c.get('rationale'))[:160]}</td></tr>")

    trap = " ".join(f"<span class='tag'>{_esc(a)}</span>" for a in res.get("crux_anchors_trap", []))
    exp = " ".join(f"<span class='tag'>{_esc(a)}</span>" for a in res.get("crux_anchors_expert", []))

    _CSS_WITH_DECISIONS = _css_with(DECISION_CSS)
    doc = f"""<!doctype html><html><head><meta charset=utf-8>
<title>{tid} · augmentation</title>{_CSS_WITH_DECISIONS}</head><body>
<h1>{tid} — Augmentation & Corrections</h1>
<p>Audit verdict: <span class='pill {vcls}'>{verdict}</span>
&nbsp; Crux size: <b>{len(crux)}</b> verifiers &nbsp; Model: <span class=mono>{_esc(res.get('model_used'))}</span></p>
{_verdict_reason_block(res, verdict)}
{"<p class='pill bad'>ERROR: "+_esc(res.get('error'))+"</p>" if res.get('error') else ""}
{"<p class='pill warn'>&#9888; DEGRADED GOLDEN — these input files could not be read and were NOT used: "+", ".join("<span class=tag>"+_esc(s)+"</span>" for s in res.get('skipped_inputs',[]))+"</p>" if res.get('skipped_inputs') else ""}

<h2>Crux anchors (from Sanity Check)</h2>
<p><b>Lazy-AI / trap side:</b> {trap or '<i>none</i>'}<br>
<b>Expert-path side:</b> {exp or '<i>none</i>'}</p>

<h2>Verifiers · DAG · crux-only Shapley weights</h2>
<table><tr><th>ID</th><th>Crux</th><th>Depends on</th><th>Base wt</th><th>Crux Shapley wt</th><th>Source of verification</th></tr>
{rows}</table>
{"<p class='pill warn'>Dropped from crux (reachable but no frozen expected value, so unscoreable): "+", ".join("<span class=tag>"+_esc(d)+"</span>" for d in res.get('crux_dropped_no_expected',[]))+"</p>" if res.get('crux_dropped_no_expected') else ""}

<h2>Corrections applied</h2>
{"<table><tr><th>Artifact</th><th>Where</th><th>Type</th><th>Old</th><th>New</th><th>Why</th></tr>"+chg_rows+"</table>" if chg_rows else "<p><i>No changes.</i></p>"}
{("<p class='pill warn'>"+str(len(res.get('judgment_changes_pending_sme',[])))+" judgment change(s) NOT applied — the artifact keeps its original wording and a question is posed for you in Decisions required below. Mechanical changes above are applied.</p>") if res.get('judgment_changes_pending_sme') else ""}

<h2>Golden trajectory</h2>
{_trajectory(res)}

<h2>Decisions required from you</h2>
{_decision_section(res)}

<h2>Verifier properties</h2>
{_property_table(res)}

<h2>Corrected artifacts</h2>
{_corrected_blocks(res)}

<h2>Augmented verifiers (canonical)</h2>
<pre>{_esc(res.get('augmented_verifiers_text'))}</pre>

{"<h2>Notes</h2><pre>"+_esc(res.get('notes'))+"</pre>" if res.get('notes') else ""}
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


def _dag_structure(dag: dict, title_note: str = "",
                   weights: dict = None) -> str:
    """Render a DAG as a readable structural view for the sealed report (no SVG
    engine here): nodes are grouped by depth (roots first), each showing what it
    depends on and, optionally, its Shapley weight. Makes the graph reviewable as
    an artifact rather than only as a 'depends on' column."""
    if not dag:
        return "<p><i>(empty)</i></p>"
    # depth = longest path from a root
    depth = {}

    def _d(n, seen=None):
        seen = seen or set()
        if n in seen:
            return 0
        if n in depth:
            return depth[n]
        deps = dag.get(n, [])
        depth[n] = 0 if not deps else 1 + max(
            _d(p, seen | {n}) for p in deps if p in dag)
        return depth[n]

    for n in dag:
        _d(n)
    by_depth = {}
    for n, d in depth.items():
        by_depth.setdefault(d, []).append(n)
    rows = ""
    for d in sorted(by_depth):
        level = "root" if d == 0 else f"depth {d}"
        cells = ""
        for n in sorted(by_depth[d]):
            deps = dag.get(n, [])
            wtxt = ""
            if weights and n in weights:
                wtxt = f" <span class='mono'>{weights[n] * 100:.0f}%</span>"
            edge = (" &larr; " + ", ".join(deps)) if deps else ""
            cells += (f"<div style='display:inline-block;border:1px solid #e7e2d8;"
                      f"border-radius:6px;padding:4px 9px;margin:3px;background:#fff'>"
                      f"<b class='mono'>{_esc(n)}</b>{wtxt}"
                      f"<span class='mono' style='color:#7a746a'>{_esc(edge)}</span>"
                      f"</div>")
        rows += (f"<div style='margin:4px 0'><span class='pill' "
                 f"style='background:#f2eee6'>{level}</span> {cells}</div>")
    note = f"<p style='color:#7a746a;margin:2px 0 8px'>{title_note}</p>" if title_note else ""
    return note + rows


def _sealed_verifier_table(res: dict) -> str:
    """Verifier · DAG · Shapley table for the sealed set: every verifier, its
    dependencies, base weight, full-DAG Shapley (all verifiers), and (crux only)
    its crux-DAG Shapley weight."""
    crux = set(res.get("crux_ids", []))
    cshap = res.get("crux_dag_shapley_weights") or res.get("crux_shapley_weights", {})
    fshap = res.get("full_dag_shapley_weights", {})
    dag = res.get("dag", {})
    base = res.get("base_weights", {})
    expected = res.get("expected_values", {})
    v2s = res.get("verifier_to_step", {})
    rows = ""
    for vid in dag.keys():
        is_crux = vid in crux
        cw = cshap.get(vid, 0.0) * 100 if is_crux else 0.0
        fw = fshap.get(vid, 0.0) * 100
        deps = ", ".join(dag.get(vid, [])) or "root"
        step = v2s.get(vid, "—")
        sov = (expected.get(vid, {}) or {}).get("source_of_verification", "")
        rows += (f"<tr><td class='mono'>{_esc(vid)}</td>"
                 f"<td>{'<span class=crux>CRUX</span>' if is_crux else ''}</td>"
                 f"<td class='mono'>{_esc(step)}</td>"
                 f"<td class='mono'>{deps}</td>"
                 f"<td class='mono'>{base.get(vid, 0) * 100:.1f}%</td>"
                 f"<td class='mono'>{fw:.1f}%</td>"
                 f"<td class='mono'>{cw:.1f}%"
                 + (f" <span class='bar' style='width:{max(2, cw):.0f}px'></span>"
                    if is_crux else "") + "</td>"
                 f"<td class='mono'>{_esc(sov)}</td></tr>")
    return ("<table><tr><th>ID</th><th>Crux</th><th>Watches step</th>"
            "<th>Depends on</th><th>Base wt</th><th>Full-DAG Shapley</th>"
            "<th>Crux-DAG Shapley</th><th>Source of verification</th></tr>"
            + rows + "</table>")


def write_golden_report(res: dict, out_path: str) -> str:
    tid = _esc(res.get("task_id"))
    fmt = _esc(res.get("gold_deliverable_format"))
    secs = res.get("gold_deliverable_sections", [])
    body = ""
    for s in secs:
        body += f"<h2>{_esc(s.get('title'))}</h2><pre>{_esc(s.get('content'))}</pre>"
    doc = f"""<!doctype html><html><head><meta charset=utf-8>
<title>{tid} · golden deliverable</title>{_CSS}</head><body>
<h1>{tid} — Golden Deliverable</h1>
<p>Format: <span class='pill ok'>{fmt}</span> &nbsp; {len(secs)} section(s)</p>
{body if body else "<p><i>No deliverable generated.</i></p>"}
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


def write_sealed_report(res: dict, out_path: str) -> str:
    """The clean FINAL prompt-package report, produced after apply_decisions seals
    the golden. Unlike the augment report, it carries NO review scaffolding (no
    'Decisions required', no 'Corrections applied', no verifier-property audit) —
    it is what the golden IS, for the SME to review as a finished artifact: the
    final prompt, solution logic, sanity check, golden deliverable, golden
    trajectory, and the sealed verifiers with the DAG and Shapley weights."""
    tid = _esc(res.get("task_id"))
    verdict = _esc(res.get("audit_verdict"))
    scoreable = res.get("scoreable")
    vcls = "ok" if scoreable else "warn"
    scls = "ok" if scoreable else "bad"
    crux = res.get("crux_ids", []) or []

    prompt = res.get("corrected_prompt") or res.get("prompt") or ""
    logic = res.get("corrected_solution_logic") or res.get("solution_logic") or ""
    sanity = res.get("corrected_sanity_check") or res.get("sanity_check") or ""
    fmt = _esc(res.get("gold_deliverable_format"))
    gold_secs = res.get("gold_deliverable_sections", []) or []
    gold_body = "".join(
        f"<h3>{_esc(s.get('title'))}</h3><pre>{_esc(s.get('content'))}</pre>"
        for s in gold_secs) or "<p><i>No deliverable.</i></p>"

    _CSS_LOCAL = _css_with("")
    doc = f"""<!doctype html><html><head><meta charset=utf-8>
<title>{tid} · final prompt package</title>{_CSS_LOCAL}</head><body>
<h1>{tid} — Final Prompt Package (sealed)</h1>
<p>Verdict: <span class='pill {vcls}'>{verdict}</span> &nbsp;
Scoreable: <span class='pill {scls}'>{'yes' if scoreable else 'no'}</span> &nbsp;
Crux: <b>{len(crux)}</b> verifier(s) &nbsp;
Sealed from: <span class=mono>{_esc(res.get('run_hash') or '')}</span></p>
{"<p class='pill warn'>Not scoreable: " + _esc(res.get('not_scoreable_reason')) + "</p>" if not scoreable and res.get('not_scoreable_reason') else ""}

<h2>Prompt</h2>
<pre>{_esc(prompt)}</pre>

<h2>Solution logic</h2>
<pre>{_esc(logic)}</pre>

<h2>Sanity check (trap / expert path)</h2>
<pre>{_esc(sanity)}</pre>

<h2>Golden deliverable</h2>
<p>Format: <span class='pill ok'>{fmt}</span> &nbsp; {len(gold_secs)} section(s)</p>
{gold_body}

<h2>Golden trajectory</h2>
{_trajectory(res)}

<h2>Verifiers · DAG · Shapley weights</h2>
{_sealed_verifier_table(res)}
{"<p class='pill ok'>Crux (sanity-anchored + immediate parents): " + ", ".join("<span class=tag>" + _esc(c) + "</span>" for c in crux) + "</p>" if crux else ""}

<h2>Full DAG</h2>
{_dag_structure(res.get('dag') or {}, "Every verifier by depth (roots first); &larr; shows dependencies. Weight = full-DAG Shapley.", res.get('full_dag_shapley_weights'))}

<h2>Crux DAG</h2>
{_dag_structure(res.get('crux_dag') or {}, "The crux verifiers only, as an induced subgraph. Weight = crux-DAG Shapley.", res.get('crux_dag_shapley_weights')) if res.get('crux_dag') else "<p><i>No crux DAG (crux not yet computed or empty).</i></p>"}

<h2>Verifiers (canonical)</h2>
<pre>{_esc(res.get('augmented_verifiers_text'))}</pre>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path