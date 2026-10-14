#!/usr/bin/env python3
"""Apply an SME decisions file to an augment package and seal it.

    python apply_decisions.py --augment output/augmented/tsk_X_augment.json \
        --decisions decisions_tsk_X_2026-08-03.json --out output/final

This is the last step of audit + augmentation. Until it runs, a package carries
unresolved judgment questions, and a package with unresolved judgment questions is
not scoreable — not because a rule says so, but because the model resolves an open
question differently on different runs. On one task the FX ambiguity resolved as
the live rate (savings 2.20 Cr) and on the next as the file rate (0.918 Cr): same
task, same code, two goldens with different answers and different frozen targets.
An answered question, written into the artifact, is what stops that.

WHAT EACH DECISION DOES
  accept          the proposal stands as-is
  reject + reason a mechanical change is REVERTED; a rewrite or split is undone;
                  a proposed verifier is not added
  other  + reason the SME's text replaces the proposal
  question        the SME's answer is applied and recorded; the question closes

SME edits reach every artifact a resolution can change: the corrected prompt,
sanity check and solution logic, the verifier block, AND the auditor's deliverable
and trajectory (a resolution that changes a value must move the deliverable with
it, or the sealed golden contradicts its own solution logic).

THE SEAL RE-DERIVES THE FROZEN GRAPH. An SME edit can change the verifier id space
(a reverted split drops V5a/V5b and restores V5; an accepted gap adds a verifier)
or a frozen target, so the DAG, base weights, crux set and Shapley weights computed
at augment time are stale by the time the package is sealed. Sealing therefore
re-runs augment_task.derive_frozen_graph on the edited artifacts — the SAME routine
the augmenter used for its preview, so the two cannot diverge — and only then marks
the package sealed. A re-derive that cannot run means the sealed set is inconsistent
and the package is marked not-scoreable rather than shipping a stale graph.

REFUSALS
  * a run_hash mismatch means the decisions were made against a different run of
    this task, so applying them would edit text the reviewer never saw
  * an incomplete file (any item undecided, or a reject/other with no reason, or a
    question with no answer) is refused; a partly-sealed package is worse than an
    unsealed one because nothing downstream can tell which it is
"""
import argparse
import json
import os
import re
import sys


def _load_dotenv(start: str = ".") -> None:
    """Load KEY=VALUE lines from a nearby .env into os.environ if not already set.

    apply_decisions is a separate entrypoint from run_augment, and nothing else
    loads .env here — so without this, ANTHROPIC_API_KEY sits in .env unread and
    every LLM-assisted resolution falls back to manual with "model unavailable",
    even though the key exists. Dependency-free (no python-dotenv): walks up from
    the working directory to find a .env and sets only vars not already exported,
    so a real shell export always wins. Silent if no .env is found.
    """
    d = os.path.abspath(start)
    for _ in range(6):                       # walk up a few levels, then stop
        p = os.path.join(d, ".env")
        if os.path.isfile(p):
            try:
                for line in open(p, encoding="utf-8"):
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
            except Exception:                # noqa: BLE001 — never fail on .env
                pass
            return
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent


_load_dotenv()

_VID_AD = r"V(?:\d+[a-z]?|_[A-Za-z0-9][A-Za-z0-9_.]*)"
_VID_BARE_AD = r"\b" + _VID_AD + r"\b"
_VID_ANCHOR_AD = r"(?:" + _VID_AD + r")$"

#: Verdicts that block scoring unless the SME explicitly re-grades at seal.
_NONPROCEEDABLE = {"BROKEN", "UNGRADEABLE", "NON_DETERMINISTIC"}
#: Verdicts an SME override may set (a re-grade must land on a proceedable one).
_PROCEEDABLE = {"SOUND", "SALVAGEABLE"}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


#: Which package field each editable artifact maps to. Defined once so the revert
#: and replace branches cannot drift apart. The deliverable and trajectory are
#: here because an SME resolution that changes a value must move them too — a
#: sealed golden whose deliverable still shows the pre-resolution value is a
#: self-contradiction the scorer cannot catch.
def _llm():
    """Return (call_llm, clean_json, model) from the auditor's evaluator, or None
    if unavailable (no key / import fails). Callers fall back to manual when None,
    so the LLM path never becomes a hard dependency of sealing."""
    try:
        from src.prompt_evaluator import (
            _call_llm, _clean_json_response, DEFAULT_JUDGE_MODEL)
        return _call_llm, _clean_json_response, DEFAULT_JUDGE_MODEL
    except Exception:                                           # noqa: BLE001
        return None


#: The LLM may reword prose but MUST NOT rewrite it wholesale — a coherent
#: application of the SME's intent stays close in length. Reject an edit whose
#: length changed by more than this factor as a likely hallucination; fall back
#: to manual so a runaway rewrite never seals silently.
_PROSE_LEN_GUARD = 1.6


def _llm_apply_prose_edit(artifact_text: str, sme_intent: str,
                          artifact_name: str):
    """Apply the SME's free-text INTENT to a prose artifact via the auditor's
    model. Returns the edited text, or None to fall back to manual.

    The model only REWORDS to satisfy the intent — it is told never to introduce
    a new numeric value, target, or fact (those are the SME's / the golden's, not
    the model's). The result is length-guarded and then shown to the SME in the
    regenerated HTML for confirmation; it is never trusted blind.
    """
    llm = _llm()
    if llm is None or not artifact_text.strip():
        return None
    call, _clean, model = llm
    prompt = (
        "You are editing one text artifact of a benchmark task to satisfy an "
        "SME's instruction. Apply the instruction faithfully and minimally.\n\n"
        "HARD RULES:\n"
        "- Return ONLY the full edited artifact text, nothing else.\n"
        "- Do NOT introduce any new numeric value, target, threshold, date, or "
        "factual claim. You may remove or reword text; you may not invent data.\n"
        "- Change as little as possible beyond what the instruction requires.\n"
        "- Preserve all content the instruction does not touch, verbatim.\n\n"
        f"ARTIFACT ({artifact_name}):\n<<<\n{artifact_text}\n>>>\n\n"
        f"SME INSTRUCTION:\n<<<\n{sme_intent}\n>>>\n\n"
        "Edited artifact text:")
    try:
        out = call(prompt, model, max_tokens=4000)
    except Exception:                                           # noqa: BLE001
        return None
    out = (out or "").strip()
    # Strip scaffolding the model sometimes echoes back around the artifact:
    # code fences, and the <<< >>> delimiters we wrap the artifact in. Remove
    # leading/trailing markers and any bare delimiter lines, so they never land
    # in the sealed artifact text.
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", out).strip()
    # leading/trailing <<< or >>> the model wrapped the whole answer in
    out = re.sub(r"^<{3,}\s*\n?", "", out)
    out = re.sub(r"\n?\s*>{3,}$", "", out)
    # any remaining bare delimiter lines anywhere
    out = "\n".join(ln for ln in out.splitlines()
                    if ln.strip() not in ("<<<", ">>>")).strip()
    if not out:
        return None
    # Reject if the model still smuggled scaffolding into the middle of the text
    # (a sign it misunderstood the wrapper) — safer to fall back to manual.
    if "<<<" in out or ">>>" in out:
        return None
    lo = len(artifact_text) / _PROSE_LEN_GUARD
    hi = len(artifact_text) * _PROSE_LEN_GUARD
    if not (lo <= len(out) <= hi):
        return None                       # length blew up/collapsed -> manual
    return out


#: A structured tolerance/value pin an SME can write in a question answer to set
#: a verifier's frozen target deterministically — no free-text parsing of a
#: golden number, no LLM. Syntax (case-insensitive), e.g.:
#:     PIN V3 tol=0.785
#:     PIN V3 value=157.039 tol=0.785
#: Applied straight into expected_values[Vid]; this is the one right way to edit a
#: scoring number: the SME names the verifier and the exact figure, code writes it.
_PIN_RE = re.compile(
    r"\bPIN\s+(?P<vid>"+_VID_AD+r")\b"
    r"(?:\s+value\s*=\s*(?P<value>-?[\d,]+\.?\d*))?"
    r"(?:\s+tol\s*=\s*(?P<tol>-?[\d,]+\.?\d*))?",
    re.IGNORECASE)


def _parse_pin(answer: str):
    """Return (vid, {value?, tol?}) from a structured PIN directive, or None.
    Only fires on the explicit `PIN Vid ...` form so ordinary prose answers are
    never mistaken for a numeric pin."""
    if not answer:
        return None
    m = _PIN_RE.search(answer)
    if not m:
        return None
    vid = m.group("vid").upper()
    fields = {}
    if m.group("value") is not None:
        try:
            fields["value"] = float(m.group("value").replace(",", ""))
        except ValueError:
            pass
    if m.group("tol") is not None:
        try:
            fields["tol"] = float(m.group("tol").replace(",", ""))
        except ValueError:
            pass
    if not fields:
        return None
    return vid, fields


def _num_key(v):
    """Coerce a value to a rounded float for numeric comparison, or None.
    Local copy of adjudicate_runs._num_key so apply_decisions has no cross-module
    dependency for this one-line helper."""
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _llm_format_step(sme_free_text: str, claims: list):
    """Structure an SME's free-text trajectory step into a claim node via the
    auditor's model. Returns a claim dict, or None to fall back to manual.

    CRITICAL: the model does NOT compute or invent the value — it EXTRACTS the
    value the SME stated. If the SME did not state a numeric value, the model
    returns null and we fall back to manual (a step with no SME-supplied value is
    never auto-added). The caller then re-derives to confirm the value is
    arithmetically consistent before it is accepted.
    """
    llm = _llm()
    if llm is None or not sme_free_text.strip():
        return None
    call, clean, model = llm
    existing = ", ".join(str(c.get("id")) for c in claims if c.get("id"))[:800]
    prompt = (
        "An SME wants to add ONE step (claim) to a benchmark task's golden "
        "derivation. Convert their free-text description into a single claim "
        "node. Respond with ONLY a JSON object, no prose.\n\n"
        "HARD RULES:\n"
        "- Extract the numeric value the SME STATED. Do NOT compute or invent a "
        "value. If the SME stated no numeric value, set \"recomputed\": null.\n"
        "- \"operation\" is a short formula/description in the SME's terms.\n"
        "- \"inputs\" lists the ids of existing claims this step consumes, chosen "
        "ONLY from the existing ids provided; use [] if none apply.\n"
        "- Choose a fresh \"id\" not already used.\n\n"
        f"EXISTING CLAIM IDS: {existing}\n\n"
        f"SME STEP DESCRIPTION:\n<<<\n{sme_free_text}\n>>>\n\n"
        "JSON with keys: id, label, operation, inputs (list of ids), recomputed "
        "(number or null):")
    try:
        raw = call(prompt, model, max_tokens=1200)
        obj = json.loads(clean(raw))
    except Exception:                                           # noqa: BLE001
        return None
    if not isinstance(obj, dict) or obj.get("recomputed") is None:
        return None                       # no SME value -> manual, never invent
    node = {
        "id": str(obj.get("id") or ""),
        "label": str(obj.get("label") or ""),
        "operation": str(obj.get("operation") or ""),
        "recomputed": obj.get("recomputed"),
        "input_provenance": [{"from_claim": i} for i in (obj.get("inputs") or [])
                             if isinstance(i, str)],
    }
    if not node["id"] or _num_key(node["recomputed"]) is None:
        return None
    existing_ids = {str(c.get("id")) for c in claims}
    if node["id"] in existing_ids:
        return None                       # id collision -> manual
    for p in node["input_provenance"]:
        if p["from_claim"] not in existing_ids:
            return None                   # references a non-existent claim
    return node


_ELIDE_MAX_SPAN = 400   # chars; refuse to auto-edit an elided anchor whose
#: hidden middle is larger than this — too much unseen text to replace safely.


def _locate_anchor_span(old: str, cur: str):
    """Locate the span in `cur` that an SME resolution's `old` anchor refers to.

    Returns (start, end) char offsets, or None if it can't be located safely.
    Two cases:
      * verbatim: `old` is an exact substring of `cur` -> its own span.
      * elided:   `old` contains an ellipsis (the auditor dropped the middle,
        e.g. "…offline retail. ... Do not use…"). Split on the ellipsis, require
        EVERY fragment to appear in `cur` in order, and return the span from the
        first fragment's start to the last fragment's end — but only if the
        hidden middle is not larger than _ELIDE_MAX_SPAN (guards against
        replacing a huge unseen span). Otherwise None -> caller flags the edit
        for manual application rather than guessing.
    """
    old = (old or "").strip()
    if not old:
        return None
    if old in cur:
        i = cur.find(old)
        return (i, i + len(old))
    if "..." not in old and "\u2026" not in old:
        return None
    frags = [f.strip() for f in re.split(r"\.{3,}|\u2026", old) if f.strip()]
    if len(frags) < 2:
        return None
    pos, start, end = 0, None, None
    for f in frags:
        i = cur.find(f, pos)
        if i == -1:
            return None                    # a fragment is missing -> unsafe
        if start is None:
            start = i
        end = i + len(f)
        pos = end
    if start is None or end is None:
        return None
    seen = sum(len(f) for f in frags)
    if (end - start) - seen > _ELIDE_MAX_SPAN:
        return None                        # hidden middle too large -> unsafe
    return (start, end)


_ARTIFACT_KEY = {
    "solution_logic": "corrected_solution_logic",
    "sanity_check": "corrected_sanity_check",
    "prompt": "corrected_prompt",
    "deliverable": "gold_deliverable_text",
    "golden_deliverable": "gold_deliverable_text",   # alias
}
#: The trajectory is not a separate text artifact in this build — it IS the
#: corrected_claim_verdicts + judgment_steps, from which claim_graph rebuilds the
#: step graph. An SME who needs to change a trajectory value therefore edits the
#: value in solution_logic (which flows into the deliverable) and, if the change
#: touches a claim's recomputed figure, the corrected_claim_verdicts entry. When a
#: standalone trajectory artifact is introduced, add its field here (e.g.
#: "trajectory": "gold_trajectory_text") and give it a producer in augment_task.
#: Adding an unbacked field now would be dead weight, so it is deliberately left
#: out until there is text for it to carry.


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _items(pkg):
    """Rebuild the same id space the report used. Must stay in step with
    augment_report._decision_items — the ids are the contract between them."""
    out = []
    va = pkg.get("verifier_audit") or {}
    # A change that targets a verifier later REWRITTEN by the property audit is
    # suppressed in the report (the rewrite supersedes it, so the SME sees one
    # decision, not two). The sealer must drop the same change items or the id
    # space drifts (sealer would carry a chg the SME never decided).
    rewritten_vids = {v.get("id") for v in (va.get("verifiers") or [])
                      if (v.get("rewrite") or "").strip()}

    def _change_superseded(c):
        import re as _re
        if str(c.get("artifact", "")).lower() != "verifiers":
            return False
        loc = str(c.get("location", "")) + " " + str(c.get("old", ""))
        return any(vid in _re.findall(_VID_BARE_AD, loc)
                   for vid in rewritten_vids)

    _ci = 0
    for c in (pkg.get("changes_applied") or []):
        if _change_superseded(c):
            _ci += 1
            continue
        out.append({"id": f"chg{_ci}", "kind": "change", "payload": c})
        _ci += 1
    for i, c in enumerate(pkg.get("judgment_changes_pending_sme") or []):
        out.append({"id": f"q{i}", "kind": "question", "payload": c})
    va = pkg.get("verifier_audit") or {}
    for i, v in enumerate(va.get("verifiers") or [] if "verifier_rewrites_pending" not in pkg else []):
        if (v.get("rewrite") or "").strip():
            out.append({"id": f"rw{i}", "kind": "rewrite", "payload": v})
    # rewrites held for the SME (same ids as augment_report: rp0, rp1, ...)
    for i, p in enumerate(pkg.get("verifier_rewrites_pending") or []):
        out.append({"id": f"rp{i}", "kind": "rewrite_proposal", "payload": p})
    for i, x in enumerate(pkg.get("verifier_splits_applied") or []):
        out.append({"id": f"sp{i}", "kind": "split", "payload": x})
    # coverage gaps MUST be built from the same deterministic
    # unwatched_load_bearing list the report uses (not the model's shorter
    # coverage_gaps list) or the id space drifts and accepted gap verifiers are
    # silently dropped at seal. Reuse the report's authoring so the proposal text
    # the SME accepted is exactly what gets applied.
    from src.augment_report import _step_index, _author_gap_verifier
    cov = pkg.get("step_coverage") or {}
    model_gaps = {str(g.get("step")): g for g in (va.get("coverage_gaps") or [])}
    step_idx = _step_index(pkg)
    for i, step in enumerate(cov.get("unwatched_load_bearing") or []):
        mg = model_gaps.get(str(step), {})
        proposed = mg.get("proposed_verifier") or _author_gap_verifier(
            str(step), step_idx.get(str(step), {}))
        out.append({"id": f"gap{i}", "kind": "gap",
                    "payload": {"step": str(step),
                                "proposed_verifier": proposed}})
    # advisory model gaps for steps NOT in the deterministic list (gapx*),
    # matching the report's second gap loop
    det_steps = {str(s) for s in (cov.get("unwatched_load_bearing") or [])}
    # numbered over the advisory steps ONLY, exactly like augment_report (enumerating all model gaps and
    # skipping left holes, so an SME decision on gapx{j} could land on a different gap)
    for j, step in enumerate(s2 for s2 in model_gaps if s2 not in det_steps):
        g = model_gaps[step]
        out.append({"id": f"gapx{j}", "kind": "gap",
                    "payload": {"step": str(step),
                                "proposed_verifier": g.get("proposed_verifier",
                                                           "")}})
    for i, c in enumerate(va.get("duplicate_clusters") or []):
        out.append({"id": f"dup{i}", "kind": "dedupe", "payload": c})
    # value-mismatch items: a numeric verifier whose value matches no step
    mapping = pkg.get("verifier_mapping_report") or {}
    for i, u in enumerate(mapping.get("unmatched") or []):
        if str(u.get("kind", "numeric")).lower() == "numeric":
            out.append({"id": f"unmatched{i}", "kind": "value_mismatch",
                        "payload": u})
    # temporal-drift items: a split child with an unpinned live value
    _seen = []
    for x in (pkg.get("verifier_splits_applied") or []):
        _seen += x.get("temporal_unpinned_children") or []
    for i, vid in enumerate(dict.fromkeys(_seen)):
        out.append({"id": f"tmp{i}", "kind": "temporal", "payload": {"verifier": vid}})
    # SME verdict override — present only when the model's verdict is
    # non-proceedable. Lets the SME re-grade the task once the defects that drove
    # the verdict are resolved, so a fully-repaired task can become scoreable. The
    # answer carries the new verdict + justification; it is applied LAST.
    if str(pkg.get("audit_verdict", "")).upper() in _NONPROCEEDABLE:
        out.append({"id": "verdict", "kind": "verdict_override",
                    "payload": {"current": pkg.get("audit_verdict")}})
    return out


def _step_value_index(pkg: dict) -> dict:
    """Map each step id to its computed value, so a gap verifier added for that
    step can be given a matching target. Claims carry a numeric recomputed value;
    judgment steps usually have no numeric value (they bind by decision, not
    value) so they are omitted here and left to text/id mapping."""
    idx = {}
    for c in pkg.get("corrected_claim_verdicts", []):
        v = c.get("recomputed")
        if isinstance(v, (int, float)):
            idx[str(c.get("id"))] = float(v)
    return idx


def _verifier_map(pkg):
    """{id: text} from the canonical block, preserving suffixed ids."""
    import re
    out = {}
    for line in (pkg.get("augmented_verifiers_text") or "").splitlines():
        m = re.match(r"\s*(V[\w]+)\s*:\s*(.*)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _next_id(vmap):
    import re
    n = max((int(re.sub(r"\D", "", k) or 0) for k in vmap), default=0)
    return f"V{n + 1}"


def apply_decisions(pkg: dict, dec: dict, force: bool = False) -> dict:
    items = _items(pkg)
    choices = dec.get("decisions") or {}
    answers = dec.get("answers") or {}

    # --- refusals -------------------------------------------------------
    ph, dh = pkg.get("run_hash"), dec.get("run_hash")
    if ph and dh and ph != dh and not force:
        raise SystemExit(
            f"run_hash mismatch: package {ph} vs decisions {dh}. These decisions "
            f"were made against a different run of this task, so applying them "
            f"would edit text the reviewer never saw. Re-review, or --force.")

    incomplete = []
    for it in items:
        c = choices.get(it["id"])
        a = (answers.get(it["id"]) or "").strip()
        if not c:
            incomplete.append(f"{it['id']} undecided")
        elif c != "accept" and not a and not (it["kind"] == "rewrite_proposal" and c == "reject"):
            # declining a held rewrite keeps the current wording: no reason required
            incomplete.append(f"{it['id']} {c} with no reason")
        elif it["kind"] == "question" and not a:
            incomplete.append(f"{it['id']} question unanswered")
        elif it["kind"] == "temporal" and c == "accept" and not a:
            # accepting a temporal item means "pin it" — the pin text is required
            incomplete.append(f"{it['id']} temporal pin not provided")
        elif it["kind"] == "value_mismatch" and c == "accept" and not a:
            # accepting means "here is the corrected verifier text / value"
            incomplete.append(f"{it['id']} value_mismatch resolution not provided")
        elif it["kind"] == "verdict_override" and c == "accept" and not a:
            # accepting means "re-grade to this verdict" — the new verdict + reason
            # is required; rejecting (keep the verdict) needs no answer
            incomplete.append(f"{it['id']} verdict override accepted but no "
                              f"new verdict given")
    if incomplete and not force:
        raise SystemExit("decisions incomplete, refusing to seal:\n  "
                         + "\n  ".join(incomplete)
                         + "\n(a partly-sealed package cannot be told apart from a "
                           "sealed one downstream; --force to override)")

    out = dict(pkg)
    vmap = _verifier_map(pkg)
    log = []
    _resolved_questions: dict = {}   # cid -> applied?  (drives the pending sweep)

    for it in items:
        cid, kind = it["id"], it["kind"]
        choice = choices.get(cid, "accept")
        reason = (answers.get(cid) or "").strip()
        pay = it["payload"]

        if kind == "change":
            if choice == "reject":
                # revert a mechanical edit: put the old text back
                art = pay.get("artifact")
                key = _ARTIFACT_KEY.get(art)
                if key and pay.get("old") and pay.get("new"):
                    cur = out.get(key) or ""
                    if pay["new"] in cur:
                        out[key] = cur.replace(pay["new"], pay["old"], 1)
                        log.append({"item": cid, "action": "reverted",
                                    "artifact": art, "reason": reason})
                    else:
                        log.append({"item": cid, "action": "revert_failed",
                                    "artifact": art,
                                    "detail": "new text not found; edit by hand",
                                    "reason": reason})
            elif choice == "other":
                art = pay.get("artifact")
                key = _ARTIFACT_KEY.get(art)
                if key and pay.get("new"):
                    cur = out.get(key) or ""
                    if pay["new"] in cur:
                        out[key] = cur.replace(pay["new"], reason, 1)
                        log.append({"item": cid, "action": "replaced_with_sme_text",
                                    "artifact": art, "text": reason})
            else:
                log.append({"item": cid, "action": "kept"})

        elif kind == "question":
            # Option B: the answer IS the resolution — infer how to APPLY it from
            # the payload, don't just file it. Three flavors, each applied
            # defensively (never blind-splice free text into an artifact):
            #   * text-artifact edit  — artifact is an editable text field
            #     (prompt/sanity/solution_logic/deliverable) AND the payload gives
            #     an `old` anchor: replace the anchor with the answer, but ONLY if
            #     the anchor is present. No anchor / anchor missing -> flag manual.
            #   * verifier pin        — artifact names a verifier (V..): set its
            #     text to the answer, like value_mismatch.
            #   * manual / design     — a binary file (pdf/xlsx) or a design-only
            #     question the pipeline can't apply: record it and add an explicit
            #     manual-edit requirement so it is never silently dropped and the
            #     not-scoreable gate stays honest.
            # `applied` gates whether this question is cleared from the pending
            # list below — an unapplied question keeps blocking, by design.
            art = pay.get("artifact")
            old = pay.get("old")
            applied = False
            manual_reason = None
            if choice in ("accept", "other") and reason:
                pin = _parse_pin(reason)
                if pin is not None:
                    # Structured numeric pin: "PIN V3 tol=0.785 [value=...]".
                    # Deterministic — the SME named the verifier and the exact
                    # figure; write it straight into the frozen target. No LLM,
                    # no free-text parsing of a golden number.
                    vid, fields = pin
                    ev = out.setdefault("expected_values", {})
                    tgt = ev.setdefault(vid, {"value": None, "tol": 0.0,
                                              "unit": "", "kind": "numeric",
                                              "source_of_verification":
                                                  "arithmetic"})
                    if "value" in fields:
                        tgt["value"] = fields["value"]
                    if "tol" in fields:
                        tgt["tol"] = fields["tol"]
                    applied = True
                    log.append({"item": cid, "action": "question_pin_applied",
                                "verifier": vid, "fields": fields})
                    key = None            # skip the text/manual dispatch below
                else:
                    key = _ARTIFACT_KEY.get(art)
                if pin is not None:
                    pass                  # already applied
                elif key and old:
                    cur = out.get(key) or ""
                    span = _locate_anchor_span(old, cur)
                    if span is not None:
                        s, e = span
                        out[key] = cur[:s] + reason + cur[e:]
                        applied = True
                        log.append({"item": cid, "action": "question_text_applied",
                                    "artifact": art})
                    else:
                        # anchor not locatable — the SME likely wrote INTENT, not
                        # exact replacement text. Apply it with the auditor's model
                        # (rewords only; never invents a value). SME confirms the
                        # result in the regenerated HTML.
                        edited = _llm_apply_prose_edit(cur, reason, art)
                        if edited is not None and edited != cur:
                            out[key] = edited
                            applied = True
                            log.append({"item": cid,
                                        "action": "question_text_applied_llm",
                                        "artifact": art})
                        else:
                            manual_reason = (
                                f"could not locate an anchor in {art} and the "
                                f"model edit was unavailable or rejected; SME "
                                f"must edit {art} by hand")
                elif key and not old:
                    # whole-artifact intent edit (no anchor at all) via the model
                    cur = out.get(key) or ""
                    edited = _llm_apply_prose_edit(cur, reason, art)
                    if edited is not None and edited != cur:
                        out[key] = edited
                        applied = True
                        log.append({"item": cid,
                                    "action": "question_text_applied_llm",
                                    "artifact": art})
                    else:
                        manual_reason = (f"no anchor and model edit unavailable "
                                         f"for {art}; SME must edit by hand")
                elif isinstance(art, str) and re.match(_VID_ANCHOR_AD, art) \
                        and art in vmap:
                    vmap[art] = reason
                    applied = True
                    log.append({"item": cid, "action": "question_verifier_pinned",
                                "verifier": art, "text": reason})
                else:
                    # binary file (pdf/xlsx) or unmapped artifact, or a pure
                    # design decision — cannot be applied programmatically.
                    manual_reason = (f"artifact {art!r} is not a programmatically "
                                     f"editable text field; SME applies by hand")
            elif choice == "reject":
                log.append({"item": cid, "action": "question_rejected",
                            "reason": reason})
                applied = True     # a reject is a resolution: stop blocking on it
            # always record for provenance
            out.setdefault("sme_resolutions", []).append({
                "artifact": art, "location": pay.get("location"),
                "question": pay.get("sme_question"), "answer": reason,
                "decided": choice, "applied": applied})
            if manual_reason:
                out.setdefault("sme_manual_edits_required", []).append({
                    "item": cid, "artifact": art, "answer": reason,
                    "why": manual_reason})
                log.append({"item": cid, "action": "question_manual_required",
                            "artifact": art, "why": manual_reason})
            # remember whether this question was resolved, for the pending sweep
            _resolved_questions[cid] = applied

        elif kind == "rewrite":
            vid = pay.get("id")
            if choice == "accept":
                log.append({"item": cid, "action": "kept", "verifier": vid})
            elif choice == "reject":
                orig = next((x for x in (pkg.get("_verifiers_before_audit") or [])
                             if x.get("id") == vid), None)
                if orig:
                    vmap[vid] = orig.get("text", vmap.get(vid, ""))
                    log.append({"item": cid, "action": "rewrite_reverted",
                                "verifier": vid, "reason": reason})
                else:
                    log.append({"item": cid, "action": "revert_unavailable",
                                "verifier": vid,
                                "detail": "pre-rewrite text not retained",
                                "reason": reason})
            else:
                vmap[vid] = reason
                log.append({"item": cid, "action": "verifier_replaced_by_sme",
                            "verifier": vid, "text": reason})

        elif kind == "rewrite_proposal":
            # held rewrite (tolerance band / untraceable numbers): applies ONLY on accept
            vid = pay.get("id")
            if choice == "accept":
                vmap[vid] = pay.get("proposed", vmap.get(vid, ""))
                log.append({"item": cid, "action": "rewrite_proposal_applied", "verifier": vid,
                            "reason": pay.get("reason")})
            elif choice == "reject":
                log.append({"item": cid, "action": "rewrite_proposal_declined", "verifier": vid,
                            "reason": reason})
            else:
                vmap[vid] = reason
                log.append({"item": cid, "action": "verifier_replaced_by_sme", "verifier": vid, "text": reason})

        elif kind == "split":
            if choice == "reject":
                parent, kids = pay.get("parent"), pay.get("children") or []
                for k in kids:
                    vmap.pop(k, None)
                vmap[parent] = pay.get("parent_text", "")
                ev = out.get("expected_values") or {}
                heir = pay.get("target_went_to")
                if heir and heir in ev:
                    ev[parent] = ev.pop(heir)
                for k in kids:
                    ev.pop(k, None)
                out["expected_values"] = ev
                log.append({"item": cid, "action": "split_undone",
                            "parent": parent, "reason": reason})
            else:
                log.append({"item": cid, "action": "kept",
                            "parent": pay.get("parent")})

        elif kind == "gap":
            if choice in ("accept", "other"):
                nid = _next_id(vmap)
                vmap[nid] = (pay.get("proposed_verifier", "") if choice == "accept"
                             else reason)
                step = pay.get("step")
                # Bind the new verifier to the step it was authored for, by seeding
                # an expected_value from that step's computed value. The seal
                # re-derive maps numeric verifiers to steps BY VALUE, so without a
                # target the added verifier stays unmapped and the coverage gap it
                # was meant to close persists. We know the step, so seed its value
                # directly (a claim's recomputed value; judgment steps bind by id).
                sidx = _step_value_index(pkg)
                sv = sidx.get(str(step))
                if sv is not None:
                    out.setdefault("_sealed_added_targets", {})[nid] = {
                        "value": sv, "tol": abs(sv) * 0.005 if sv else 0,
                        "source_of_verification": "arithmetic", "step": str(step)}
                out.setdefault("_sealed_step_bindings", {})[nid] = str(step)
                log.append({"item": cid,
                            "action": ("verifier_added" if choice == "accept"
                                       else "verifier_added_sme_text"),
                            "id": nid, "step": step})
            else:
                log.append({"item": cid, "action": "gap_left_open",
                            "step": pay.get("step"), "reason": reason})

        elif kind == "dedupe":
            # Two+ verifiers assert the same quantity. On accept, keep the one the
            # audit recommends and DROP the rest, so the set is MECE. The keep id
            # is parsed from recommended_action ("keep V17; V18 restates it"); if
            # it can't be parsed, fall back to keeping the lowest-numbered id. The
            # seal re-derive rebuilds the graph for the reduced set.
            ids = [i for i in (pay.get("verifier_ids") or []) if i in vmap]
            if choice == "accept" and len(ids) >= 2:
                import re as _re
                rec = str(pay.get("recommended_action") or "")
                m = _re.search(r"keep\s+("+_VID_AD+r")", rec, _re.I)
                keep = m.group(1) if m and m.group(1) in ids else min(
                    ids, key=lambda x: (int(_re.sub(r"\D", "", x) or 0), x))
                ev = out.get("expected_values") or {}
                dropped = []
                for vid in ids:
                    if vid != keep:
                        vmap.pop(vid, None)
                        ev.pop(vid, None)
                        dropped.append(vid)
                out["expected_values"] = ev
                out.setdefault("_crux_excluded_ids", []).extend(dropped)
                log.append({"item": cid, "action": "deduped",
                            "kept": keep, "dropped": dropped})
            elif choice == "reject":
                log.append({"item": cid, "action": "dedupe_declined",
                            "verifiers": ids, "reason": reason})
            else:
                log.append({"item": cid, "action": "kept_all",
                            "verifiers": ids})

        elif kind == "value_mismatch":
            # A numeric verifier whose value matches no computed step. Accept +
            # answer = the corrected verifier text (the SME fixes the number to the
            # derivation's value, or rewords it to the step it should check).
            # Reject = the golden is genuinely missing a step; left not-scoreable
            # so the gap is not forgotten. The seal re-derive re-maps the fixed
            # verifier — if it now matches a step, the mismatch clears itself.
            vid = pay.get("verifier")
            if choice == "accept" and reason and vid in vmap:
                vmap[vid] = reason
                log.append({"item": cid, "action": "value_corrected",
                            "verifier": vid, "text": reason})
            elif choice == "other" and reason:
                # "Something else" on a value_mismatch = the golden is missing the
                # step this verifier should watch; the SME describes it in free
                # text. Format it into a claim node with the auditor's model (which
                # EXTRACTS the SME's stated value, never invents one) and add it to
                # the trajectory. The seal re-derive below recomputes the graph and
                # the arithmetic verifier confirms the value chains — a node whose
                # value does not reconcile shows up as a failed claim, not a silent
                # corruption. If the model can't produce a valid node (no value,
                # bad chaining, id clash), fall back to manual.
                claims = out.get("corrected_claim_verdicts") or []
                node = _llm_format_step(reason, claims)
                if node is not None:
                    claims.append(node)
                    out["corrected_claim_verdicts"] = claims
                    log.append({"item": cid, "action": "golden_step_added_llm",
                                "claim": node["id"], "value": node["recomputed"],
                                "verifier": vid})
                else:
                    out.setdefault("sme_manual_edits_required", []).append({
                        "item": cid, "artifact": "corrected_claim_verdicts",
                        "answer": reason,
                        "why": "could not format a valid, chaining claim from the "
                               "description (missing value, bad inputs, or model "
                               "unavailable); SME must add the golden step by hand"})
                    log.append({"item": cid, "action": "golden_step_manual",
                                "verifier": vid})
            elif choice == "reject":
                log.append({"item": cid, "action": "value_mismatch_golden_gap",
                            "verifier": vid, "reason": reason})
            else:
                log.append({"item": cid, "action": "value_mismatch_unresolved",
                            "verifier": vid})

        elif kind == "verdict_override":
            # SME re-grades a non-proceedable verdict after resolving the defects
            # that drove it. Requires acceptance AND a target verdict + reason. The
            # new verdict must be a proceedable one; anything else is ignored and
            # the original verdict stands (recorded, so the attempt is auditable).
            if choice == "accept":
                new_v = ""
                # answer may be "SOUND: justification" or just the verdict word
                a = (reason or "").strip()
                head = a.split(":", 1)[0].strip().upper() if a else ""
                if head in _PROCEEDABLE:
                    new_v = head
                if new_v:
                    out["_verdict_before_override"] = out.get("audit_verdict")
                    out["audit_verdict"] = new_v
                    out["proceedable"] = True
                    out["verdict_override"] = {
                        "from": out.get("_verdict_before_override"),
                        "to": new_v, "justification": a,
                        "decided_by": dec.get("decided_by", "SME")}
                    log.append({"item": cid, "action": "verdict_overridden",
                                "from": out["_verdict_before_override"],
                                "to": new_v})
                else:
                    log.append({"item": cid, "action": "verdict_override_invalid",
                                "detail": f"answer did not name a proceedable "
                                          f"verdict {sorted(_PROCEEDABLE)}",
                                "reason": a})
            else:
                log.append({"item": cid, "action": "verdict_override_declined",
                            "reason": reason})

        elif kind == "temporal":
            # A verifier with an unpinned live value. Accept + answer = the SME's
            # pin, which REPLACES the verifier text (e.g. adds "as-of 2026-01-15
            # per market_wage_reference.txt"). Reject = leave it, but it stays
            # not-scoreable. The seal re-derive picks up the pinned text.
            vid = pay.get("verifier")
            if choice == "accept" and reason and vid in vmap:
                vmap[vid] = reason
                log.append({"item": cid, "action": "temporal_pinned",
                            "verifier": vid, "text": reason})
            elif choice == "reject":
                log.append({"item": cid, "action": "temporal_left_unpinned",
                            "verifier": vid, "reason": reason})
            else:
                log.append({"item": cid, "action": "temporal_no_pin",
                            "verifier": vid})

    out["augmented_verifiers_text"] = "\n".join(
        f"{k}: {vmap[k]}" for k in sorted(
            vmap, key=lambda x: (int(re.sub(r"\D", "", x) or 0), x)))
    out["sme_decisions"] = {"decisions": choices, "answers": answers,
                            "saved_at": dec.get("saved_at"),
                            "run_hash": dh}
    out["sme_applied_log"] = log
    # Clear only the questions that were actually APPLIED (or explicitly
    # rejected). A question whose answer could not be applied programmatically —
    # it needs a manual edit to a binary file, or its anchor was not found — stays
    # in the pending list so the package remains not-scoreable until the SME makes
    # the edit. This replaces the old unconditional wipe, which sealed the task as
    # scoreable even when the resolution never reached any artifact.
    pending = pkg.get("judgment_changes_pending_sme") or []
    if _resolved_questions:
        kept = []
        for i, c in enumerate(pending):
            cid_q = f"q{i}"
            if _resolved_questions.get(cid_q, False):
                continue                      # applied/rejected -> resolved, drop
            kept.append(c)                    # unapplied -> keep blocking
        out["judgment_changes_pending_sme"] = kept
    else:
        out["judgment_changes_pending_sme"] = []

    # Re-derive the frozen graph from the SEALED artifacts. Every SME edit above
    # is now applied — verifier text (incl. reverted splits/rewrites and added
    # gap verifiers), solution logic, prompt, sanity, deliverable and trajectory —
    # so the verifier id space and/or the targets may differ from what the
    # augmenter froze. The dag / base weights / crux set / Shapley weights it
    # produced are stale; recompute them here, on the sealed set, as the single
    # authoritative freeze. Same routine the augmenter used for its preview, so
    # the two cannot diverge; only the inputs (post- vs pre-SME) differ.
    # Seed targets for gap verifiers added at seal so the re-derive maps them to
    # the step they were authored for (they close the coverage gap they resolved).
    # Without this, an added numeric verifier has no frozen target, stays unmapped,
    # and the gap the SME just closed reappears as unwatched.
    if out.get("_sealed_added_targets"):
        ev = dict(out.get("expected_values") or {})
        for vid, tgt in out["_sealed_added_targets"].items():
            ev.setdefault(vid, tgt)
        out["expected_values"] = ev

    from src.augment_task import derive_frozen_graph
    try:
        # compute_shapley=True: the seal is the ONLY place crux Shapley is
        # computed now — on the final sealed set, once, so its value is stable and
        # meaningful rather than a pre-seal preview the SME's edits invalidate.
        out = derive_frozen_graph(
            out, compute_shapley=True,
            forced_verifier_to_step=out.get("_sealed_step_bindings"),
            seal_crux=True)
        out["frozen_graph_source"] = "sealed"
        # A successful seal re-derive supersedes any stale 'error' carried in from
        # an earlier (pre-fix) build of the input package — otherwise the final
        # scoreability gate keys off a stale error and misreports the reason.
        if out.get("error"):
            out["_cleared_stale_error"] = out.pop("error")
    except Exception as e:
        # A re-derive that cannot run means the sealed set is inconsistent, so the
        # package must not present itself as scoreable with a stale graph.
        out["scoreable"] = False
        out["not_scoreable_reason"] = f"seal-time graph re-derive failed: {e}"
        out["frozen_graph_source"] = "STALE_rederive_failed"

    # Post-seal coverage closure. The seal re-derive rebuilds the DAG from the
    # POST-DECISION artifacts, and the SME's own decisions (splits, drops, added
    # verifiers) can reshape the graph so that steps NOT in the pre-seal decision
    # list now show as unwatched. The SME resolved every gap they were shown, yet
    # the sealed task could still have holes they never saw. Rather than ship a
    # silently-not-scoreable task or force a second decision round, auto-author a
    # verifier for each newly-surfaced gap (the same content-derived proposal used
    # pre-seal), bind it, and re-derive ONCE more. Bounded to a single pass; any
    # gap still open after that is reported, not hidden. Everything auto-added is
    # recorded in _auto_authored_gaps for the final report and audit.
    try:
        newly = list((out.get("step_coverage") or {}).get(
            "unwatched_load_bearing") or [])
    except Exception:                                           # noqa: BLE001
        newly = []
    if newly and out.get("frozen_graph_source") == "sealed":
        from src.augment_report import _step_index, _author_gap_verifier
        vmap2 = _verifier_map(out)
        sidx = _step_index(out)
        svidx = _step_value_index(out)
        auto = []
        for step in newly:
            nid = _next_id(vmap2)
            proposed = _author_gap_verifier(str(step), sidx.get(str(step), {}))
            vmap2[nid] = proposed
            out.setdefault("_sealed_step_bindings", {})[nid] = str(step)
            sv = svidx.get(str(step))
            if sv is not None:
                out.setdefault("_sealed_added_targets", {})[nid] = {
                    "value": sv, "tol": abs(sv) * 0.005 if sv else 0,
                    "source_of_verification": "arithmetic", "step": str(step)}
            auto.append({"id": nid, "step": str(step), "verifier": proposed})
        # write the augmented verifier set back and re-seed targets
        out["augmented_verifiers_text"] = "\n".join(
            f"{vid}: {txt}" for vid, txt in vmap2.items())
        if out.get("_sealed_added_targets"):
            ev = dict(out.get("expected_values") or {})
            for vid, tgt in out["_sealed_added_targets"].items():
                ev.setdefault(vid, tgt)
            out["expected_values"] = ev
        out["_auto_authored_gaps"] = (out.get("_auto_authored_gaps") or []) + auto
        try:
            out = derive_frozen_graph(
                out, compute_shapley=True,
                forced_verifier_to_step=out.get("_sealed_step_bindings"),
                seal_crux=True)
            still = list((out.get("step_coverage") or {}).get(
                "unwatched_load_bearing") or [])
            log.append({"item": "_post_seal_coverage",
                        "action": "auto_authored_gap_verifiers",
                        "added": [a["id"] for a in auto],
                        "still_unwatched": still})
        except Exception as e:                                  # noqa: BLE001
            out["scoreable"] = False
            out["not_scoreable_reason"] = (
                f"post-seal gap-closure re-derive failed: {e}")

    out["sealed"] = True
    # Complete source_of_verification map for EVERY verifier, so GEAR's per-edge
    # lambda (sov mode) and any downstream recomputation can be done from the
    # sealed record alone. SOV lives in expected_values for verifiers with a
    # frozen value; a verifier WITHOUT one is a decision/judgment check, whose
    # verification source is llm_judgment (the weak-lambda edge in GEAR). Building
    # the full map here means the CSV can carry SOV for the whole DAG, not just
    # the numeric subset.
    ev_all = out.get("expected_values") or {}
    dag_all = out.get("dag") or {}
    sov_map = {}
    for vid in dag_all:
        s = (ev_all.get(vid) or {}).get("source_of_verification")
        sov_map[vid] = s or "llm_judgment"
    out["sov_map"] = sov_map
    # Seal-time Shapley on BOTH graphs, plus the crux subgraph. crux_shapley_weights
    # (crux over the full DAG, non-crux as context) is already set by the
    # re-derive; here we add the full-DAG distribution over ALL verifiers and the
    # crux-subgraph distribution (crux vs crux only), and store the crux DAG.
    try:
        from src.crux_shapley import (full_dag_shapley, crux_dag_shapley,
                                       crux_subgraph)
    except Exception:                                           # noqa: BLE001
        full_dag_shapley = crux_dag_shapley = crux_subgraph = None
    if full_dag_shapley and out.get("dag") and out.get("base_weights"):
        all_vs = [{"id": v} for v in out["dag"].keys()]
        try:
            out["full_dag_shapley_weights"] = full_dag_shapley(
                all_vs, out["dag"], out["base_weights"])
            crux = out.get("crux_ids") or []
            out["crux_dag"] = crux_subgraph(out["dag"], crux)
            out["crux_dag_shapley_weights"] = crux_dag_shapley(
                all_vs, out["dag"], out["base_weights"], crux)
        except Exception as e:                                  # noqa: BLE001
            out["shapley_note"] = f"seal Shapley (full/crux-dag) failed: {e}"
    # scoreable only if the audit was proceedable, there is no error, AND the
    # re-derive did not itself veto (e.g. a crux verifier still judgment_flagged
    # after the SME pass, or the re-derive failed above).
    out["scoreable"] = (bool(out.get("proceedable"))
                        and not out.get("error")
                        and out.get("scoreable", True))
    if not out["scoreable"]:
        out["not_scoreable_reason"] = (
            out.get("not_scoreable_reason")
            or f"audit verdict {out.get('audit_verdict')} is not proceedable")
    else:
        out["not_scoreable_reason"] = ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--out", default="output/final")
    ap.add_argument("--force", action="store_true",
                    help="apply despite a run_hash mismatch or an incomplete file")
    args = ap.parse_args()

    pkg, dec = load(args.augment), load(args.decisions)
    if pkg.get("task_id") != dec.get("task_id"):
        raise SystemExit(f"task mismatch: {pkg.get('task_id')} vs {dec.get('task_id')}")

    sealed = apply_decisions(pkg, dec, force=args.force)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{pkg['task_id']}_final.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sealed, f, indent=2, ensure_ascii=False, default=str)
    # clean final prompt-package HTML for SME review (no decision scaffolding)
    html_path = os.path.join(args.out, f"{pkg['task_id']}_final.html")
    try:
        from src.augment_report import write_sealed_report
        write_sealed_report(sealed, html_path)
    except Exception as e:                                       # noqa: BLE001
        html_path = None
        print(f"  (sealed HTML not written: {e})")

    print(f"\n=== {pkg['task_id']} sealed ===")
    for e in sealed["sme_applied_log"]:
        extra = e.get("verifier") or e.get("id") or e.get("artifact") or ""
        print(f"  {e['item']:6s} {e['action']:26s} {extra}")
    print(f"\n  resolutions recorded : {len(sealed.get('sme_resolutions') or [])}")
    print(f"  verifiers            : "
          f"{len(sealed['augmented_verifiers_text'].splitlines())}")
    print(f"  scoreable            : {sealed['scoreable']}"
          + (f"  ({sealed['not_scoreable_reason']})"
             if not sealed["scoreable"] else ""))
    print(f"\nwrote {path}")
    if html_path:
        print(f"wrote {html_path}")


if __name__ == "__main__":
    main()