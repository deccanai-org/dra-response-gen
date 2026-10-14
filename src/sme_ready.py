"""SME-ready adjudicated packages (ported from the sme-ready-package skill, 6 Oct).

Adjudicated reports carried ~86 decision cards per task (863 across the 10 EDGAR tasks), ~85% mechanical.
make_sme_ready() applies the mechanical ones deterministically — split repair (trap children named and
target-free, value child carries the target), bundled show-your-work clauses folded into one verifier,
tolerance bands HELD as SME proposals, gap verifiers for auditor-confirmed steps, agreeing duplicates
merged, false value-mismatches cleared, already-applied changes listed — and leaves only genuine
judgements (863 -> 249 on the EDGAR batch). Called at the end of adjudicate_runs.build_sme_package.
Disable with SME_READY=0.
"""
import copy, html as _html, json, os, re

VID = re.compile(r"^(V\d+[a-z]?)\s*:\s*(.*)$")
TRAP = re.compile(r"^\s*(?:reporting|stating|a response (?:that )?reports?)\s+(.+?)\s+fails\b|\(trap value", re.I)
TRAP_VAL = re.compile(r"^\s*(?:reporting|stating|a response (?:that )?reports?)\s+(.+?)\s+fails\b", re.I)
BUNDLES = [r",?\s*with inputs,? sources,? and formula shown", r",?\s*with inputs and sources? shown",
           r",?\s*inputs,? (?:sources? )?and (?:formula|source)s? shown", r",?\s*with (?:the )?component lines cited",
           r",?\s*with (?:the )?source(?:s)? cited", r",?\s*with (?:the )?formula shown",
           r",?\s*with (?:the |their |its )?sources? and (?:the )?formula shown"]
SHOW_ONLY = re.compile(r"^\s*inputs?\b.*\b(sources?|formula)\b.*\bshown\b", re.I)
BAND = re.compile(r"\s*\((?:±|\+/-)\s*[\d.,]+\s*(?:pp|%|x|K|M|bps?)?\)|\s*(?:±|\+/-)\s*[\d.,]+\s*(?:pp|%|x|K|M|bps?)?", re.I)
ALT_STEP = re.compile(r"alternative|reading [ab]\b|\bif\b|hypothetical|counterfactual|\bwould\b|under the .{0,40} reading", re.I)
SHOW_WORK = ("For every figure the response reports, it states the inputs used and the source document "
             "each input comes from.")


def _num(s):
    try:
        return float(str(s).replace(",", "").replace("K", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def _fmt(v):
    f = _num(v)
    if f is None:
        return str(v)
    return f"{f:,.0f}" if float(f).is_integer() else f"{f:,}"


def _in_golden(tgt, dec, golden):
    """The verifier's value is printed in the corrected solution logic (small integers excluded: too
    likely to match by accident)."""
    if tgt is None or not golden or (abs(tgt) < 100 and not dec):
        return False
    rend = {f"{abs(tgt):.{dec}f}"} if dec else {f"{abs(tgt):,.0f}", f"{abs(tgt):.0f}"}
    return any(re.search(r"(?<![\d.,])" + re.escape(r) + r"(?![\d])", golden) for r in rend)


def parse_verifiers(text):
    out = []
    for line in (text or "").splitlines():
        m = VID.match(line.strip())
        if m:
            out.append([m.group(1), m.group(2).strip()])
        elif line.strip() and out:
            out[-1][1] += " " + line.strip()
    return out


def render_verifiers(vs):
    return "\n".join(f"{i}: {t}" for i, t in vs)


def _subject(text):
    s = re.split(r"\s=\s|\s+(?:is|equals)\s+-?[\d$]", text, maxsplit=1)[0]
    for b in BUNDLES:
        s = re.sub(b, "", s, flags=re.I)
    return s.strip(" ,;:.")


def _next_id(vs):
    """Next id in the normal sequence (stray 4-digit ids in some packages are ignored)."""
    used = {i for i, _ in vs}
    n = max([int(re.match(r"V(\d+)", i).group(1)) for i, _ in vs if len(re.match(r"V(\d+)", i).group(1)) <= 3] + [0]) + 1
    while f"V{n}" in used:
        n += 1
    return f"V{n}"


def _clean_proposal(t):
    """A proposed verifier: no tolerance band, no [target ..] annotation, no bundled FAIL-IF trap.
    Returns (value_text, trap_value_or_None)."""
    t = re.sub(r"\s*\[target [^\]]*\]", "", str(t))
    trap = None
    m = re.search(r"\s*FAIL IF (?:reported as |the response reports )?(-?[\d.,]+\s*(?:%|x|K|M)?)\.?", t)
    if m and "absent" not in m.group(0).lower():
        trap = m.group(1).strip(); t = t[:m.start()] + t[m.end():]
    t = BAND.sub("", t)
    t = re.sub(r"\s*\(within tolerance\)", " at the stated precision", t)
    t = re.sub(r"\s*FAIL IF absent or a different value\.?", "; any other value fails.", t)
    t = re.sub(r"=\s*(-?\d{4,}(?:\.\d+)?)", lambda mm: "= " + _fmt(mm.group(1)), t)
    t = re.sub(r"\.\s*;", ";", re.sub(r"\s{2,}", " ", t)).strip()
    return t, trap


def make_sme_ready(pkg, derive=True):
    """Apply every auto-approvable item to an adjudicated package; leave only genuine judgements as
    decision cards (apply_decisions builds its card list from the same fields). Returns a new package
    with the record of what was applied under pkg["sme_ready"]. derive=False skips the graph
    recomputation (tests)."""
    from src.augment_report import _step_index, _author_gap_verifier
    p = copy.deepcopy(pkg)
    log = {"split_repairs": [], "bundles_removed": [], "tolerance_held": [], "gap_verifiers_added": [],
           "duplicates_merged": [], "mismatches_cleared": [], "changes_already_applied": [],
           "rewrites_already_applied": []}
    vs = parse_verifiers(p.get("augmented_verifiers_text"))
    ev = p.setdefault("expected_values", {})
    text = {i: t for i, t in vs}

    # 1. split repair
    trap_ids = set()
    for s in p.get("verifier_splits_applied") or []:
        kids = [k for k in s.get("children") or [] if k in text]
        value_kids = [k for k in kids if not TRAP.search(text[k])]
        subj = next((_subject(text[k]) for k in value_kids if "=" in text[k]), "") or \
            _subject(s.get("parent_text", ""))
        rep = {"parent": s.get("parent"), "children": {}}
        for k in kids:
            t = text[k]
            if TRAP.search(t):
                trap_ids.add(k)
                m = TRAP_VAL.search(t)
                if subj and m and subj.lower()[:25] not in t.lower():
                    t = f"The response does not report {subj} as {m.group(1).strip()}."
                ev.pop(k, None)
            text[k] = t
            rep["children"][k] = t
        for k in value_kids:                           # the value child carries the parent's target
            if k not in ev and "=" in text[k]:
                m = re.search(r"=\s*(-?[\d,]+(?:\.\d+)?)\s*(%|x|K|M)?", text[k])
                if m and _num(m.group(1)) is not None:
                    v = _num(m.group(1))
                    ev[k] = {"value": v, "tol": round(abs(v) * 0.005, 6), "unit": m.group(2) or "",
                             "kind": "numeric", "source_of_verification": "split_repair",
                             "derived_from": "verifier_text"}
                    rep.setdefault("target_set", {})[k] = v
        log["split_repairs"].append(rep)
    p["verifier_splits_applied"] = []

    # 2. bundled clauses -> one show-your-work verifier
    for i in list(text):
        t0 = text[i]
        t = t0
        for b in BUNDLES:
            t = re.sub(b, "", t, flags=re.I)
        # a removed clause can leave its connective behind ("…million), with. FAIL IF …")
        t = re.sub(r",?\s*\bwith\s*(?=[.;,]|$)", "", t)
        if t != t0:
            text[i] = re.sub(r"\s{2,}", " ", t).strip()
            log["bundles_removed"].append(i)
    for i in [i for i in list(text) if SHOW_ONLY.search(text[i])]:
        text.pop(i, None); ev.pop(i, None); log.setdefault("show_work_merged", []).append(i)
    vs = [[i, text[i]] for i, _ in vs if i in text]
    if (log["bundles_removed"] or log.get("show_work_merged")) and not any("states the inputs used" in t for _, t in vs):
        sw = _next_id(vs); vs.append([sw, SHOW_WORK]); text[sw] = SHOW_WORK
        log["show_work_verifier"] = sw

    # 3. tolerance bands -> held for the SME
    pending = list(p.get("verifier_rewrites_pending") or [])
    for i, t in vs:
        if BAND.search(t):
            stripped = re.sub(r"\s{2,}", " ", BAND.sub("", t)).strip()
            text[i] = stripped
            pending.append({"id": i, "original": stripped, "proposed": t, "reason": "tolerance"})
            log["tolerance_held"].append(i)
    p["verifier_rewrites_pending"] = pending
    vs = [[i, text[i]] for i, _ in vs]
    if not p.get("verifier_rewrites_auto_applied"):
        p["verifier_rewrites_auto_applied"] = []
    va = p.setdefault("verifier_audit", {})
    for v in va.get("verifiers") or []:              # older packages: rewrites applied in place
        if (v.get("rewrite") or "").strip() and v.get("id") not in log["tolerance_held"]:
            log["rewrites_already_applied"].append({"id": v.get("id"), "new": v.get("rewrite")})

    # 5. duplicates whose values agree -> keep the first
    keep_dups = []
    for c in va.get("duplicate_clusters") or []:
        ids = [x for x in c.get("verifier_ids") or [] if x in text]
        if c.get("values_agree") and len(ids) > 1:
            for d in ids[1:]:
                text.pop(d, None); ev.pop(d, None)
            log["duplicates_merged"].append({"kept": ids[0], "dropped": ids[1:], "quantity": c.get("quantity")})
        else:
            keep_dups.append(c)
    va["duplicate_clusters"] = keep_dups
    vs = [[i, t] for i, t in vs if i in text]

    # 7. already-applied mechanical changes: record, not a decision
    log["changes_already_applied"] = list(p.get("changes_applied") or [])
    p["changes_applied"] = []

    for r in log["split_repairs"]:
        r["children"] = {k: text[k] for k in r["children"] if k in text}
    p["augmented_verifiers_text"] = render_verifiers(vs)
    p["sme_ready"] = log
    p = _derive(p, log, derive)                        # graph on the edited set
    vs = parse_verifiers(p.get("augmented_verifiers_text")); text = {i: t for i, t in vs}
    ev = p.setdefault("expected_values", {}); va = p.setdefault("verifier_audit", {})
    # 4. definite gaps on CONFIRMED steps -> verifiers added
    sidx = _step_index(p)
    status = {str(c.get("id")): str(c.get("status", "")).upper() for c in (p.get("corrected_claim_verdicts") or [])}
    cov = p.setdefault("step_coverage", {})
    model_gaps = {str(g.get("step")): g for g in (va.get("coverage_gaps") or [])}
    still = []
    for step in [str(x) for x in cov.get("unwatched_load_bearing") or []]:
        info = sidx.get(step, {})
        label = str(info.get("label") or info.get("what") or "")
        if ALT_STEP.search(label):                     # the auditor's alternative reading: never a verifier
            log.setdefault("gaps_dismissed", []).append({"step": step, "label": label, "why": "alternative reading"})
            continue
        # MISLABELLED_INPUT: value recomputed and confirmed by the auditor, only its source label was wrong
        ok = status.get(step, "CONFIRMED") in ("CONFIRMED", "CORRECTED", "MISLABELLED_INPUT", "")
        prop = (model_gaps.get(step) or {}).get("proposed_verifier") or _author_gap_verifier(step, info)
        val = info.get("value", info.get("claimed"))
        if not ok or _num(val) is None or not prop:
            still.append(step); continue
        t, trapv = _clean_proposal(prop)
        nid = _next_id(vs); vs.append([nid, t]); text[nid] = t
        if trapv:
            tid = _next_id(vs); tt = f"The response does not report {_subject(t)} as {trapv}."
            vs.append([tid, tt]); text[tid] = tt
        ev[nid] = {"value": _num(val), "tol": round(abs(_num(val)) * 0.005, 6), "unit": "", "kind": "numeric",
                   "source_of_verification": "gap_verifier", "derived_from": "step_value"}
        p.setdefault("verifier_to_step", {})[nid] = step
        log["gap_verifiers_added"].append({"id": nid, "step": step, "text": t})
    cov["unwatched_load_bearing"] = still

    p["augmented_verifiers_text"] = render_verifiers(vs)
    p = _derive(p, log, derive)                        # graph with the gap verifiers
    text = {i: t for i, t in parse_verifiers(p.get("augmented_verifiers_text"))}
    sidx = _step_index(p)
    det_now = {str(x) for x in (p.get("step_coverage") or {}).get("unwatched_load_bearing") or []}
    va = p.setdefault("verifier_audit", {})
    va["coverage_gaps"] = [g for g in (va.get("coverage_gaps") or []) if str(g.get("step")) not in det_now]
    for g in va["coverage_gaps"]:                      # proposals left for the SME: no tolerance wording
        if g.get("proposed_verifier"):
            t = BAND.sub("", re.sub(r"\s*\(within tolerance\)", " at the stated precision", str(g["proposed_verifier"])))
            g["proposed_verifier"] = re.sub(r"\s*\[target [^\]]*\]", "", re.sub(r"\s{2,}", " ", t)).strip()
    # 6. value mismatches: drop trap checks and roundings at the verifier's stated precision
    step_vals = [_num(v.get("value", v.get("claimed"))) for v in sidx.values()]
    step_vals += [_num(c.get(k)) for c in (p.get("corrected_claim_verdicts") or []) for k in ("claimed", "recomputed")]
    step_vals = [v for v in step_vals if v is not None]
    keep_um = []
    mp = p.setdefault("verifier_mapping_report", {})
    for u in mp.get("unmatched") or []:
        vid = u.get("verifier"); t = text.get(vid, "")
        if vid not in text:
            log["mismatches_cleared"].append({"verifier": vid, "why": "verifier merged as a duplicate"}); continue
        if vid in trap_ids or TRAP.search(t):
            log["mismatches_cleared"].append({"verifier": vid, "why": "trap check: a wrong value by design"}); continue
        tgt = _num(u.get("target"))
        m = re.search(r"=\s*(-?[\d,]+(?:\.(\d+))?)", t)
        dec = len(m.group(2)) if m and m.group(2) else 0
        tol = 0.5 * 10 ** (-dec)
        if tgt is not None and any(abs(sv * f - tgt) <= max(tol + 1e-9, 0.005 * max(abs(sv * f), abs(tgt), 1.0))
                                   for sv in step_vals for f in (1, 100, 0.01)):
            log["mismatches_cleared"].append({"verifier": vid, "why": f"rounds to a computed step at {dec} decimal(s)"}); continue
        if _in_golden(tgt, dec, str(p.get("corrected_solution_logic") or "")):
            log["mismatches_cleared"].append({"verifier": vid, "why": "value is in the corrected solution logic"}); continue
        keep_um.append(u)
    mp["unmatched"] = keep_um

    p["sme_ready"] = log
    return p


def _derive(p, log, enabled=True):
    if not enabled:
        log["graph_recomputed"] = "skipped"; p["sme_ready"] = log; return p
    try:                                               # recompute graph / crux / weights on the current set
        from src.augment_task import derive_frozen_graph
        p = derive_frozen_graph(p, compute_shapley=True)
        log["graph_recomputed"] = True
    except Exception as e:                             # noqa: BLE001
        log["graph_recomputed"] = f"kept previous graph ({type(e).__name__}: {str(e)[:120]})"
    p["sme_ready"] = log
    return p


def auto_block_html(log):
    e = _html.escape
    def li(x): return f"<li>{x}</li>"
    rows = []
    if log["split_repairs"]:
        rows.append(f"<h4>Splits repaired and applied ({len(log['split_repairs'])})</h4><ul>" + "".join(
            li(f"<b>{e(r['parent'])}</b> → " + "; ".join(f"<b>{e(k)}</b>: {e(v)}" for k, v in r["children"].items()))
            for r in log["split_repairs"]) + "</ul>")
    if log["gap_verifiers_added"]:
        rows.append(f"<h4>Verifiers added for unwatched confirmed steps ({len(log['gap_verifiers_added'])})</h4><ul>"
                    + "".join(li(f"<b>{e(g['id'])}</b> (step {e(g['step'])}): {e(g['text'])}") for g in log["gap_verifiers_added"]) + "</ul>")
    if log["duplicates_merged"]:
        rows.append(f"<h4>Duplicates with agreeing values merged ({len(log['duplicates_merged'])})</h4><ul>"
                    + "".join(li(f"kept <b>{e(d['kept'])}</b>, dropped {e(', '.join(d['dropped']))} — {e(str(d.get('quantity') or ''))}") for d in log["duplicates_merged"]) + "</ul>")
    if log["mismatches_cleared"]:
        rows.append(f"<h4>Value mismatches cleared ({len(log['mismatches_cleared'])})</h4><ul>"
                    + "".join(li(f"<b>{e(m['verifier'])}</b>: {e(m['why'])}") for m in log["mismatches_cleared"]) + "</ul>")
    if log["bundles_removed"]:
        rows.append(f"<h4>Bundled clauses removed ({len(log['bundles_removed'])} verifiers)</h4><p>"
                    + e(", ".join(log["bundles_removed"])) + (f" — covered once by <b>{e(log.get('show_work_verifier',''))}</b> (show-your-work)" if log.get("show_work_verifier") else "") + "</p>")
    if log["changes_already_applied"]:
        rows.append(f"<h4>Mechanical corrections already applied ({len(log['changes_already_applied'])})</h4><ul>"
                    + "".join(li(f"<b>{e(str(c.get('artifact')))} — {e(str(c.get('location')))}</b>: was <i>{e(str(c.get('old')))}</i> → now <i>{e(str(c.get('new')))}</i>") for c in log["changes_already_applied"]) + "</ul>")
    if log["rewrites_already_applied"]:
        rows.append(f"<h4>Rewordings already applied ({len(log['rewrites_already_applied'])})</h4><ul>"
                    + "".join(li(f"<b>{e(r['id'])}</b>: {e(r['new'])}") for r in log["rewrites_already_applied"]) + "</ul>")
    n = (len(log["split_repairs"]) + len(log["gap_verifiers_added"]) + len(log["duplicates_merged"]) +
         len(log["mismatches_cleared"]) + len(log["changes_already_applied"]) + len(log["rewrites_already_applied"]))
    return (f"<details class='card'><summary><b>Applied automatically — {n} item(s), no decision needed</b> "
            f"(click to review)</summary>" + "".join(rows) + "</details>")