#!/usr/bin/env python3
"""
run_augment.py — augment prompt packages directly (no SME-approval gate).

For each row: audit -> apply corrections -> generate golden deliverable + DAG +
crux set + crux-only Shapley (single augment call) -> write an AUGMENTED CSV plus
per-task {task_id}_augment.html and {task_id}_golden.html.

    python run_augment.py --csv prompt_data.csv --out-dir output/augmented
    python run_augment.py --csv prompt_data.csv --row 1
    python run_augment.py --csv prompt_data.csv --from 1 --to 20 --no-files

The augmented CSV keeps every original column and ADDS:
  corrected_solution_logic, golden_deliverable, augmented_verifiers, dag_json,
  crux_verifier_ids, crux_shapley_weights_json, base_weights_json,
  audit_verdict, changes_applied_json, judgment_pending_json, augment_error
"""

import argparse, csv, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import configure_api
from src.auditor import (build_header_map, get_field, HeaderError,
                        read_task_csv)
from src.prompt_evaluator import DEFAULT_JUDGE_MODEL
from src.augment_task import augment_task
from src.augment_report import write_augment_report, write_golden_report

NEW_COLS = [
    "corrected_solution_logic", "golden_deliverable", "augmented_verifiers",
    "dag_json", "crux_verifier_ids", "crux_shapley_weights_json",
    "base_weights_json", "expected_values_json", "audit_verdict", "changes_applied_json",
    "judgment_pending_json", "skipped_inputs", "scoreable", "not_scoreable_reason",
    "augment_error",
]


def _blank_new_cols(row):
    """Guarantee every augmented column exists on a row (even on failure) so the
    CSV never silently loses columns for rows that hit the except branch."""
    for c in NEW_COLS:
        row.setdefault(c, "")
    return row


def _adjudicate_task(run_dir_base, manifest, model, no_llm=False, no_html=False):
    """Reconcile a task's N runs (from its manifest) into one final JSON + HTML,
    written alongside the runs. Wires the master-LLM cluster+judge to Opus unless
    no_llm. Kept resilient: a failure here never aborts the augment batch."""
    try:
        from adjudicate_runs import adjudicate, render_html, build_sme_package
    except Exception as e:                                       # noqa: BLE001
        print(f"    !! adjudicator unavailable ({e}); runs stored, not merged")
        return
    run_jsons = []
    for r in manifest.get("runs", []):
        p = r.get("json")
        if p and os.path.exists(p):
            try:
                run_jsons.append(json.load(open(p, encoding="utf-8")))
            except Exception as e:                               # noqa: BLE001
                print(f"    warn: could not load {p}: {e}")
    if len(run_jsons) < 3:
        print(f"    !! only {len(run_jsons)} usable run(s); need >=3 to "
              f"adjudicate — skipping")
        return

    cluster = judge = None
    if not no_llm:
        try:
            from src.prompt_evaluator import _call_llm, DEFAULT_JUDGE_MODEL

            def _call(prompt):
                return _call_llm(prompt, model or DEFAULT_JUDGE_MODEL,
                                 max_tokens=4000,
                                 system_prompt="You reconcile multiple runs of a "
                                 "task. You only SELECT among options given; you "
                                 "never invent.")
            cluster = judge = _call
        except Exception as e:                                   # noqa: BLE001
            print(f"    warn: LLM client unavailable ({e}); deterministic "
                  f"adjudication")

    final, adj = None, None
    if cluster is not None:
        try:
            final, adj = adjudicate(run_jsons, llm_cluster=cluster,
                                    llm_judge=judge)
        except Exception as e:                                   # noqa: BLE001
            print(f"    warn: LLM adjudication failed ({e}); falling back to "
                  f"deterministic")
            final, adj = None, None
    if final is None:
        # deterministic: keyword clustering + majority-wins, fallbacks logged
        final, adj = adjudicate(run_jsons, llm_cluster=None, llm_judge=None)
    # Build the SME package ONCE (Option-A1 representative-run assembly with the
    # value overrides applied and structure re-derived). Save THAT as JSON and
    # render THAT as HTML, so adjudicated.json and adjudicated.html describe the
    # same object — with the full dag / crux / shapley, not the thin merge.
    pkg = build_sme_package(final, adj, run_jsons=run_jsons)
    # prefix outputs with the task id so files are self-identifying once moved out
    # of their per-task folder (e.g. {task}_adjudicated.json, not adjudicated.json).
    tid = manifest.get("task_id") or os.path.basename(run_dir_base.rstrip("/"))
    fjson = os.path.join(run_dir_base, f"{tid}_adjudicated.json")
    json.dump(pkg, open(fjson, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    if not no_html:
        open(os.path.join(run_dir_base, f"{tid}_adjudicated.html"),
             "w", encoding="utf-8").write(
                 render_html(final, adj, run_jsons=run_jsons, pkg=pkg))
        # trimmed 5-section summary companion
        try:
            from summary_html import build as _build_summary
            open(os.path.join(run_dir_base, f"{tid}_adjudicated_summary.html"),
                 "w", encoding="utf-8").write(_build_summary(pkg))
        except Exception as e:                                   # noqa: BLE001
            print(f"    warn: summary html not written: {e}")
    print(f"    => adjudicated {len(run_jsons)} runs -> {tid}_adjudicated.json"
          f"{'' if no_html else ' + .html'} | verdict "
          f"{final.get('audit_verdict')} | {len(adj.overrides)} override(s)")


def rows_from_json(paths):
    """Build (headers, rows) from deep-case package JSON(s), matching the CSV row-dict shape so the
    existing pipeline runs unchanged. Verifiers / solution logic are rendered to the text columns the
    auditor parses; a `json` column carries the absolute package path so audit_task reads full text
    straight from the JSON (no CSV cell limit)."""
    import json as _json, glob as _glob, os as _os
    headers = ["task_id","prompt_id","Primary Prompt Type","Secondary Prompt Type","Prompt",
               "Sanity Check","Solution Logic","Drive Link","Verifiers","Estimated Time",
               "Sub-Domain","Task Focus","json"]
    def _claims_text(cl):
        out=[]
        for c in cl:
            ins=", ".join(f"{i.get('name')}={i.get('value')}" for i in (c.get('inputs') or []))
            out.append(f"{c.get('id')} — {c.get('label','')}: {c.get('operation','')} = "
                       f"{c.get('claimed_result','')}" + (f" [inputs: {ins}]" if ins else "")
                       + (f" (source: {c.get('source','')})" if c.get('source') else ""))
        return "\n".join(out)
    def _vers_text(vs):
        out=[]
        for v in vs:
            crux=" [crux]" if v.get("crux") else ""
            out.append(f"{v.get('id')}{crux}: {v.get('check','')}")
        return "\n".join(out)
    rows=[]
    expanded=[]
    for p in paths:
        expanded += _glob.glob(p) or [p]
    for p in expanded:
        pkg=_json.load(open(p, encoding="utf-8"))
        dr=pkg.get("draft", pkg)
        prompt=dr.get("prompt","")
        rules=dr.get("governing_rules",[])
        if rules:
            prompt=prompt.rstrip()+"\n\nSourcing conventions:\n"+"\n".join(f"- {r}" for r in rules)
        rows.append({
            "task_id": dr.get("task_id",""),
            "prompt_id": "", "Primary Prompt Type": dr.get("prompt_type","FSP"),
            "Secondary Prompt Type": "LDP", "Prompt": prompt, "Sanity Check": "",
            "Solution Logic": _claims_text(dr.get("solution_logic",[])),
            "Drive Link": "", "Verifiers": _vers_text(dr.get("verifiers",[])),
            "Estimated Time": "", "Sub-Domain": dr.get("sub_domain",""), "Task Focus": "",
            "json": _os.path.abspath(p),
        })
    return headers, rows


def load_rows(path):
    """Banner-row tolerant; see src.auditor.read_task_csv."""
    return read_task_csv(path)


def main():
    ap = argparse.ArgumentParser(description="Direct prompt-package augmenter")
    ap.add_argument("--csv", default=None, help="task CSV (banner-tolerant). Use this OR --json.")
    ap.add_argument("--json", nargs="+", default=None,
                    help="one or more deep-case package JSON files (globs ok) to audit directly, "
                         "no CSV needed. The auditor reads full verifier/solution text from the JSON.")
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--task", action="append", default=None,
                    help="select by task_id instead of row number; repeatable. "
                         "A task appearing on several rows (one per model) is "
                         "augmented once — the verifiers and golden are per task.")
    ap.add_argument("--from", dest="row_from", type=int, default=None)
    ap.add_argument("--to", dest="row_to", type=int, default=None)
    ap.add_argument("--out-dir", default="output/augmented")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-files", action="store_true",
                    help="skip Drive input-file fetch (auditor provenance layer off)")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--runs", type=int, default=1,
                    help="run the auditor/augmenter N times per task (default 1). "
                         "With N>1, each run's JSON+HTML is written under "
                         "{out_dir}/{task_id}/run_{k}/ and a runs_manifest.json is "
                         "written per task listing the run JSONs, so the "
                         "adjudicator can consume them downstream.")
    ap.add_argument("--adjudicate", action="store_true",
                    help="after N runs, immediately adjudicate them into "
                         "{out_dir}/{task_id}/adjudicated.json + .html "
                         "(requires --runs >= 3).")
    args = ap.parse_args()

    if args.adjudicate and args.runs < 3:
        ap.error("--adjudicate needs --runs >= 3 (default 5 is typical)")

    if not args.csv and not args.json:
        ap.error("provide --csv or --json")
    if args.csv and args.json:
        ap.error("use --csv OR --json, not both")
    if args.json:
        headers, rows = rows_from_json(args.json)
        # guard: two packages with the SAME task_id would collide in {out_dir}/{task_id}/. Disambiguate
        # by suffixing a short hash of the source filename, and warn — never silently overwrite.
        import hashlib as _hl, os as _os
        seen_ids = {}
        for r in rows:
            tid = r["task_id"]
            if tid in seen_ids:
                suffix = _hl.sha1(r["json"].encode()).hexdigest()[:6]
                new_tid = f"{tid}__{suffix}"
                print(f"  ! duplicate task_id {tid!r} across JSONs — disambiguating "
                      f"{_os.path.basename(r['json'])} as {new_tid}", flush=True)
                r["task_id"] = new_tid
            seen_ids[r["task_id"]] = r["json"]
        print(f"loaded {len(rows)} task(s) from JSON", flush=True)
    else:
        headers, rows = load_rows(args.csv)
    try:
        hmap = build_header_map(headers)
    except HeaderError as e:
        print(f"ERROR: {e}"); sys.exit(2)

    n = len(rows)
    if args.task:
        wanted = set(args.task)
        # rows are 1-indexed to match --row; dedupe by task since the verifier set
        # and golden are per task, not per response
        seen, idxs = set(), []
        for i, r in enumerate(rows, 1):
            tid = get_field(r, hmap, "task_id")
            if tid in wanted and tid not in seen:
                seen.add(tid)
                idxs.append(i)
        missing = wanted - seen
        if missing:
            print(f"ERROR: task_id(s) not in the csv: {sorted(missing)}")
            print(f"       first few present: "
                  f"{[get_field(r, hmap, 'task_id') for r in rows[:5]]}")
            sys.exit(2)
    elif args.row is not None:
        idxs = [args.row]
    elif args.row_from or args.row_to:
        idxs = list(range((args.row_from or 1), (args.row_to or n) + 1))
    else:
        idxs = list(range(1, n + 1))
    idxs = [i for i in idxs if 1 <= i <= n]

    configure_api()
    os.makedirs(args.out_dir, exist_ok=True)
    model = args.model or DEFAULT_JUDGE_MODEL

    out_fields = list(headers) + [c for c in NEW_COLS if c not in headers]
    out_rows = []

    for i in idxs:
        row = dict(rows[i - 1])
        task_id = get_field(row, hmap, "task_id") or f"row{i}"
        print(f"[{i}] augmenting {task_id} ...", flush=True)

        # fetch input files (raw) so the auditor's provenance layer runs
        files_text, files_names, skipped = "", [], []
        corpus = None
        if not args.no_files:
            dl = get_field(row, hmap, "drive_link")
            if dl.strip():
                try:
                    # build_corpus_from_drive replaces the truncating fetcher:
                    # it keeps the COMPLETE text for code to search and a bounded
                    # index-plus-excerpts view for the prompt, and caches
                    # extraction so a 21.8 MB pdf is parsed once.
                    from src.input_corpus import build_corpus_from_drive
                    corpus = build_corpus_from_drive(
                        dl, get_field(row, hmap, "solution_logic"))
                    files_names = [f["name"] for f in corpus.files
                                   if f.get("extracted")]
                    skipped = list(corpus.skipped)
                    files_text = corpus.full_text
                    # Report this positively. A whole batch once audited with no
                    # input files because a /u/1/ account segment in the folder
                    # URL made every reference unparseable, and nothing in the
                    # output said so — the run looked normal.
                    if files_names:
                        print(f"    input files: {len(files_names)} fetched | "
                              f"{corpus.full_chars:,} chars searchable | "
                              f"prompt view {corpus.prompt_chars:,} chars "
                              f"({corpus.n_excerpts} excerpt(s)) | cache "
                              f"{corpus.cache_hits}hit/{corpus.cache_misses}miss")
                    else:
                        print(f"    !! NO INPUT FILES READ from {dl[:60]} — every "
                              f"figure in the golden will be unverifiable against "
                              f"source")
                    if skipped:
                        print(f"    !! SKIPPED INPUTS (golden may be degraded): {skipped}")
                except Exception as e:
                    print(f"    !! drive fetch failed: {e}")
            else:
                print("    (no drive link on this row)")

        try:
            res = augment_task(row, hmap, input_files_text=files_text,
                               input_corpus=corpus,
                               input_files_names=files_names, model_name=model,
                               skipped_inputs=skipped)
            rd = res.to_dict()
        except Exception as e:
            print(f"    AUGMENT FAILED: {e}")
            _blank_new_cols(row)
            row["skipped_inputs"] = json.dumps(skipped, ensure_ascii=False)
            row["scoreable"] = False
            row["not_scoreable_reason"] = f"augment crashed: {e}"
            row["augment_error"] = str(e)
            out_rows.append(row); continue

        # ---- N-RUN WRAPPER --------------------------------------------------
        # The auditor/augmenter are LLM calls and vary run to run. When --runs>1,
        # produce N independent runs and store them so the adjudicator can
        # reconcile them. The fetch above is deterministic, so it is done ONCE and
        # its inputs reused for every run; only the augment call repeats.
        # `rd` from the first call above is reused as run 1 (no wasted call).
        if args.runs > 1:
            run_dir_base = os.path.join(args.out_dir, task_id)
            os.makedirs(run_dir_base, exist_ok=True)
            manifest = {"task_id": task_id, "n_runs": args.runs, "runs": []}
            for k in range(1, args.runs + 1):
                if k == 1:
                    rd_k = rd            # reuse the run already computed
                else:
                    try:
                        res_k = augment_task(
                            row, hmap, input_files_text=files_text,
                            input_corpus=corpus, input_files_names=files_names,
                            model_name=model, skipped_inputs=skipped)
                        rd_k = res_k.to_dict()
                    except Exception as e:
                        print(f"    run {k}/{args.runs} FAILED: {e}")
                        manifest["runs"].append(
                            {"run": k, "error": str(e), "json": None})
                        continue
                rdir = os.path.join(run_dir_base, f"run_{k}")
                os.makedirs(rdir, exist_ok=True)
                jpath = os.path.join(rdir, f"{task_id}_run{k}_augment.json")
                try:
                    json.dump(rd_k, open(jpath, "w", encoding="utf-8"),
                              ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"    warn: run {k} json write failed: {e}")
                    jpath = None
                if not args.no_html and jpath:
                    try:
                        write_augment_report(
                            rd_k, os.path.join(rdir, f"{task_id}_run{k}_augment.html"))
                        write_golden_report(
                            rd_k, os.path.join(rdir, f"{task_id}_run{k}_golden.html"))
                    except Exception as e:
                        print(f"    warn: run {k} html write failed: {e}")
                manifest["runs"].append(
                    {"run": k, "json": jpath,
                     "verdict": rd_k.get("audit_verdict"),
                     "scoreable": rd_k.get("scoreable"),
                     "gate_ok": bool((rd_k.get("gate") or {}).get("passed"))})
                print(f"    run {k}/{args.runs}: {rd_k.get('audit_verdict')} "
                      f"| crux {len(rd_k.get('crux_ids', []))}")
            # per-task manifest the adjudicator consumes
            mpath = os.path.join(run_dir_base, "runs_manifest.json")
            json.dump(manifest, open(mpath, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print(f"    -> {args.runs} runs stored under {run_dir_base}/ "
                  f"(manifest: runs_manifest.json)")

            if args.adjudicate:
                _adjudicate_task(run_dir_base, manifest, model,
                                 no_llm=False, no_html=args.no_html)
            # fall through: the first run still fills the CSV row below, so the
            # augmented CSV remains a valid single-run artifact alongside the runs.

        # permanent safeguard: lossless per-task JSON so the CSV can always be
        # rebuilt (via rebuild_csv.py) without re-calling Opus.
        try:
            json.dump(rd, open(os.path.join(args.out_dir, f"{task_id}_augment.json"),
                               "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"    warn: could not write {task_id}_augment.json: {e}")

        # fill new columns
        row["corrected_solution_logic"] = rd["corrected_solution_logic"]
        row["golden_deliverable"] = rd["gold_deliverable_text"]
        row["augmented_verifiers"] = rd["augmented_verifiers_text"]
        row["dag_json"] = json.dumps(rd["dag"], ensure_ascii=False)
        row["crux_verifier_ids"] = json.dumps(rd["crux_ids"], ensure_ascii=False)
        row["crux_shapley_weights_json"] = json.dumps(rd["crux_shapley_weights"], ensure_ascii=False)
        row["base_weights_json"] = json.dumps(rd["base_weights"], ensure_ascii=False)
        row["expected_values_json"] = json.dumps(rd["expected_values"], ensure_ascii=False)
        row["audit_verdict"] = rd["audit_verdict"]
        row["changes_applied_json"] = json.dumps(rd["changes_applied"], ensure_ascii=False)
        row["judgment_pending_json"] = json.dumps(rd["judgment_changes_pending_sme"], ensure_ascii=False)
        row["skipped_inputs"] = json.dumps(rd.get("skipped_inputs", []), ensure_ascii=False)
        row["scoreable"] = rd.get("scoreable", True)
        row["not_scoreable_reason"] = rd.get("not_scoreable_reason", "")
        row["augment_error"] = rd.get("error", "")
        out_rows.append(row)

        if not args.no_html:
            write_augment_report(rd, os.path.join(args.out_dir, f"{task_id}_augment.html"))
            write_golden_report(rd, os.path.join(args.out_dir, f"{task_id}_golden.html"))

        cov = rd.get("input_coverage") or {}
        vt = (rd.get("augmented_verifiers") or "")
        print(f"    -> {rd['audit_verdict']} | crux {len(rd['crux_ids'])} verifiers"
              + (f"/{len(vt.splitlines())}" if vt else "")
              + (f" | files {len(cov.get('files_read') or [])}"
                 + ("" if cov.get("complete") else " INCOMPLETE") if cov else "")
              + (f" | {len(rd.get('judgment_pending_json') and json.loads(rd['judgment_pending_json']) or [])} question(s)"
                 if rd.get("judgment_pending_json") else "")
              + (f" | not scoreable: {rd.get('not_scoreable_reason','')[:44]}"
                 if not rd.get("scoreable", True) else "")
              + (f" | ERROR {rd['error']}" if rd.get("error") else ""))

    out_csv = os.path.join(args.out_dir, "augmented_prompt_packages.csv")
    # Diagnostic: confirm rows carry the augmented keys before writing.
    if out_rows:
        present = [c for c in NEW_COLS if c in out_rows[0]]
        print(f"[debug] out_rows={len(out_rows)} | new cols on row0: "
              f"{len(present)}/{len(NEW_COLS)} | header cols={len(out_fields)}")

    # MERGE by task_id rather than replacing the file. This used to be a plain
    # "w" of only the rows processed, so a single-task re-run silently destroyed
    # every other task's augmented columns — a whole batch reduced to one row.
    existing, kept = [], 0
    if os.path.exists(out_csv):
        try:
            with open(out_csv, encoding="utf-8-sig", newline="") as f:
                existing = list(csv.DictReader(f))
        except Exception as e:
            print(f"[merge] could not read the existing csv ({e}); it will be "
                  f"replaced. A copy is at {out_csv}.bak")
        if existing:
            import shutil
            shutil.copyfile(out_csv, out_csv + ".bak")

    def _tid(r):
        return str(r.get("task_id") or "").strip()

    # Preserve the ORIGINAL row order and substitute in place, so the file still
    # reads in sheet order and anything downstream that assumes it keeps working.
    updated = {_tid(r): r for r in out_rows if _tid(r)}
    merged, seen = [], set()
    for r in existing:
        t = _tid(r)
        if t and t in updated:
            merged.append(updated[t])
        else:
            merged.append(r)
            kept += 1
        if t:
            seen.add(t)
    for r in out_rows:                      # tasks not previously in the file
        if _tid(r) not in seen:
            merged.append(r)

    # union the columns so an older row missing a newly-added field still writes
    fields = list(out_fields)
    for r in existing:
        for k in r:
            if k and k not in fields:
                fields.append(k)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in merged:
            w.writerow(r)
    print(f"\nWrote {len(out_rows)} new/updated row(s), kept {kept} existing "
          f"-> {out_csv}"
          + (f"  (previous version saved as {os.path.basename(out_csv)}.bak)"
             if existing else ""))


if __name__ == "__main__":
    main()