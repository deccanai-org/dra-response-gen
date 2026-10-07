#!/usr/bin/env python3
"""Adjudicate N run JSONs of one task into a final JSON + HTML.

    python run_adjudicate.py run1.json run2.json ... [--out final] [--no-llm]

By default it wires the two master-LLM entry points (cluster + judge) to the
project's Opus client. --no-llm runs fully deterministic (keyword clustering,
majority-wins with LLM-off fallbacks logged), which is what the tests use.

Default N is 5; any N >= 3 is accepted. Runs that failed the gate are kept for
structural votes but excluded from value votes.
"""
import argparse
import json
import os
import sys

from adjudicate_runs import adjudicate, render_html, build_sme_package


def _make_llm_callables():
    """Wire cluster + judge to the Opus client. Returns (cluster, judge) or
    (None, None) if the client is unavailable."""
    try:
        from src.prompt_evaluator import _call_llm, DEFAULT_JUDGE_MODEL
    except Exception:                                           # noqa: BLE001
        return None, None

    def _call(prompt):
        return _call_llm(prompt, DEFAULT_JUDGE_MODEL, max_tokens=4000,
                         system_prompt="You reconcile multiple runs of a task. "
                         "You only SELECT among options given; you never invent.")
    return _call, _call


def _resolve_run_paths(inputs):
    """Accept: explicit run JSONs, a runs_manifest.json, or a task dir containing
    one. Returns the list of run JSON paths."""
    import glob
    if len(inputs) == 1:
        one = inputs[0]
        if os.path.isdir(one):
            man = os.path.join(one, "runs_manifest.json")
            if os.path.exists(man):
                inputs = [man]
            else:  # a dir of run_*/ subfolders, no manifest
                found = sorted(glob.glob(os.path.join(one, "run_*",
                                                      "*_augment.json")))
                if found:
                    return found
        if inputs and inputs[0].endswith("runs_manifest.json"):
            man = json.load(open(inputs[0]))
            return [r["json"] for r in man.get("runs", []) if r.get("json")]
    return inputs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+",
                    help="run JSON files (>=3, default 5), OR a runs_manifest.json, "
                         "OR a task directory containing runs_manifest.json")
    ap.add_argument("--out", default="adjudicated", help="output basename")
    ap.add_argument("--no-llm", action="store_true",
                    help="deterministic only (no master LLM)")
    args = ap.parse_args(argv)

    run_paths = _resolve_run_paths(args.runs)
    run_jsons = [json.load(open(p)) for p in run_paths]
    cluster, judge = (None, None) if args.no_llm else _make_llm_callables()
    if not args.no_llm and cluster is None:
        print("warning: LLM client unavailable, falling back to deterministic",
              file=sys.stderr)

    final, adj = adjudicate(run_jsons, llm_cluster=cluster, llm_judge=judge)

    # save the assembled SME package (full structure) as JSON, and render the
    # same object as HTML, so the two files agree
    pkg = build_sme_package(final, adj, run_jsons=run_jsons, resplit=judge is not None)
    with open(f"{args.out}.json", "w") as f:
        json.dump(pkg, f, indent=1, default=str)
    with open(f"{args.out}.html", "w") as f:
        f.write(render_html(final, adj, run_jsons=run_jsons, pkg=pkg))
    # trimmed summary view (5 sections: trajectory, decisions, solution logic,
    # prompt, verifiers) — a quick-read companion to the full report.
    try:
        from summary_html import build as _build_summary
        with open(f"{args.out}_summary.html", "w", encoding="utf-8") as f:
            f.write(_build_summary(pkg))
    except Exception as e:                                        # noqa: BLE001
        print(f"  warn: summary html not written: {e}")

    print(f"adjudicated {len(run_jsons)} runs -> {args.out}.json / .html")
    print(f"  verdict: {final.get('audit_verdict')}")
    print(f"  overrides (LLM beat majority): {len(adj.overrides)}")
    print(f"  dropped (minority): {len(adj.dropped_minority)}")
    print(f"  residual decisions: {len(adj.residual_decisions)}")


if __name__ == "__main__":
    main()