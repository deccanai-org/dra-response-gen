#!/usr/bin/env python3
"""
crux_judge.py — LLM-as-judge crux selection, GOLDEN-ANCHORED.

The crux = verifiers whose value the golden REPORTS AS AN ANSWER: the final
decision plus the key computed results the deliverable/solution-logic actually
states as conclusions. NOT raw inputs, NOT internal intermediates the memo never
surfaces, NOT filename/format checks, NOT restatements with no distinct value.

The judge is given, per task:
  - golden solution logic + golden deliverable text  (the reported answers)
  - the verifiers with their expected values + kinds
  - a STRUCTURAL HINT from the DAG: the final-deliverable node(s) (deepest) and
    their immediate parents — to orient attention on the answer region.
It returns the verifier IDs holding the reported answers.

Reads augment JSONs (which carry corrected_solution_logic, gold_deliverable_text,
expected_values, dag, depths). Works on a directory of {task}_augment.json.

Emits:
  crux_labels.json        {task_id: [crux ids]}
  crux_judge_detail.json  {task_id: {reported_answers:[...], reasoning:...}}

Usage:
  export ANTHROPIC_API_KEY=...      # and/or OPENROUTER_API_KEY (fallback / AUDITOR_PROVIDER=openrouter)
  python crux_judge.py --aug-dir ./output_2/augmented --out crux_labels.json
"""
import argparse, json, os, glob, re, sys

MODEL = "claude-opus-5"

SYSTEM = """You identify the CRUX verifiers of a hard analytical task.

CRUX = the verifiers whose value the GOLDEN reports as an ANSWER: the final \
decision/recommendation, plus the key computed results the deliverable actually \
STATES as conclusions (e.g. the per-option totals it compares, the headline figure, \
the final decision). 

NOT crux:
- Raw inputs / given assumptions. This is the most common error to avoid: any figure \
that is READ from an input file rather than COMPUTED is not crux, even if the memo \
restates it. This includes given rates (repo rate, carrying %, FX premium), given \
capex figures, given demand/volume, given per-unit prices, and given time periods. \
If the value's source_of_verification is "source_file" it is almost always a given \
input, NOT a reported answer — exclude it.
- Internal intermediate steps the memo does not surface as a reported result.
- Filename / file-format / section-present / formatting checks.
- Pure restatements that carry no distinct computed value.
- Tautologies (e.g. "probabilities sum to 1").

A reported ANSWER is something the analysis COMPUTES and the memo states as a \
conclusion (a total, a delta, a ratio, the final decision) — not a number lifted \
directly from an input file. When unsure, prefer values whose \
source_of_verification is "arithmetic" or "llm_judgment" (computed/judged) over \
"source_file" (given).

Method: read the golden's stated conclusions. For each verifier, ask "is this \
verifier's value one of the answers the golden REPORTS?" Use the DAG hint (the final \
deliverable node and its immediate parents) to orient toward the answer region, but \
the GOLDEN's reported conclusions are the authority, not the DAG shape. Typically \
3-6 crux per task.

Return STRICT JSON only:
{"crux":["V5","V6","V12"],"reasoning":"<=25 words on what the golden reports as answers"}"""

def load(path):
    j = json.load(open(path))
    return j

def deepest_and_parents(dag, depths):
    if not depths:
        # compute depths
        def dep(n, memo):
            if n in memo: return memo[n]
            ps = dag.get(n, [])
            memo[n] = 0 if not ps else 1 + max(dep(p, memo) for p in ps)
            return memo[n]
        memo = {}; depths = {n: dep(n, memo) for n in dag}
    md = max(depths.values())
    deepest = [v for v, d in depths.items() if d == md]
    parents = sorted({p for d in deepest for p in dag.get(d, [])}, key=lambda x: int(x[1:]))
    return deepest, parents

def verifier_block(ev):
    lines = []
    for v in sorted(ev, key=lambda x: int(x[1:])):
        e = ev[v] or {}
        lines.append(f"{v}: kind={e.get('kind')}, value={e.get('value')}, "
                     f"source={e.get('source_of_verification')}")
    return "\n".join(lines)

def call(client, tid, soln, deliverable, ev, deepest, parents):
    user = f"""TASK {tid}

GOLDEN SOLUTION LOGIC:
{(soln or '')[:2500]}

GOLDEN DELIVERABLE (what the memo states):
{(deliverable or '')[:2000]}

VERIFIERS (id: kind, value, source):
{verifier_block(ev)}

DAG HINT: final-deliverable node(s) = {deepest}; their immediate parents = {parents}.

Return the verifiers whose value the golden REPORTS AS AN ANSWER (final decision + key stated results)."""
    # Anthropic direct, else the same model on OpenRouter (src/llm_route.py)
    from src.llm_route import claude_complete
    txt = claude_complete(user, model=MODEL, system=SYSTEM, max_tokens=8000,
                          anthropic_extra={"temperature": 1, "thinking": {"type": "adaptive"},
                                           "output_config": {"effort": "medium"}},
                          effort="medium")
    txt = re.sub(r"^```(json)?|```$", "", txt.strip(), flags=re.M).strip()
    return json.loads(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-dir", required=True)
    ap.add_argument("--out", default="crux_labels.json")
    ap.add_argument("--detail", default="crux_judge_detail.json")
    ap.add_argument("--only", default="", help="comma-sep task ids to limit to")
    ap.add_argument("--runs", type=int, default=3, help="runs per task for stability (majority vote)")
    a = ap.parse_args()
    client = None          # provider routing lives in src/llm_route.py

    files = sorted(glob.glob(os.path.join(a.aug_dir, "*_augment.json")))
    only = set(t.strip() for t in a.only.split(",") if t.strip())
    labels, detail = {}, {}
    for f in files:
        j = load(f)
        tid = j.get("task_id") or os.path.basename(f).replace("_augment.json", "")
        if only and tid not in only:
            continue
        if not j.get("scoreable", True):
            print(f"{tid}: SKIP (not scoreable: {j.get('not_scoreable_reason','')})")
            detail[tid] = {"skipped": j.get("not_scoreable_reason", "not scoreable")}
            continue
        ev = j.get("expected_values", {})
        dag = j.get("dag", {})
        depths = j.get("depths", {})
        deepest, parents = deepest_and_parents(dag, depths)
        runs = []
        for i in range(a.runs):
            try:
                res = call(client, tid, j.get("corrected_solution_logic", ""),
                           j.get("gold_deliverable_text", ""), ev, deepest, parents)
                runs.append([v for v in res.get("crux", []) if v in ev])
            except Exception as e:
                print(f"{tid}: run {i+1} error {e}")
        if not runs:
            continue
        # stability: count how often each verifier appears across runs
        from collections import Counter
        cnt = Counter(v for r in runs for v in r)
        # majority set: appears in > half the runs
        majority = sorted([v for v, c in cnt.items() if c > a.runs / 2], key=lambda x: int(x[1:]))
        intersection = sorted(set.intersection(*[set(r) for r in runs]), key=lambda x: int(x[1:]))
        union = sorted(set.union(*[set(r) for r in runs]), key=lambda x: int(x[1:]))
        labels[tid] = majority
        stable = intersection == union
        detail[tid] = {"majority": majority, "intersection": intersection,
                       "union": union, "runs": runs, "stable": stable,
                       "unstable_verifiers": sorted(set(union) - set(intersection), key=lambda x: int(x[1:])),
                       "deepest": deepest, "parents": parents}
        flag = "" if stable else f"  UNSTABLE: {sorted(set(union)-set(intersection))}"
        print(f"{tid}: majority crux {majority}{flag}")

    json.dump(labels, open(a.out, "w"), indent=1)
    json.dump(detail, open(a.detail, "w"), indent=1)
    n = len(labels)
    avg = sum(len(v) for v in labels.values()) / max(n, 1)
    unstable = [t for t, d in detail.items() if d.get("stable") is False]
    print(f"\n{n} tasks judged ({a.runs} runs each), mean crux {avg:.1f} -> {a.out}")
    if unstable:
        print(f"UNSTABLE tasks (crux varied across runs — review these): {unstable}")
    else:
        print("all tasks stable across runs")

if __name__ == "__main__":
    main()