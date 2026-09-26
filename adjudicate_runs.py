"""Reconcile N pipeline runs of the SAME task into one final JSON + HTML.

The audit/augment stages are LLM calls and vary run to run: 3 vs 5 claims, V4 vs
V4a/V4b, a trap value of 3 one run and 1 the next, a split that appears or not.
Scoring a single run inherits whichever way that run fell. This module takes
several runs and produces one reconciled golden by SELECT AND MERGE — it chooses
among the real alternatives the runs produced and never authors new content.

Pipeline (agreed design):
  1. CLUSTER (master LLM): group each run's claims/verifiers into conceptual
     ROLES. A keyword key fragments ~6 real roles into ~15 (labels vary run to
     run), so clustering is semantic, done once by the LLM. The LLM only assigns
     existing items to groups; it writes no verifier text.
  2. VOTE within each cluster (deterministic): the modal value / text wins, and
     the tally is recorded. Majority = >= ceil(N/2) agreeing.
  3. ADJUDICATE VALUES (master LLM, can OVERRIDE majority): majority counts, it
     does not judge correctness — if the model repeats a mistake in most runs,
     majority entrenches it (observed: 2 runs say wait=38, 1 says 42; 38 is
     wrong). So for each value the LLM is shown the derivation context and the
     vote, and may override the majority. Every override is logged with its
     reason, in BOTH the JSON (`adjudication.overrides`) and the HTML.
  4. MERGE + EMIT: assemble the winning items into one package, re-id verifiers
     V1..Vn, and render an HTML with the decision + override log visible.

The master LLM does two jobs (cluster, adjudicate) — both SELECT/JUDGE over real
run content, never author. Each is one non-deterministic call; to make the
adjudicator itself reproducible, run it K times and majority-vote its outputs
(a later option, not done here). Both LLM entry points accept an injected
callable, so the module is fully testable without a live model.
"""
from __future__ import annotations

#: Verifier-id pattern — classic (V1, V5a) and semantic (V_P1_pat_2004) ids.
_VID = r"V(?:\d+[a-z]?|_[A-Za-z0-9][A-Za-z0-9_.]*)"
_VID_LINE = (r"\s*(?P<vid>" + _VID + r")\s*(?:\[[^\]]*\])?"
             r"\s*(?::\s*|\s+-\s+)(?P<text>.*)")

import html as _html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


def _majority_needed(n: int) -> int:
    return (n // 2) + 1


def _num_key(v) -> Optional[float]:
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _first_number(text: str) -> Optional[float]:
    """The first standalone number in a verifier's text — the value it asserts.
    Used to tell a value disagreement ('wait = 38' vs '= 42') from a pure wording
    difference. Skips ids like 'V3' and pure ordinals."""
    for m in re.finditer(r"(?<![A-Za-z])(\d[\d,]*\.?\d*)", text or ""):
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            continue
    return None


def _vote(values: List, n_runs: int) -> Tuple[Optional[object], bool, dict]:
    """(winner, has_majority, tally). Numeric values compared by rounded key."""
    norm = []
    for v in values:
        nk = _num_key(v)
        norm.append(nk if nk is not None else str(v).strip())
    tally = Counter(norm)
    if not tally:
        return None, False, {}
    winner, count = tally.most_common(1)[0]
    return winner, count >= _majority_needed(n_runs), {str(k): c for k, c in tally.items()}


_ROLE_VOCAB = [
    "utilization", "offered load", "minimum server", "required", "fte", "clerk",
    "redesign", "residual", "wait", "recommend", "hire", "stability", "wage",
    "market", "target", "queue", "erlang", "arrival", "service", "capacity",
]


def _keyword_role(label: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", " ", (label or "").lower())
    keys = sorted({kw for kw in _ROLE_VOCAB if kw in t})
    if keys:
        return "|".join(keys)
    toks = [w for w in t.split() if len(w) > 3]
    return "|".join(toks[:3]) or "_empty_"


@dataclass
class Run:
    idx: int
    data: dict

    @property
    def gate_ok(self) -> bool:
        return bool((self.data.get("gate") or {}).get("passed"))

    @property
    def claims(self) -> List[dict]:
        return self.data.get("corrected_claim_verdicts") or []

    def verifiers(self) -> List[dict]:
        out = []
        for line in (self.data.get("augmented_verifiers_text") or "").splitlines():
            m = re.match(_VID_LINE, line)
            if m:
                out.append({"id": m.group("vid"),
                            "text": m.group("text").strip()})
        return out


@dataclass
class Adjudication:
    n_runs: int = 0
    gate_ok_runs: List[int] = field(default_factory=list)
    clustering_method: str = ""
    claim_clusters: List[dict] = field(default_factory=list)
    verifier_clusters: List[dict] = field(default_factory=list)
    majority_decisions: List[dict] = field(default_factory=list)
    overrides: List[dict] = field(default_factory=list)
    residual_decisions: List[dict] = field(default_factory=list)
    dropped_minority: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "n_runs", "gate_ok_runs", "clustering_method", "claim_clusters",
            "verifier_clusters", "majority_decisions", "overrides",
            "residual_decisions", "dropped_minority", "notes")}


def _loads(raw: str):
    if not raw:
        return None
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except (json.JSONDecodeError, AttributeError):
            return None


def _cluster_with_llm(items, kind, llm_cluster) -> Dict[int, str]:
    listing = "\n".join(f"{i}: {it['label']}" for i, it in enumerate(items))
    prompt = (
        f"These are {kind} drawn from several runs of the SAME task. Group them by "
        f"the CONCEPTUAL ROLE each one checks — the quantity or claim it is about, "
        f"NOT its wording.\n"
        f"CRITICAL: merge DIFFERENT NAMES for the SAME quantity into one group. "
        f"Runs phrase the same thing differently, e.g. 'utilization with 3 clerks' "
        f"and 'stability check with 3 clerks' are the SAME role (both are rho at "
        f"3 servers); 'offered load' and 'arrival load in Erlangs' are the same; "
        f"'required FTE' and 'minimum feasible servers' are the same. If two items "
        f"compute or assert the same underlying quantity, they go in ONE group "
        f"even if the words differ. Prefer FEWER, broader groups over many narrow "
        f"ones. Do NOT rewrite any item.\n"
        f"{listing}\n\n"
        'Return ONLY JSON: {"groups":[{"role":"short name","items":[indices]}]}. '
        "Every index appears in exactly one group.")
    obj = _loads(llm_cluster(prompt))
    mapping: Dict[int, str] = {}
    for g in (obj or {}).get("groups", []):
        role = str(g.get("role") or "").strip() or "unnamed"
        for i in g.get("items", []):
            if isinstance(i, int) and 0 <= i < len(items):
                mapping[i] = role
    return mapping


def _group(items, kind, llm_cluster, value_of=None) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    if llm_cluster and items:
        try:
            mapping = _cluster_with_llm(items, kind, llm_cluster)
        except Exception:                                       # noqa: BLE001
            mapping = {}
        if len(mapping) == len(items):
            for i, role in mapping.items():
                groups[role].append(i)
            groups = _value_postmerge(dict(groups), items, value_of)
            groups = _conflation_split(groups, items, value_of)
            return groups
    for i, it in enumerate(items):
        groups[_keyword_role(it["label"])].append(i)
    return _conflation_split(
        _value_postmerge(dict(groups), items, value_of), items, value_of)


def _value_postmerge(groups: Dict[str, List[int]], items,
                     value_of) -> Dict[str, List[int]]:
    """Deterministic safety net for under-merged clustering. Two clusters that
    carry the SAME computed value are the same quantity the clusterer split under
    different names ('utilization with 3 clerks' vs 'stability check with 3
    clerks', both rho=1.111). Merge clusters whose value-sets overlap on a shared
    value. Only runs when a value_of(item)->number extractor is supplied (claims,
    trap values); text-only items (verifiers) are left to the LLM + keyword key.
    """
    if value_of is None or len(groups) < 2:
        return groups
    # signature of each group = the set of rounded values it contains, plus the
    # set of runs it spans (to avoid over-merging two DIFFERENT quantities that
    # coincidentally share a value)
    sig: Dict[str, set] = {}
    runs_of: Dict[str, set] = {}
    for role, idxs in groups.items():
        vals = set()
        rns = set()
        for i in idxs:
            v = value_of(items[i])
            if v is not None:
                vals.add(round(float(v), 4))
            ref = items[i].get("ref")
            if isinstance(ref, tuple):
                rns.add(ref[0])
        sig[role] = vals
        runs_of[role] = rns
    roles = list(groups)
    parent = {r: r for r in roles}

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    for a in range(len(roles)):
        for b in range(a + 1, len(roles)):
            ra, rb = roles[a], roles[b]
            # merge only if they share a concrete value AND do NOT co-occur in the
            # same run — co-occurrence means they are distinct claims in that run
            # (e.g. two real quantities that happen to equal), not one role the
            # clusterer split across runs.
            if (sig[ra] and sig[rb] and (sig[ra] & sig[rb])
                    and not (runs_of[ra] & runs_of[rb])):
                parent[find(rb)] = find(ra)
    merged: Dict[str, List[int]] = defaultdict(list)
    # keep the name of the largest original cluster in each merged component
    comp_name: Dict[str, str] = {}
    for r in sorted(roles, key=lambda x: -len(groups[x])):
        comp_name.setdefault(find(r), r)
    for r in roles:
        merged[comp_name[find(r)]].extend(groups[r])
    return dict(merged)


def _conflation_split(groups: Dict[str, List[int]], items,
                      value_of) -> Dict[str, List[int]]:
    """Deterministic safety net for OVER-merged clustering — the complement of
    _value_postmerge. The LLM clusterer sometimes sweeps two genuinely different
    quantities into one role because their labels look alike (observed: "PPI Dec
    2025" = 157.039 and "Inflation factor" = 1.0272 merged into one role, so the
    value vote ran across both and the wrong value could seal onto the claim).

    A cluster is SPLIT into per-value-group roles only when the groups are
    distinct QUANTITIES, judged by two robust signals:
      * co-occurrence: two claims from the SAME run with different values are
        necessarily distinct quantities (a run does not compute one quantity
        twice with two values) — a decisive split signal; or
      * operation signature: the value-groups are produced by different
        `operation` strings (e.g. 'ppi_dec2025' vs 'ppi_dec2025/ppi_dec2024').

    It deliberately does NOT split when the groups look like the same quantity
    under a different convention — a pure SIGN FLIP (a == -b) with a matching
    operation — because that is a reconciliation the value vote/representative
    handles, not two quantities. Those remain one cluster (and the
    CLUSTER_CONFLATION diagnostic still flags them for SME review).
    """
    if value_of is None or not groups:
        return groups

    def num(it):
        v = value_of(it)
        return _num_key(v)

    def op_of(it):
        return (it.get("ref", (None, {}))[1] or {}).get("operation", "")

    def run_of(it):
        return it.get("ref", (None, None))[0]

    out: Dict[str, List[int]] = {}
    for role, idxs in groups.items():
        vals = [(i, num(items[i])) for i in idxs]
        numeric = [(i, v) for i, v in vals if v is not None]
        if len(numeric) < 2:
            out[role] = idxs
            continue
        # bucket indices into value-groups (within relative tolerance)
        vgroups: List[dict] = []
        for i, v in numeric:
            hit = None
            for g in vgroups:
                if abs(v - g["v"]) <= (abs(g["v"]) * 1e-3 + 1e-9):
                    hit = g
                    break
            if hit is None:
                vgroups.append({"v": v, "idxs": [i]})
            else:
                hit["idxs"].append(i)
        non_numeric = [i for i, v in vals if v is None]
        if len(vgroups) < 2:
            out[role] = idxs
            continue
        # decide split vs keep
        runs_seen = [set(run_of(items[i]) for i in g["idxs"]) for g in vgroups]
        co_occur = any(runs_seen[a] & runs_seen[b]
                       for a in range(len(vgroups))
                       for b in range(a + 1, len(vgroups)))
        ops = [set(op_of(items[i]) for i in g["idxs"]) for g in vgroups]
        ops_differ = any(not (ops[a] & ops[b])
                         for a in range(len(vgroups))
                         for b in range(a + 1, len(vgroups)))
        # pure sign-flip of one quantity (same |value|, one op) -> do NOT split
        all_ops = set().union(*ops) if ops else set()
        vs = [g["v"] for g in vgroups]
        sign_flip = (len(vgroups) == 2 and len(all_ops) <= 1
                     and abs(abs(vs[0]) - abs(vs[1])) <= (abs(vs[0]) * 1e-3 + 1e-9)
                     and (vs[0] * vs[1]) < 0)
        if sign_flip or not (co_occur or ops_differ):
            out[role] = idxs          # keep as one cluster (reconcile / SME-flag)
            continue
        # SPLIT: one role per value-group; attach non-numeric items to the first
        for k, g in enumerate(sorted(vgroups, key=lambda x: -len(x["idxs"]))):
            name = role if k == 0 else f"{role} ⟨{g['v']:g}⟩"
            members = list(g["idxs"])
            if k == 0:
                members += non_numeric
            out[name] = members
    return out


def adjudicate(
    run_jsons: List[dict],
    llm_cluster: Optional[Callable[[str], str]] = None,
    llm_judge: Optional[Callable[[str], str]] = None,
) -> Tuple[dict, Adjudication]:
    """Reconcile runs into one final package + an Adjudication log."""
    if len(run_jsons) < 3:
        raise ValueError("adjudication needs at least 3 runs (default 5)")

    runs = [Run(i, d) for i, d in enumerate(run_jsons)]
    n = len(runs)
    adj = Adjudication(n_runs=n, gate_ok_runs=[r.idx for r in runs if r.gate_ok])
    value_runs = [r for r in runs if r.gate_ok] or runs
    vr_idx = {r.idx for r in value_runs}
    if len(value_runs) < n:
        adj.notes.append(f"{n - len(value_runs)} run(s) failed the gate; kept for "
                         f"structural votes, excluded from value votes")
    adj.clustering_method = "llm" if llm_cluster else "keyword_fallback"

    final: dict = {"task_id": run_jsons[0].get("task_id", ""),
                   "adjudicated": True, "n_runs": n}

    # CLAIMS
    claim_items = [{"ref": (r.idx, c), "label": c.get("label", "")}
                   for r in runs for c in r.claims]
    final_claims = []
    for role, idxs in _group(claim_items, "claims", llm_cluster,
                             value_of=lambda it: it["ref"][1].get("recomputed")).items():
        occ = [claim_items[i]["ref"] for i in idxs]
        present = {ri for ri, _ in occ}
        adj.claim_clusters.append({"role": role, "runs_present": sorted(present)})
        if len(present) < _majority_needed(n):
            adj.dropped_minority.append(
                {"kind": "claim", "role": role, "runs_present": sorted(present)})
            continue
        vals = [c.get("recomputed") for ri, c in occ
                if ri in vr_idx and c.get("recomputed") is not None]
        # DIAGNOSTIC (TODO-3): detect a CONFLATED cluster — one role that has
        # swept together claims of two genuinely different quantities (observed:
        # "PPI Dec 2025" = 157.039 and "Inflation factor" = 1.0272 clustered into
        # one role, so the value vote runs across both and the wrong value wins,
        # corrupting the sealed claim). Signature: the cluster's numeric values
        # fall into 2+ groups that differ by more than a rounding tolerance.
        _num = [_num_key(v) for v in vals if _num_key(v) is not None]
        if len(_num) >= 2:
            _groups = []
            for v in _num:
                if not any(abs(v - g) <= (abs(g) * 1e-3 + 1e-9) for g in _groups):
                    _groups.append(v)
            if len(_groups) >= 2:
                _labels = sorted({c.get("label", "") for ri, c in occ
                                  if ri in vr_idx})
                adj.notes.append(
                    "CLUSTER_CONFLATION role=%r holds %d distinct value-groups "
                    "%r across labels %r — a value vote here mixes different "
                    "quantities; the sealed claim may take the wrong value" % (
                        role, len(_groups), _groups, _labels))
        winner, has_maj, tally = _vote(vals, len(value_runs)) if vals else (None, False, {})
        chosen = winner
        if vals:
            chosen = _judge_value(adj, llm_judge, role=role, tally=tally,
                                  majority=winner, has_majority=has_maj,
                                  occ=occ, kind="claim_value")
        final_claims.append(_rep(occ, vr_idx, chosen, adj=adj, role=role))
    final["corrected_claim_verdicts"] = final_claims

    # VERIFIERS
    ver_items = [{"ref": (r.idx, v), "label": v["text"]}
                 for r in runs for v in r.verifiers()]
    final_verifiers = []
    for role, idxs in _group(ver_items, "verifiers", llm_cluster).items():
        occ = [ver_items[i]["ref"] for i in idxs]
        present = {ri for ri, _ in occ}
        adj.verifier_clusters.append({"role": role, "runs_present": sorted(present)})
        if len(present) < _majority_needed(n):
            adj.dropped_minority.append(
                {"kind": "verifier", "role": role, "runs_present": sorted(present)})
            continue
        texts = [v["text"] for _, v in occ]
        winner, has_maj, tally = _vote(texts, n)
        # Does the disagreement carry a NUMERIC value that differs across
        # phrasings? If so it is a value question, not a wording one — the model
        # can embed a wrong number in most runs' phrasings (observed: "wait = 38"
        # in 2 runs, "= 42" in 1; 38 is the planted error). Route it through value
        # judgment, which can OVERRIDE the majority, rather than picking the modal
        # phrasing (which would entrench 38). A pure wording difference (same
        # numbers) stays on the phrasing path.
        nums = [_first_number(t) for t in dict.fromkeys(texts)]
        distinct_nums = {x for x in nums if x is not None}
        if len(distinct_nums) > 1:
            num_tally = _vote([_first_number(t) for t in texts], n)[2]
            num_winner, num_maj, _ = _vote([_first_number(t) for t in texts], n)
            picked_val = _judge_value(
                adj, llm_judge, role=f"{role} (embedded value)", tally=num_tally,
                majority=num_winner, has_majority=num_maj, occ=occ,
                kind="verifier_embedded_value")
            # keep the phrasing whose number matches the adjudicated value
            chosen = next((t for _, v in occ for t in [v["text"]]
                           if _num_key(_first_number(t)) == _num_key(picked_val)),
                          winner)
        elif has_maj:
            chosen = winner
            adj.majority_decisions.append(
                {"kind": "verifier_text", "role": role, "text": winner,
                 "tally": tally})
        else:
            chosen = _judge_text(adj, llm_judge, role=role, tally=tally,
                                 options=texts)
        final_verifiers.append({"role": role, "text": chosen})
    final["augmented_verifiers_text"] = "\n".join(
        f"V{i+1}: {v['text']}" for i, v in enumerate(final_verifiers))
    final["adjudicated_verifier_roles"] = [v["role"] for v in final_verifiers]

    # STRUCTURAL
    for kind, series in (
            ("any_split", [bool(r.data.get("verifier_splits_applied")) for r in runs]),
            ("verdict", [r.data.get("audit_verdict") for r in runs])):
        w, maj, tally = _vote(series, n)
        adj.majority_decisions.append(
            {"kind": kind, "value": w, "majority": maj, "tally": tally})
        if kind == "verdict":
            final["audit_verdict"] = w

    # TRAP VALUES
    trap_items = [{"ref": (r.idx, c), "label": c.get("label", "")}
                  for r in value_runs for c in r.claims
                  if c.get("trap_value") is not None]
    trap_final = {}
    for role, idxs in _group(trap_items, "trap values", llm_cluster,
                             value_of=lambda it: it["ref"][1].get("trap_value")).items():
        occ = [trap_items[i]["ref"] for i in idxs]
        vals = [c.get("trap_value") for _, c in occ]
        w, maj, tally = _vote(vals, len(value_runs))
        chosen = _judge_value(adj, llm_judge, role=role, tally=tally, majority=w,
                              has_majority=maj, occ=occ, kind="trap_value")
        trap_final[role] = {"value": chosen, "tally": tally}
    final["adjudicated_trap_values"] = trap_final

    # JUDGMENT STEPS: align by role, keep a representative from a value run ----
    j_items = [{"ref": (r.idx, j), "label": j.get("question", "")}
               for r in runs for j in (r.data.get("judgment_steps") or [])]
    final_judgments = []
    for role, idxs in _group(j_items, "judgment steps", llm_cluster).items():
        occ = [j_items[i]["ref"] for i in idxs]
        present = {ri for ri, _ in occ}
        if len(present) < _majority_needed(n):
            adj.dropped_minority.append(
                {"kind": "judgment", "role": role, "runs_present": sorted(present)})
            continue
        rep = next((dict(j) for ri, j in occ if ri in vr_idx), dict(occ[0][1]))
        final_judgments.append(rep)
    final["judgment_steps"] = final_judgments

    # Carry the fields a full re-derivation + SME report need. These are taken
    # from a representative gate-ok run (structure is reconciled; these are the
    # supporting artifacts the report renders). The re-derivation below rebuilds
    # dag / weights / crux / trajectory from the MERGED claims+verifiers, so the
    # adjudicated HTML has the same sections a single-run report does.
    rep_run = next((r for r in value_runs), runs[0]).data
    # expected_values must be re-derived from the MERGED, re-ided verifier text —
    # the runs' own expected_values key on their old ids (V4a etc.) and would be
    # dropped by the re-derivation's id filter. derive_expected_values reads the
    # canonical text and produces fresh V1..Vn targets that match the merged set.
    try:
        from src.verifier_grammar import derive_expected_values
        # Collect the task's known trap values from the runs' claims so the
        # widened compute-result target reader is enabled here too (guarded — it
        # never freezes a trap value). Without this the merge freezes only the
        # strict "= N" / "must be N" targets and the compute-phrased interior
        # verifiers ("Calculate X as N") get no target, so they cannot map to a
        # step and the overlaid crux/coverage collapses downstream.
        _trap_values = sorted({
            c.get("trap_value")
            for r in runs
            for c in (r.data.get("corrected_claim_verdicts") or [])
            if c.get("trap_value") is not None
        })
        ev, _ = derive_expected_values(final["augmented_verifiers_text"],
                                       trap_values=_trap_values)
        final["expected_values"] = ev
    except Exception:                                           # noqa: BLE001
        final["expected_values"] = {}
    final["trap_anchor_ids"] = rep_run.get("trap_anchor_ids", [])
    final["expert_anchor_ids"] = rep_run.get("expert_anchor_ids", [])
    final["gold_deliverable_format"] = rep_run.get("gold_deliverable_format", "")
    final["gold_deliverable_sections"] = rep_run.get("gold_deliverable_sections", [])
    final["gold_deliverable_text"] = rep_run.get("gold_deliverable_text", "")
    final["model_used"] = rep_run.get("model_used", "")
    final["input_coverage"] = rep_run.get("input_coverage", {})

    final["adjudication"] = adj.to_dict()
    return final, adj


def _rep(occ, vr_idx, chosen, adj=None, role=None):
    """Return the representative claim whose recomputed value matches the
    adjudicated `chosen`. DIAGNOSTIC: if no gate-ok claim in this cluster carries
    the chosen value, we fall through to the first claim in the cluster — which
    silently stamps the WRONG value onto the claim (observed: a "PPI Dec 2025"
    cluster adjudicated to 157.039 fell through and returned the inflation-factor
    claim, so the sealed PPI claim read 1.0272). Record every such fall-through so
    a re-run reveals exactly which role/value crossed."""
    key = _num_key(chosen)
    if key is not None:
        for ri, c in occ:
            if ri in vr_idx and _num_key(c.get("recomputed")) == key:
                return dict(c)
    # no value match — this is the bug path. Log what we're about to mis-assign.
    if adj is not None:
        picked = next((c for ri, c in occ if ri in vr_idx), None)
        if picked is None:
            picked = occ[0][1]
        adj.notes.append(
            "REP_MISMATCH role=%r chosen=%r no gate-ok claim in cluster carries "
            "that value; falling back to claim id=%r label=%r recomputed=%r "
            "(cluster values=%r)" % (
                role, chosen, picked.get("id"), picked.get("label"),
                picked.get("recomputed"),
                [c.get("recomputed") for ri, c in occ if ri in vr_idx]))
    for ri, c in occ:
        if ri in vr_idx:
            return dict(c)
    return dict(occ[0][1])


def _judge_value(adj, llm_judge, *, role, tally, majority, has_majority, occ, kind):
    if llm_judge is None:
        if not has_majority:
            adj.residual_decisions.append(
                {"kind": kind, "role": role, "tally": tally,
                 "resolved_by": "majority_fallback", "choice": str(majority)})
        return majority
    sample = occ[0][1] if occ else {}
    ctx = (f"role={role}; operation={sample.get('operation','')!r}; "
           f"inputs={[(i.get('name'), i.get('value')) for i in (sample.get('input_provenance') or [])]}")
    prompt = (
        "Independent runs of the same task disagree on a value. Majority vote is "
        "NOT proof of correctness — the model can repeat a mistake. Judge which "
        "option is CORRECT from the derivation, and pick it FROM THE LIST (do not "
        "invent a value).\n"
        f"{kind} for {ctx}\n"
        f"Options (value: #runs): {tally}\n"
        f"Majority-by-count = {majority}.\n"
        'Return ONLY JSON: {"choice": <one option value>, "reason": "...", '
        '"overrides_majority": true/false}.')
    obj = _loads(llm_judge(prompt)) or {}
    choice = obj.get("choice")
    if choice is None or str(choice) not in {str(k) for k in tally}:
        adj.residual_decisions.append(
            {"kind": kind, "role": role, "tally": tally,
             "resolved_by": "llm_off_menu_rejected", "llm_said": choice,
             "choice": str(majority)})
        return majority
    if str(choice) != str(majority):
        adj.overrides.append(
            {"kind": kind, "role": role, "tally": tally,
             "majority_was": str(majority), "chosen": str(choice),
             "reason": obj.get("reason", "")})
    else:
        adj.majority_decisions.append(
            {"kind": kind, "role": role, "value": str(choice), "tally": tally,
             "llm_confirmed": True})
    return _num_key(choice) if _num_key(choice) is not None else choice


def _judge_text(adj, llm_judge, *, role, tally, options):
    if llm_judge is None:
        w = Counter(options).most_common(1)[0][0]
        adj.residual_decisions.append(
            {"kind": "verifier_text", "role": role, "tally": tally,
             "resolved_by": "majority_fallback", "choice": w})
        return w
    prompt = (
        "Independent runs phrase the same verifier differently. Pick the single "
        "clearest, most precise phrasing FROM THE LIST — do not rewrite it.\n"
        f"role={role}\n" + "\n".join(f"- {t}" for t in dict.fromkeys(options)) +
        '\nReturn ONLY JSON: {"choice": "<exact option text>"}.')
    obj = _loads(llm_judge(prompt)) or {}
    choice = obj.get("choice")
    if choice in options:
        adj.residual_decisions.append(
            {"kind": "verifier_text", "role": role, "resolved_by": "llm",
             "choice": choice})
        return choice
    w = Counter(options).most_common(1)[0][0]
    adj.residual_decisions.append(
        {"kind": "verifier_text", "role": role,
         "resolved_by": "llm_off_menu_rejected", "choice": w})
    return w


def _pick_representative(run_jsons: List[dict]) -> Tuple[int, dict]:
    """Choose the run whose STRUCTURE (trajectory/DAG) is the skeleton for the
    adjudicated report. Ranked by: gate passed, scoreable, has a derived DAG,
    crux size, then claim+judgment count. The runs disagree structurally and no
    merged trajectory is coherent (their claim-id spaces differ), so one real
    run's trajectory is used as the skeleton and the adjudicated verifiers/values
    are overlaid on it (Option A)."""
    def score(d):
        return (
            1 if (d.get("gate") or {}).get("passed") else 0,
            1 if d.get("scoreable") else 0,
            1 if d.get("dag") else 0,
            len(d.get("crux_ids") or []),
            len(d.get("corrected_claim_verdicts") or [])
            + len(d.get("judgment_steps") or []),
        )
    best_i = max(range(len(run_jsons)), key=lambda i: score(run_jsons[i]))
    return best_i, run_jsons[best_i]


def build_sme_package(final: dict, adj: Adjudication,
                      run_jsons: Optional[List[dict]] = None) -> dict:
    """Option A: use a representative run's TRAJECTORY (claims, judgment steps,
    DAG skeleton) as a coherent whole, overlay the ADJUDICATED (merged) verifier
    set onto it, and apply the value overrides and majority verdict on top.

    Why overlay the merged set rather than keep the representative's own
    verifiers: a single run often authors only a few verifiers, so using its set
    alone yields a crux of 1-2 even when the trajectory is rich. The merged set is
    the reconciled union across runs. Overlaying it used to flatten the DAG —
    the merged verifiers had no single claim graph to attach to — but the
    containment-based verifier→step mapper now attaches them to the chosen
    trajectory by name+value, so the dependency graph survives the overlay. The
    representative is chosen for a rich, scoreable trajectory (see
    _pick_representative); the merged verifiers are mapped onto it and anchors are
    re-derived in the merged id space. The value overrides (the reconciliation
    that most affects correctness) are still applied, and the banner records the
    full adjudication so nothing is hidden.
    """
    if not run_jsons:
        pkg = dict(final)
    else:
        rep_i, rep = _pick_representative(run_jsons)
        pkg = dict(rep)                        # the representative run's TRAJECTORY
        adj.notes.append(f"structure from run index {rep_i} (representative); "
                         f"merged verifier set mapped onto it; value overrides + "
                         f"verdict applied")
        pkg["_structure_from_run"] = rep_i

        # OPTION A: overlay the MERGED verifier set onto the representative's
        # trajectory, instead of keeping only the representative run's own (often
        # thin) verifiers. The merged set is the reconciled union across runs; the
        # representative alone may carry very few verifiers, which yields a crux of
        # 1-2 even on a rich trajectory. The containment-based verifier→step mapper
        # (derive_dag.name_agreement) attaches the merged verifiers to this
        # trajectory's steps by name+value, so overlaying no longer flattens the
        # DAG the way it did under the old Jaccard mapper — measured on
        # tsk_1104027835: 4 mapped / 2 edges / 0.33 coverage (rep-only) ->
        # 12 mapped / 9 edges / 0.625 coverage (merged overlay). If the merge came
        # back empty (e.g. a degenerate no-LLM run), fall back to the
        # representative's own verifiers so we are never worse than before.
        merged_vtext = (final.get("augmented_verifiers_text") or "").strip()
        if merged_vtext:
            pkg["augmented_verifiers_text"] = merged_vtext
            # Carry the merge's frozen targets onto the overlaid package. The
            # merge already derived expected_values from the merged text (with the
            # task's trap values, so the compute-form targets are included);
            # letting the re-derivation below re-freeze from scratch would drop
            # them on a run whose own claims carry no trap_value, thinning the
            # target set and stranding most verifiers unmapped. derive_frozen_graph
            # keeps existing expected_values and fills only the gaps.
            if final.get("expected_values"):
                pkg["expected_values"] = dict(final["expected_values"])
            # The merged set lives in the merge's re-IDed V1..Vn space, NOT the
            # representative run's id space. The rep run's persisted anchors
            # (crux_anchors_*) therefore do NOT resolve against the merged ids, and
            # carrying them over would feed select_crux stale ids that filter to
            # nothing. Drop them and let derive_frozen_graph re-derive anchors in
            # the merged id space (expected-value + final-answer + mapped-interior),
            # which is what produces a correct crux over the overlaid set.
            for _k in ("trap_anchor_ids", "expert_anchor_ids",
                       "crux_anchors_trap", "crux_anchors_expert"):
                pkg.pop(_k, None)
            # verifier_splits_applied is a PER-RUN artifact copied from the
            # representative via dict(rep). Its children use that run's suffixed
            # ids (V4a, V4b). The merge re-ids every verifier to a fresh V1..Vn
            # space in which those suffixed ids do not exist — the split's
            # conjuncts survive the merge as SEPARATE clustered roles (e.g. V4a's
            # "PPI ref = 152.88" and V4b's "inflation factor = 1.027204" become two
            # ordinary merged verifiers), so the split has already been realized in
            # the merged set. Carrying the old log forward leaves it describing
            # children the merged set no longer names, and the report then renders
            # phantom "V4a split ... text not found / NO TARGET" cards for verifiers
            # that are in fact present and targeted under new ids. Reconcile the log
            # against the merged text: keep only entries all of whose children still
            # appear as ids in the merged set (after re-id, none do), so the stale
            # entries are dropped rather than shown as unscoreable orphans.
            _merged_ids = set(re.findall(r"(?m)^\s*(" + _VID + r")\s*"
                                         r"(?:\[[^\]]*\])?\s*:",
                                         merged_vtext))
            _splits = pkg.get("verifier_splits_applied") or []
            _kept = [s for s in _splits
                     if s.get("children")
                     and all(c in _merged_ids for c in s["children"])]
            if len(_kept) != len(_splits):
                pkg["verifier_splits_applied"] = _kept
                adj.notes.append(
                    f"dropped {len(_splits) - len(_kept)} stale split-log "
                    f"entry(ies) whose children were re-ided away by the merge; "
                    f"their conjuncts remain as separate merged verifiers")
        else:
            # No merged set to overlay — keep the representative's own verifiers.
            # As in the original path, the persisted run stores anchors under the
            # OUTPUT keys (crux_anchors_*) while derive_frozen_graph reads the INPUT
            # keys (*_anchor_ids); restore them so the rebuild reproduces the
            # representative run's own crux instead of collapsing to the
            # final-answer verifier(s).
            if not pkg.get("trap_anchor_ids") and pkg.get("crux_anchors_trap"):
                pkg["trap_anchor_ids"] = pkg["crux_anchors_trap"]
            if not pkg.get("expert_anchor_ids") and pkg.get("crux_anchors_expert"):
                pkg["expert_anchor_ids"] = pkg["crux_anchors_expert"]

        # majority verdict (structural) overlays the representative's
        pkg["audit_verdict"] = final.get("audit_verdict", pkg.get("audit_verdict"))
        pkg["task_id"] = final.get("task_id", pkg.get("task_id"))
        # keep the merged set + overrides visible in the package for the banner
        pkg["adjudicated_verifier_set"] = final.get("augmented_verifiers_text", "")
        pkg["adjudicated_trap_values"] = final.get("adjudicated_trap_values", {})
        # apply value overrides into BOTH the trajectory's claim values and,
        # where the value appears in the (now merged) verifier text, that text — so
        # the coherent trajectory shows the adjudicated (corrected) numbers
        _apply_overrides_to_claims(pkg, adj)
        _apply_overrides_to_verifier_text(pkg, adj)

    # POST-ADJUDICATION RE-SPLIT (Option 2): the value-clustering merge can leave a few compound
    # verifiers and redundant parent/child pairs. Re-atomize the CONSENSUS set with the concern-split
    # audit before deriving the frozen graph, so DAG/crux/Shapley are computed over the atomic set.
    try:
        from src.post_adjudication_split import resplit_adjudicated
        pkg = resplit_adjudicated(pkg)
    except Exception as _e:                                      # noqa: BLE001
        pkg.setdefault("notes", []).append(f"post-adjudication re-split skipped: {_e}") \
            if isinstance(pkg.get("notes"), list) else None

    try:
        from src.augment_task import derive_frozen_graph
        pkg = derive_frozen_graph(pkg, compute_shapley=True)
    except Exception as e:                                       # noqa: BLE001
        pkg["error"] = f"adjudicated re-derivation failed: {e}"
    pkg["_adjudication"] = adj.to_dict()
    pkg["adjudicated"] = True
    return pkg


def _apply_overrides_to_verifier_text(pkg: dict, adj: Adjudication):
    """Where a value override's old number appears in the representative run's
    verifier text, replace it with the adjudicated value — so the verifier the
    SME reads carries the corrected figure (e.g. 'wait = 38' -> 'wait = 42') while
    keeping its DAG position. Only replaces a standalone number to avoid touching
    ids or unrelated figures."""
    text = pkg.get("augmented_verifiers_text") or ""
    if not text:
        return
    for ov in adj.overrides:
        mv, cv = ov.get("majority_was"), ov.get("chosen")
        mvf, cvf = _num_key(mv), _num_key(cv)
        if mvf is None or cvf is None:
            continue
        # format the numbers as they'd appear (int if whole)
        old_s = str(int(mvf)) if mvf == int(mvf) else str(mvf)
        new_s = str(int(cvf)) if cvf == int(cvf) else str(cvf)
        text = re.sub(rf"(?<![.\d]){re.escape(old_s)}(?![.\d])", new_s, text)
    pkg["augmented_verifiers_text"] = text


def _apply_overrides_to_claims(pkg: dict, adj: Adjudication):
    """Write each value override into the representative run's matching claim, so
    the skeleton trajectory shows the adjudicated value (e.g. a claim that read 38
    becomes 42).

    The override is matched to its claim by ROLE (label agreement), then confirmed
    by the overridden 'majority_was' value — NOT by value alone. Matching on value
    alone silently corrupts the trajectory whenever two distinct claims share a
    number: an override for the 'Inflation factor' role (…-> 1.0272) would scan
    every claim for the majority value and overwrite the 'PPI Dec 2025' claim too,
    stamping the factor's value onto the PPI input (observed on tsk_4140790588:
    C0a PPI 157.039 -> 1.0272, C0c rate 4.1 -> 4.3615, C8 shortfall 33688 ->
    260000, each claim taking a DIFFERENT claim's value). Each override now edits
    at most ONE claim — the best role match whose current value equals
    majority_was — so a shared value can no longer cross-assign.
    """
    claims = pkg.get("corrected_claim_verdicts") or []
    from src.derive_dag import name_agreement, NAME_MIN
    for ov in adj.overrides:
        mv = _num_key(ov.get("majority_was"))
        cv = _num_key(ov.get("chosen"))
        if mv is None or cv is None:
            continue
        role = ov.get("role", "") or ""
        # candidates: claims that currently hold the majority value
        cands = [c for c in claims if _num_key(c.get("recomputed")) == mv]
        if not cands:
            continue                      # nothing to change; banner surfaces it
        if len(cands) == 1 and not role:
            target = cands[0]
        else:
            # pick the claim whose label best matches the override's role; require
            # a real match so a value-only collision never wins by default.
            scored = sorted(
                ((name_agreement(role, c.get("label", "")), c) for c in cands),
                key=lambda x: -x[0])
            best_score, target = scored[0]
            if best_score < NAME_MIN:
                # role does not clearly identify one of the value-matches — do not
                # guess; leave the trajectory unchanged and let the banner surface
                # the override for SME review rather than risk a wrong write.
                adj.notes.append(
                    "OVERRIDE_UNAPPLIED role=%r majority_was=%r chosen=%r: "
                    "%d claims share that value, none matches the role by name "
                    "(best %.2f) — left unwritten to avoid cross-assignment"
                    % (role, ov.get("majority_was"), ov.get("chosen"),
                       len(cands), best_score))
                continue
        target["recomputed"] = cv
        target["_adjudicated_from"] = ov.get("majority_was")
        target["_adjudicated_reason"] = ov.get("reason", "")


def _override_banner(adj: Adjudication) -> str:
    """An SME-facing banner listing where the adjudicator overrode the majority,
    plus the reconciliation summary. Prepended to the standard report."""
    parts = [
        "<div style='border:1px solid #e7e2d8;border-radius:8px;padding:12px 14px;"
        "margin:10px 0;background:#fff'>",
        f"<b>Adjudicated from {adj.n_runs} runs</b> "
        f"(gate-ok: {_html.escape(str(adj.gate_ok_runs))}; "
        f"clustering: {_html.escape(adj.clustering_method)}). "
        f"Trajectory/DAG shown is from the representative run; verifiers and "
        f"values below are the reconciled (adjudicated) set.",
    ]
    # Surface TODO-3 diagnostics (cluster conflation / rep mismatch) prominently,
    # so a corrupted claim value is visible in the report and not buried in JSON.
    _diag = [n for n in (adj.notes or [])
             if n.startswith("CLUSTER_CONFLATION") or n.startswith("REP_MISMATCH")]
    if _diag:
        parts.append("<p style='margin:8px 0 4px'><b>⚠ Adjudication "
                     "diagnostics (possible value corruption):</b></p>")
        for d in _diag:
            parts.append(
                f"<div style='background:#fff3cd;border-left:4px solid #d39e00;"
                f"padding:6px 10px;margin:5px 0;border-radius:6px'>"
                f"<span class=mono>{_html.escape(d)}</span></div>")
    if adj.overrides:
        parts.append("<p style='margin:8px 0 4px'><b>Value overrides "
                     "(adjudicator judged against the majority):</b></p>")
        for ov in adj.overrides:
            parts.append(
                f"<div style='background:#fdecea;border-left:4px solid #b4413c;"
                f"padding:6px 10px;margin:5px 0;border-radius:6px'>"
                f"<span class=mono>{_html.escape(str(ov['role']))}</span> — "
                f"majority-by-count was <b>{_html.escape(str(ov['majority_was']))}"
                f"</b>, chose <b>{_html.escape(str(ov['chosen']))}</b> · "
                f"tally {_html.escape(str(ov['tally']))}<br>"
                f"<i>{_html.escape(str(ov.get('reason','')))}</i></div>")
    else:
        parts.append(" No value overrides — majority and adjudicator agreed.")
    if adj.dropped_minority:
        drops = ", ".join(f"{d['kind']}:{d['role']}" for d in adj.dropped_minority)
        parts.append(f"<p style='margin:6px 0 0;color:#9a7400'><b>Dropped "
                     f"(minority, too few runs):</b> {_html.escape(drops)}</p>")
    parts.append("</div>")
    return "".join(parts)


def render_html(final: dict, adj: Adjudication, out_path: Optional[str] = None,
                run_jsons: Optional[List[dict]] = None,
                pkg: Optional[dict] = None) -> str:
    """Render the adjudicated golden as an SME report IDENTICAL in shape to a
    per-run augment report, with an override banner prepended. If out_path is
    given, writes there; always returns the HTML string.

    Pass `pkg` (the output of build_sme_package) to render the SAME object saved
    as JSON — keeps adjudicated.json and adjudicated.html in sync and avoids
    building the package twice. If None, builds from final + run_jsons (Option
    A1)."""
    import os
    import tempfile
    if pkg is None:
        pkg = build_sme_package(final, adj, run_jsons=run_jsons)
    try:
        from src.augment_report import write_augment_report
        tmp = out_path or os.path.join(tempfile.mkdtemp(), "adjudicated.html")
        write_augment_report(pkg, tmp)
        html = open(tmp, encoding="utf-8").read()
    except Exception as e:                                       # noqa: BLE001
        # fallback: minimal doc so we never lose the adjudication if the shared
        # renderer is unavailable
        html = (f"<!doctype html><meta charset=utf-8><h1>"
                f"{_html.escape(str(final.get('task_id')))} — adjudicated</h1>"
                f"<pre>{_html.escape(json.dumps(final, indent=1, default=str))}</pre>"
                f"<p>renderer unavailable: {_html.escape(str(e))}</p>")
        if out_path:
            open(out_path, "w", encoding="utf-8").write(html)
        return html
    # inject the override banner right after the <h1> line
    banner = _override_banner(adj)
    marker = "</h1>"
    idx = html.find(marker)
    if idx != -1:
        cut = idx + len(marker)
        html = html[:cut] + "\n" + banner + html[cut:]
    if out_path:
        open(out_path, "w", encoding="utf-8").write(html)
    return html