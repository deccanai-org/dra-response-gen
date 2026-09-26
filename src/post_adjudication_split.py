"""Post-adjudication re-split — Option 2.

The adjudicator merges N runs' verifiers by clustering on value, re-IDs to a flat V1..Vn space, and
prunes the per-run split log. That mostly preserves atomicity (split children survive as separate
flat verifiers), but the value-clustering merge is imperfect: it can leave a few COMPOUND verifiers
(a parent bundling result + components + trap) AND keep redundant parent+child PAIRS (the compound
net-debt parent alongside its own LT-debt and cash children).

Rather than teach the fragile merge to preserve splits, we RE-ATOMIZE the final consensus set: run
the concern-split audit once on the merged verifier text, apply the splits + dedupe, and return the
re-atomized text. The split audit is now reliable and cheap (Sonnet workers + Opus consolidator), so
this is a clean final pass that:
  * splits any straggler compound verifiers the merge left,
  * dedupes parent/child redundancy (the compound parent duplicates its children's components),
on the CONSENSUS set (better than merging per-run splits that may disagree).

Runs AFTER the merge sets augmented_verifiers_text and BEFORE derive_frozen_graph, so the DAG / crux
/ Shapley are computed over the re-atomized set.
"""
from __future__ import annotations
import re
from typing import Optional

_VID = r"V\d+[a-z]?"


def _parse_verifiers(text: str):
    """text -> [{id, text}] from 'Vn: ...' lines (tolerant to '[..]' tag between id and ':')."""
    out = []
    for line in (text or "").splitlines():
        m = re.match(r"\s*(" + _VID + r")\s*(?:\[[^\]]*\])?\s*:\s*(.*)", line)
        if m:
            out.append({"id": m.group(1), "text": m.group(2).strip()})
    return out


def _to_text(verifiers) -> str:
    return "\n".join(f"{v['id']}: {v['text']}" for v in verifiers)


def resplit_adjudicated(pkg: dict, provider: Optional[callable] = None) -> dict:
    """Re-atomize pkg['augmented_verifiers_text'] with the concern-split audit. Mutates and returns
    pkg. No-op (returns unchanged) on any failure — never worse than the merged set."""
    text = (pkg.get("augmented_verifiers_text") or "").strip()
    if not text:
        return pkg
    verifiers = _parse_verifiers(text)
    if not verifiers:
        return pkg

    # inputs the split audit needs, from the adjudicated package
    ev = pkg.get("expected_values") or {}
    step_graph = pkg.get("step_graph") or {}
    # step_graph is an adjacency map {id: [edges]}, NOT the {id: {node-dict}} shape the audit's
    # trap_passes_band expects (it calls node.get("trap_value")). Only pass it as step_nodes when its
    # values are dicts; otherwise pass {} — trap_passes_band is a diagnostic, safely empty, and the
    # split itself does not need it. This was the AttributeError ('list' has no .get) that failed the
    # re-split silently on adjudicated packages.
    step_nodes = step_graph if (isinstance(step_graph, dict)
                                and all(isinstance(v, dict) for v in step_graph.values())) else {}
    v2s = pkg.get("verifier_to_step") or {}
    sol = pkg.get("corrected_solution_logic") or ""
    if isinstance(sol, list):
        sol = "\n".join(str(x) for x in sol)

    try:
        from src.verifier_audit_split import audit_verifiers_split
        from src.verifier_audit import apply_splits, apply_rewrites
        va = audit_verifiers_split(
            task_id=str(pkg.get("task_id", "adjudicated")),
            verifiers=verifiers, expected_values=ev, step_nodes=step_nodes,
            solution_logic=sol, verifier_to_step=v2s, provider=provider)
    except Exception as e:                                        # noqa: BLE001
        _note = f"post-adjudication re-split: audit call failed ({type(e).__name__}); set unchanged"
        if isinstance(pkg.get("notes"), list):
            pkg["notes"].append(_note)
        else:
            pkg["post_adjudication_resplit"] = _note
        return pkg

    if getattr(va, "error", ""):
        _note = f"post-adjudication re-split: audit returned error ({va.error}); set unchanged"
        (pkg["notes"].append(_note) if isinstance(pkg.get("notes"), list)
         else pkg.update({"post_adjudication_resplit": _note}))
        return pkg

    # apply rewrites then splits, exactly as the auditor does
    n_before = len(verifiers)
    if va.rewrites:
        verifiers = apply_rewrites(verifiers, va.rewrites)
    dropped = []
    if va.duplicate_clusters:
        # drop the restated verifier(s) each cluster recommends removing (MECE dedupe)
        for c in va.duplicate_clusters:
            keep = None
            act = str(c.get("recommended_action", ""))
            m = re.search(r"keep\s+(" + _VID + r")", act)
            keep = m.group(1) if m else (c.get("verifier_ids") or [None])[0]
            for vid in (c.get("verifier_ids") or []):
                if vid != keep:
                    dropped.append(vid)
    if dropped:
        verifiers = [v for v in verifiers if v["id"] not in set(dropped)]
    if va.splits:
        verifiers, ev, split_log = apply_splits(verifiers, va.splits, ev)
        pkg["expected_values"] = ev
        pkg["verifier_splits_applied"] = split_log

    pkg["augmented_verifiers_text"] = _to_text(verifiers)
    note = (f"post-adjudication re-split RAN: {n_before} -> {len(verifiers)} verifiers "
            f"({len(va.splits or {})} split, {len(dropped)} deduped) on the consensus set"
            + ("" if (va.splits or dropped) else " — nothing to atomize (already atomic)"))
    if isinstance(pkg.get("notes"), list):
        pkg["notes"].append(note)
    else:
        pkg["post_adjudication_resplit"] = note
    return pkg