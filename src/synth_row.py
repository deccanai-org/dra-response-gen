"""Render a deep-case package JSON into the auditor's task row — EXACTLY as the synthesizer's
deepcase/to_csv.py does (package_to_row, solution_logic_text, verifiers_text, _fmt), so
`run_augment.py --json` and `run_augment.py --csv` give the auditor identical input.

Previously --json used its own renderer, which dropped the trap values ("trap (shallow-agent wrong
value): X"), the anchors/tolerances, and the verifier dependencies ("(depends on: ...)"), and left
Sub-Domain / Task Focus empty — so the auditor's trap-validity checks ran blind and its step graph had
no dependencies. Keep this file in step with deepcase/to_csv.py; test_synth_row.py checks parity.
"""
import random
import string

HEADERS = ["task_id", "prompt_id", "Primary Prompt Type", "Secondary Prompt Type", "Prompt", "Sanity Check",
           "Solution Logic", "Drive Link", "json", "Verifiers", "Estimated Time", "Sub-Domain", "Task Focus"]


def _prompt_id():
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def _fmt(v):
    """Format a number like the existing CSV (commas, no trailing .0 for ints)."""
    try:
        f = float(v)
        if f == int(f):
            return f"{int(f):,}"
        return f"{f:,}"
    except (TypeError, ValueError):
        return str(v)


def solution_logic_text(claims):
    """Render structured claims as numbered step text (matches the P1/P2 style the auditor expects)."""
    lines = []
    for c in claims:
        cid = c.get("id", "")
        label = c.get("label", "")
        op = c.get("operation", "")
        res = c.get("claimed_result")
        inputs = c.get("inputs", [])
        # a derived claim (multi-input / an operation over names) vs a read
        if inputs and op and op not in [i.get("name") for i in inputs]:
            inpart = ", ".join(f"{i.get('name')}={_fmt(i.get('value'))}" for i in inputs)
            lines.append(f"{cid} — {label}: {op} = {_fmt(res)}  (inputs: {inpart})")
        else:
            src = c.get("source", "")
            lines.append(f"{cid} — {label} = {_fmt(res)}  (source: {src})")
        # anchored tolerance / trap annotations
        if c.get("target") is not None and c.get("tolerance") is not None:
            lines.append(f"     anchor: {_fmt(c['target'])} ± {_fmt(c['tolerance'])}; reconcile if outside")
        if c.get("trap_value") is not None:
            lines.append(f"     trap (shallow-agent wrong value): {_fmt(c['trap_value'])}")
    return "\n".join(lines)


def verifiers_text(verifiers):
    """Render verifiers as 'V_xxx [crux]: check  (depends on: ...)'."""
    lines = []
    for v in verifiers:
        crux = " [crux]" if v.get("crux") else ""
        dep = (f"  (depends on: {', '.join(x for x in v.get('depends_on', []) if x)})"
               if v.get("depends_on") and any(v["depends_on"]) else "")
        lines.append(f"{v.get('id', '')}{crux}: {v.get('check', '')}{dep}")
    return "\n".join(lines)


def package_to_row(pkg, drive_link="", json_path=""):
    d = pkg["draft"] if "draft" in pkg else pkg
    # the mandate EXACTLY as the synthesizer wrote it: governing rules are grading context, not agent
    # input (appending them as "Sourcing conventions" announced the planted conflicts)
    prompt = d.get("prompt", "")
    return {
        "task_id": d.get("task_id", ""),
        "prompt_id": _prompt_id(),
        "Primary Prompt Type": d.get("prompt_type", "FSP"),
        "Secondary Prompt Type": "LDP",   # deep-case tasks are latent-decomposition by nature
        "Prompt": prompt,
        "Sanity Check": "",               # authored downstream by the augmenter
        "Solution Logic": solution_logic_text(d.get("solution_logic", [])),
        "Drive Link": drive_link,
        "json": json_path,                # provenance only: the audit reads the rendered columns
        "Verifiers": verifiers_text(d.get("verifiers", [])),
        "Estimated Time": "",
        "Sub-Domain": d.get("sub_domain") or "Due Diligence",
        "Task Focus": d.get("schema_or_case_type", "").replace("_", " "),
    }