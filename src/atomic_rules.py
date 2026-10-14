"""ONE atomicity definition for every prompt that judges or splits verifiers.

Before (Oct 2026): PROPERTY_SPEC (verifier_audit.py) told the per-run audit "do NOT flag a fail-if half
plus a target half of the SAME quantity" while the split worker (verifier_audit_split.py) said a trap
MUST split — so the per-run audit exempted the most common bundle ("= 7.0%; reporting 12.0% fails"), and
neither prompt knew show-your-work clauses, value+outcome or final-line+binding bundles. Result: ~1 split
per run against 43-62% of verifiers rated non-atomic by SMEs. Both prompts now embed these constants.

No curly braces anywhere in this file: the texts are concatenated into str.format templates.
"""

ATOMIC_RULES = """\
DEFINITION. A verifier is ATOMIC when it checks exactly ONE thing, so a response either fully passes or
fully fails it. One verifier = one predicate on one referent (a single value, an outcome line, a label,
a statement, or a structural element; the referent may be a named set — see rule 5).

THE TEST: could a response satisfy one part of this verifier and fail another? If YES, it is NOT atomic:
split it.

RULES
1. One target per verifier.
2. Show-your-work is never part of a value verifier. "X = 7.0% ... with inputs, sources and formula
   shown" is TWO checks: the value, and showing the work. The show-work requirement belongs in ONE
   response-level verifier, not in every value verifier: the value child must not keep the clause.
3. A trap / fail-if / "must not use" clause is ALWAYS its own verifier, even for the SAME quantity:
   "reports the correct value" and "does not report the decoy" are separately attributable (a response
   can avoid the decoy and still get the value wrong).
4. A value and the outcome derived from it are separate: "coverage = 2.50x, so Criterion 3: FAIL" is two
   checks (the figure; the outcome line). Likewise the final recommendation line and the binding
   criteria it names are separate ("ends with FINAL RECOMMENDATION: DECLINE and names Criterion 3").
5. Lists and sets. Ask: IS A PARTIAL FAILURE WORTH ATTRIBUTING?
   - NO  -> atomic: the set itself is the unit ("the bridge contains exactly these 7 rows, in order";
            "exactly these three line items, nothing else").
   - YES -> split: members could each be right or wrong and any member carries diagnostic weight — it is
            a trap element, an input to a decision where members can disagree, or separately reported in
            the deliverable ("the table contains revenue, operating income and margin").
6. These do NOT break atomicity: a formula describing how the single value is defined ("operating margin
   = operating income / revenue = 17.0%"); a tolerance band; a period, unit or window qualifier; a number
   inside a label ("FY20XX", "Phase 2").
7. Atomic is not self-contained. "Reporting 15.2% fails" is atomic but does not name its metric: that is a
   self_contained FAIL, not an atomic FAIL.
"""

CHILD_RULES = """\
WHEN YOU SPLIT, EVERY CHILD MUST:
- be readable alone: restate the metric, its period and its unit (never "it", "this value", "the figure");
- for a trap child, read exactly: "The response does not report <metric, period> as <decoy value>." and
  declare NO target;
- give the parent's frozen target to the child that STATES the parent's value (inherits_target true) —
  never to a trap child;
- never add, widen or change a tolerance band (tolerance is the SME's decision);
- never keep a show-your-work clause ("with inputs, sources and formula shown" and variants).
"""

ATOMIC_EXAMPLES = """\
WORKED EXAMPLES (placeholder figures for an imaginary company - never copy them):
- "Fuel cost, FY20XX, as % of revenue = 7.0% at the stated precision, with inputs, sources and formula
   shown; reporting 12.0% fails."  -> NOT atomic (rules 2, 3). Children: "Fuel expense, FY20XX, as % of
   revenue = 7.0% at the stated precision." (inherits target) + "The response does not report fuel
   cost, FY20XX, as % of revenue as 12.0%." (no target).
- "Fuel cost, FY20XX, as % of revenue = 7.0% at the stated precision."  -> atomic.
- "Criterion 3 (coverage at or above 3.00x) is tested on the response's own figure; golden 2.50x -> FAIL;
   the line reads 'Criterion 3: FAIL'."  -> NOT atomic (rule 4): the coverage value + the outcome line.
- "The line 'Criterion 3: PASS/FAIL' states the outcome that follows from the response's own coverage
   figure."  -> atomic.
- "The memo ends with 'FINAL RECOMMENDATION: DECLINE' and names Criterion 3 as binding."  -> NOT atomic
   (rule 4).
- "Operating margin FY20XX = operating income / revenue = 17.0% (+/- 0.05pp)."  -> atomic (rule 6).
- "Table contains rows: Operating income, Revenue, Operating margin, Total equity, ROE."
   -> NOT atomic (rule 5: each row can be right or wrong and is reported separately).
- "The bridge contains exactly these 7 rows, in order: (1) ... (7) ..."  -> atomic (rule 5: the bridge
   is the unit).
- "Reporting 15.2% fails (trap value)."  -> atomic, but NOT self-contained (rule 7).
"""

ATOMICITY_BLOCK = ATOMIC_RULES + "\n" + CHILD_RULES + "\n" + ATOMIC_EXAMPLES