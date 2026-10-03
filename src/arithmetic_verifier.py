# src/arithmetic_verifier.py
"""
Hybrid arithmetic verification — Option C (structured claims) with Option A
(safe formula evaluation) as the inner layer.

The auditor LLM does NOT do arithmetic in its head. Instead it emits, for each
load-bearing figure, a structured claim:

    {
      "id": "C1",
      "label": "peak demand",
      "inputs": [
        {"name": "arrivals", "value": 170, "source": "input file, 9-11am window"},
        {"name": "hours",    "value": 2,   "source": "prompt"}
      ],
      "operation": "arrivals / hours",
      "claimed_result": 85,
      "trap_value": 67.5,
      "notes": "..."
    }

This module then does two independent checks per claim:

  (A) ARITHMETIC: re-evaluate `operation` against the declared input values using
      a safe expression evaluator (no eval/exec), and confirm it equals
      `claimed_result` within tolerance.

  (C) PROVENANCE: check each declared input `value` actually appears in the
      model-facing inputs at the stated source. (Optional / phased — when the
      input text is available; otherwise reported UNVERIFIABLE rather than passed.)

Outcomes per claim:
  CONFIRMED          arithmetic correct AND (provenance ok OR not checked)
  ARITHMETIC_ERROR   recomputed result != claimed_result
  INPUT_ERROR        a declared input value not found in source text
  WRONG_PERIOD       input value exists in source but under a different period than claimed
  MISLABELLED_INPUT  an input declared source_type='file' is absent from every
                     file — usually a computed subtotal written as a read, which
                     hides the computation that produced it
  UNVERIFIABLE       operation not safely evaluable, or source not parseable

No code from the LLM is executed. Only arithmetic expressions over numbers are
evaluated, via a restricted AST walker.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Safe arithmetic expression evaluator (the "A" layer)
# ---------------------------------------------------------------------------

# Allowed binary/unary operators — arithmetic only, no names/calls/attributes
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# A tiny whitelist of math functions the auditor may legitimately use
def _flat(args):
    """Accept either median(a, b, c) or median([a, b, c])."""
    out = []
    for a in args:
        out.extend(a) if isinstance(a, (list, tuple)) else out.append(a)
    return [float(x) for x in out]


def _median(*args):
    xs = sorted(_flat(args))
    if not xs:
        raise UnsafeExpression("median of nothing")
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _mean(*args):
    xs = _flat(args)
    if not xs:
        raise UnsafeExpression("mean of nothing")
    return sum(xs) / len(xs)


#: Whitelisted functions. MEDIAN AND MEAN WERE MISSING, which is not a small gap:
#: comparable-company valuation is built on the median multiple, and with no way
#: to express it the model emitted an EMPTY operation. That became UNVERIFIABLE,
#: the gate blocked, and the task returned zero verifiers — on a claim whose
#: arithmetic was in fact correct (median of six multiples = 20.71).
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "min": min, "max": max,
    "round": round, "sum": sum, "pow": pow, "log": math.log, "exp": math.exp,
    # ceil/floor were missing, so a round-UP — required-FTE, headcount, batch
    # sizing, any "round up to the next whole unit" — had no way to be expressed
    # and the model emitted an unevaluable operation. Same failure class as the
    # median/mean gap above: correct arithmetic reported UNVERIFIABLE, gate blocks.
    "ceil": lambda x: float(math.ceil(x)), "floor": lambda x: float(math.floor(x)),
    "median": _median, "mean": _mean, "average": _mean, "avg": _mean,
    "count": lambda *a: float(len(_flat(a))), "len": lambda *a: float(len(_flat(a))),
}


class UnsafeExpression(Exception):
    pass


def normalize_expr(expr: str) -> str:
    """Rewrite spreadsheet/finance notation into Python before parsing.

    "^" means EXPONENTIATION in Excel and in every finance write-up, and in Python
    it is bitwise XOR. A discount factor written the natural way —
    "1 / (1 + wacc/100) ^ years" — therefore failed with "operator not allowed:
    BitXor", the claim came back UNVERIFIABLE, the gate blocked, and the task
    returned zero verifiers. Bitwise operators have no meaning over the floats
    these expressions carry, so the rewrite cannot change a valid expression's
    value; it only makes an intended one readable.

    Also handles the unicode forms of the arithmetic operators, which appear when
    an expression is copied out of a document.
    """
    e = expr or ""
    for a, b in (("\u00d7", "*"), ("\u00f7", "/"), ("\u2212", "-"),
                 ("\u2013", "-"), ("\u2014", "-"), ("\u2044", "/")):
        e = e.replace(a, b)
    # ** first so an already-correct expression is untouched
    e = e.replace("**", "\x00POW\x00").replace("^", "**").replace("\x00POW\x00", "**")
    return e


def _sanitize_reserved_names(expr: str, variables: Dict[str, float]):
    """Rename any variable whose name is a Python keyword so ast.parse can read it.

    A claim written the natural way — operation "lambda / mu" with inputs named
    lambda and mu — parses as a broken lambda-expression and fails with "invalid
    syntax", so a correct division came back UNVERIFIABLE and the gate blocked.
    Greek-letter and reserved names are common in real tasks (lambda in queueing,
    is/in/for as ad-hoc names), so rewrite each reserved name to a safe alias in
    BOTH the expression and the variable dict. Word-boundary matched so a variable
    like "lambda_peak" is untouched. The value is unchanged, so a valid expression
    keeps its value; only an unparseable-but-intended one becomes readable.
    """
    import keyword
    if not variables:
        return expr, variables
    e, out = expr or "", {}
    for name, val in variables.items():
        if keyword.iskeyword(name):
            alias = f"_kw_{name}"
            e = re.sub(rf"\b{re.escape(name)}\b", alias, e)
            out[alias] = val
        else:
            out[name] = val
    return e, out


def safe_eval(expr: str, variables: Dict[str, float]) -> float:
    """Evaluate an arithmetic expression over the given variables. Raises
    UnsafeExpression for anything outside arithmetic + whitelisted functions."""
    expr = normalize_expr(expr)
    expr, variables = _sanitize_reserved_names(expr, variables)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpression(f"syntax error: {e}")

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise UnsafeExpression(f"non-numeric constant: {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise UnsafeExpression(f"unknown variable: {node.id}")
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise UnsafeExpression(f"operator not allowed: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise UnsafeExpression(f"unary op not allowed: {type(node.op).__name__}")
            return op(_eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise UnsafeExpression("only whitelisted math functions allowed")
            args = [_eval(a) for a in node.args]
            return _FUNCS[node.func.id](*args)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [_eval(e) for e in node.elts]
        raise UnsafeExpression(f"node type not allowed: {type(node).__name__}")

    return _eval(tree)


# ---------------------------------------------------------------------------
# Number normalization — handles Indian/Western grouping, %, currency, ranges
# ---------------------------------------------------------------------------

_CURRENCY = "₹$€£¥"
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_number(raw: Any) -> Optional[float]:
    """Coerce a value that may be a number, or a string like '₹3,90,00,000',
    '85', '7.5%', '2.23Cr' into a float. Returns None if not parseable.

    Handles:
      - Indian grouping (3,90,00,000) and Western (390,000,00) — commas stripped
      - currency symbols
      - trailing % (returned as the literal number, e.g. '7.5%' -> 7.5)
      - lakh/crore suffixes (Cr, Lakh, L) expanded
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return None

    # Strip currency symbols and spaces
    for c in _CURRENCY:
        s = s.replace(c, "")
    s = s.strip()

    # Crore / lakh multipliers
    mult = 1.0
    low = s.lower()
    cr = re.search(r"([\d,]*\.?\d+)\s*cr", low)
    lakh = re.search(r"([\d,]*\.?\d+)\s*(lakh|lac|l)\b", low)
    if cr:
        base = cr.group(1).replace(",", "")
        try:
            return float(base) * 1e7
        except ValueError:
            return None
    if lakh:
        base = lakh.group(1).replace(",", "")
        try:
            return float(base) * 1e5
        except ValueError:
            return None

    # Plain number (drop trailing %, keep the numeral)
    s = s.replace("%", "").strip()
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Claim data structures
# ---------------------------------------------------------------------------

@dataclass
class ClaimInput:
    name: str
    value: Any
    source: str = ""
    #: The claim whose output produced this input, when the producer states it.
    #: None for a value read from a source file. Set by the auditor, never here.
    from_claim: Optional[str] = None
    #: "file" | "claim" | "metadata". A METADATA value is arithmetic on something
    #: the task states rather than a data cell — "9:00-10:30 duration = 1.5
    #: hours". It will never appear verbatim in a file, so provenance-checking it
    #: produces a false INPUT_ERROR. Observed on real tasks: hours_first=1.5,
    #: hours_last=0.5.
    source_type: str = ""


@dataclass
class ArithmeticClaim:
    id: str
    label: str
    inputs: List[ClaimInput]
    operation: str
    claimed_result: Any
    trap_value: Any = None
    notes: str = ""
    #: How this claim's value is established. "arithmetic" (default) means the
    #: operation is a whitelisted expression this module recomputes. A value that
    #: is correct but produced by a method OUTSIDE arithmetic — a queueing formula
    #: (Erlang-C M/M/c wait), an integral, a simulation, a solver — is marked
    #: non-arithmetic so an empty/absent operation is not misread as a slip. Such a
    #: claim is recorded and trusted, not recomputed, and does not block the gate.
    source_of_verification: str = "arithmetic"

    @staticmethod
    def from_dict(d: dict) -> "ArithmeticClaim":
        return ArithmeticClaim(
            id=str(d.get("id", "")),
            label=str(d.get("label", "")),
            inputs=[ClaimInput(name=str(i.get("name", "")),
                               value=i.get("value"),
                               source=str(i.get("source", "")),
                               from_claim=(str(i["from_claim"])
                                           if i.get("from_claim") else None),
                               source_type=str(i.get("source_type", "") or ""))
                    for i in d.get("inputs", [])],
            operation=str(d.get("operation", "")),
            claimed_result=d.get("claimed_result"),
            trap_value=d.get("trap_value"),
            notes=str(d.get("notes", "")),
            source_of_verification=str(
                d.get("source_of_verification")
                or d.get("verification_method") or "arithmetic").lower(),
        )


@dataclass
class ClaimVerdict:
    id: str
    label: str
    status: str                      # CONFIRMED | ARITHMETIC_ERROR | INPUT_ERROR | UNVERIFIABLE
    recomputed: Optional[float] = None
    claimed: Optional[float] = None
    matches_trap: bool = False       # did the claimed value equal the trap value?
    #: The operation as declared. Previously dropped after verification, which
    #: left a derivation with values but no arithmetic.
    operation: str = ""
    #: What a solver who fell for the trap would get instead. Also previously
    #: dropped, keeping only the matches_trap boolean — so nothing downstream
    #: could NAME the wrong answer, which is exactly what a falsifiable verifier
    #: has to do ("must be 92.60, not 82").
    trap_value: Optional[float] = None
    detail: str = ""
    input_provenance: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _rel_close(a: float, b: float, rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


# Matches a source string that points at a prior claim's result rather than a
# raw input file: "C1 result", "from C2", "result of C3", "C4 output",
# "(C5)", "derived in C6", or an explicit "derived"/"computed"/"intermediate".
_DERIVED_SOURCE_RE = re.compile(
    r"\bC\d+[A-Za-z0-9_]*\b|\bderived\b|\bcomputed\b|\bintermediate\b"
    r"|\bresult of\b|\bfrom step\b",
    re.IGNORECASE,
)

#: Any token that could be a claim id, so prose can be tested against the ids
#: that ACTUALLY exist rather than against a guess at their shape. Real goldens
#: use C1, C8SUM, C10A; a regex over id shape is always one convention behind.
_ID_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _is_derived_source(source: str,
                       known_ids: Optional[Set[str]] = None,
                       self_id: str = "") -> bool:
    """True if this input is the output of a prior claim/step (so it should NOT
    be expected to appear verbatim in the source files).

    With `known_ids` this is a set-membership test against the ids that exist in
    this task, which works for any naming convention. The regex is the fallback.
    """
    if not source:
        return False
    if known_ids:
        for tok in _ID_TOKEN_RE.findall(source):
            if tok != self_id and tok in known_ids:
                return True
        return bool(re.search(r"\bderived\b|\bcomputed\b|\bintermediate\b"
                              r"|\bresult of\b|\bfrom step\b", source, re.I))
    return bool(_DERIVED_SOURCE_RE.search(source))


# An inline arithmetic expression embedded in a source string, e.g.
# "sum 10:00-11:00: 23+22+19+18" -> "23+22+19+18". Requires at least one
# operator between numbers so we don't mistake a single value or a time range.
_INLINE_EXPR_RE = re.compile(r"\d[\d,]*\.?\d*\s*[-+*/]\s*\d[\d,]*\.?\d*(?:\s*[-+*/]\s*\d[\d,]*\.?\d*)*")
# Time tokens like "10:00-11:00" look like subtraction; strip them before scanning.
_TIME_RANGE_RE = re.compile(r"\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}")


def _extract_inline_expression(source: str) -> Optional[str]:
    """If the source string contains an inline arithmetic expression (a subtotal
    computed from raw numbers), return it; else None.

    Conservative: a bare two-number subtraction ("9-11", "9-11am") is almost
    always a time/window range, not a subtotal, so we require EITHER a +,*,/
    operator OR three-or-more operands before treating it as an expression."""
    if not source:
        return None
    cleaned = _TIME_RANGE_RE.sub(" ", source)            # drop HH:MM-HH:MM ranges
    cleaned = re.sub(r"\d{1,2}\s*-\s*\d{1,2}\s*(am|pm)", " ", cleaned, flags=re.I)  # 9-11am
    m = _INLINE_EXPR_RE.search(cleaned)
    if not m:
        return None
    expr = m.group(0).replace(",", "")
    operands = re.findall(r"\d+\.?\d*", expr)
    has_addmuldiv = bool(re.search(r"\d\s*[+*/]\s*\d", expr))
    # Accept as an expression only if it has a +/*// operator, or it's a
    # subtraction with 3+ operands (a real chain, not a 2-number range).
    if has_addmuldiv or (len(operands) >= 3):
        return expr
    return None


def _verify_inline_expression(expr: str, claimed_value: float, source_text: str,
                              rel_tol: float) -> tuple:
    """Verify an inline subtotal: its component numbers should appear in source
    AND it should evaluate to the claimed value. Returns (ok_bool_or_None, detail)."""
    # Pull the operand numbers out of the expression.
    operands = [float(x) for x in re.findall(r"\d+\.?\d*", expr)]
    missing = [o for o in operands if not _number_appears(o, source_text)]
    try:
        evaluated = float(safe_eval(expr, {}))
    except (UnsafeExpression, ZeroDivisionError, ValueError, OverflowError):
        return None, f"could not evaluate inline expression '{expr}'"
    eval_ok = _rel_close(evaluated, claimed_value, rel_tol)
    if missing:
        return False, (f"inline subtotal components {[ _fmt(o) for o in missing ]} "
                       f"not found in source")
    if not eval_ok:
        return False, (f"inline expression '{expr}' evaluates to {evaluated:g}, "
                       f"not the stated {claimed_value:g}")
    return True, f"subtotal '{expr}'={evaluated:g} verified from source components"


def _fmt(x: float) -> str:
    return str(int(x)) if x == int(x) else f"{x:g}"



_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_QMON = re.compile(r"\bQ[1-4]\b|\b[1-4]Q\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", re.I)
def _periods_of(text):
    text = text or ""
    out = set(_YEAR.findall(text))
    out.update(m.lower() for m in _QMON.findall(text) if m)
    return out
def _adjacent_period(num, source_text):
    """Period token(s) bound to num: a parenthetical right after it, else nearest within a tight window."""
    for c in ({str(int(num)), f"{int(num):,}"} if num==int(num) else {f"{num:.1f}", f"{num:.2f}", f"{num:g}"}):
        for m in re.finditer(r"(?<![\d.,])"+re.escape(c)+r"(?!\d)(?![.,]\d)", source_text):
            paren = re.match(r"\s*\(([^)]{0,40})\)", source_text[m.end():m.end()+42])
            if paren:
                ps=_periods_of(paren.group(1))
                if ps: return ps
            ps=_periods_of(source_text[max(0,m.start()-45):m.end()+45])
            if ps: return ps
    return set()

def verify_claim(
    claim: ArithmeticClaim,
    source_text: Optional[str] = None,
    rel_tol: float = 1e-3,
    known_claim_ids: Optional[Set[str]] = None,
) -> ClaimVerdict:
    """Run the A (arithmetic) and C (provenance) checks on one claim."""
    # --- C-layer: provenance (only if source text supplied) ---------------
    # A "derived" input is the result of a PRIOR claim (its source references
    # another claim id, e.g. "C1 result" / "from C2" / "result of C3"), or is
    # explicitly marked derived. Derived inputs are NOT expected to appear
    # verbatim in the source files, so we must NOT source-check them — doing so
    # produces false INPUT_ERRORs on legitimate intermediate values.
    provenance: List[dict] = []
    input_error = False
    mislabelled: List[str] = []          # file-typed inputs absent from source
    variables: Dict[str, float] = {}
    for inp in claim.inputs:
        num = parse_number(inp.value)
        if num is not None:
            variables[inp.name] = num
        # An explicit from_claim is authoritative; the regex is only the fallback.
        is_derived = bool(inp.from_claim) or _is_derived_source(
            inp.source, known_claim_ids, claim.id)
        # A METADATA value is arithmetic on something the task states, not a data
        # cell: "9:00-10:30 duration = 1.5 hours". It cannot appear verbatim in a
        # file, so provenance-checking it manufactures a false INPUT_ERROR.
        is_metadata = (inp.source_type or "").lower() == "metadata"
        entry = {"name": inp.name, "value": inp.value, "source": inp.source,
                 "derived": is_derived, "from_claim": inp.from_claim,
                 "source_type": inp.source_type,
                 "source_kind": "raw", "found_in_source": None}

        if is_metadata:
            entry["source_kind"] = "metadata"      # stated, not stored; skip
        elif is_derived:
            entry["source_kind"] = "cross_claim"   # result of a prior claim; skip
        elif source_text is not None and num is not None:
            inline_expr = _extract_inline_expression(inp.source)
            if inline_expr is not None:
                # The source itself is an arithmetic expression (e.g. an inline
                # subtotal "23+22+19+18"). The TOTAL won't appear verbatim, so
                # verify instead that (a) its components are in source and (b) it
                # evaluates to the stated value. This catches a mis-summed subtotal
                # while not false-flagging a correct one.
                entry["source_kind"] = "inline_expr"
                ok, detail = _verify_inline_expression(inline_expr, num, source_text, rel_tol)
                entry["found_in_source"] = ok
                entry["inline_detail"] = detail
                if ok is False:
                    input_error = True
            else:
                # Genuine raw input — must appear verbatim in source.
                entry["found_in_source"] = _number_appears(num, source_text)
                if entry["found_in_source"] is False:
                    input_error = True
                elif entry["found_in_source"] is True:
                    # PERIOD CHECK: value is present — confirm it is under the period the claim
                    # attributes it to. Catches a real figure lifted from the wrong period column
                    # (e.g. a Dec-2006 comparative used as the Sep-2007 current value).
                    claim_periods = _periods_of((inp.source or "") + " " + (claim.label or ""))
                    if claim_periods:
                        adj = _adjacent_period(num, source_text)
                        if adj and not (claim_periods & adj):
                            entry["period_mismatch"] = True
                            entry["period_detail"] = (f"value under {sorted(adj)} but claim attributes "
                                                      f"it to {sorted(claim_periods)}")
                    # A value the producer DECLARED as coming from a file, which
                    # is not in the file, is a different animal from a value we
                    # simply could not place. It is almost always a computed
                    # subtotal mislabelled as a read — "arrivals_9_11 = 170",
                    # which is the sum of eight buckets. That collapse hides an
                    # unverified computation: neither layer ever checks the eight
                    # buckets or the sum, so a misread bucket would pass.
                    if (inp.source_type or "").lower() == "file":
                        mislabelled.append(inp.name)
        provenance.append(entry)

    claimed = parse_number(claim.claimed_result)
    trap = parse_number(claim.trap_value)

    # --- A-layer: arithmetic re-evaluation --------------------------------
    # A claim whose value is produced OUTSIDE arithmetic (a queueing formula, an
    # integral, a solver) declares itself non-arithmetic. Its operation is
    # legitimately absent, so do not misread that as a slip or block the gate on
    # it: record the claimed value, mark it NOT_ARITHMETIC, and move on. It is
    # verified by other means (source file, judgment, or an accepted formula), not
    # by this module.
    _nonarith = {"non_arithmetic", "nonarithmetic", "not_arithmetic", "formula",
                 "source_file", "llm_judgment", "judgment", "external"}
    if (claim.source_of_verification or "").lower() in _nonarith:
        return ClaimVerdict(
            id=claim.id, label=claim.label, status="NOT_ARITHMETIC",
            claimed=claimed, recomputed=claimed,
            detail=(f"value established by {claim.source_of_verification}, not "
                    f"arithmetic; recorded and trusted, not recomputed"),
            input_provenance=provenance,
        )
    if not claim.operation.strip():
        return ClaimVerdict(
            id=claim.id, label=claim.label, status="UNVERIFIABLE",
            claimed=claimed, detail="no operation supplied",
            input_provenance=provenance,
        )
    # LEAF / IDENTITY claim: the operation is a single bare variable (e.g. "ar_rev_K") — a value
    # RETRIEVED from a document, not computed. safe_eval would raise "unknown variable" on it (the
    # re-split path hit this and mislabelled such leaves UNVERIFIABLE, though the raw pass confirmed
    # them). A leaf is correct iff its value appears in source (provenance) and equals its single
    # input — verify it that way instead of arithmetic-evaluating a passthrough.
    _op = claim.operation.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", _op):
        # it's a lone identifier. If it names the single declared input (or the input value equals the
        # claimed result), this is a retrieved leaf.
        _single = len(claim.inputs) == 1
        _val_matches = False
        if _single:
            try:
                _val_matches = abs(float(claim.inputs[0].value) - float(claimed)) < 1e-6
            except (TypeError, ValueError):
                _val_matches = False
        if _single and (_op == claim.inputs[0].name or _val_matches):
            # verified by provenance: the leaf value must appear in source
            found = provenance[0]["found_in_source"] if provenance else None
            if found is False:
                return ClaimVerdict(
                    id=claim.id, label=claim.label, status="INPUT_ERROR",
                    claimed=claimed, recomputed=claimed,
                    detail=f"retrieved value {claimed} not found in source (leaf claim)",
                    input_provenance=provenance)
            return ClaimVerdict(
                id=claim.id, label=claim.label, status="CONFIRMED",
                claimed=claimed, recomputed=claimed,
                detail="retrieved leaf value (identity passthrough); confirmed by provenance",
                input_provenance=provenance)
    try:
        recomputed = float(safe_eval(claim.operation, variables))
    except UnsafeExpression as e:
        return ClaimVerdict(
            id=claim.id, label=claim.label, status="UNVERIFIABLE",
            claimed=claimed, detail=f"operation not safely evaluable: {e}",
            input_provenance=provenance,
        )
    except (ZeroDivisionError, ValueError, OverflowError) as e:
        return ClaimVerdict(
            id=claim.id, label=claim.label, status="UNVERIFIABLE",
            claimed=claimed, detail=f"evaluation failed: {e}",
            input_provenance=provenance,
        )

    matches_trap = trap is not None and _rel_close(recomputed, trap, rel_tol)

    # Decide status. Arithmetic correctness is primary; provenance gates it.
    if claimed is None:
        status = "UNVERIFIABLE"
        detail = "claimed_result not numeric"
    elif not _rel_close(recomputed, claimed, rel_tol):
        status = "ARITHMETIC_ERROR"
        detail = (f"recomputed {recomputed:g} from operation '{claim.operation}' "
                  f"!= claimed {claimed:g}")
    elif mislabelled:
        status = "MISLABELLED_INPUT"
        detail = (
            f"input(s) {mislabelled} are declared source_type='file' but do not "
            f"appear in any supplied file. Almost always a computed subtotal "
            f"written as a read: the arithmetic here evaluates fine, but the "
            f"computation that PRODUCED the value is checked by nobody — neither "
            f"its own inputs nor its sum. Emit that computation as its own claim "
            f"and reference it via from_claim, or retype the input as "
            f"'metadata' if it is genuinely derivable from a stated fact.")
    elif input_error:
        status = "INPUT_ERROR"
        bad = [p["name"] for p in provenance if p["found_in_source"] is False]
        detail = (f"arithmetic checks out, but declared input(s) {bad} not found "
                  f"in source — possible fabricated/wrong input used")
    elif any(p.get("period_mismatch") for p in provenance):
        status = "WRONG_PERIOD"
        pm = [p["name"] for p in provenance if p.get("period_mismatch")]
        det = "; ".join(p.get("period_detail","") for p in provenance if p.get("period_mismatch"))
        detail = (f"arithmetic checks out and inputs exist, but input(s) {pm} appear under a "
                  f"different period than the claim attributes them to — {det}")
    else:
        status = "CONFIRMED"
        detail = f"recomputed {recomputed:g} matches claimed {claimed:g}"

    return ClaimVerdict(
        id=claim.id, label=claim.label, status=status,
        recomputed=recomputed, claimed=claimed,
        matches_trap=matches_trap, detail=detail,
        operation=claim.operation, trap_value=trap,
        input_provenance=provenance,
    )


def _number_appears(num: float, source_text: str) -> bool:
    """Heuristic: does `num` appear in source_text, tolerant of comma grouping,
    decimals, and currency? Conservative — returns True on a reasonable match."""
    # Build a few string renderings of the number to look for.
    candidates = set()
    if num == int(num):
        n = int(num)
        candidates.add(str(n))
        candidates.add(f"{n:,}")
        candidates.add(_indian_group(n))
        # integer-valued but may be written with a trailing decimal in the source (184 -> 184.0)
        candidates.add(f"{n}.0")
        candidates.add(f"{n:,}.0")
    else:
        candidates.add(f"{num:g}")
        candidates.add(f"{num:.1f}")
        candidates.add(f"{num:.2f}")
    # Boundary-guarded match: the rendering must not sit inside a longer number (so 22474 does not
    # match as a fragment of 122474 or an accession/CIK/date). This replaces the old
    # comma-stripped substring fallback, which matched coincidental digit runs anywhere in a
    # multi-MB filing and let fabricated figures pass provenance.
    for c in candidates:
        # allow an optional trailing .0/.00 in the source (184 matches 184.0); still guard the left
        # side and the right side beyond the decimals so 184 does not match 1840 or 184.57.
        pat = r"(?<![\d.,])" + re.escape(c) + (r"(?:\.0+)?(?!\d)(?![.,]\d)" if "." not in c else r"(?!\d)(?![.,]\d)")
        if re.search(pat, source_text):
            return True
    # thousands-scale tolerance ($M claim vs $K source): also try x1000 / ÷1000 renderings, guarded.
    for scale in (1000.0, 0.001):
        sv = num * scale
        if sv == int(sv):
            for c in {str(int(sv)), f"{int(sv):,}"}:
                if len(re.sub(r"[^\d]","",c))>=2 and re.search(r"(?<![\d.,])"+re.escape(c)+r"(?!\d)(?![.,]\d)", source_text):
                    return True
    return False


def _indian_group(n: int) -> str:
    """Format an integer with Indian digit grouping (e.g. 39000000 -> 3,90,00,000)."""
    s = str(abs(n))
    if len(s) <= 3:
        grouped = s
    else:
        head, tail = s[:-3], s[-3:]
        # group head in pairs from the right
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        grouped = ",".join(parts) + "," + tail
    return ("-" if n < 0 else "") + grouped


def verify_claims(
    claims: List[ArithmeticClaim],
    source_text: Optional[str] = None,
    rel_tol: float = 1e-3,
) -> List[ClaimVerdict]:
    known = {c.id for c in claims if c.id}
    return [verify_claim(c, source_text=source_text, rel_tol=rel_tol,
                         known_claim_ids=known) for c in claims]


def summarize(verdicts: List[ClaimVerdict]) -> dict:
    """Roll up claim verdicts into counts the auditor uses for its overall verdict."""
    counts: Dict[str, int] = {}
    for v in verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1
    return {
        "total": len(verdicts),
        "confirmed": counts.get("CONFIRMED", 0),
        "arithmetic_error": counts.get("ARITHMETIC_ERROR", 0),
        "input_error": counts.get("INPUT_ERROR", 0),
        "mislabelled_input": counts.get("MISLABELLED_INPUT", 0),
        "unverifiable": counts.get("UNVERIFIABLE", 0),
        "any_error": counts.get("ARITHMETIC_ERROR", 0) + counts.get("INPUT_ERROR", 0) > 0,
    }