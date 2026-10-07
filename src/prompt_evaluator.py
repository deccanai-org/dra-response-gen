# src/prompt_evaluator.py
"""
Domain-agnostic prompt quality evaluator.
Reads any rubric JSON and scores prompts against it using LLM-as-Judge.

Modes:
    - batch:      Score all prompts, produce summary report
    - individual: Deep-dive single prompt with improvement suggestions
    - audit:      Compare LLMAJ scores against existing human scores
"""

import json
import math
import os
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.prompt_eval_templates import (
    BATCH_IMPROVEMENT_FIELDS,
    CROSS_VALIDATION_INSTRUCTIONS,
    CROSS_VALIDATION_OUTPUT,
    GDRIVE_FILES_SECTION,
    INDIVIDUAL_EXTRA_INSTRUCTIONS,
    INDIVIDUAL_IMPROVEMENT_FIELDS,
    NO_CROSS_VALIDATION_INSTRUCTIONS,
    NO_CROSS_VALIDATION_OUTPUT,
    NO_SUPPORTING_EVIDENCE,
    PROMPT_AUDIT_TEMPLATE,
    PROMPT_RUBRIC_EXTRACTION_TEMPLATE,
    PROMPT_SCORE_TEMPLATE,
    SUPPORTING_EVIDENCE_SECTION,
    build_decision_rules_text,
    build_existing_scores_text,
    build_gate_criteria_text,
    build_rubric_text,
    get_criteria_ids,
    get_gate_criteria,
    get_score_range,
)


# =============================================================================
# Rate Limiting
# =============================================================================

DEFAULT_DELAY_SECONDS = 3.0
MAX_RETRIES = 3
INITIAL_BACKOFF = 10.0
MAX_BACKOFF = 60.0
_last_api_call_time = 0.0


def _rate_limit_delay():
    """Enforce minimum delay between API calls."""
    global _last_api_call_time
    now = time.time()
    elapsed = now - _last_api_call_time
    if elapsed < DEFAULT_DELAY_SECONDS:
        time.sleep(DEFAULT_DELAY_SECONDS - elapsed)
    _last_api_call_time = time.time()


# Default judge model for the auditor + rubric scorer.
DEFAULT_JUDGE_MODEL = "claude-opus-5"

# NOTE on the judge pool and self-evaluation bias:
#   The original pipeline excluded Claude and Gemini from the judge pool because
#   they are in the DRA *response* evaluation set — using them to score agent
#   RESPONSES would be self-evaluation. The task auditor and prompt-quality
#   scorer do something different: they assess TASK validity (arithmetic,
#   leakage, trap structure, prompt quality), not agent responses. That is
#   prompt-QC, not response-scoring, so Claude (Opus 5) is admissible here.
#   Gemini/DeepSeek remain excluded for response scoring elsewhere.
#
# Fallback chains for the open-weight Together judges only. The Claude judge
# deliberately has NO fallback: if the Anthropic API fails, the pipeline must
# hard-stop rather than silently degrade to a weaker, different-family judge.
# Mixing judge models within a single benchmark run would contaminate results
# in a way that is hard to detect after the fact, so we fail loudly instead.
# PROVIDER fallback is different and allowed: src/llm_route.py serves the SAME
# Claude model through OpenRouter when Anthropic is down or over its spend cap.
MODEL_FALLBACK_CHAIN = {
    "claude-opus-5": [],
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": [
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ],
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": [
        "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    ],
}

_exhausted_models: set = set()
_last_model_used: str = ""


def get_last_model_used() -> str:
    return _last_model_used


def get_exhausted_models() -> set:
    return _exhausted_models.copy()


def _call_llm(prompt: str, model_name: str, max_tokens: int = 4000,
              system_prompt: Optional[str] = None) -> str:
    """Call LLM API with exponential backoff retry and model fallback.

    system_prompt is only honored by the Anthropic (claude-) provider branch;
    for Together/Gemini models it is prepended to the user prompt so behavior is
    consistent across providers.
    """
    global _last_model_used
    # For non-Anthropic providers, fold any system prompt into the user message
    # so the instruction is not silently dropped on fallback.
    effective_prompt = prompt
    models_to_try = [model_name] + MODEL_FALLBACK_CHAIN.get(model_name, [])
    models_to_try = [m for m in models_to_try if m not in _exhausted_models]

    if not models_to_try:
        raise Exception(
            f"All models exhausted for this session: {_exhausted_models}. "
            f"Wait for quota reset or add billing."
        )

    for model_idx, current_model in enumerate(models_to_try):
        backoff = INITIAL_BACKOFF

        for attempt in range(MAX_RETRIES):
            try:
                _rate_limit_delay()

                if current_model.startswith("claude-"):
                    # Anthropic direct, else the SAME model through OpenRouter (src/llm_route.py).
                    # Both providers stream; Opus 5 takes no sampling params here (as before).
                    from src.llm_route import claude_complete
                    text = claude_complete(prompt, model=current_model, system=system_prompt,
                                           max_tokens=max_tokens)
                    _last_model_used = current_model
                    if model_idx > 0:
                        print(f"  → Using fallback model: {current_model}")
                    return text
                elif "gemini" in current_model.lower():
                    import google.generativeai as genai
                    if system_prompt:
                        effective_prompt = f"{system_prompt}\n\n{prompt}"
                    generation_config = genai.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.1,
                    )
                    model = genai.GenerativeModel(current_model)
                    response = model.generate_content(
                        effective_prompt, generation_config=generation_config
                    )
                    _last_model_used = current_model
                    if model_idx > 0:
                        print(f"  → Using fallback model: {current_model}")
                    return response.text
                else:
                    from together import Together
                    if system_prompt:
                        effective_prompt = f"{system_prompt}\n\n{prompt}"
                    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
                    response = client.chat.completions.create(
                        model=current_model,
                        messages=[{"role": "user", "content": effective_prompt}],
                        max_tokens=max_tokens,
                        temperature=0.1,
                    )
                    _last_model_used = current_model
                    if model_idx > 0:
                        print(f"  → Using fallback model: {current_model}")
                    return response.choices[0].message.content

            except Exception as e:
                error_str = str(e).lower()

                is_model_unavailable = any(
                    x in error_str
                    for x in ["model_not_available", "non-serverless", "not found",
                              "does not exist", "model not available", "invalid model",
                              "not supported"]
                )
                if is_model_unavailable:
                    _exhausted_models.add(current_model)
                    remaining = [m for m in models_to_try[model_idx + 1:] if m not in _exhausted_models]
                    if remaining:
                        print(f"  ⚠ {current_model} unavailable. Falling back to {remaining[0]}...")
                    else:
                        print(f"  ⚠ {current_model} unavailable. No fallback models remaining.")
                    break

                is_quota_exhausted = any(
                    x in error_str for x in ["quota", "resource exhausted"]
                ) and "per day" in error_str.replace("perday", "per day").replace("per_day", "per day")

                is_retriable = any(
                    x in error_str
                    for x in ["quota", "rate limit", "resource exhausted", "429", "too many", "retry"]
                )

                if is_quota_exhausted or (is_retriable and attempt >= MAX_RETRIES - 1):
                    _exhausted_models.add(current_model)
                    remaining = [m for m in models_to_try[model_idx + 1:] if m not in _exhausted_models]
                    if remaining:
                        print(f"  ⚠ {current_model} quota exhausted. Falling back to {remaining[0]}...")
                    else:
                        print(f"  ⚠ {current_model} quota exhausted. No fallback models remaining.")
                    break

                if is_retriable and attempt < MAX_RETRIES - 1:
                    jitter = random.uniform(0, backoff * 0.1)
                    wait_time = backoff + jitter
                    print(f"  Rate limited. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                else:
                    # Claude judge has no fallback by design — make the hard-stop explicit
                    if current_model.startswith("claude-"):
                        raise RuntimeError(
                            f"Claude judge ({current_model}) failed and has NO fallback "
                            f"by design (hard-stop to avoid mixing judge models within a "
                            f"benchmark run). Original error: {e}. "
                            f"Fix the Anthropic API issue and re-run — do not let the batch "
                            f"continue on a different judge."
                        ) from e
                    raise e

    # If we reach here with a Claude model and an empty chain, surface it clearly
    if model_name.startswith("claude-"):
        raise RuntimeError(
            f"Claude judge ({model_name}) unavailable and has no fallback by design. "
            f"Exhausted: {_exhausted_models}. The run is halted intentionally rather "
            f"than degrading to a weaker judge."
        )
    raise Exception(
        f"All models failed. Exhausted: {_exhausted_models}. Tried: {models_to_try}"
    )


def _clean_json_response(text: str) -> str:
    """Clean LLM response to extract valid JSON.

    Handles an optional leading <analysis>...</analysis> reasoning scratchpad
    (used by the auditor): everything up to and including </analysis> is removed.
    If the block is unclosed, we fall back to taking text from the first '{'.
    """
    text = text.strip()

    # Strip a leading <analysis> scratchpad if present.
    if "<analysis>" in text:
        if "</analysis>" in text:
            text = text.split("</analysis>", 1)[1].strip()
        else:
            # Unclosed scratchpad — drop everything before the first JSON brace.
            brace = text.find("{")
            if brace != -1:
                text = text[brace:].strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()


def _repair_and_parse_json(text: str, prompt_id: str = "") -> Optional[Dict]:
    """
    Attempt multiple strategies to parse potentially malformed JSON from LLM.

    Strategies (in order):
    1. Direct parse after cleaning
    2. Remove trailing commas before } and ]
    3. Fix unescaped newlines inside strings
    4. Remove control characters
    5. Truncate at last valid closing brace and parse
    6. Use regex to extract individual score fields
    """
    cleaned = _clean_json_response(text)

    # Strategy 1: Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Remove trailing commas
    try:
        fixed = re.sub(r',\s*([}\]])', r'\1', cleaned)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Fix unescaped control characters inside strings
    try:
        fixed = cleaned.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Remove non-ASCII and control characters
    try:
        fixed = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 5: Truncate at last valid closing brace
    try:
        depth = 0
        last_valid = -1
        in_string = False
        escape_next = False
        for i, c in enumerate(cleaned):
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    last_valid = i
                    break
        if last_valid > 0:
            truncated = cleaned[:last_valid + 1]
            truncated = re.sub(r',\s*([}\]])', r'\1', truncated)
            return json.loads(truncated)
    except json.JSONDecodeError:
        pass

    # Strategy 6: Regex extraction as last resort
    try:
        scores = {}
        score_pattern = re.compile(
            r'"(\w+)"\s*:\s*\{\s*"score"\s*:\s*(\d+)\s*,\s*"justification"\s*:\s*"([^"]*(?:\\"[^"]*)*)"',
            re.DOTALL
        )
        for match in score_pattern.finditer(cleaned):
            criterion_id = match.group(1)
            score_val = int(match.group(2))
            justification = match.group(3).replace('\\"', '"')
            scores[criterion_id] = {"score": score_val, "justification": justification}

        if scores:
            print(f"  Repair: extracted {len(scores)} scores via regex for {prompt_id}")
            decision_match = re.search(r'"decision"\s*:\s*"(ACCEPT|REVISE|REJECT)"', cleaned)
            avg_match = re.search(r'"average_score"\s*:\s*([\d.]+)', cleaned)
            tag_match = re.search(r'"research_depth_tag"\s*:\s*"(\w+)"', cleaned)
            gate_match = re.search(r'"gate_passed"\s*:\s*(true|false)', cleaned)
            return {
                "prompt_id": prompt_id,
                "scores": scores,
                "decision": decision_match.group(1) if decision_match else "REVISE",
                "average_score": float(avg_match.group(1)) if avg_match else 0.0,
                "research_depth_tag": tag_match.group(1) if tag_match else None,
                "gate_passed": gate_match.group(1) == "true" if gate_match else False,
                "decision_logic": "Reconstructed from partial JSON via regex repair",
                "_repaired": True,
            }
    except Exception:
        pass

    return None


# =============================================================================
# Rubric Generation
# =============================================================================

def generate_prompt_rubric_from_document(
    instruction_document: str,
    model_name: str = DEFAULT_JUDGE_MODEL,
) -> Optional[Dict]:
    """Extract prompt evaluation rubric from an instruction document."""
    print(f"\n--- Extracting Prompt Evaluation Rubric using {model_name} ---")

    try:
        prompt = PROMPT_RUBRIC_EXTRACTION_TEMPLATE.format(
            instruction_document_text=instruction_document
        )
        response_text = _call_llm(prompt, model_name, max_tokens=16000)
        cleaned = _clean_json_response(response_text)
        rubric = json.loads(cleaned)

        rubric_data = rubric.get("rubric", rubric)
        categories = rubric_data.get("categories", [])
        if not categories:
            print("Warning: No categories extracted from document.")
            return None

        n_cats = len(categories)
        n_gates = sum(1 for c in categories if c.get("is_gate_criterion"))
        n_tags = len(rubric_data.get("binary_tags", []))
        n_types = len(rubric_data.get("prompt_types", []))
        n_examples = len(rubric_data.get("calibration_examples", []))

        print(f"  Criteria extracted:     {n_cats}")
        print(f"  Gate criteria:          {n_gates}")
        print(f"  Binary tags:            {n_tags}")
        print(f"  Prompt types:           {n_types}")
        print(f"  Calibration examples:   {n_examples}")
        print("Rubric extracted successfully.")
        return rubric

    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse LLM response as JSON: {e}")
        print(f"  Raw response (first 500 chars): {response_text[:500]}")
        return None
    except Exception as e:
        print(f"Error extracting rubric: {e}")
        return None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PromptScore:
    """Score for a single criterion."""
    criterion_id: str
    criterion_name: str
    score: int
    justification: str
    is_gate: bool = False
    gate_minimum: Optional[int] = None
    gate_passed: Optional[bool] = None


@dataclass
class ImprovementSuggestion:
    """Actionable improvement suggestion for a weak criterion."""
    criterion_id: str
    current_score: int
    target_score: int
    issue: str
    fix: str
    example_rewrite_fragment: str = ""


@dataclass
class PromptEvalResult:
    """Complete evaluation result for one prompt."""
    prompt_id: str
    sme_name: str = ""
    prompt_type: str = ""
    domain: str = ""

    # Scores
    criterion_scores: List[PromptScore] = field(default_factory=list)
    average_score: float = 0.0

    # Tags
    research_depth_tag: Optional[str] = None
    research_depth_justification: str = ""

    # Decision
    gate_passed: bool = True
    decision: str = ""           # ACCEPT, REVISE, REJECT, ERROR
    decision_logic: str = ""

    # Improvements (individual mode only)
    improvement_suggestions: List[ImprovementSuggestion] = field(default_factory=list)

    # Type validation
    type_appropriate: bool = True
    type_reasoning: str = ""
    suggested_type: Optional[str] = None

    # Cross-validation (when Logic/SC/GDrive provided)
    cross_validation: Optional[Dict] = None

    # Multi-run metadata
    n_runs: int = 1
    best_run_index: int = 0
    per_run_averages: List[float] = field(default_factory=list)

    # Raw LLM output (for debugging)
    raw_llm_response: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "prompt_id": self.prompt_id,
            "sme_name": self.sme_name,
            "prompt_type": self.prompt_type,
            "domain": self.domain,
            "criterion_scores": [asdict(s) for s in self.criterion_scores],
            "average_score": self.average_score,
            "research_depth_tag": self.research_depth_tag,
            "research_depth_justification": self.research_depth_justification,
            "gate_passed": self.gate_passed,
            "decision": self.decision,
            "decision_logic": self.decision_logic,
            "improvement_suggestions": [asdict(s) for s in self.improvement_suggestions],
            "type_appropriate": self.type_appropriate,
            "type_reasoning": self.type_reasoning,
            "suggested_type": self.suggested_type,
            "cross_validation": self.cross_validation,
            "n_runs": self.n_runs,
            "best_run_index": self.best_run_index,
            "per_run_averages": self.per_run_averages,
        }


@dataclass
class AuditResult:
    """Audit comparison between LLMAJ and human scores."""
    prompt_id: str
    sme_name: str = ""
    llm_scores: Dict[str, int] = field(default_factory=dict)
    human_scores: Dict[str, int] = field(default_factory=dict)
    disagreements: List[Dict] = field(default_factory=list)
    llm_average: float = 0.0
    human_average: float = 0.0
    llm_decision: str = ""
    human_decision: str = ""
    decisions_agree: bool = True
    calibration_note: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Core Scoring Functions
# =============================================================================

def evaluate_prompt(
    prompt_text: str,
    rubric: Dict,
    prompt_id: str = "unknown",
    prompt_type: str = "",
    sme_name: str = "",
    domain: str = "",
    extra_metadata: str = "",
    logic_text: str = "",
    sc_text: str = "",
    gdrive_text: str = "",
    model_name: str = DEFAULT_JUDGE_MODEL,
    mode: str = "batch",
) -> PromptEvalResult:
    """
    Score a single prompt against the rubric.

    Args:
        prompt_text:    Full text of the prompt to evaluate.
        rubric:         Rubric JSON dict (with 'rubric' key or flat).
        prompt_id:      Identifier for this prompt.
        prompt_type:    SME-assigned type (e.g., FSP, CRP, LDP).
        sme_name:       Name of SME who wrote the prompt.
        domain:         Domain label (e.g., "Management Consulting").
        extra_metadata: Any additional context to include.
        logic_text:     SME's Solution Logic (calculation steps, golden answer).
        sc_text:        SME's Sanity Check (failure trap description).
        gdrive_text:    Pre-extracted text from GDrive reference files.
        model_name:     LLM model for evaluation.
        mode:           "batch" (scores only) or "individual" (scores + suggestions).

    Returns:
        PromptEvalResult with all scores, decision, cross-validation, and suggestions.
    """
    rubric_text = build_rubric_text(rubric)
    gate_text = build_gate_criteria_text(rubric)
    decision_text = build_decision_rules_text(rubric)
    criteria_ids = get_criteria_ids(rubric)
    gates = get_gate_criteria(rubric)

    if mode == "individual":
        additional_instructions = INDIVIDUAL_EXTRA_INSTRUCTIONS
        improvement_fields = INDIVIDUAL_IMPROVEMENT_FIELDS
        max_tokens = 8000
    else:
        additional_instructions = ""
        improvement_fields = BATCH_IMPROVEMENT_FIELDS
        max_tokens = 4000

    # Build the GDrive files subsection (only when content is present)
    if gdrive_text.strip():
        gdrive_files_section = GDRIVE_FILES_SECTION.format(gdrive_text=gdrive_text.strip())
    else:
        gdrive_files_section = ""

    # Logic/SC cross-validation (active when at least one of the three sources is present)
    has_supporting = bool(logic_text.strip()) or bool(sc_text.strip()) or bool(gdrive_text.strip())
    if has_supporting:
        supporting_evidence_section = SUPPORTING_EVIDENCE_SECTION.format(
            logic_text=logic_text.strip() if logic_text.strip() else "(Not provided by SME)",
            sc_text=sc_text.strip() if sc_text.strip() else "(Not provided by SME)",
            gdrive_files_section=gdrive_files_section,
        )
        cross_validation_instructions = CROSS_VALIDATION_INSTRUCTIONS
        cross_validation_output = CROSS_VALIDATION_OUTPUT
        max_tokens += 2000
    else:
        supporting_evidence_section = NO_SUPPORTING_EVIDENCE
        cross_validation_instructions = NO_CROSS_VALIDATION_INSTRUCTIONS
        cross_validation_output = NO_CROSS_VALIDATION_OUTPUT

    llm_prompt = PROMPT_SCORE_TEMPLATE.format(
        rubric_text=rubric_text,
        gate_criteria_text=gate_text,
        decision_rules_text=decision_text,
        prompt_id=prompt_id,
        prompt_type=prompt_type or "Not specified",
        domain=domain or "Not specified",
        extra_metadata=extra_metadata,
        prompt_text=prompt_text,
        supporting_evidence_section=supporting_evidence_section,
        cross_validation_instructions=cross_validation_instructions,
        additional_instructions=additional_instructions,
        improvement_fields=improvement_fields,
        cross_validation_output=cross_validation_output,
    )

    MAX_JSON_RETRIES = 2
    print(f"  Evaluating prompt {prompt_id}...")

    result_json = None
    last_error = None
    response_text = ""

    for attempt in range(1 + MAX_JSON_RETRIES):
        try:
            if attempt > 0:
                print(f"  Retry {attempt}/{MAX_JSON_RETRIES} for {prompt_id}...")
                _rate_limit_delay()

            response_text = _call_llm(llm_prompt, model_name, max_tokens=max_tokens)
            result_json = _repair_and_parse_json(response_text, prompt_id)

            if result_json is not None:
                if result_json.get("_repaired"):
                    scores = result_json.get("scores", {})
                    if len(scores) < len(criteria_ids):
                        print(f"  Partial repair: only {len(scores)}/{len(criteria_ids)} scores recovered — retrying...")
                        result_json = None
                        continue
                break
            else:
                last_error = "All repair strategies failed"

        except Exception as e:
            last_error = str(e)
            error_str = str(e).lower()
            print(f"  Warning: LLM call failed (attempt {attempt + 1}): {e}")
            if any(x in error_str for x in [
                "429", "quota", "rate",
                "model_not_available", "non-serverless", "not found",
                "does not exist", "model not available", "invalid model",
            ]):
                print(f"  Skipping further retries — _call_llm exhausted all options.")
                break

    if result_json is None:
        print(f"  ERROR: Failed after {1 + MAX_JSON_RETRIES} attempts for {prompt_id}: {last_error}")
        if response_text:
            print(f"  Last raw response (first 500 chars): {response_text[:500]}")
        return _default_eval_result(prompt_id, sme_name, prompt_type, domain, criteria_ids)

    return _parse_score_response(
        result_json, rubric, prompt_id, sme_name, prompt_type, domain, criteria_ids, gates
    )


def evaluate_prompt_multi_run(
    prompt_text: str,
    rubric: Dict,
    n_runs: int = 3,
    prompt_id: str = "unknown",
    prompt_type: str = "",
    sme_name: str = "",
    domain: str = "",
    extra_metadata: str = "",
    logic_text: str = "",
    sc_text: str = "",
    gdrive_text: str = "",
    model_name: str = DEFAULT_JUDGE_MODEL,
    mode: str = "batch",
    per_run_log_path: Optional[str] = None,
) -> PromptEvalResult:
    """
    Run evaluate_prompt n_runs times, average criterion scores across runs,
    and return a single PromptEvalResult with:
      - Averaged (rounded) criterion scores
      - Justifications, cross-validation, and decision logic from the best run
        (best = highest average_score among non-ERROR runs)
      - Multi-run metadata: n_runs, best_run_index, per_run_averages

    Per-run raw data is optionally appended to a JSONL log file.

    If n_runs == 1, falls through directly to evaluate_prompt().
    If all runs error, returns the first error result with multi-run metadata.
    """
    if n_runs <= 1:
        result = evaluate_prompt(
            prompt_text=prompt_text, rubric=rubric, prompt_id=prompt_id,
            prompt_type=prompt_type, sme_name=sme_name, domain=domain,
            extra_metadata=extra_metadata, logic_text=logic_text,
            sc_text=sc_text, gdrive_text=gdrive_text,
            model_name=model_name, mode=mode,
        )
        result.n_runs = 1
        result.per_run_averages = [result.average_score]
        result.best_run_index = 0
        if per_run_log_path:
            _append_run_log(per_run_log_path, result, run_index=0)
        return result

    print(f"\n  ┌─ Multi-run: {n_runs} runs for prompt {prompt_id}")
    run_results: List[PromptEvalResult] = []

    for run_idx in range(n_runs):
        print(f"  │  Run {run_idx + 1}/{n_runs}...")
        result = evaluate_prompt(
            prompt_text=prompt_text, rubric=rubric, prompt_id=prompt_id,
            prompt_type=prompt_type, sme_name=sme_name, domain=domain,
            extra_metadata=extra_metadata, logic_text=logic_text,
            sc_text=sc_text, gdrive_text=gdrive_text,
            model_name=model_name, mode=mode,
        )
        run_results.append(result)
        if per_run_log_path:
            _append_run_log(per_run_log_path, result, run_index=run_idx)
        print(f"  │  Run {run_idx + 1} → {result.decision} (avg={result.average_score:.2f})")

    per_run_averages = [r.average_score for r in run_results]
    valid_runs: List[Tuple[int, PromptEvalResult]] = [
        (i, r) for i, r in enumerate(run_results) if r.decision != "ERROR"
    ]

    print(f"  └─ Runs complete. Averages: {[f'{a:.2f}' for a in per_run_averages]}")

    # All runs failed
    if not valid_runs:
        print(f"  ⚠ All {n_runs} runs returned ERROR for {prompt_id}")
        final = run_results[0]
        final.n_runs = n_runs
        final.best_run_index = 0
        final.per_run_averages = per_run_averages
        return final

    # Best run = highest average_score
    best_run_idx, best_result = max(valid_runs, key=lambda x: x[1].average_score)
    print(f"     Best run: #{best_run_idx + 1} (avg={best_result.average_score:.2f})")

    # Average criterion scores across all valid runs
    rubric_data = rubric.get("rubric", rubric)
    cat_lookup = {c["id"]: c for c in rubric_data.get("categories", [])}

    # Map: criterion_id → list of scores across valid runs
    cid_score_lists: Dict[str, List[int]] = {}
    for _, r in valid_runs:
        for ps in r.criterion_scores:
            cid_score_lists.setdefault(ps.criterion_id, []).append(ps.score)

    # Build final result — use best run for metadata, averaged scores for scoring
    final = PromptEvalResult(
        prompt_id=prompt_id,
        sme_name=sme_name,
        prompt_type=prompt_type,
        domain=domain,
        n_runs=n_runs,
        best_run_index=best_run_idx,
        per_run_averages=per_run_averages,
        research_depth_tag=best_result.research_depth_tag,
        research_depth_justification=best_result.research_depth_justification,
        type_appropriate=best_result.type_appropriate,
        type_reasoning=best_result.type_reasoning,
        suggested_type=best_result.suggested_type,
        cross_validation=best_result.cross_validation,
        improvement_suggestions=best_result.improvement_suggestions,
    )

    # Averaged (rounded) criterion scores; justifications from best run
    float_averages: List[float] = []
    best_run_score_map = {ps.criterion_id: ps for ps in best_result.criterion_scores}

    for cid, scores_list in cid_score_lists.items():
        float_avg = sum(scores_list) / len(scores_list)
        rounded = round(float_avg)
        float_averages.append(float_avg)

        cat_info = cat_lookup.get(cid, {})
        is_gate = cat_info.get("is_gate_criterion", False)
        gate_min = cat_info.get("gate_minimum")
        gate_ok = rounded >= gate_min if (is_gate and gate_min is not None) else None

        best_ps = best_run_score_map.get(cid)
        final.criterion_scores.append(PromptScore(
            criterion_id=cid,
            criterion_name=best_ps.criterion_name if best_ps else cid,
            score=rounded,
            justification=best_ps.justification if best_ps else "",
            is_gate=is_gate,
            gate_minimum=gate_min,
            gate_passed=gate_ok,
        ))

    # Overall average uses float averages (before rounding) for accuracy
    final.average_score = sum(float_averages) / len(float_averages) if float_averages else 0.0

    # Gate check on averaged scores
    gate_results = [
        ps.gate_passed for ps in final.criterion_scores
        if ps.is_gate and ps.gate_passed is not None
    ]
    final.gate_passed = all(gate_results) if gate_results else True

    # Decision and logic from best run, prefixed with multi-run context
    final.decision = best_result.decision
    final.decision_logic = (
        f"[Multi-run {n_runs}x: best run #{best_run_idx + 1} "
        f"avg={best_result.average_score:.2f}, "
        f"overall avg={final.average_score:.2f}] "
        + best_result.decision_logic
    )
    # Re-verify decision on the averaged scores
    final.decision = _verify_decision(final, rubric)

    return final


def audit_prompt(
    prompt_text: str,
    rubric: Dict,
    existing_scores: Dict[str, int],
    prompt_id: str = "unknown",
    prompt_type: str = "",
    sme_name: str = "",
    logic_text: str = "",
    sc_text: str = "",
    gdrive_text: str = "",
    model_name: str = DEFAULT_JUDGE_MODEL,
) -> AuditResult:
    """Score a prompt and compare against existing human scores."""
    rubric_text = build_rubric_text(rubric)
    gate_text = build_gate_criteria_text(rubric)
    criteria_ids = get_criteria_ids(rubric)
    existing_text = build_existing_scores_text(existing_scores, criteria_ids)

    if gdrive_text.strip():
        gdrive_files_section = GDRIVE_FILES_SECTION.format(gdrive_text=gdrive_text.strip())
    else:
        gdrive_files_section = ""

    has_supporting = bool(logic_text.strip()) or bool(sc_text.strip()) or bool(gdrive_text.strip())
    if has_supporting:
        supporting_evidence_section = SUPPORTING_EVIDENCE_SECTION.format(
            logic_text=logic_text.strip() if logic_text.strip() else "(Not provided by SME)",
            sc_text=sc_text.strip() if sc_text.strip() else "(Not provided by SME)",
            gdrive_files_section=gdrive_files_section,
        )
    else:
        supporting_evidence_section = ""

    llm_prompt = PROMPT_AUDIT_TEMPLATE.format(
        rubric_text=rubric_text,
        gate_criteria_text=gate_text,
        prompt_id=prompt_id,
        prompt_type=prompt_type or "Not specified",
        sme_name=sme_name or "Unknown",
        prompt_text=prompt_text,
        supporting_evidence_section=supporting_evidence_section,
        existing_scores_text=existing_text,
    )

    print(f"  Auditing prompt {prompt_id} (vs {sme_name})...")
    try:
        response_text = _call_llm(llm_prompt, model_name, max_tokens=6000)
        result_json = _repair_and_parse_json(response_text, prompt_id)
        if result_json is None:
            print(f"  Warning: All JSON repair strategies failed for audit {prompt_id}")
            return AuditResult(prompt_id=prompt_id, sme_name=sme_name)
    except Exception as e:
        print(f"  Warning: Audit failed for {prompt_id}: {e}")
        return AuditResult(prompt_id=prompt_id, sme_name=sme_name)

    return _parse_audit_response(result_json, prompt_id, sme_name, rubric, criteria_ids)


# =============================================================================
# Response Parsing
# =============================================================================

def _parse_score_response(
    result_json: Dict,
    rubric: Dict,
    prompt_id: str,
    sme_name: str,
    prompt_type: str,
    domain: str,
    criteria_ids: List[str],
    gates: List[Dict],
) -> PromptEvalResult:
    """Parse LLM JSON response into PromptEvalResult."""
    rubric_data = rubric.get("rubric", rubric)
    cat_lookup = {c["id"]: c for c in rubric_data.get("categories", [])}

    result = PromptEvalResult(
        prompt_id=prompt_id,
        sme_name=sme_name,
        prompt_type=prompt_type,
        domain=domain,
        raw_llm_response=result_json,
    )

    scores_dict = result_json.get("scores", {})
    score_values = []

    for cid in criteria_ids:
        score_data = scores_dict.get(cid, {})
        cat_info = cat_lookup.get(cid, {})

        score_val = score_data.get("score", 0) if isinstance(score_data, dict) else 0
        justification = score_data.get("justification", "") if isinstance(score_data, dict) else ""

        try:
            score_val = int(score_val)
        except (ValueError, TypeError):
            score_val = 0

        is_gate = cat_info.get("is_gate_criterion", False)
        gate_min = cat_info.get("gate_minimum")
        gate_ok = score_val >= gate_min if (is_gate and gate_min is not None) else None

        result.criterion_scores.append(PromptScore(
            criterion_id=cid,
            criterion_name=cat_info.get("name", cid),
            score=score_val,
            justification=justification,
            is_gate=is_gate,
            gate_minimum=gate_min,
            gate_passed=gate_ok,
        ))
        score_values.append(score_val)

    result.average_score = sum(score_values) / len(score_values) if score_values else 0.0

    gate_results = [
        ps.gate_passed for ps in result.criterion_scores
        if ps.is_gate and ps.gate_passed is not None
    ]
    result.gate_passed = all(gate_results) if gate_results else True

    result.research_depth_tag = result_json.get("research_depth_tag")
    result.research_depth_justification = result_json.get("research_depth_justification", "")

    result.decision = result_json.get("decision", "REJECT")
    result.decision_logic = result_json.get("decision_logic", "")
    result.decision = _verify_decision(result, rubric)

    for sugg in result_json.get("improvement_suggestions", []):
        result.improvement_suggestions.append(
            ImprovementSuggestion(
                criterion_id=sugg.get("criterion_id", ""),
                current_score=sugg.get("current_score", 0),
                target_score=sugg.get("target_score", 3),
                issue=sugg.get("issue", ""),
                fix=sugg.get("fix", ""),
                example_rewrite_fragment=sugg.get("example_rewrite_fragment", ""),
            )
        )

    type_val = result_json.get("prompt_type_validation", {})
    result.type_appropriate = type_val.get("type_appropriate", True)
    result.type_reasoning = type_val.get("reasoning", "")
    result.suggested_type = type_val.get("suggested_type")

    cv = result_json.get("cross_validation")
    if cv:
        result.cross_validation = cv

    return result


def _verify_decision(result: PromptEvalResult, rubric: Dict) -> str:
    """Verify and potentially override the LLM's decision using the rubric's rules."""
    rubric_data = rubric.get("rubric", rubric)

    scores = [ps.score for ps in result.criterion_scores]
    avg = result.average_score

    if any(s == 0 for s in scores):
        if result.decision != "REJECT":
            result.decision_logic += " [OVERRIDE: criterion scored 0 → REJECT]"
        return "REJECT"

    for ps in result.criterion_scores:
        if ps.is_gate and ps.gate_passed is False:
            if ps.gate_minimum is not None and ps.score < ps.gate_minimum - 1:
                if result.decision != "REJECT":
                    result.decision_logic += (
                        f" [OVERRIDE: {ps.criterion_id}={ps.score} "
                        f"< gate_min={ps.gate_minimum} → REJECT]"
                    )
                return "REJECT"
            else:
                if result.decision == "ACCEPT":
                    result.decision_logic += (
                        f" [OVERRIDE: {ps.criterion_id}={ps.score} "
                        f"< gate_min={ps.gate_minimum} → REVISE]"
                    )
                    return "REVISE"

    min_score, max_score = get_score_range(rubric)
    accept_threshold = max_score * 0.833
    if result.decision == "ACCEPT" and avg < accept_threshold:
        result.decision_logic += (
            f" [OVERRIDE: avg={avg:.2f} < {accept_threshold:.2f} → REVISE]"
        )
        return "REVISE"

    all_gates_passed = all(
        ps.gate_passed is not False
        for ps in result.criterion_scores
        if ps.is_gate
    )
    if result.decision == "REVISE" and all_gates_passed and avg >= accept_threshold:
        result.decision_logic += (
            f" [OVERRIDE: avg={avg:.2f} >= {accept_threshold:.2f} "
            f"and all gates passed → ACCEPT]"
        )
        return "ACCEPT"

    return result.decision


def _parse_audit_response(
    result_json: Dict,
    prompt_id: str,
    sme_name: str,
    rubric: Dict,
    criteria_ids: List[str],
) -> AuditResult:
    """Parse LLM audit response into AuditResult."""
    result = AuditResult(prompt_id=prompt_id, sme_name=sme_name)

    llm_scores_raw = result_json.get("llm_scores", {})
    for cid in criteria_ids:
        score_data = llm_scores_raw.get(cid, {})
        if isinstance(score_data, dict):
            try:
                result.llm_scores[cid] = int(score_data.get("score", 0))
            except (ValueError, TypeError):
                result.llm_scores[cid] = 0
        else:
            try:
                result.llm_scores[cid] = int(score_data)
            except (ValueError, TypeError):
                result.llm_scores[cid] = 0

    human_scores_raw = result_json.get("human_scores", {})
    for cid in criteria_ids:
        try:
            result.human_scores[cid] = int(human_scores_raw.get(cid, 0))
        except (ValueError, TypeError):
            result.human_scores[cid] = 0

    result.disagreements = result_json.get("disagreements", [])

    llm_vals = list(result.llm_scores.values())
    human_vals = list(result.human_scores.values())
    result.llm_average = sum(llm_vals) / len(llm_vals) if llm_vals else 0
    result.human_average = sum(human_vals) / len(human_vals) if human_vals else 0

    result.llm_decision = result_json.get("llm_decision", "")
    result.human_decision = result_json.get("human_decision", "")
    result.decisions_agree = result_json.get("decisions_agree", False)
    result.calibration_note = result_json.get("overall_calibration_note", "")

    return result


def _default_eval_result(
    prompt_id: str,
    sme_name: str,
    prompt_type: str,
    domain: str,
    criteria_ids: List[str],
) -> PromptEvalResult:
    """Return default result when evaluation fails."""
    result = PromptEvalResult(
        prompt_id=prompt_id,
        sme_name=sme_name,
        prompt_type=prompt_type,
        domain=domain,
        decision="ERROR",
        decision_logic="Evaluation failed — prompt was NOT scored (re-run to retry)",
    )
    for cid in criteria_ids:
        result.criterion_scores.append(
            PromptScore(
                criterion_id=cid,
                criterion_name=cid,
                score=-1,
                justification="Evaluation failed — not scored",
            )
        )
    return result


# =============================================================================
# Per-run logging
# =============================================================================

def _append_run_log(log_path: str, result: PromptEvalResult, run_index: int):
    """Append a single run's data as one JSON line to a JSONL log file."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "prompt_id": result.prompt_id,
        "sme_name": result.sme_name,
        "run_index": run_index,
        "decision": result.decision,
        "average_score": result.average_score,
        "gate_passed": result.gate_passed,
        "criterion_scores": {
            ps.criterion_id: ps.score for ps in result.criterion_scores
        },
        "research_depth_tag": result.research_depth_tag,
        "timestamp": datetime.now().isoformat(),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def save_per_run_log(results: List[PromptEvalResult], output_dir: str):
    """
    Write per-run log CSV for all multi-run results (n_runs > 1).

    Columns: prompt_id, sme_name, n_runs, run_0_avg, run_1_avg, ...,
             best_run_index, final_avg, final_decision
    """
    import csv

    multi_run_results = [r for r in results if r.n_runs > 1]
    if not multi_run_results:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find max n_runs to build run_N_avg columns
    max_runs = max(r.n_runs for r in multi_run_results)

    base_cols = ["prompt_id", "sme_name", "n_runs", "best_run_index", "final_avg", "final_decision"]
    run_cols = [f"run_{i}_avg" for i in range(max_runs)]
    all_cols = base_cols + run_cols

    rows = []
    for r in multi_run_results:
        row = {
            "prompt_id": r.prompt_id,
            "sme_name": r.sme_name,
            "n_runs": r.n_runs,
            "best_run_index": r.best_run_index,
            "final_avg": f"{r.average_score:.3f}",
            "final_decision": r.decision,
        }
        for i in range(max_runs):
            row[f"run_{i}_avg"] = (
                f"{r.per_run_averages[i]:.3f}" if i < len(r.per_run_averages) else ""
            )
        rows.append(row)

    csv_path = output_path / "per_run_averages.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Per-run CSV saved to: {csv_path}")


# =============================================================================
# Failure Reason Extraction
# =============================================================================

def _extract_failure_reasons(result: PromptEvalResult, rubric: Dict) -> List[str]:
    """
    Extract a succinct list of failure reason strings for REVISE/REJECT prompts.

    Returns a list of human-readable reason strings, e.g.:
      "[GATE FAIL]  dra_necessity = 1 (min = 2)"
      "[ZERO SCORE] constraint_rigidity = 0"
      "[LOW SCORE]  inference_necessity = 1"
      "[CV GAP]     LOGIC_GAP: prompt doesn't require external data retrieval"
      "[OVERRIDE]   avg=1.83 < 2.50 → REVISE"
    """
    _, max_score = get_score_range(rubric)
    low_threshold = max_score / 2  # scores at or below this are flagged as LOW

    reasons: List[str] = []

    # Gate failures and zero/low scores
    for ps in result.criterion_scores:
        if ps.score <= 0:
            reasons.append(f"[ZERO SCORE] {ps.criterion_id} = {ps.score}")
        elif ps.is_gate and ps.gate_passed is False:
            reasons.append(
                f"[GATE FAIL]  {ps.criterion_id} = {ps.score} "
                f"(min required = {ps.gate_minimum})"
            )
        elif ps.score <= low_threshold and ps.score > 0:
            reasons.append(f"[LOW SCORE]  {ps.criterion_id} = {ps.score}")

    # Decision override messages embedded in decision_logic
    override_matches = re.findall(r'\[OVERRIDE:[^\]]+\]', result.decision_logic)
    for m in override_matches:
        reasons.append(f"[OVERRIDE]   {m[10:-1].strip()}")  # strip "[OVERRIDE: " and "]"

    # Cross-validation gaps
    if result.cross_validation:
        for gap in result.cross_validation.get("gaps", []):
            if isinstance(gap, dict):
                gap_type = gap.get("type", "GAP")
                gap_desc = gap.get("description", "")[:100]
                reasons.append(f"[CV GAP]     {gap_type}: {gap_desc}")
            else:
                reasons.append(f"[CV GAP]     {str(gap)[:100]}")

        if result.cross_validation.get("is_meta_prompt"):
            reasons.append("[META PROMPT] Prompt describes itself rather than being a task")

    return reasons


# =============================================================================
# Reporting
# =============================================================================

def generate_batch_report(
    results: List[PromptEvalResult],
    rubric: Dict,
    output_dir: str,
) -> str:
    """Generate summary report from batch evaluation results."""

    criteria_ids = get_criteria_ids(rubric)
    _, max_score = get_score_range(rubric)

    n_error = sum(1 for r in results if r.decision == "ERROR")
    n_scored = len(results) - n_error

    lines = [
        "=" * 70,
        "PROMPT QUALITY EVALUATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "## OVERVIEW",
        f"Total Prompts:   {len(results)}",
        f"Scored:          {n_scored}",
        f"  Accepted:      {sum(1 for r in results if r.decision == 'ACCEPT')}",
        f"  Revise:        {sum(1 for r in results if r.decision == 'REVISE')}",
        f"  Rejected:      {sum(1 for r in results if r.decision == 'REJECT')}",
        f"Not Scored:      {n_error}  (ERROR — re-run to retry these)",
        "",
    ]

    # Multi-run summary (only when n_runs > 1 was used)
    multi_run_results = [r for r in results if r.n_runs > 1]
    if multi_run_results:
        n_runs_val = multi_run_results[0].n_runs
        lines.extend([
            "## MULTI-RUN CONFIGURATION",
            f"  Runs per prompt: {n_runs_val}",
            f"  Prompts with multi-run data: {len(multi_run_results)}/{len(results)}",
            "",
        ])

    # Score summary (exclude ERROR)
    scored_results = [r for r in results if r.decision != "ERROR"]
    avg_scores = [r.average_score for r in scored_results]
    if avg_scores:
        lines.extend([
            "## SCORE SUMMARY",
            f"  Mean Average Score:   {statistics.mean(avg_scores):.3f} / {max_score}",
            (f"  Std Dev:              {statistics.stdev(avg_scores):.3f}"
             if len(avg_scores) > 1 else ""),
            f"  Min:                  {min(avg_scores):.3f}",
            f"  Max:                  {max(avg_scores):.3f}",
            "",
        ])

    # Per-criterion breakdown
    lines.append("## PER-CRITERION BREAKDOWN")
    lines.append(f"{'Criterion':<35} {'Mean':>6} {'Min':>5} {'Max':>5} {'%@Max':>6}")
    lines.append("-" * 60)

    for cid in criteria_ids:
        crit_scores = []
        for r in scored_results:
            for ps in r.criterion_scores:
                if ps.criterion_id == cid:
                    crit_scores.append(ps.score)

        if crit_scores:
            mean_s = statistics.mean(crit_scores)
            pct_max = 100 * sum(1 for s in crit_scores if s == max_score) / len(crit_scores)
            lines.append(
                f"  {cid:<33} {mean_s:>6.2f} {min(crit_scores):>5} "
                f"{max(crit_scores):>5} {pct_max:>5.1f}%"
            )

    lines.append("")

    # Gate criterion results
    gates = get_gate_criteria(rubric)
    if gates:
        lines.append("## GATE CRITERION RESULTS")
        for gate in gates:
            passed = sum(
                1 for r in scored_results
                for ps in r.criterion_scores
                if ps.criterion_id == gate["id"] and ps.gate_passed
            )
            lines.append(
                f"  {gate['name']} (min={gate['minimum']}): "
                f"{passed}/{len(results)} passed ({100 * passed / len(results):.1f}%)"
            )
        lines.append("")

    # Research depth distribution
    tags = [r.research_depth_tag for r in results if r.research_depth_tag]
    if tags:
        from collections import Counter
        tag_counts = Counter(tags)
        lines.append("## RESEARCH DEPTH DISTRIBUTION")
        for tag, count in tag_counts.most_common():
            lines.append(f"  {tag}: {count} ({100 * count / len(results):.1f}%)")
        lines.append("")

    # Type distribution and validation
    types = [r.prompt_type for r in results if r.prompt_type]
    if types:
        from collections import Counter
        type_counts = Counter(types)
        mistyped = sum(1 for r in results if not r.type_appropriate)
        lines.append("## PROMPT TYPE DISTRIBUTION")
        for pt, count in type_counts.most_common():
            lines.append(f"  {pt}: {count}")
        lines.append(f"  Type misassignments: {mistyped}/{len(results)}")
        lines.append("")

    # Per-prompt decisions
    lines.append("## PER-PROMPT DECISIONS")
    lines.append(
        f"{'ID':<8} {'SME':<22} {'Type':<6} {'Avg':>5} "
        f"{'Decision':<8} {'Gate':>5} {'Runs':>5}"
    )
    lines.append("-" * 65)
    for r in sorted(results, key=lambda x: x.average_score):
        gate_str = "PASS" if r.gate_passed else "FAIL"
        runs_str = f"{r.n_runs}x" if r.n_runs > 1 else "1x"
        lines.append(
            f"  {str(r.prompt_id)[:6]:<8} {r.sme_name[:20]:<22} "
            f"{r.prompt_type[:5]:<6} {r.average_score:>5.2f} "
            f"{r.decision:<8} {gate_str:>5} {runs_str:>5}"
        )
    lines.append("")

    # ── FAILURE REASONS ──────────────────────────────────────────────────────
    failed_results = [r for r in results if r.decision in ("REJECT", "REVISE", "ERROR")]
    if failed_results:
        lines.append("## FAILURE REASONS")
        lines.append(
            "  Succinct failure summary for each non-ACCEPT prompt. "
            "Use these to prioritise revisions."
        )
        lines.append("")

        for r in sorted(failed_results, key=lambda x: x.average_score):
            sme_tag = f" ({r.sme_name})" if r.sme_name else ""
            lines.append(f"  --- {r.prompt_id}{sme_tag} → {r.decision} ---")

            if r.decision == "ERROR":
                lines.append(f"    Scoring failed: {r.decision_logic}")
            else:
                reasons = _extract_failure_reasons(r, rubric)
                if reasons:
                    for reason in reasons:
                        lines.append(f"    {reason}")
                else:
                    # Fallback: no specific reason flags found
                    lines.append(
                        f"    avg={r.average_score:.2f} — see decision_logic: "
                        f"{r.decision_logic[:120]}"
                    )
            lines.append("")

    # Cross-validation summary
    cv_results = [r for r in results if r.cross_validation]
    if cv_results:
        from collections import Counter as _Counter

        lines.append("## CROSS-VALIDATION SUMMARY (Logic/SC/GDrive vs Prompt)")
        lines.append(f"  Prompts with supporting evidence: {len(cv_results)}/{len(results)}")

        lq = _Counter(r.cross_validation.get("logic_quality", "MISSING") for r in cv_results)
        sq = _Counter(r.cross_validation.get("sc_quality", "MISSING") for r in cv_results)
        lines.append(f"  Logic quality: {dict(lq)}")
        lines.append(f"  SC quality:    {dict(sq)}")

        dl = sum(1 for r in cv_results if r.cross_validation.get("prompt_delivers_on_logic"))
        ds = sum(1 for r in cv_results if r.cross_validation.get("prompt_delivers_on_sc"))
        lines.append(f"  Prompt delivers on Logic: {dl}/{len(cv_results)}")
        lines.append(f"  Prompt delivers on SC:    {ds}/{len(cv_results)}")

        metas = [r for r in cv_results if r.cross_validation.get("is_meta_prompt")]
        if metas:
            lines.append(f"  ⚠ META-PROMPTS DETECTED: {[r.prompt_id for r in metas]}")

        all_gaps = []
        for r in cv_results:
            for gap in r.cross_validation.get("gaps", []):
                all_gaps.append(gap.get("type", "UNKNOWN") if isinstance(gap, dict) else "UNKNOWN")
        if all_gaps:
            lines.append(f"  Gap types found: {dict(_Counter(all_gaps))}")
        lines.append("")

        lines.append("  PER-PROMPT GAPS:")
        for r in cv_results:
            gaps = r.cross_validation.get("gaps", [])
            if gaps:
                lines.append(f"    {r.prompt_id} ({r.sme_name[:15]}):")
                for gap in gaps:
                    if isinstance(gap, dict):
                        lines.append(
                            f"      [{gap.get('type','?')}] {gap.get('description','')[:100]}"
                        )
                    else:
                        lines.append(f"      {str(gap)[:100]}")
        lines.append("")

    report_text = "\n".join(lines)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "prompt_eval_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def generate_audit_report(
    audit_results: List[AuditResult],
    rubric: Dict,
    output_dir: str,
) -> str:
    """Generate calibration audit report."""
    criteria_ids = get_criteria_ids(rubric)

    lines = [
        "=" * 70,
        "PROMPT SCORING CALIBRATION AUDIT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        f"Total Prompts Audited: {len(audit_results)}",
        "",
    ]

    agree_count = sum(1 for r in audit_results if r.decisions_agree)
    lines.append("## DECISION AGREEMENT")
    lines.append(
        f"  Decisions agree: {agree_count}/{len(audit_results)} "
        f"({100 * agree_count / len(audit_results):.1f}%)"
    )
    lines.append("")

    lines.append("## PER-CRITERION DISAGREEMENT")
    lines.append(f"{'Criterion':<35} {'Mean Δ':>7} {'# Disagree':>11} {'LLM Bias':>9}")
    lines.append("-" * 65)

    for cid in criteria_ids:
        deltas = []
        for ar in audit_results:
            llm_s = ar.llm_scores.get(cid, 0)
            hum_s = ar.human_scores.get(cid, 0)
            deltas.append(llm_s - hum_s)

        mean_delta = statistics.mean(deltas) if deltas else 0
        disagree = sum(1 for d in deltas if d != 0)
        bias = "stricter" if mean_delta < -0.3 else ("lenient" if mean_delta > 0.3 else "aligned")
        lines.append(
            f"  {cid:<33} {mean_delta:>+7.2f} {disagree:>11} {bias:>9}"
        )

    lines.append("")

    llm_avgs = [r.llm_average for r in audit_results]
    hum_avgs = [r.human_average for r in audit_results]
    if llm_avgs and hum_avgs:
        lines.append("## OVERALL SCORE COMPARISON")
        lines.append(f"  LLM mean average:   {statistics.mean(llm_avgs):.3f}")
        lines.append(f"  Human mean average: {statistics.mean(hum_avgs):.3f}")
        lines.append(
            f"  Mean delta:         {statistics.mean(llm_avgs) - statistics.mean(hum_avgs):+.3f}"
        )
        lines.append("")

    lines.append("## PER-PROMPT AUDIT DETAILS")
    for ar in audit_results:
        lines.append(f"\n  --- {ar.prompt_id} (SME: {ar.sme_name}) ---")
        lines.append(f"  LLM avg: {ar.llm_average:.2f} | Human avg: {ar.human_average:.2f}")
        lines.append(f"  LLM decision: {ar.llm_decision} | Human decision: {ar.human_decision}")
        if ar.disagreements:
            for dis in ar.disagreements:
                lines.append(
                    f"    {dis.get('criterion_id','?')}: "
                    f"LLM={dis.get('llm_score','?')} vs Human={dis.get('human_score','?')} "
                    f"(Δ={dis.get('delta','?')}) — {dis.get('explanation','')[:100]}"
                )
        if ar.calibration_note:
            lines.append(f"  Note: {ar.calibration_note}")

    report_text = "\n".join(lines)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / "calibration_audit_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def save_eval_results_csv(results: List[PromptEvalResult], output_dir: str):
    """Save evaluation results as CSV."""
    import csv

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    rows = []
    all_criteria = set()
    for r in results:
        row = {
            "prompt_id": r.prompt_id,
            "sme_name": r.sme_name,
            "prompt_type": r.prompt_type,
            "domain": r.domain,
            "average_score": f"{r.average_score:.3f}",
            "decision": r.decision,
            "gate_passed": r.gate_passed,
            "research_depth_tag": r.research_depth_tag or "",
            "type_appropriate": r.type_appropriate,
            "suggested_type": r.suggested_type or "",
            "n_runs": r.n_runs,
            "best_run_index": r.best_run_index,
            "per_run_averages": ";".join(f"{a:.3f}" for a in r.per_run_averages),
        }
        for ps in r.criterion_scores:
            row[ps.criterion_id] = ps.score
            all_criteria.add(ps.criterion_id)
        rows.append(row)

    base_cols = [
        "prompt_id", "sme_name", "prompt_type", "domain",
        "average_score", "decision", "gate_passed",
    ]
    crit_cols = sorted(all_criteria)
    extra_cols = [
        "research_depth_tag", "type_appropriate", "suggested_type",
        "n_runs", "best_run_index", "per_run_averages",
    ]
    all_cols = base_cols + crit_cols + extra_cols

    csv_path = output_path / "prompt_eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV saved to: {csv_path}")


def save_audit_results_csv(results: List[AuditResult], rubric: Dict, output_dir: str):
    """Save audit results as CSV."""
    import csv

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    criteria_ids = get_criteria_ids(rubric)
    rows = []
    for ar in results:
        row = {
            "prompt_id": ar.prompt_id,
            "sme_name": ar.sme_name,
            "llm_average": f"{ar.llm_average:.3f}",
            "human_average": f"{ar.human_average:.3f}",
            "delta_average": f"{ar.llm_average - ar.human_average:+.3f}",
            "llm_decision": ar.llm_decision,
            "human_decision": ar.human_decision,
            "decisions_agree": ar.decisions_agree,
            "num_disagreements": len(ar.disagreements),
        }
        for cid in criteria_ids:
            row[f"{cid}_llm"] = ar.llm_scores.get(cid, "")
            row[f"{cid}_human"] = ar.human_scores.get(cid, "")
            row[f"{cid}_delta"] = ar.llm_scores.get(cid, 0) - ar.human_scores.get(cid, 0)
        rows.append(row)

    base_cols = [
        "prompt_id", "sme_name", "llm_average", "human_average", "delta_average",
        "llm_decision", "human_decision", "decisions_agree", "num_disagreements",
    ]
    crit_cols = []
    for cid in criteria_ids:
        crit_cols.extend([f"{cid}_llm", f"{cid}_human", f"{cid}_delta"])

    csv_path = output_path / "calibration_audit_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_cols + crit_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Audit CSV saved to: {csv_path}")