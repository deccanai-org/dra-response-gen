"""Provider routing for the auditor / augmenter / adjudicator / crux judge.

One Claude model (Opus 5) is called direct on Anthropic first and, when Anthropic cannot serve it, through
OpenRouter — the SAME model, so the "never mix judge models within a run" guarantee in
prompt_evaluator.MODEL_FALLBACK_CHAIN still holds. Only the provider changes; if both fail, the caller's
hard-stop still fires.

Routing:
  * spend / usage / billing / quota limit on Anthropic -> OpenRouter for the REST OF THE PROCESS
    (a workspace cap does not clear by itself; e.g. "You have reached your specified workspace API usage limits")
  * overload / rate limit / 5xx / timeout              -> OpenRouter for AUDITOR_FALLBACK_COOLDOWN seconds
    (default 600), then Anthropic again
  * AUDITOR_PROVIDER=openrouter                         -> OpenRouter only
  * AUDITOR_OPENROUTER_MODEL                            -> override the OpenRouter model id
    (default 'anthropic/<model>', checked against OpenRouter's model list; an unknown id fails loudly)

Call shape parity: streaming on both providers, system prompt as a system message, max_tokens passed through,
no reasoning parameter unless the caller sets one (crux_judge sets effort='medium'). A response that comes back
empty because reasoning consumed max_tokens is retried once with 1.5x the budget, then raised — never returned
as an empty string.

Live check:  python -m src.llm_route --probe
"""
import os
import sys
import time

DEFAULT_MODEL = "claude-opus-5"

_PERSISTENT = ("usage limit", "spend", "billing", "credit", "quota", "reached your", "insufficient")
_TRANSIENT = ("overloaded", "529", "rate limit", "rate_limit", "429", "500", "502", "503", "504",
              "timeout", "timed out", "connection", "temporarily")

_state = {"until": 0.0, "announced": None, "or_model_checked": {}}


def _log(msg):
    print(f"  [llm-route] {msg}", file=sys.stderr, flush=True)


def _anthropic_down():
    return time.time() < _state["until"]


def _mark_down(exc):
    msg = str(exc).lower()
    persistent = any(s in msg for s in _PERSISTENT)
    secs = 10 ** 9 if persistent else float(os.environ.get("AUDITOR_FALLBACK_COOLDOWN", "600"))
    _state["until"] = time.time() + secs
    reason = "spend/usage limit" if persistent else "overload/rate limit/server error"
    if _state["announced"] != reason:
        _state["announced"] = reason
        _log(f"Anthropic unavailable ({reason}) — routing Claude calls to OpenRouter "
             + ("for the rest of this run" if persistent else f"for {int(secs)}s")
             + f". Error: {str(exc)[:160]}")


def _is_fallback_error(exc):
    msg = str(exc).lower()
    return any(s in msg for s in _PERSISTENT + _TRANSIENT)


def openrouter_model(model):
    """'claude-opus-5' -> 'anthropic/claude-opus-5', verified once against OpenRouter's model list."""
    import re as _re
    mid = os.environ.get("AUDITOR_OPENROUTER_MODEL") or (model if "/" in model else f"anthropic/{model}")
    if mid in _state["or_model_checked"]:
        return mid
    try:
        import requests
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
        ids = {m.get("id") for m in (r.json().get("data") or [])}
        # OpenRouter writes some versions with a dot: claude-sonnet-4-6 -> claude-sonnet-4.6
        dotted = _re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", mid)
        if ids and mid not in ids and dotted in ids and not os.environ.get("AUDITOR_OPENROUTER_MODEL"):
            _state["or_model_checked"][dotted] = True
            return dotted
        if ids and mid not in ids:
            raise RuntimeError(f"OpenRouter has no model '{mid}'. Set AUDITOR_OPENROUTER_MODEL to the exact id "
                               f"(candidates: {sorted(i for i in ids if 'claude' in str(i))[:8]}).")
    except RuntimeError:
        raise
    except Exception as e:                                  # listing unavailable: the call itself will fail loudly
        _log(f"could not verify the OpenRouter model id ({str(e)[:80]}); using '{mid}'")
    _state["or_model_checked"][mid] = True
    return mid


# ---------------------------------------------------------------------------------------------------
# provider calls (module-level so tests can stub them)
# ---------------------------------------------------------------------------------------------------
def _anthropic_call(model, prompt, system, max_tokens, extra):
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    kw = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if system:
        kw["system"] = system
    kw.update(extra or {})
    parts = []
    with client.messages.stream(**kw) as stream:
        for text in stream.text_stream:
            parts.append(text)
        try:
            stop = stream.get_final_message().stop_reason
        except Exception:
            stop = None
    return "".join(parts), ("length" if stop == "max_tokens" else stop)


def _openrouter_call(model, prompt, system, max_tokens, extra):
    from openai import OpenAI
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — cannot fall back from Anthropic")
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"usage": {"include": True}}
    effort = (extra or {}).get("_effort")
    if effort:
        body["reasoning"] = {"effort": effort}
    stream = client.chat.completions.create(model=openrouter_model(model), messages=msgs, max_tokens=max_tokens,
                                            stream=True, extra_body=body)
    parts, finish = [], None
    for ch in stream:
        if ch.choices:
            d = ch.choices[0].delta
            if getattr(d, "content", None):
                parts.append(d.content)
            finish = ch.choices[0].finish_reason or finish
    return "".join(parts), finish


def claude_complete(prompt, model=DEFAULT_MODEL, system=None, max_tokens=4000, anthropic_extra=None, effort=None):
    """Text from `model`: Anthropic direct, else OpenRouter (same model). Raises if both fail."""
    # Opus 5 thinks on BOTH providers and the thinking counts against max_tokens, so a call can stop
    # mid-answer (CCL via OpenRouter: 46,779 chars; BKNG direct on Anthropic, 6 Oct: 38-43k chars under a
    # 32k budget, all three runs). A response that stops at the token limit — empty OR cut off — is
    # retried ONCE with 1.5x the budget, capped at AUDITOR_MAX_OUTPUT_CEILING (default 96000 =
    # auditor.MAX_OUTPUT_CEILING), on whichever provider is serving the call.
    ceiling = int(os.environ.get("AUDITOR_MAX_OUTPUT_CEILING", "96000"))

    def _one(mt):
        if os.environ.get("AUDITOR_PROVIDER", "").lower() != "openrouter" and not _anthropic_down():
            try:
                return _anthropic_call(model, prompt, system, mt, anthropic_extra)
            except Exception as e:
                if not _is_fallback_error(e):
                    raise                                     # e.g. a malformed request: not a provider problem
                _mark_down(e)
        return _openrouter_call(model, prompt, system, mt, {"_effort": effort} if effort else None)

    mt = max_tokens
    text, finish = "", None
    for attempt in (1, 2):
        text, finish = _one(mt)
        empty = not (text and text.strip())
        cut = finish == "length"
        if not empty and not cut:
            return text
        if attempt == 1:
            new_mt = min(ceiling, int(mt * 1.5))
            if new_mt <= mt:
                break                                     # already at the ceiling: no room to retry
            _log(f"{provider_in_use()} response {'empty' if empty else 'cut off at the token limit'} "
                 f"(finish={finish}, {len(text or ''):,} chars, max_tokens={mt:,}) — retrying once with "
                 f"max_tokens={new_mt:,}")
            mt = new_mt
    if text and text.strip():
        _log(f"{provider_in_use()} response still cut off at max_tokens={mt:,}; returning it (the caller flags truncation)")
        return text
    raise RuntimeError(f"{provider_in_use()} returned no visible text for {model} even with max_tokens={mt} "
                       f"(finish={finish}). Raise max_tokens for this call.")


def provider_in_use():
    if os.environ.get("AUDITOR_PROVIDER", "").lower() == "openrouter" or _anthropic_down():
        return "openrouter"
    return "anthropic"


if __name__ == "__main__":
    if "--probe" in sys.argv:
        q = "Reply with exactly one word: OK"
        for label, env in (("auto (Anthropic first)", None), ("forced OpenRouter", "openrouter")):
            if env:
                os.environ["AUDITOR_PROVIDER"] = env
            try:
                t = claude_complete(q, max_tokens=2000, system="You answer tersely.")
                print(f"  OK    {label:<24} provider={provider_in_use():<10} reply={t.strip()[:20]!r}")
            except Exception as e:
                print(f"  FAIL  {label:<24} {type(e).__name__}: {str(e)[:160]}")