"""
provider.py — One unified driver for ALL providers via OpenRouter.

Because Claude, OpenAI, Gemini and Qwen are all reached through OpenRouter,
they share the OpenAI-compatible Chat Completions API. So a SINGLE driver
handles every provider — only the model slug changes.

    MODEL_REGISTRY   logical name → OpenRouter slug + capability flags (editable)
    resolve_slug()   pick the slug for a provider (config override wins)
    OpenRouterDriver async .chat() → normalized ChatResponse

Web search and tool calling are passed through as standard OpenRouter
features, controlled entirely by GenParams.
"""

from __future__ import annotations

import os
import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

try:
    from .config import GenParams
except ImportError:  # run as a script from inside the package dir
    from config import GenParams


logger = logging.getLogger("dra.provider")


# ─── Model registry (EDIT THESE SLUGS AS NEEDED) ──────────────────────────────
# Logical provider name → OpenRouter model slug + capabilities.
# Slugs are placeholders for sensible current models; change freely.

MODEL_REGISTRY: dict[str, dict] = {
    # Slugs verified against OpenRouter's live /api/v1/models listing on 13 Oct 2026.

    # ── Ceiling reference (run separately: CEILING_PROVIDERS in config.py) ──────
    "fable51":     {"slug": "anthropic/claude-fable-5.1",       "supports_tools": True},

    # ── Tier 1 — frontier closed ─────────────────────────────────────────────────
    "opus55":      {"slug": "anthropic/claude-opus-5.5",        "supports_tools": True},
    "gpt6_astra":  {"slug": "openai/gpt-6-astra",               "supports_tools": True},
    "gpt61_sol":   {"slug": "openai/gpt-6.1-sol",               "supports_tools": True},
    "gemini31_pro": {"slug": "google/gemini-3.1-pro-preview",   "supports_tools": True},   # still Google's newest Pro
    "gemini38_flash": {"slug": "google/gemini-3.8-flash",       "supports_tools": True},
    "grok47":      {"slug": "x-ai/grok-4.7",                    "supports_tools": True},

    # ── Tier 2 — balanced closed ─────────────────────────────────────────────────
    "gpt6_luna":   {"slug": "openai/gpt-6-luna",                "supports_tools": True},

    # ── Tier 3 — open weight ─────────────────────────────────────────────────────
    "kimi_k3":     {"slug": "moonshotai/kimi-k3",               "supports_tools": True},
    "glm53":       {"slug": "z-ai/glm-5.3",                     "supports_tools": True},
    "qwen38_max":  {"slug": "qwen/qwen3.8-max-0902",            "supports_tools": True},   # reasoning is mandatory on this endpoint
    "qwen38_27b":  {"slug": "qwen/qwen3.8-27b",                 "supports_tools": True},
    # 'deepseek/deepseek-v4-pro' is the April 0423 build; the August GA model is pinned below.
    "deepseek_v4_pro": {"slug": "deepseek/deepseek-v4-pro-0813", "supports_tools": True},
    "deepseek_v41_flash": {"slug": "deepseek/deepseek-v4.1-flash", "supports_tools": True},

    # ── Incumbent reference slot (Tencent deal context) — Model_A / Model_B ─────
    # Kept as fixed reference generators regardless of leaderboard rank.
    "hunyuan":     {"slug": "tencent/hy4-preview",              "supports_tools": True},
    # Seed 2.1 PRO is not on OpenRouter (only Seed 2.1 Turbo is): stays on CometAPI.
    "doubao":      {"slug": "doubao-seed-2-1-pro-260628",       "supports_tools": True,
                    "base_url": "https://api.cometapi.com/v1",
                    "api_key_env": "COMETAPI_KEY"},

    # ── August 2026 generation (kept so earlier runs stay reproducible) ──────────
    "opus5":       {"slug": "anthropic/claude-opus-5",          "supports_tools": True},
    "gpt56_sol":   {"slug": "openai/gpt-5.6-sol",               "supports_tools": True},
    "gpt56_terra": {"slug": "openai/gpt-5.6-terra",             "supports_tools": True},
    "sonnet":      {"slug": "anthropic/claude-sonnet-5",        "supports_tools": True},
    "grok46":      {"slug": "x-ai/grok-4.6",                    "supports_tools": True},
    "deepseek_v4_flash": {"slug": "deepseek/deepseek-v4-flash", "supports_tools": True},
    "glm52":       {"slug": "z-ai/glm-5.2",                     "supports_tools": True},
    "qwen27b":     {"slug": "qwen/qwen3.6-27b",                 "supports_tools": True},
    "hunyuan_hy3": {"slug": "tencent/hy3",                      "supports_tools": True},
    # DISQUALIFIED (Aug 2026: fabricates boilerplate, narrates tool use as prose, never reads files) - never in a sweep
    "qwen":        {"slug": "qwen/qwen3.6-35b-a3b",             "supports_tools": True},

    # ── Legacy aliases ("claude"/"openai"/"gemini") → the current tier for that vendor ──
    "claude":      {"slug": "anthropic/claude-opus-5.5",        "supports_tools": True},
    "openai":      {"slug": "openai/gpt-6.1-sol",               "supports_tools": True},
    "gemini":      {"slug": "google/gemini-3.1-pro-preview",    "supports_tools": True},
}

def driver_for(provider: str, dry_run: bool = False) -> "OpenRouterDriver":
    """Create the right driver for a provider (OpenRouter or CometAPI)."""
    entry = MODEL_REGISTRY.get(provider, {})
    base_url = entry.get("base_url", "https://openrouter.ai/api/v1")
    key_env = entry.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(key_env)
    return OpenRouterDriver(api_key=api_key, base_url=base_url, dry_run=dry_run)


def resolve_slug(provider: str, model_override: Optional[str] = None) -> str:
    """Resolve the OpenRouter slug for a provider. Override wins over registry."""
    if model_override:
        return model_override
    entry = MODEL_REGISTRY.get(provider)
    if not entry:
        raise ValueError(
            f"Unknown provider '{provider}'. Known: {list(MODEL_REGISTRY)}. "
            f"Add it to MODEL_REGISTRY or pass a model_override."
        )
    return entry["slug"]


def supports_tools(provider: str) -> bool:
    return bool(MODEL_REGISTRY.get(provider, {}).get("supports_tools", False))


# ─── Normalized response ──────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)     # [{id, name, arguments(dict)}]
    assistant_message: dict = field(default_factory=dict)  # to append to messages
    citations: list = field(default_factory=list)      # [{url, title, snippet}]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str = ""


def _extract_text(msg) -> str:
    """Pull the assistant's visible text out of an OpenAI-compatible message,
    robustly across providers. Three shapes seen on OpenRouter:
      1. content is a plain string (OpenAI, Anthropic) — use it.
      2. content is a LIST of parts [{type:'text', text:...}] (Gemini, others) —
         concatenate the text parts.
      3. content is empty/None but a THINKING model answered during reasoning
         (DeepSeek, GLM, Kimi, Qwen, Gemini w/ reasoning_effort) — the visible
         text is in 'reasoning' / 'reasoning_content'. Fall back to it, so the
         turn is not silently empty (nonzero output tokens but text == "").
    Falls back to reasoning ONLY when content is genuinely empty, so a real
    answer is never overwritten by the thinking trace.
    """
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text") or p.get("content") or ""
            else:
                t = getattr(p, "text", "") or getattr(p, "content", "") or ""
            if t:
                parts.append(t)
        if parts:
            return "\n".join(parts)
    for attr in ("reasoning_content", "reasoning"):
        r = getattr(msg, attr, None)
        if isinstance(r, str) and r.strip():
            return r
        if isinstance(r, list):
            rp = []
            for x in r:
                t = (x.get("text") if isinstance(x, dict)
                     else getattr(x, "text", None))
                if t:
                    rp.append(t)
            if rp:
                return "\n".join(rp)
    return content if isinstance(content, str) else ""


# ─── The driver ───────────────────────────────────────────────────────────────

class OpenRouterDriver:
    """
    Thin async wrapper over the OpenAI-compatible OpenRouter endpoint.

    One instance is reused across all providers and passes (connection pooling).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        dry_run: bool = False,
    ):
        self.base_url = base_url
        self.dry_run = dry_run
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._client = None

        if not dry_run:
            if not self.api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY not set. Export it, put it in a .env "
                    "(see config.load_env), or pass api_key=..., or use dry_run."
                )
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("pip install openai  (used for the OpenRouter API)")
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            self._openai = __import__("openai")

    # ── Public call ───────────────────────────────────────────────────────

    async def chat(
        self,
        model_slug: str,
        messages: list[dict],
        params: GenParams,
        tools_schemas: Optional[list[dict]] = None,
    ) -> ChatResponse:
        """One OpenAI-compatible chat completion, with retries and normalization."""
        if self.dry_run:
            raise RuntimeError("chat() called in dry_run mode — runner should mock instead")

        slug = model_slug
        extra_body: dict = {"usage": {"include": True}}   # ask OpenRouter for cost

        # ── Web search ────────────────────────────────────────────────
        if params.web_search:
            if params.web_method == "suffix":
                if not slug.endswith(":online"):
                    slug = f"{slug}:online"
            else:  # "plugins"
                extra_body["plugins"] = [
                    {"id": "web", "max_results": params.web_max_results}
                ]

        kwargs: dict = dict(
            model=slug,
            messages=messages,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            timeout=params.request_timeout,
            extra_body=extra_body,
        )

        # ── Tools ─────────────────────────────────────────────────────
        if tools_schemas:
            kwargs["tools"] = tools_schemas
            kwargs["tool_choice"] = params.tool_choice
        
        # ── Reasoning effort ──────────────────────────────────────────
        if params.reasoning_effort:
            kwargs["extra_body"]["reasoning_effort"] = params.reasoning_effort

        # ── Call with retries ─────────────────────────────────────────
        last_err: Optional[Exception] = None
        for attempt in range(params.max_retries + 1):
            try:
                resp = await self._client.chat.completions.create(**kwargs)
                return self._normalize(resp)
            except Exception as e:  # noqa: BLE001 — surface after retries
                last_err = e
                transient = self._is_transient(e)
                logger.warning(
                    "chat attempt %d/%d failed (%s): %s",
                    attempt + 1, params.max_retries + 1,
                    "transient" if transient else "fatal", e,
                )
                if not transient or attempt == params.max_retries:
                    break
                await asyncio.sleep(min(2 ** attempt, 15))

        raise last_err if last_err else RuntimeError("chat failed with no exception")

    # ── Normalization ─────────────────────────────────────────────────────

    def _normalize(self, resp) -> ChatResponse:
        out = ChatResponse()

        choice = (resp.choices or [None])[0]
        if choice is None:
            return out

        msg = choice.message
        out.text = _extract_text(msg)
        out.finish_reason = getattr(choice, "finish_reason", "") or ""

        # Tool calls
        raw_tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in raw_tool_calls:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn else ""
            args_raw = getattr(fn, "arguments", "") if fn else ""
            try:
                args = json.loads(args_raw) if args_raw else {}
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": args_raw}
            out.tool_calls.append({
                "id": getattr(tc, "id", ""),
                "name": name,
                "arguments": args,
            })

        # Assistant message to append back to the conversation
        out.assistant_message = self._assistant_message_dict(msg)

        # Citations (web search annotations / top-level)
        out.citations = self._extract_citations(msg, resp)

        # Usage + cost
        usage = getattr(resp, "usage", None)
        if usage:
            out.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            out.output_tokens = getattr(usage, "completion_tokens", 0) or 0
            cost = getattr(usage, "cost", None)
            if cost is None and isinstance(usage, dict):
                cost = usage.get("cost")
            out.cost_usd = float(cost) if cost is not None else 0.0

        return out

    @staticmethod
    def _assistant_message_dict(msg) -> dict:
        """Rebuild the assistant message as a dict for message history."""
        d: dict = {"role": "assistant", "content": _extract_text(msg)}
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            d["tool_calls"] = []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                d["tool_calls"].append({
                    "id": getattr(tc, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(fn, "name", "") if fn else "",
                        "arguments": getattr(fn, "arguments", "") if fn else "",
                    },
                })
        return d

    @staticmethod
    def _extract_citations(msg, resp) -> list:
        citations: list = []
        seen: set = set()

        def add(url, title="", snippet=""):
            if url and url not in seen:
                citations.append({"url": url, "title": title or "", "snippet": snippet or ""})
                seen.add(url)

        # 1) Message annotations (OpenRouter web plugin)
        for ann in (getattr(msg, "annotations", None) or []):
            url_cit = getattr(ann, "url_citation", None)
            if url_cit is not None:
                add(getattr(url_cit, "url", ""),
                    getattr(url_cit, "title", ""),
                    getattr(url_cit, "content", ""))
            elif isinstance(ann, dict):
                uc = ann.get("url_citation", {})
                add(uc.get("url", ""), uc.get("title", ""), uc.get("content", ""))

        # 2) Top-level citations array (some providers)
        for cit in (getattr(resp, "citations", None) or []):
            if isinstance(cit, str):
                add(cit)
            elif isinstance(cit, dict):
                add(cit.get("url", ""), cit.get("title", ""), cit.get("snippet", ""))

        return citations

    def _is_transient(self, e: Exception) -> bool:
        """Heuristic: retry on rate limits, timeouts, and 5xx."""
        oa = getattr(self, "_openai", None)
        if oa is not None:
            for cls_name in ("RateLimitError", "APITimeoutError",
                             "InternalServerError", "APIConnectionError"):
                cls = getattr(oa, cls_name, None)
                if cls and isinstance(e, cls):
                    return True
            api_err = getattr(oa, "APIStatusError", None)
            if api_err and isinstance(e, api_err):
                status = getattr(e, "status_code", 0) or 0
                return status >= 500 or status == 429
        return False