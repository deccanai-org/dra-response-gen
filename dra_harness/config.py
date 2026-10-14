"""
config.py — Single source of truth for every pipeline hyperparameter.

Design goal: ONE config object that exposes every knob the pipeline has,
with per-provider overrides so anything can change for a specific model.

    PipelineConfig
      ├── defaults: GenParams        ← base knobs for every provider
      ├── agent_overrides            ← {"qwen": {"max_turns": 2}, ...}
      ├── model_overrides            ← {"claude": "anthropic/claude-..."}
      └── batch/runtime knobs        ← providers, passes, concurrency, IO

Resolve effective knobs for a provider with:

    params = cfg.params_for("claude")   # GenParams with overrides applied
    slug   = cfg.model_for("claude")    # OpenRouter slug

Nothing here imports from the repo root — fully standalone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace, asdict
from typing import Optional


# ─── Minimal .env loader (standalone) ─────────────────────────────────────────

def load_env(extra_paths: Optional[list[str]] = None) -> None:
    """
    Populate os.environ from the first .env file found among common locations.

    Existing environment variables are never overwritten. Lines are simple
    KEY=VALUE pairs; blank lines and '#' comments are ignored; surrounding
    single/double quotes are stripped.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(here, ".env"),
        os.path.join(repo_root, ".env"),
        os.path.join(repo_root, "imp_files", ".env"),
    ]
    if extra_paths:
        candidates = list(extra_paths) + candidates

    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            continue
        break  # only load the first file found


# ─── Generation / per-provider knobs ──────────────────────────────────────────

@dataclass
class GenParams:
    """
    Every per-call knob. These are the things you tune per provider.
    Override any subset per provider via PipelineConfig.agent_overrides.
    """
    # ── Sampling ──────────────────────────────────────────────────────
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 32000
    reasoning_effort: str = "high"   # APEX "Thinking=High" parity

    # ── Agentic loop ──────────────────────────────────────────────────
    max_turns: int = 250            # APEX parity (was 150; last run used 20)

    # ── Web search (via OpenRouter) ───────────────────────────────────
    web_search: bool = False
    web_max_results: int = 5
    web_method: str = "plugins"   # "plugins" or "suffix" (model ":online")

    # ── Local tool calling ────────────────────────────────────────────
    enabled_tools: list[str] = field(default_factory=lambda: ["all"])  # names in tools.TOOL_REGISTRY
    tool_choice: str = "auto"     # "auto" | "none" | "required"

    # ── File creation (local code execution) ──────────────────────────
    file_output: bool = True       # honor task.output_formats when set
    file_fix_attempts: int = 3    # retries when generated code fails
    code_exec_timeout: int = 300  # seconds for the local subprocess

    # ── Reliability ───────────────────────────────────────────────────
    request_timeout: int = 1800    # per-request timeout (seconds)
    max_retries: int = 2          # transient-error retries per request
    max_cost_usd: float = 50.0    # budget guard per task/pass (safety net)

    def merged(self, overrides: dict) -> "GenParams":
        """Return a copy with the given field overrides applied."""
        valid = {k: v for k, v in (overrides or {}).items()
                 if k in self.__dataclass_fields__}
        return replace(self, **valid)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Top-level pipeline config ────────────────────────────────────────────────

DEFAULT_PROVIDERS = [
    # candidate models under test, October 2026 (all via OpenRouter → uniform local tools)
    "opus55", "gpt6_astra", "gpt61_sol", "gemini31_pro", "gemini38_flash", "grok47",   # frontier closed
    "kimi_k3", "glm53", "qwen38_max", "deepseek_v4_pro",                                # open weight
]
# Reference generators (Model_A / Model_B for the golden) — run these SEPARATELY,
# not mixed into a candidate sweep: providers=REFERENCE_PROVIDERS on that run.
REFERENCE_PROVIDERS = ["hunyuan", "doubao"]
# Capability ceiling — one pass, separately (most expensive model).
CEILING_PROVIDERS = ["fable51"]


@dataclass
class PipelineConfig:
    """
    The complete pipeline configuration. Construct one and pass it around.
    """
    # ── Batch shape ───────────────────────────────────────────────────
    providers: list[str] = field(default_factory=lambda: list(DEFAULT_PROVIDERS))
    passes_per_provider: int = 1
    max_concurrent: int = 4

    # ── Execution mode ────────────────────────────────────────────────
    dry_run: bool = False

    # ── IO ────────────────────────────────────────────────────────────
    staging_dir: str = "./staging"     # where GDrive files are downloaded
    output_dir: str = "./results"      # where results JSON + files are written
    resolve_files: bool = False        # download GDrive links to local files
    run_root: str = "./runs_dir"        # single root; each run gets a timestamped subfolder
    run_id: Optional[str] = None    # set at run start; names the per-run subfolder
    cleanup_inputs: bool = True     # delete input files from run folders and the batch staging folder after the runs

    # ── OpenRouter ────────────────────────────────────────────────────
    api_key: Optional[str] = None      # falls back to OPENROUTER_API_KEY
    base_url: str = "https://openrouter.ai/api/v1"

    # ── Per-provider knobs ────────────────────────────────────────────
    defaults: GenParams = field(default_factory=GenParams)
    # Qwen 27B emits empty output at temp=1.0; it needs 0.3 for agentic tasks.
    # Keys MUST match MODEL_REGISTRY keys exactly - an override under any other key is silently ignored
    # (Aug 2026: "qwen" did not apply to "qwen27b").
    agent_overrides: dict = field(
        default_factory=lambda: {
            # Anthropic / OpenAI reasoning models default to reasoning ON; on long agentic loops "high"
            # effort can run away on cost. Start at "medium" and raise only if quality drops on a real task.
            "fable51":    {"reasoning_effort": "medium"},
            "opus55":     {"reasoning_effort": "medium"},
            "gpt6_astra": {"reasoning_effort": "medium", "max_tokens": 64000},
            "gpt61_sol":  {"reasoning_effort": "medium", "max_tokens": 64000},
            "gpt6_luna":  {"reasoning_effort": "medium"},
            # Qwen 3.6 27B emitted empty output at temp=1.0 and needed 0.3 for agentic tasks; the same setting
            # is the starting point for the 3.8 generation until test_models.py shows otherwise.
            "qwen38_27b": {"temperature": 0.3},
            "qwen38_max": {"temperature": 0.3},   # reasoning is mandatory on this endpoint: keep reasoning_effort set
            # ── August 2026 generation (unchanged) ──
            "opus5":       {"reasoning_effort": "medium"},
            "sonnet":      {"reasoning_effort": "medium"},
            "gpt56_sol":   {"reasoning_effort": "medium", "max_tokens": 64000},
            "gpt56_terra": {"reasoning_effort": "medium"},
            "qwen27b":     {"temperature": 0.3},
        }
    )   # {provider: {field: val}}
    model_overrides: dict = field(default_factory=dict)   # {provider: "slug"}

    # ── Resolution helpers ────────────────────────────────────────────

    def params_for(self, provider: str) -> GenParams:
        """Effective GenParams for a provider (defaults + agent_overrides)."""
        return self.defaults.merged(self.agent_overrides.get(provider, {}))

    def model_for(self, provider: str) -> Optional[str]:
        """
        Override slug for a provider, if set. When None, the caller falls
        back to provider.MODEL_REGISTRY. Kept here so config has no import
        dependency on the provider module.
        """
        return self.model_overrides.get(provider)

    def to_dict(self) -> dict:
        d = asdict(self)
        # api_key is a secret — never serialize it
        d.pop("api_key", None)
        return d