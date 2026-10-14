"""
cli.py — Command-line entry point for the unified pipeline.

Examples:
    # Dry run (no API calls) on 3 rows, all default providers
    python -m dra_harness.cli --csv prompts.csv --max-rows 3

    # Live run, Claude + Qwen, web search on, 2 turns, save under ./out
    python -m dra_harness.cli --csv prompts.csv --live \
        --providers claude qwen --web-search --max-turns 2 --output-dir ./out

    # Override a model slug and enable the calculator tool
    python -m dra_harness.cli --csv prompts.csv --live \
        --model qwen=qwen/qwen3-235b-a22b --tools calculator
"""

from __future__ import annotations

from dataclasses import replace
import asyncio
import logging
import argparse

try:
    from .config import PipelineConfig, GenParams, load_env, DEFAULT_PROVIDERS
    from .pipeline import run_batch, save_results
except ImportError:  # run as a script from inside the package dir
    from config import PipelineConfig, GenParams, load_env, DEFAULT_PROVIDERS
    from pipeline import run_batch, save_results


def _build_config(args) -> PipelineConfig:
    defaults = GenParams()

    # Only override what was explicitly passed on CLI
    if args.temperature is not None:
        defaults = replace(defaults, temperature=args.temperature)
    if args.max_tokens is not None:
        defaults = replace(defaults, max_tokens=args.max_tokens)
    if args.max_turns is not None:
        defaults = replace(defaults, max_turns=args.max_turns)
    if args.max_cost is not None:
        defaults = replace(defaults, max_cost_usd=args.max_cost)
    if args.timeout is not None:
        defaults = replace(defaults, request_timeout=args.timeout)
    if args.web_search:
        defaults = replace(defaults, web_search=True)
    if args.no_file_output:
        defaults = replace(defaults, file_output=False)
    if args.tools is not None:
        defaults = replace(defaults, enabled_tools=args.tools or [])

    model_overrides: dict = {}
    for pair in (args.model or []):
        if "=" in pair:
            prov, slug = pair.split("=", 1)
            model_overrides[prov.strip()] = slug.strip()

    return PipelineConfig(
        providers=args.providers or list(DEFAULT_PROVIDERS),
        passes_per_provider=args.passes,
        max_concurrent=args.concurrency,
        dry_run=not args.live,
        staging_dir=args.staging_dir,
        output_dir=args.output_dir,
        resolve_files=args.resolve_files,
        defaults=defaults,
        model_overrides=model_overrides,
        cleanup_inputs=not args.keep_inputs,
    )


def _print_summary(output: dict) -> None:
    s = output["summary"]
    print(f"\n{'=' * 64}")
    print(f"  PIPELINE COMPLETE — {output['csv']}")
    print(f"{'=' * 64}")
    print(f"  Runs:      {s['total_runs']}  "
          f"(succeeded {s['succeeded']}, failed {s['failed']})")
    print(f"  Cost:      ${s['total_cost_usd']:.4f}")
    print(f"  Duration:  {output['duration_sec']:.1f}s")
    print(f"{'-' * 64}")
    for prov, slot in s["by_provider"].items():
        print(f"  {prov:10s}  {slot['succeeded']}/{slot['runs']} ok   "
              f"${slot['cost']:.4f}")
    print(f"{'=' * 64}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified OpenRouter response-generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="Path to the prompt CSV")
    parser.add_argument("--providers", nargs="*", default=None,
                        help=f"Providers (default: {DEFAULT_PROVIDERS})")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--task-ids", default=None, help="Comma-separated task ids")

    parser.add_argument("--live", action="store_true",
                        help="Make real API calls (default is dry-run)")
    parser.add_argument("--web-search", action="store_true")
    parser.add_argument("--no-file-output", action="store_true",
                        help="Skip file generation even if the prompt asks for it")
    parser.add_argument("--tools", nargs="*", default=None,
                        help="Local tool names to enable (e.g. calculator)")

    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-cost", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)

    parser.add_argument("--model", nargs="*", default=None,
                        help="Per-provider slug overrides, e.g. claude=anthropic/...")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--staging-dir", default="./staging")
    parser.add_argument("--resolve-files", action="store_true",
                        help="Download GDrive links to local files before running")
    parser.add_argument("--keep-inputs", action="store_true",
                        help="Keep the input files in the run folders and staging (default: delete them after the runs)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write the results JSON to disk")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    load_env()

    cfg = _build_config(args)
    task_ids = [t.strip() for t in args.task_ids.split(",")] if args.task_ids else None

    output = asyncio.run(run_batch(args.csv, cfg, max_rows=args.max_rows, task_ids=task_ids))

    _print_summary(output)

    if not args.no_save:
        path = save_results(output)
        print(f"  Results → {path}\n")


if __name__ == "__main__":
    main()