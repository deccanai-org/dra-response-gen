# src/augment_score_config.py
"""
Single source of truth for augment + score paths and knobs. Every folder/file
name is here (or a CLI override) so renaming staging/results/csv touches ONE
place, never module code. Mirrors the repo's config.py + cli.py convention:
config is the default, CLI flags are selective overrides via dataclasses.replace.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class AugScoreConfig:
    # inputs
    csv_path: str = "prompt_data.csv"          # SME prompt-package CSV (augmenter input)
    augmented_csv: str = "output/augmented/augmented_prompt_packages.csv"
    results_glob: str = "results/*.json"       # harness results (scorer input); path OR glob

    # staging resolution (see path resolver): stored paths look like
    # "./staging/<task>/runs/<provider>__p<n>/<file>". staging_prefix is the
    # leading segment we rebase onto staging_root when the folder is renamed.
    staging_remap_from: str = "staging"        # stored staging folder name in results JSON
    staging_remap_to: str = "staging_1"        # actual staging folder name on disk

    # outputs
    out_dir_augment: str = "output/augmented"
    out_dir_score: str = "output/scores"

    # model + concurrency
    model: str = "claude-opus-5"
    workers: int = 4                           # parallel grading threads (I/O bound)

    def override(self, **kw) -> "AugScoreConfig":
        clean = {k: v for k, v in kw.items() if v is not None}
        return replace(self, **clean)


DEFAULT = AugScoreConfig()