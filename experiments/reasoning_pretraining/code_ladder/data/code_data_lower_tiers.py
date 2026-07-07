# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize Stack-Edu Python source for the LOWER quality bands ([2.7, 2.8)
and [2.5, 2.7)) — feeds the 25 B code-only base training (`run_1_4b_code25b.py`).

Per-band caches:
  c5_25b_se_python_mid : score in [2.7, 2.8)  (fetched in `stack-edu-python-content-mid`)
  c5_25b_se_python_low2: score in [2.5, 2.7)  (fetched in `stack-edu-python-content-lower2`)

Each band is fetched via `experiments.reasoning_pretraining.code_ladder.data.fetch_stack_edu_python_score_range`
in 8 rank subdirs holding `se_python_low_*.jsonl.gz` shards (the fetch script
uses the `se_python_low_` filename prefix regardless of band — we override by
pointing at the band-specific out-dir).

Run via:
    cd /fsx/users/dongweij/marin
    MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python -m experiments.reasoning_pretraining.code_ladder.data.code_data_lower_tiers
"""

import glob
import os

from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer


def _ensure_validation_symlink(prefix: str) -> None:
    """Levanter opens validation split even when num_validation_sequences=0.

    Default tokenize only writes train/. Add validation -> train symlink to
    each cache matching `<prefix>-*` after tokenize.
    """
    for cache in glob.glob(f"/fsx/users/dongweij/marin/outputs/tokenized/{prefix}-*"):
        link = os.path.join(cache, "validation")
        if not os.path.lexists(link):
            os.symlink("train", link)


SE_PYTHON_MID_GLOB = "/fsx/users/dongweij/marin/outputs/raw/stack-edu-python-content-mid/rank_*/se_python_low_*.jsonl.gz"
SE_PYTHON_LOW2_GLOB = "/fsx/users/dongweij/marin/outputs/raw/stack-edu-python-content-lower2/rank_*/se_python_low_*.jsonl.gz"


stack_edu_python_mid = default_tokenize(
    name="c5_25b_se_python_mid",
    dataset=SE_PYTHON_MID_GLOB,
    tokenizer=llama3_tokenizer,
)


stack_edu_python_low2 = default_tokenize(
    name="c5_25b_se_python_low2",
    dataset=SE_PYTHON_LOW2_GLOB,
    tokenizer=llama3_tokenizer,
)


if __name__ == "__main__":
    executor_main(steps=[stack_edu_python_mid, stack_edu_python_low2])
    _ensure_validation_symlink("c5_25b_se_python_mid")
    _ensure_validation_symlink("c5_25b_se_python_low2")
