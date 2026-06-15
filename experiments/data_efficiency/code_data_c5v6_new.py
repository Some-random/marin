# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the fresh Stack-Edu Python source for C5-v6-NEW phase 2.

We fetched Stack-Edu Python blobs in score-range [2.8, 3.0) via SWH — DISJOINT
from the existing c5v2_stack_edu_python_clean cache (score >= 3.0). Tokenizing
those into a separate cache lets C5-v6-NEW phase 2 read genuinely-new Python
docs starting at sequence-index 0 (no offset wrap-around).

The other phase-2 code+markup components reuse the existing caches with an
explicit per-component `offset` (set in run_1_4b_c5v6_phase2_new.py).

Run via:
    cd /fsx/users/dongweij/marin
    MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python -m experiments.data_efficiency.code_data_c5v6_new
"""

from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer


# 8 rank subdirs each holding se_python_low_*.jsonl.gz shards
SE_PYTHON_LOW_GLOB = "/fsx/users/dongweij/marin/outputs/raw/stack-edu-python-content-low/rank_*/se_python_low_*.jsonl.gz"


stack_edu_python_low = default_tokenize(
    name="c5v6new_stack_edu_python_low",
    dataset=SE_PYTHON_LOW_GLOB,
    tokenizer=llama3_tokenizer,
)


if __name__ == "__main__":
    executor_main(steps=[stack_edu_python_low])
