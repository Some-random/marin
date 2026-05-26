# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tokenize local code datasets for Aryabumi-style code-mix experiments at 1.4B.

Three local jsonl.gz sources (produced by /tmp/convert_to_jsonl.py):
  1. code_web.jsonl.gz             — codeparrot/github-code style, 5 parquets, mixed languages
  2. code_synth_solution.jsonl.gz  — OpenCodeReasoning `solution` field (pure code, formally-verified-ish)
  3. code_synth_full.jsonl.gz      — OpenCodeReasoning `input + output + solution` (Phi-style: code+reasoning+NL)

Outputs to /fsx/users/dongweij/marin/outputs/tokenized/aryabumi_*

Run via:
    cd /fsx/users/dongweij/marin
    MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python -m experiments.data_efficiency.code_data
"""

from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer


RAW = "/fsx/users/dongweij/marin/outputs/raw"


def _local(file_path: str):
    """Return a local file URI that marin's tokenize step can consume."""
    return file_path


# === Tokenize each code source ===
# Each step reads a local jsonl.gz and writes a tokenized cache under outputs/tokenized/aryabumi_<name>/

code_web_tokenized = default_tokenize(
    name="aryabumi_code_web",
    dataset=_local(f"{RAW}/code_web.jsonl.gz"),
    tokenizer=llama3_tokenizer,
)

code_synth_solution_tokenized = default_tokenize(
    name="aryabumi_code_synth_solution",
    dataset=_local(f"{RAW}/code_synth_solution.jsonl.gz"),
    tokenizer=llama3_tokenizer,
)

code_synth_full_tokenized = default_tokenize(
    name="aryabumi_code_synth_full",
    dataset=_local(f"{RAW}/code_synth_full.jsonl.gz"),
    tokenizer=llama3_tokenizer,
)


if __name__ == "__main__":
    executor_main(steps=[
        code_web_tokenized,
        code_synth_solution_tokenized,
        code_synth_full_tokenized,
    ])
