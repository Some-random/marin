# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the clean code sources for C5-v2.

Four sources fed into the 80% code + 20% markup recipe:
  1. stack_edu_python_clean        — Stack-Edu Python @ score > 3.0, content fetched from SWH
  2. stack_edu_markdown_clean      — Stack-Edu Markdown @ score > 3.0, content fetched from SWH
  3. nemotron_code_concepts        — Nemotron synthetic Python from concept taxonomies
  4. nemotron_unconditional_alg    — Nemotron synthetic Python from minimal prompts

Each step produces a tokenized cache under outputs/tokenized/c5v2_<name>/.

Run via:
    cd /fsx/users/dongweij/marin
    MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python -m experiments.reasoning_pretraining.code_ladder.data.code_data_c5v2
"""

from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer


RAW = "/fsx/users/dongweij/marin/outputs/raw"


stack_edu_python_clean = default_tokenize(
    name="c5v2_stack_edu_python_clean",
    dataset=f"{RAW}/stack-edu-python-content/content.jsonl.gz",
    tokenizer=llama3_tokenizer,
)

stack_edu_markdown_clean = default_tokenize(
    name="c5v2_stack_edu_markdown_clean",
    dataset=f"{RAW}/stack-edu-markdown-content/content.jsonl.gz",
    tokenizer=llama3_tokenizer,
)

nemotron_code_concepts = default_tokenize(
    name="c5v2_nemotron_code_concepts",
    dataset=f"{RAW}/nemotron_code_concepts.jsonl.gz",
    tokenizer=llama3_tokenizer,
)

nemotron_unconditional_alg = default_tokenize(
    name="c5v2_nemotron_unconditional_algorithmic",
    dataset=f"{RAW}/nemotron_unconditional_algorithmic.jsonl.gz",
    tokenizer=llama3_tokenizer,
)


if __name__ == "__main__":
    executor_main(steps=[
        stack_edu_python_clean,
        stack_edu_markdown_clean,
        nemotron_code_concepts,
        nemotron_unconditional_alg,
    ])
