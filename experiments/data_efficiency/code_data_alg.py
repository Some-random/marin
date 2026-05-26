# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tokenize the OpenCoder algorithmic_corpus (Python QA pairs).

Source: gzipped JSONL produced by /tmp/convert_alg_to_jsonl.py from
OpenCoder-LLM/opc-annealing-corpus/algorithmic_corpus/* (54 files combined).

Content: ~all rows are type='qa', lang='python' — competitive-programming style
"Write a Python function to ..." prompts paired with markdown-formatted code
answers. Closest open analogue to Aryabumi's proprietary verified-Python set.

Run via:
    cd /fsx/users/dongweij/marin
    MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python -m \
        experiments.data_efficiency.code_data_alg
"""

from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer


RAW = "/fsx/users/dongweij/marin/outputs/raw/opc_algorithmic.jsonl.gz"


opc_algorithmic_tokenized = default_tokenize(
    name="opc_algorithmic",
    dataset=RAW,
    tokenizer=llama3_tokenizer,
)


if __name__ == "__main__":
    executor_main(steps=[opc_algorithmic_tokenized])
