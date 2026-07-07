"""Tokenize the filtered SlimPajama NL jsonl.gz files into a Levanter cache.

Inputs:
  /fsx/users/dongweij/marin/outputs/raw/slimpajama_nl/*.jsonl.gz (128 files,
  ~22 B Llama-3.1 tokens after filtering Github + StackExchange).

Output:
  /fsx/users/dongweij/marin/outputs/tokenized/slimpajama_nl/...

Run:
  cd /fsx/users/dongweij/marin
  MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python -m experiments.reasoning_pretraining.code_ladder.data.slimpajama_tokenize
"""

from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer


RAW_GLOB = "/fsx/users/dongweij/marin/outputs/raw/slimpajama_nl/*.jsonl.gz"


slimpajama_nl_tokenized = default_tokenize(
    name="slimpajama_nl",
    dataset=RAW_GLOB,
    tokenizer=llama3_tokenizer,
)


if __name__ == "__main__":
    executor_main(steps=[slimpajama_nl_tokenized])
