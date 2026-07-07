# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize per-language StarCoderData jsonl.gz files (produced by
`download_starcoderdata.py`) with the Llama-3.1 tokenizer (same as A5/B4).

Inputs (15 jsonl.gz files):
    {RAW}/starcoderdata/<lang>.jsonl.gz  for each lang in
        Stack:  java, javascript, php, python, c-sharp, typescript, c, cpp, go, ruby
        Markup: markdown, yaml, json, html, css

Outputs (15 tensorstore caches, one per language):
    {TOKENIZED}/stack_<lang>-<hash>/        (10 languages)
    {TOKENIZED}/markup_<lang>-<hash>/       (5 markup types)

Run via:
    cd /fsx/users/dongweij/marin
    MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python -m \\
        experiments.reasoning_pretraining.code_ladder.data.tokenize_starcoderdata
"""

from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer


RAW = "/fsx/users/dongweij/marin/outputs/raw/starcoderdata"

STACK_LANGS = [
    "java", "javascript", "php", "python", "c-sharp", "typescript",
    "c", "cpp", "go", "ruby",
]
MARKUP_LANGS = ["markdown", "yaml", "json", "html", "css"]


def _stack_step(lang: str):
    safe_name = lang.replace("-", "_")
    return default_tokenize(
        name=f"stack_{safe_name}",
        dataset=f"{RAW}/{lang}.jsonl.gz",
        tokenizer=llama3_tokenizer,
    )


def _markup_step(lang: str):
    return default_tokenize(
        name=f"markup_{lang}",
        dataset=f"{RAW}/{lang}.jsonl.gz",
        tokenizer=llama3_tokenizer,
    )


steps = (
    [_stack_step(l) for l in STACK_LANGS]
    + [_markup_step(l) for l in MARKUP_LANGS]
)


if __name__ == "__main__":
    executor_main(steps=steps)
