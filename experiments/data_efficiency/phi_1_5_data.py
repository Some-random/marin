# Tokenize the three components of the phi-1.5-style training mix.
#
# Components (raw data already on disk under outputs/raw/phi_1_5_mix/):
#   1. cosmopedia-v2          (~28 B tokens, synthetic NL textbooks, parquet)
#   2. python_edu_content     (~3-4 B tokens, filtered Python edu code, jsonl.gz)
#   3. code_exercises_text    (~120 M tokens, GPT-3.5 Python problem+solution, jsonl.gz)
#
# All three use the same Llama-3.1-8B tokenizer as our prior DCLM runs so we can
# share components across data mixes and pretokenize once.
#
# Run via:
#   cd /fsx/users/dongweij/marin
#   MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python -m \
#       experiments.data_efficiency.phi_1_5_data

from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

RAW_ROOT = "/fsx/users/dongweij/marin/outputs/raw/phi_1_5_mix"


cosmopedia_v2_tokenized = default_tokenize(
    name="phi_1_5/cosmopedia_v2",
    dataset=f"{RAW_ROOT}/cosmopedia-v2",
    tokenizer=llama3_tokenizer,
)

python_edu_tokenized = default_tokenize(
    name="phi_1_5/python_edu",
    # rank_NN/python_edu_NNNNN.jsonl.gz from the S3 fetcher
    dataset=f"{RAW_ROOT}/python_edu_content",
    tokenizer=llama3_tokenizer,
)

code_exercises_tokenized = default_tokenize(
    name="phi_1_5/code_exercises",
    dataset=f"{RAW_ROOT}/code_exercises_text",
    tokenizer=llama3_tokenizer,
)


if __name__ == "__main__":
    executor_main(
        steps=[
            cosmopedia_v2_tokenized,
            python_edu_tokenized,
            code_exercises_tokenized,
        ]
    )
