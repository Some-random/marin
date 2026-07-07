# Tokenize the matched-epochs corrected slices for the v2 code-mix experiment.
#
# Inputs (produced by preprocess_dclm_opc_subsamples.py):
#   dclm_150m.jsonl.gz           — 121,500 docs, ~150 M tokens (Llama-3.1-8B tokenizer)
#   opc_algorithmic_50m.jsonl.gz — 282,500 docs, ~50 M tokens

from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer


dclm_150m_tokenized = default_tokenize(
    name="dclm_150m",
    dataset="/fsx/users/dongweij/marin/outputs/raw/dclm_150m.jsonl.gz",
    tokenizer=llama3_tokenizer,
)

opc_algorithmic_50m_tokenized = default_tokenize(
    name="opc_algorithmic_50m",
    dataset="/fsx/users/dongweij/marin/outputs/raw/opc_algorithmic_50m.jsonl.gz",
    tokenizer=llama3_tokenizer,
)


if __name__ == "__main__":
    executor_main(
        steps=[
            dclm_150m_tokenized,
            opc_algorithmic_50m_tokenized,
        ]
    )
