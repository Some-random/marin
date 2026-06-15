# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
1.4B Aryabumi-inspired code-mix probe at our scale.

Tests the H1 active hypothesis: does adding 25% high-quality Python code data
to a DCLM-only pretraining mix improve NL performance at 1.4B / 3.34B-token scale?

Data: OpenCoder-LLM/opc-annealing-corpus/algorithmic_corpus subset only —
Python-only competitive-programming style Q&A pairs (closest open analogue to
Aryabumi's proprietary "Python programming problems formally verified").

Diffs from run_1_4b_wd1_6_x16_nocrossblock.py (our non-looping text-only baseline):
  - Add `opc_algorithmic` component with train_weight 0.25
  - Reduce dclm_200m train_weight from 1.0 to 0.75
  - Checkpoint path: checkpoints/1_4b_25code_alg/
  - Tags: aryabumi-probe, code-alg

All other hyperparams identical to the wd=1.6/x16/block=False baseline.

Reference baseline for comparison: `peach-thunder-100` / `6xx0hu3l`
(run_1_4b_wd1_6_x16_nocrossblock.py — same recipe, 0% code).

Success criteria (per CURRENT GOALS section of EXPERIMENT_LOG):
  - Paloma macro PPL strictly improves (lower loss vs baseline) AND
  - ≥2 of {arc_easy, sciq, piqa, boolq} strictly improve AND
  - No selected benchmark regresses below baseline-minus-noise

These 4 are the only above-random benchmarks at our scale (see Step 12 of
the experiment log for the analysis).

Scale caveats acknowledged:
  - Aryabumi trained 470M/2.8B for 200B tokens (60x more than ours)
  - The aggregate +8.2% NL reasoning claim is NOT measurable at our scale
    because most benchmarks are at-random
  - A null result here would NOT refute Aryabumi — different data, different scale
"""

import os
from datetime import timedelta
from pathlib import Path

_secrets = Path(__file__).resolve().parents[2] / ".secrets"
if _secrets.exists():
    for line in _secrets.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig, DatasetComponent
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from levanter.trainer import TrainerConfig

from experiments.data_efficiency.models import model_dict

BASE_TOKENIZED = "/fsx/users/dongweij/marin/outputs/tokenized"

DCLM_TRAIN = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_train-d321eb"
DCLM_VAL = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_val-415aea"


def _code_cache_dir(name_prefix: str) -> str:
    base = Path(BASE_TOKENIZED)
    matches = sorted(base.glob(f"{name_prefix}-*"))
    if not matches:
        raise FileNotFoundError(
            f"Tokenized code dataset '{name_prefix}-*' not found in {BASE_TOKENIZED}. "
            f"Run `MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python -m "
            f"experiments.data_efficiency.code_data_alg` first."
        )
    return str(matches[-1])


OPC_ALGORITHMIC = _code_cache_dir("opc_algorithmic")

PALOMA_SUBSETS = [
    "4chan", "c4_100_domains", "c4_en", "dolma-v1_5",
    "dolma_100_programing_languages", "dolma_100_subreddits", "falcon-refinedweb",
    "gab", "m2d2_s2orc_unsplit", "m2d2_wikipedia_unsplit", "manosphere_meta_sep",
    "mc4", "ptb", "redpajama", "twitterAAE_HELM_fixed", "wikitext_103",
]


def _paloma_components() -> dict[str, DatasetComponent]:
    paloma_dir = Path(f"{BASE_TOKENIZED}/paloma")
    found = {}
    for sub in PALOMA_SUBSETS:
        matches = sorted(paloma_dir.glob(f"{sub}-*"))
        if not matches:
            raise FileNotFoundError(f"Paloma subset '{sub}' not tokenized.")
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple caches match {sub!r}: " + str([p.name for p in matches]) + ". "
                "Pin one explicitly (include hash in prefix)."
            )
        found[f"paloma_{sub}"] = DatasetComponent(cache_dir=str(matches[0]))
    return found


paloma_components = _paloma_components()

data_config = LmDataConfig(
    components={
        "dclm_200m": DatasetComponent(cache_dir=DCLM_TRAIN),
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        "opc_algorithmic": DatasetComponent(cache_dir=OPC_ALGORITHMIC),
        **paloma_components,
    },
    train_weights={
        "dclm_200m": 0.75,
        "opc_algorithmic": 0.25,
        "dclm_200m_val": 0.0,
        **{k: 0.0 for k in paloma_components},
    },
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    shuffle=True,
    block_cross_document_attention=False,
    shuffle_before_trainval_split=False,
    stop_strategy="restart",
    enforce_eos=True,
)

model_config = model_dict["1_4b4k"]

train_config = TrainLmConfig(
    data=data_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency",
            entity="dongwei_jiang",
            tags=["aryabumi-probe", "1.4b", "code-alg", "code-mix-25",
                  "wd_1.6", "x16-epoch", "data_seed_0", "no-cross-doc-block"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=64,
        num_train_steps=12800,
        steps_per_eval=3200,
        per_device_parallelism=8,
        per_device_eval_parallelism=8,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_25code_alg/",
            save_interval=timedelta(minutes=10),
            keep=[{"every": 10000}],
        ),
        ray=RayConfig(auto_start_cluster=False),
    ),
    model=model_config,
    train_seq_len=4096,
    optimizer=AdamConfig(
        learning_rate=1e-3,
        weight_decay=1.6,
        lr_schedule="cosine",
        min_lr_ratio=0.0,
        beta1=0.9,
        beta2=0.95,
        warmup=0.01,
        max_grad_norm=1.0,
    ),
    data_seed=0,
)

if __name__ == "__main__":
    from levanter.main import train_lm
    train_lm.main(train_config)
