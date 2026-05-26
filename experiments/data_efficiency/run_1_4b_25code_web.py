# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
1.4B Aryabumi replication, variant A: 25% web code + 75% text.

Tests Aryabumi (2408.10914) headline finding "25% code is optimal" at 1.4B / our scale.
Web code = code_python parquets (mixed languages: Java, JS, PHP, Python, etc.) — analogue
of Aryabumi's "The Stack" web-scraped code.

Diffs from run_1_4b_wd1_6_x16_nocrossblock.py (our non-looping baseline):
  - Add aryabumi_code_web component, weight 0.25
  - Reduce dclm_200m weight from 1.0 to 0.75
  - Tags: aryabumi-replication, code-mix-25, code-web

All else identical: 1.4B, lr=1e-3 cosine→0, wd=1.6, batch=64, 12800 steps (16 epoch
equivalent on 209M base + cycled code), seed=0, data_seed=0,
block_cross_document_attention=False.

NOTE on scale: Aryabumi trained 470M/2.8B for 200B tokens. We're at 1.4B for ~3.34B total
(64 batch × 4096 seq × 12800 steps). So absolute code tokens consumed: ~836M vs Aryabumi's
~50B. The mix ratio matches but total scale is ~60x smaller. Whether the effect persists
at this scale is part of what we're testing.
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

# Code source — resolved at script load. Glob picks up the hashed dir produced by code_data.py.
def _code_cache_dir(name_prefix: str) -> str:
    base = Path(BASE_TOKENIZED)
    matches = sorted(base.glob(f"{name_prefix}-*"))
    if not matches:
        raise FileNotFoundError(
            f"Tokenized code dataset '{name_prefix}-*' not found in {BASE_TOKENIZED}. "
            f"Run `MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python -m "
            f"experiments.data_efficiency.code_data` first."
        )
    return str(matches[-1])


CODE_WEB = _code_cache_dir("aryabumi_code_web")

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
        found[f"paloma_{sub}"] = DatasetComponent(cache_dir=str(matches[-1]))
    return found


paloma_components = _paloma_components()

data_config = LmDataConfig(
    components={
        "dclm_200m": DatasetComponent(cache_dir=DCLM_TRAIN),
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        "aryabumi_code_web": DatasetComponent(cache_dir=CODE_WEB),
        **paloma_components,
    },
    train_weights={
        "dclm_200m": 0.75,
        "aryabumi_code_web": 0.25,
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
            tags=["aryabumi-replication", "1.4b", "code-mix-25", "code-web",
                  "wd_1.6", "x16-epoch", "data_seed_0", "no-cross-doc-block"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=64,
        num_train_steps=12800,
        steps_per_eval=3200,
        per_device_parallelism=8,
        per_device_eval_parallelism=8,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_25code_web/",
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
