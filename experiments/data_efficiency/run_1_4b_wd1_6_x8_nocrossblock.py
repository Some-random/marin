# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
1.4B wd=1.6/x8 standalone — the experiment our May 23 log identified as missing
("To isolate, we'd need to ablate wd=3.2/x16 and wd=1.6/x8 separately. Step 11
the next day did exactly this."  Step 11 only did the wd=3.2/x16 half.)

Matches konwoo's `1_4b4k-209Mx8-dclm-cos-lr0.0010-wd1.60-bs64` recipe (he has
this on wandb as 11 finished runs but did NOT upload the weights to HF — we
need our own copy to settle the looping question for this config).

Diffs from run_1_4b_wd1_6_x16_nocrossblock.py:
  - num_train_steps: 12800 (x16 epochs) -> 6400 (x8 epochs)
  - Checkpoint path: 1_4b_wd1_6_x16_nocrossblock/ -> 1_4b_wd1_6_x8_nocrossblock/
  - Run tags: x16-epoch -> x8-epoch
  - Cosine LR will decay to 0 at step 6400 (not 12800) — so the model gets the
    actual low-LR anneal phase that step-6400 of an x16 run never does.

Everything else identical: model 1_4b4k, data = konwoo/dclm-164k-docs-train
(203 M unique tokens), WD=1.6, LR=1e-3 cosine to min_lr=0, batch=64, seed=0,
data_seed=0, β₁/β₂=0.9/0.95, warmup=0.01, max_grad_norm=1.0, seq_len=4096,
block_cross_document_attention=False.

Purpose: test whether wd=1.6/x8 loops on gsm8k_cot. May 23-25 found wd=3.2/x8
loops and wd=1.6/x16 does NOT (the eval here is a quantitative recount — 25%
loop rate at step 12800 of x16; ~0-25% range across earlier checkpoints).
We need an actual wd=1.6/x8 model to know whether 8 epochs at wd=1.6 loops.
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
            raise FileNotFoundError(f"Paloma subset '{sub}' not tokenized yet.")
        found[f"paloma_{sub}"] = DatasetComponent(cache_dir=str(matches[-1]))
    return found


paloma_components = _paloma_components()

data_config = LmDataConfig(
    components={
        "dclm_200m": DatasetComponent(cache_dir=DCLM_TRAIN),
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights={
        "dclm_200m": 1.0,
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
            tags=["data-efficiency", "1.4b", "wd_1.6", "x8-epoch", "data_seed_0", "no-cross-doc-block", "looping-ablation"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=64,
        num_train_steps=6400,
        steps_per_eval=1600,
        per_device_parallelism=8,
        per_device_eval_parallelism=8,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_wd1_6_x8_nocrossblock/",
            save_interval=timedelta(minutes=10),
            keep=[{"every": 5000}],
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
