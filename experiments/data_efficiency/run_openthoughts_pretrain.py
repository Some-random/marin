# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Pretrain 1.4B on flattened OpenThoughts reasoning data, eval on dclm_val.

Data: 795M tokens of OpenThoughts-114k (math/code/science CoT traces)
Training: 8 epochs of 200M tokens = 6400 steps
Eval: dclm_200m_val (held-out web text)

Compare against baseline: 1.4B on dclm_200m, 8 epochs, val loss 3.413
"""

import os
from datetime import timedelta

# Load secrets from .secrets file
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

OPENTHOUGHTS = "/fsx/users/dongweij/marin/outputs/tokenized/openthoughts_flat"
DCLM_VAL = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_200m_val-415aea"

model_config = model_dict["1_4b4k"]

data_config = LmDataConfig(
    components={
        "openthoughts": DatasetComponent(cache_dir=OPENTHOUGHTS),
        "dclm_val": DatasetComponent(cache_dir=DCLM_VAL),
    },
    train_weights={"openthoughts": 1.0, "dclm_val": 0.0},
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    shuffle=True,
    block_cross_document_attention=True,
    shuffle_before_trainval_split=False,
)

train_config = TrainLmConfig(
    data=data_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency",
            entity="dongwei_jiang",
            tags=["reasoning-exp", "openthoughts-pretrain", "1.4b", "8-epoch"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=64,
        num_train_steps=6400,
        steps_per_eval=1600,
        checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10)),
        ray=RayConfig(auto_start_cluster=False),
    ),
    model=model_config,
    train_seq_len=4096,
    optimizer=AdamConfig(
        learning_rate=1e-3,
        weight_decay=3.2,
        lr_schedule="cosine",
        min_lr_ratio=0.0,
    ),
    data_seed=42,
)

if __name__ == "__main__":
    from levanter.main import train_lm
    train_lm.main(train_config)
