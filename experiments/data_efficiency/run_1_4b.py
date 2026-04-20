# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
1.4B model, ~200M tokens of DCLM, 8 epochs, WD=3.2 (paper's regularized recipe).
Proper validation on held-out dclm_200m_val.
"""

import os
from datetime import timedelta

os.environ.setdefault("WANDB_API_KEY", "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")

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig, DatasetComponent
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from levanter.trainer import TrainerConfig

from experiments.data_efficiency.models import model_dict

TRAIN_PATH = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_200m_train-d321eb"
VAL_PATH = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_200m_val-415aea"

model_config = model_dict["1_4b4k"]

data_config = LmDataConfig(
    components={
        "dclm_200m_train": DatasetComponent(cache_dir=TRAIN_PATH),
        "dclm_200m_val": DatasetComponent(cache_dir=VAL_PATH),
    },
    train_weights={"dclm_200m_train": 1.0, "dclm_200m_val": 0.0},
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    shuffle=True,
    block_cross_document_attention=True,
    shuffle_before_trainval_split=False,
)

# 800 base_steps * 8 epochs = 6400 total steps
train_config = TrainLmConfig(
    data=data_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency",
            entity="dongwei_jiang",
            tags=["data-efficiency", "dclm_200m", "1.4b", "8-epoch", "regularized", "proper-val"],
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
