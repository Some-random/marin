# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
8B model, ~200M tokens of DCLM, 1 epoch. For comparison with the 300M run.
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

TOKENIZED_PATH = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_200m_train-d321eb"

model_config = model_dict["l8b"]

# batch=8, seq_len=4096 => 8*4096=32768 tokens/step => 200M/32768 ≈ 6104 steps
data_config = LmDataConfig(
    components={
        "dclm_200m": DatasetComponent(cache_dir=TOKENIZED_PATH),
    },
    train_weights={"dclm_200m": 1.0},
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
            tags=["data-efficiency", "dclm_200m", "8b"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=8,
        num_train_steps=6104,
        steps_per_eval=10000,
        checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10)),
        ray=RayConfig(auto_start_cluster=False),
    ),
    model=model_config,
    train_seq_len=4096,
    optimizer=AdamConfig(
        learning_rate=3e-3,
        weight_decay=0.1,
        lr_schedule="cosine",
    ),
    data_seed=42,
)

if __name__ == "__main__":
    from levanter.main import train_lm
    train_lm.main(train_config)
