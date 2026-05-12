#!/usr/bin/env python
"""300M experiment: 80% DCLM + 20% OpenWebMath mixed simultaneously."""

import os
from pathlib import Path
from datetime import timedelta

_secrets = Path(".secrets")
if _secrets.exists():
    for line in _secrets.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig, DatasetComponent
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from levanter.trainer import TrainerConfig
from experiments.data_efficiency.models import model_dict
from experiments.evals.task_configs import EvalTaskConfig, convert_to_levanter_task_config

DCLM_TRAIN = "outputs/tokenized/data_efficiency/dclm_200m_train-d321eb"
DCLM_VAL = "outputs/tokenized/data_efficiency/dclm_200m_val-415aea"
OWM_PATH = "outputs/tokenized/openwebmath"
EVAL_TASKS = (EvalTaskConfig("arc_easy", 0), EvalTaskConfig("piqa", 0), EvalTaskConfig("sciq", 0))

config = TrainLmConfig(
    data=LmDataConfig(
        components={
            "dclm": DatasetComponent(cache_dir=DCLM_TRAIN),
            "owm": DatasetComponent(cache_dir=OWM_PATH),
            "dclm_val": DatasetComponent(cache_dir=DCLM_VAL),
        },
        train_weights={"dclm": 0.8, "owm": 0.2, "dclm_val": 0.0},
        tokenizer="meta-llama/Meta-Llama-3.1-8B",
        shuffle=True,
        block_cross_document_attention=True,
        shuffle_before_trainval_split=False,
    ),
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency",
            entity="dongwei_jiang",
            tags=["mixed-exp", "dclm80-owm20", "300m4k"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=64,
        num_train_steps=6400,
        steps_per_eval=1600,
        checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10)),
        ray=RayConfig(auto_start_cluster=False),
    ),
    model=model_dict["300m4k"],
    train_seq_len=4096,
    optimizer=AdamConfig(learning_rate=3e-3, weight_decay=3.2, lr_schedule="cosine", min_lr_ratio=0.0),
    eval_harness=LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(EVAL_TASKS)),
    data_seed=42,
)

from levanter.main import train_lm

train_lm.main(config)
