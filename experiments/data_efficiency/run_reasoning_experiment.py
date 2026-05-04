# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Reasoning data experiments with 300M model.
Each phase runs as a separate process to avoid JAX reinitialization issues.

Usage:
  .venv/bin/python experiments/data_efficiency/run_reasoning_experiment.py --run B
  .venv/bin/python experiments/data_efficiency/run_reasoning_experiment.py --run C_phase1
  .venv/bin/python experiments/data_efficiency/run_reasoning_experiment.py --run C_phase2 --checkpoint /path/to/ckpt
  .venv/bin/python experiments/data_efficiency/run_reasoning_experiment.py --run D_phase1
  .venv/bin/python experiments/data_efficiency/run_reasoning_experiment.py --run D_phase2 --checkpoint /path/to/ckpt
"""

import argparse
import os
from datetime import timedelta

os.environ.setdefault("WANDB_API_KEY", "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")

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

OPENTHOUGHTS_FILTERED = "/fsx/users/dongweij/marin/outputs/tokenized/openthoughts_filtered"
DCLM_TRAIN = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_200m_train-d321eb"
DCLM_VAL = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_200m_val-415aea"

EVAL_TASKS = (
    EvalTaskConfig("arc_challenge", 0),
    EvalTaskConfig("hellaswag", 0),
    EvalTaskConfig("winogrande", 0),
    EvalTaskConfig("mmlu", 5),
)

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, required=True,
                    choices=["B", "C_phase1", "C_phase2", "D_phase1", "D_phase2"])
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--model", type=str, default="300m4k")
args = parser.parse_args()

model_config = model_dict[args.model]


def make_data_config(train_path):
    return LmDataConfig(
        components={
            "train": DatasetComponent(cache_dir=train_path),
            "dclm_val": DatasetComponent(cache_dir=DCLM_VAL),
        },
        train_weights={"train": 1.0, "dclm_val": 0.0},
        tokenizer="meta-llama/Meta-Llama-3.1-8B",
        shuffle=True,
        block_cross_document_attention=True,
        shuffle_before_trainval_split=False,
    )


def make_train_config(data_config, tags, num_steps, checkpoint_path=None, run_eval_harness=True):
    eval_harness = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(EVAL_TASKS)) if run_eval_harness else None

    config = TrainLmConfig(
        data=data_config,
        trainer=TrainerConfig(
            seed=0,
            tracker=WandbConfig(
                project="dongwei-data-efficiency",
                entity="dongwei_jiang",
                tags=tags,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=64,
            num_train_steps=num_steps,
            steps_per_eval=1600,
            checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10)),
            ray=RayConfig(auto_start_cluster=False),
        ),
        model=model_config,
        train_seq_len=4096,
        optimizer=AdamConfig(
            learning_rate=3e-3,
            weight_decay=3.2,
            lr_schedule="cosine",
            min_lr_ratio=0.0,
        ),
        eval_harness=eval_harness,
        data_seed=42,
    )
    if checkpoint_path:
        config = TrainLmConfig(**{**config.__dict__, "initialize_from_checkpoint_path": checkpoint_path})
    return config


RUN_CONFIGS = {
    "B": (OPENTHOUGHTS_FILTERED, ["reasoning-exp", "openthoughts-filtered", args.model, "run-B"], 6400, True),
    "C_phase1": (OPENTHOUGHTS_FILTERED, ["reasoning-exp", "owm-then-dclm", args.model, "run-C", "phase1"], 3200, False),
    "C_phase2": (DCLM_TRAIN, ["reasoning-exp", "owm-then-dclm", args.model, "run-C", "phase2"], 3200, True),
    "D_phase1": (DCLM_TRAIN, ["reasoning-exp", "dclm-then-owm", args.model, "run-D", "phase1"], 3200, False),
    "D_phase2": (OPENTHOUGHTS_FILTERED, ["reasoning-exp", "dclm-then-owm", args.model, "run-D", "phase2"], 3200, True),
}

train_path, tags, num_steps, run_eval = RUN_CONFIGS[args.run]
train_config = make_train_config(
    make_data_config(train_path),
    tags,
    num_steps,
    checkpoint_path=args.checkpoint,
    run_eval_harness=run_eval,
)

if __name__ == "__main__":
    from levanter.main import train_lm
    train_lm.main(train_config)
