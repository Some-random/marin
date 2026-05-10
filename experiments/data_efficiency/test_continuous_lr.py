"""
Smoke test: verify that initialize_from_step produces a continuous LR schedule.

Test plan:
  1. Train 40 steps in one go (num_train_steps=40) -> log losses
  2. Train 20 steps (num_train_steps=40, stop_step=20) -> checkpoint
  3. Resume from that checkpoint with initialize_from_step=20, num_train_steps=40 -> log losses for steps 20-39
  4. Compare losses from (1) steps 20-39 vs (3) steps 0-19 — should match within tolerance.

Usage:
  .venv/bin/python experiments/data_efficiency/test_continuous_lr.py --phase single
  .venv/bin/python experiments/data_efficiency/test_continuous_lr.py --phase split_phase1
  .venv/bin/python experiments/data_efficiency/test_continuous_lr.py --phase split_phase2 --checkpoint /path/to/step-20
"""

import argparse
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

DCLM_TRAIN = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_200m_train-d321eb"
DCLM_VAL = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_200m_val-415aea"

TOTAL_STEPS = 40
SPLIT_STEP = 20

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str, required=True,
                    choices=["single", "split_phase1", "split_phase2"])
parser.add_argument("--checkpoint", type=str, default=None)
args = parser.parse_args()

model_config = model_dict["300m4k"]

data_config = LmDataConfig(
    components={
        "train": DatasetComponent(cache_dir=DCLM_TRAIN),
        "dclm_val": DatasetComponent(cache_dir=DCLM_VAL),
    },
    train_weights={"train": 1.0, "dclm_val": 0.0},
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    shuffle=True,
    block_cross_document_attention=False,
    shuffle_before_trainval_split=False,
)

base_trainer = TrainerConfig(
    seed=0,
    tracker=WandbConfig(
        project="dongwei-data-efficiency",
        entity="dongwei_jiang",
    ),
    mp=jmp.get_policy("p=f32,c=bfloat16"),
    train_batch_size=64,
    num_train_steps=TOTAL_STEPS,
    steps_per_eval=9999,
    checkpointer=CheckpointerConfig(
        save_interval=timedelta(minutes=1),
    ),
    ray=RayConfig(auto_start_cluster=False),
)

optimizer = AdamConfig(
    learning_rate=3e-3,
    weight_decay=1.6,
    lr_schedule="cosine",
    min_lr_ratio=0.0,
)

if args.phase == "single":
    train_config = TrainLmConfig(
        data=data_config,
        trainer=TrainerConfig(**{**base_trainer.__dict__,
            "tracker": WandbConfig(project="dongwei-data-efficiency", entity="dongwei_jiang",
                                   tags=["lr-smoke-test", "single-40"]),
        }),
        model=model_config,
        train_seq_len=4096,
        optimizer=optimizer,
        data_seed=42,
    )

elif args.phase == "split_phase1":
    train_config = TrainLmConfig(
        data=data_config,
        trainer=TrainerConfig(**{**base_trainer.__dict__,
            "tracker": WandbConfig(project="dongwei-data-efficiency", entity="dongwei_jiang",
                                   tags=["lr-smoke-test", "split-phase1"]),
            "stop_step": SPLIT_STEP,
        }),
        model=model_config,
        train_seq_len=4096,
        optimizer=optimizer,
        data_seed=42,
    )

elif args.phase == "split_phase2":
    if args.checkpoint is None:
        parser.error("--checkpoint required for split_phase2")
    train_config = TrainLmConfig(
        data=data_config,
        trainer=TrainerConfig(**{**base_trainer.__dict__,
            "tracker": WandbConfig(project="dongwei-data-efficiency", entity="dongwei_jiang",
                                   tags=["lr-smoke-test", "split-phase2"]),
        }),
        model=model_config,
        train_seq_len=4096,
        optimizer=optimizer,
        data_seed=42,
        initialize_from_checkpoint_path=args.checkpoint,
        initialize_from_step=SPLIT_STEP,
    )

if __name__ == "__main__":
    from levanter.main import train_lm
    train_lm.main(train_config)
