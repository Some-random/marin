"""
H1 Experiment: Does injecting reasoning data mid-training help?

Treatment data: OpenWebMath (219M) + Code Procedural (218M) = 437M tokens combined.
OWM showed strong SciQ gains (73.2% vs 63.2% baseline), Code adds procedural knowledge.

Design (continuous cosine LR across phases 1+2):
  Phase 0 (shared): Train from scratch on 203M DCLM, 4 epochs = 3,096 steps
  Phase 1 (1,667 steps / 437M tokens):
    Treatment: OWM+Code mix (219M + 218M tokens)
    Control: Disjoint DCLM (dclm_h1_phase1, ~440M tokens)
  Phase 2 (3,052 steps / 800M tokens):
    Both arms: Disjoint DCLM (dclm_h1_phase2, ~810M tokens)

LR schedule: Phases 1+2 share one continuous cosine over 4,719 total steps.
  Phase 1 uses stop_step=1667, num_train_steps=4719.
  Phase 2 uses initialize_from_step=1667, num_train_steps=4719.

Steps math: batch_size=64, seq_len=4096 = 262,144 tokens/step
  437M tokens = 1,667 steps, 800M = 3,052 steps

Usage:
  .venv/bin/python experiments/data_efficiency/run_h1_continuous.py --phase phase0
  .venv/bin/python experiments/data_efficiency/run_h1_continuous.py --phase treatment_p1 --checkpoint <phase0_ckpt>
  .venv/bin/python experiments/data_efficiency/run_h1_continuous.py --phase control_p1 --checkpoint <phase0_ckpt>
  .venv/bin/python experiments/data_efficiency/run_h1_continuous.py --phase treatment_p2 --checkpoint <p1_ckpt>
  .venv/bin/python experiments/data_efficiency/run_h1_continuous.py --phase control_p2 --checkpoint <p1_ckpt>
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
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from levanter.trainer import TrainerConfig

from experiments.data_efficiency.models import model_dict
from experiments.evals.task_configs import EvalTaskConfig, convert_to_levanter_task_config

# --- Data paths ---
DCLM_200M = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_200m_train-d321eb"
DCLM_H1_PHASE1 = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_h1_phase1"
DCLM_H1_PHASE2 = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_h1_phase2"
DCLM_VAL = "/fsx/users/dongweij/marin/outputs/tokenized/data_efficiency/dclm_200m_val-415aea"
OPENWEBMATH = "/fsx/users/dongweij/marin/outputs/tokenized/openwebmath"
CODE_PROCEDURAL = "/fsx/users/dongweij/marin/outputs/tokenized/code_procedural"

# --- Steps ---
PHASE0_STEPS = 3096     # 4 epochs of 203M tokens (774 steps/epoch)
PHASE1_STEPS = 1667     # 437M tokens (OWM 219M + Code 218M)
PHASE2_STEPS = 3052     # 800M tokens
H1_TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS  # 4719, cosine spans this

EVAL_TASKS = (
    EvalTaskConfig("arc_easy", 0),
    EvalTaskConfig("arc_challenge", 0),
    EvalTaskConfig("piqa", 0),
    EvalTaskConfig("sciq", 0),
    EvalTaskConfig("hellaswag", 0),
    EvalTaskConfig("winogrande", 0),
    EvalTaskConfig("mmlu", 5),
)

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=str, required=True,
                    choices=["phase0", "treatment_p1", "control_p1", "treatment_p2", "control_p2"])
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Checkpoint from prior phase")
args = parser.parse_args()

model_config = model_dict["300m4k"]


def make_data_config(train_path, val_path=DCLM_VAL):
    return LmDataConfig(
        components={
            "train": DatasetComponent(cache_dir=train_path),
            "dclm_val": DatasetComponent(cache_dir=val_path),
        },
        train_weights={"train": 1.0, "dclm_val": 0.0},
        tokenizer="meta-llama/Meta-Llama-3.1-8B",
        shuffle=True,
        block_cross_document_attention=False,
        shuffle_before_trainval_split=False,
    )


def make_mixed_data_config():
    """OWM + Code mixed equally for treatment phase 1."""
    return LmDataConfig(
        components={
            "owm": DatasetComponent(cache_dir=OPENWEBMATH),
            "code": DatasetComponent(cache_dir=CODE_PROCEDURAL),
            "dclm_val": DatasetComponent(cache_dir=DCLM_VAL),
        },
        train_weights={"owm": 1.0, "code": 1.0, "dclm_val": 0.0},
        tokenizer="meta-llama/Meta-Llama-3.1-8B",
        shuffle=True,
        block_cross_document_attention=False,
        shuffle_before_trainval_split=False,
    )


# --- Phase configs ---
if args.phase == "phase0":
    train_config = TrainLmConfig(
        data=make_data_config(DCLM_200M),
        trainer=TrainerConfig(
            seed=0,
            tracker=WandbConfig(
                project="dongwei-data-efficiency",
                entity="dongwei_jiang",
                tags=["h1-continuous", "300m", "phase0", "pretrain"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=64,
            num_train_steps=PHASE0_STEPS,
            steps_per_eval=PHASE0_STEPS,
            checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10)),
            ray=RayConfig(auto_start_cluster=False),
        ),
        model=model_config,
        train_seq_len=4096,
        optimizer=AdamConfig(
            learning_rate=3e-3,
            weight_decay=1.6,
            lr_schedule="cosine",
            min_lr_ratio=0.0,
        ),
        data_seed=42,
    )

elif args.phase == "treatment_p1":
    if not args.checkpoint:
        parser.error("--checkpoint required for treatment_p1")
    train_config = TrainLmConfig(
        data=make_mixed_data_config(),
        trainer=TrainerConfig(
            seed=0,
            tracker=WandbConfig(
                project="dongwei-data-efficiency",
                entity="dongwei_jiang",
                tags=["h1-continuous", "300m", "treatment", "phase1", "owm+code"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=64,
            num_train_steps=H1_TOTAL_STEPS,
            stop_step=PHASE1_STEPS,
            steps_per_eval=H1_TOTAL_STEPS,
            checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10)),
            ray=RayConfig(auto_start_cluster=False),
        ),
        model=model_config,
        train_seq_len=4096,
        optimizer=AdamConfig(
            learning_rate=3e-3,
            weight_decay=1.6,
            lr_schedule="cosine",
            min_lr_ratio=0.0,
        ),
        data_seed=42,
        initialize_from_checkpoint_path=args.checkpoint,
    )

elif args.phase == "control_p1":
    if not args.checkpoint:
        parser.error("--checkpoint required for control_p1")
    train_config = TrainLmConfig(
        data=make_data_config(DCLM_H1_PHASE1),
        trainer=TrainerConfig(
            seed=0,
            tracker=WandbConfig(
                project="dongwei-data-efficiency",
                entity="dongwei_jiang",
                tags=["h1-continuous", "300m", "control", "phase1", "dclm"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=64,
            num_train_steps=H1_TOTAL_STEPS,
            stop_step=PHASE1_STEPS,
            steps_per_eval=H1_TOTAL_STEPS,
            checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10)),
            ray=RayConfig(auto_start_cluster=False),
        ),
        model=model_config,
        train_seq_len=4096,
        optimizer=AdamConfig(
            learning_rate=3e-3,
            weight_decay=1.6,
            lr_schedule="cosine",
            min_lr_ratio=0.0,
        ),
        data_seed=42,
        initialize_from_checkpoint_path=args.checkpoint,
    )

elif args.phase == "treatment_p2":
    if not args.checkpoint:
        parser.error("--checkpoint required for treatment_p2")
    eval_harness = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(EVAL_TASKS))
    train_config = TrainLmConfig(
        data=make_data_config(DCLM_H1_PHASE2),
        trainer=TrainerConfig(
            seed=0,
            tracker=WandbConfig(
                project="dongwei-data-efficiency",
                entity="dongwei_jiang",
                tags=["h1-continuous", "300m", "treatment", "phase2", "dclm"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=64,
            num_train_steps=H1_TOTAL_STEPS,
            steps_per_eval=H1_TOTAL_STEPS,
            checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10)),
            ray=RayConfig(auto_start_cluster=False),
        ),
        model=model_config,
        train_seq_len=4096,
        optimizer=AdamConfig(
            learning_rate=3e-3,
            weight_decay=1.6,
            lr_schedule="cosine",
            min_lr_ratio=0.0,
        ),
        eval_harness=eval_harness,
        data_seed=42,
        initialize_from_checkpoint_path=args.checkpoint,
        initialize_from_step=PHASE1_STEPS,
    )

elif args.phase == "control_p2":
    if not args.checkpoint:
        parser.error("--checkpoint required for control_p2")
    eval_harness = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(EVAL_TASKS))
    train_config = TrainLmConfig(
        data=make_data_config(DCLM_H1_PHASE2),
        trainer=TrainerConfig(
            seed=0,
            tracker=WandbConfig(
                project="dongwei-data-efficiency",
                entity="dongwei_jiang",
                tags=["h1-continuous", "300m", "control", "phase2", "dclm"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=64,
            num_train_steps=H1_TOTAL_STEPS,
            steps_per_eval=H1_TOTAL_STEPS,
            checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=10)),
            ray=RayConfig(auto_start_cluster=False),
        ),
        model=model_config,
        train_seq_len=4096,
        optimizer=AdamConfig(
            learning_rate=3e-3,
            weight_decay=1.6,
            lr_schedule="cosine",
            min_lr_ratio=0.0,
        ),
        eval_harness=eval_harness,
        data_seed=42,
        initialize_from_checkpoint_path=args.checkpoint,
        initialize_from_step=PHASE1_STEPS,
    )

if __name__ == "__main__":
    from levanter.main import train_lm
    train_lm.main(train_config)
