# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-node smoke test for the C5 (Aryabumi code->text) training config.

Verifies in ~5 minutes:
  1. All 15 per-language code caches load through Levanter's MixtureDataset.
  2. The staged-weight schedule (list of (start_seq_index, weights)) parses
     and switches at the boundary.
  3. The model + data + optimizer compile and produce a few real training
     steps on 1 node (no full A100 ring needed).

Differences from `run_1_4b_c5_code_then_text.py`:
  - 32 train steps (smoke), 16-step stage boundary.
  - Single node (no JAX_DIST_* env vars assumed).
  - No checkpointer (smoke).

Run via (single node):
    cd /fsx/users/dongweij/marin
    .venv/bin/python -m experiments.data_efficiency.run_1_4b_c5_smoke
"""

import os
from pathlib import Path

_secrets = Path(__file__).resolve().parents[2] / ".secrets"
if _secrets.exists():
    for line in _secrets.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

# Import the full config from the production script.
from experiments.data_efficiency.run_1_4b_c5_code_then_text import (  # noqa: E402
    data_config,
    model_config,
    STACK_LANG_CACHES,
    MARKUP_LANG_CACHES,
    TRAIN_BATCH_SIZE,
)
import jmp  # noqa: E402
from levanter.distributed import DistributedConfig, RayConfig  # noqa: E402
from levanter.main.train_lm import TrainLmConfig  # noqa: E402
from levanter.optim import AdamConfig  # noqa: E402
from levanter.tracker.wandb import WandbConfig  # noqa: E402
from levanter.trainer import TrainerConfig  # noqa: E402

# Single-node: 1 GPU host x 8 GPUs = 8 devices, per_device_parallelism=8 ->
# global batch = 64 (not the production 256). For the smoke test we just need
# correctness, not match the production batch.
SMOKE_BATCH_SIZE = 8 * 8  # 64
SMOKE_STEPS = 32
SMOKE_STAGE2_START_SEQ = 16 * SMOKE_BATCH_SIZE  # halfway, batch-aligned

# Override the staged weights so the boundary lands inside the smoke run.
from experiments.data_efficiency.run_1_4b_c5_code_then_text import (  # noqa: E402
    _stage1_weights,
    _stage2_weights,
)
from dataclasses import replace  # noqa: E402

smoke_data_config = replace(
    data_config,
    train_weights=[
        (0, _stage1_weights()),
        (SMOKE_STAGE2_START_SEQ, _stage2_weights()),
    ],
)


train_config = TrainLmConfig(
    data=smoke_data_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency",
            entity="dongwei_jiang",
            tags=["1.4b", "smoke", "c5", "aryabumi-code-then-text"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=SMOKE_BATCH_SIZE,
        num_train_steps=SMOKE_STEPS,
        steps_per_eval=SMOKE_STEPS + 1,  # skip eval
        per_device_parallelism=8,
        per_device_eval_parallelism=8,
        ray=RayConfig(auto_start_cluster=False),
        distributed=DistributedConfig(),
        jax_compilation_cache_dir="/fsx/users/dongweij/marin/outputs/jax_compile_cache",
    ),
    model=model_config,
    train_seq_len=4096,
    optimizer=AdamConfig(
        learning_rate=3e-4,
        weight_decay=0.1,
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
    print("=== C5 smoke test (1 node, 32 steps) ===")
    print(f"  global batch: {SMOKE_BATCH_SIZE} (production uses 256)")
    print(f"  stage 2 starts at seq {SMOKE_STAGE2_START_SEQ}")
    print(f"  stage 1 weights:")
    for k, v in _stage1_weights().items():
        if v > 0:
            print(f"    {k}: {v:.4f}")
    print(f"  stage 2 weights:")
    for k, v in _stage2_weights().items():
        if v > 0:
            print(f"    {k}: {v:.4f}")
    if not STACK_LANG_CACHES or not MARKUP_LANG_CACHES:
        print("\nERROR: STACK_LANG_CACHES / MARKUP_LANG_CACHES still empty in the prod config.")
        print("Fill them in run_1_4b_c5_code_then_text.py after tokenization.")
        raise SystemExit(1)
    from levanter.main import train_lm
    train_lm.main(train_config)
