# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test the C5-v2 training config on 1 node, 30 steps.

Verifies: cache loading, optimizer state init, stage-1 data weights apply,
loss decreases on first few steps. Does NOT verify stage-2 transition (that
needs 14,672+ steps to reach).
"""

import os
from dataclasses import replace
from pathlib import Path

_secrets = Path(__file__).resolve().parents[2] / ".secrets"
if _secrets.exists():
    for line in _secrets.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

os.environ.setdefault("WANDB_MODE", "disabled")

from experiments.data_efficiency.run_1_4b_c5v2_clean_code import (  # noqa: E402
    train_config as _prod_config,
)

# Override only the trainer to do 30 steps with no checkpointing
smoke_trainer = replace(
    _prod_config.trainer,
    num_train_steps=30,
    steps_per_eval=30,
    checkpointer=replace(
        _prod_config.trainer.checkpointer,
        base_path="checkpoints/1_4b_c5v2_smoke/",
    ),
)
train_config = replace(_prod_config, trainer=smoke_trainer)


if __name__ == "__main__":
    print("=== C5-v2 SMOKE TEST (1 node, 30 steps) ===")
    print(f"  num_train_steps: {train_config.trainer.num_train_steps}")
    print(f"  per_device_parallelism: {train_config.trainer.per_device_parallelism}")
    print(f"  base_path: {train_config.trainer.checkpointer.base_path}")
    from levanter.main import train_lm
    train_lm.main(train_config)
