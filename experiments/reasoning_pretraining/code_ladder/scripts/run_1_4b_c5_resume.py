# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume the C5 training from step-20914.

Imports the production C5 config and only overrides load_checkpoint_path
so the trainer auto-loads the latest checkpoint at the existing run dir.
LR schedule and data position are restored from the checkpoint (they're
both deterministic functions of the step count which Levanter persists).
"""

import os
from pathlib import Path

_secrets = Path(__file__).resolve().parents[4] / ".secrets"
if _secrets.exists():
    for line in _secrets.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

# Force wandb to resume the original run so checkpointer writes to the same dir.
os.environ.setdefault("WANDB_RUN_ID", "7mnu0nch")
os.environ.setdefault("WANDB_RESUME", "allow")

# Import the production config; the train_config object is fully built.
from dataclasses import replace  # noqa: E402

from experiments.reasoning_pretraining.code_ladder.scripts.run_1_4b_c5_code_then_text import (  # noqa: E402
    train_config as _prod_config,
)

RESUME_FROM = "/fsx/users/dongweij/marin/checkpoints/1_4b_1ep_c5_code_then_text/7mnu0nch"

# Override only the trainer config to point at the existing checkpoint dir.
resume_trainer = replace(_prod_config.trainer, load_checkpoint_path=RESUME_FROM)
train_config = replace(_prod_config, trainer=resume_trainer)


if __name__ == "__main__":
    print(f"=== C5 RESUME from {RESUME_FROM} ===")
    print(f"  WANDB_RUN_ID: {os.environ.get('WANDB_RUN_ID')}")
    print(f"  WANDB_RESUME: {os.environ.get('WANDB_RESUME')}")
    print(f"  load_checkpoint_path: {train_config.trainer.load_checkpoint_path}")
    print(f"  num_train_steps: {train_config.trainer.num_train_steps:,}")
    from levanter.main import train_lm
    train_lm.main(train_config)
