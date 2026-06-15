# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
1.4B 25% code-mix experiment, MATCHED-EPOCHS v2.

Fix vs the May 26 run_1_4b_25code_alg.py:
  - DCLM subsampled to 121,500 docs (~150 M tokens), opc_algorithmic subsampled
    to 282,500 docs (~50 M tokens). At 75/25 sampling weights over the same
    12,800-step (3.36 B token) budget as the baseline, both data sources are
    seen ~16.8 epochs — matching how the baseline saw its 203 M DCLM slice.
  - This isolates the "swap 50 M DCLM for 50 M code" intervention from the
    "DCLM exposure dropped" confound that the original run had.

Actual tokenized sizes (verified):
  dclm_150m: 146.97 M tokens (target 150 M)
  opc_algorithmic_50m: 54.59 M tokens (target 50 M)

To make epochs EXACTLY matched, weights are set in proportion to unique sizes
(NOT 75/25, which would give an 11% epoch mismatch given the actual sizes):
  Total unique = 146.97 + 54.59 = 201.56 M
  w_dclm = 146.97 / 201.56 = 0.729
  w_opc  =  54.59 / 201.56 = 0.271

Numerical sanity check:
  Total trained tokens = 12,800 × 64 × 4096 = 3.355 B
  DCLM drawn = 0.729 × 3.355 B = 2.446 B → 2.446 / 0.14697 = 16.64 epochs
  opc drawn  = 0.271 × 3.355 B = 0.910 B → 0.910 / 0.05459 = 16.66 epochs ✓ matched
  Same total budget as `peach-thunder-100` / `1_4b_wd1_6_x16_nocrossblock`.

Reference baseline (unchanged): `peach-thunder-100` / `6xx0hu3l` — 200 M DCLM,
zero code, same hyperparameters. The comparison this run enables is "same
budget, same epoch count, 50 M of DCLM replaced with 50 M of opc code".
"""

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

BASE_TOKENIZED = "/fsx/users/dongweij/marin/outputs/tokenized"

DCLM_VAL = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_val-415aea"


def _cache_dir(name_prefix: str) -> str:
    base = Path(BASE_TOKENIZED)
    matches = sorted(base.glob(f"{name_prefix}-*"))
    if not matches:
        raise FileNotFoundError(
            f"Tokenized dataset '{name_prefix}-*' not found in {BASE_TOKENIZED}. "
            f"Run `MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python -m "
            f"experiments.data_efficiency.data_150_50` first."
        )
    return str(matches[-1])


DCLM_150M = _cache_dir("dclm_150m")
OPC_50M = _cache_dir("opc_algorithmic_50m")

PALOMA_SUBSETS = [
    "4chan", "c4_100_domains", "c4_en", "dolma-v1_5",
    "dolma_100_programing_languages", "dolma_100_subreddits", "falcon-refinedweb",
    "gab", "m2d2_s2orc_unsplit", "m2d2_wikipedia_unsplit", "manosphere_meta_sep",
    "mc4", "ptb", "redpajama", "twitterAAE_HELM_fixed", "wikitext_103",
]


def _paloma_components() -> dict[str, DatasetComponent]:
    paloma_dir = Path(f"{BASE_TOKENIZED}/paloma")
    found = {}
    for sub in PALOMA_SUBSETS:
        matches = sorted(paloma_dir.glob(f"{sub}-*"))
        if not matches:
            raise FileNotFoundError(f"Paloma subset '{sub}' not tokenized.")
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple caches match {sub!r}: " + str([p.name for p in matches]) + ". "
                "Pin one explicitly (include hash in prefix)."
            )
        found[f"paloma_{sub}"] = DatasetComponent(cache_dir=str(matches[0]))
    return found


paloma_components = _paloma_components()

data_config = LmDataConfig(
    components={
        "dclm_150m": DatasetComponent(cache_dir=DCLM_150M),
        "opc_algorithmic_50m": DatasetComponent(cache_dir=OPC_50M),
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights={
        "dclm_150m": 0.729,
        "opc_algorithmic_50m": 0.271,
        "dclm_200m_val": 0.0,
        **{k: 0.0 for k in paloma_components},
    },
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    shuffle=True,
    block_cross_document_attention=False,
    shuffle_before_trainval_split=False,
    stop_strategy="restart",
    enforce_eos=True,
)

model_config = model_dict["1_4b4k"]

train_config = TrainLmConfig(
    data=data_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency",
            entity="dongwei_jiang",
            tags=["aryabumi-probe", "1.4b", "code-alg", "code-mix-25",
                  "wd_1.6", "x16-epoch", "data_seed_0", "no-cross-doc-block",
                  "matched-epochs", "v2"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=64,
        num_train_steps=12800,
        steps_per_eval=3200,
        per_device_parallelism=8,
        per_device_eval_parallelism=8,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_25code_alg_v2/",
            save_interval=timedelta(minutes=10),
            keep=[{"every": 10000}],
        ),
        ray=RayConfig(auto_start_cluster=False),
    ),
    model=model_config,
    train_seq_len=4096,
    optimizer=AdamConfig(
        learning_rate=1e-3,
        weight_decay=1.6,
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
    from levanter.main import train_lm
    train_lm.main(train_config)
