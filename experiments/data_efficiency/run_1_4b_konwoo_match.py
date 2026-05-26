# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
1.4B Konwoo replication run: matches stanford-mercury/suhas-data-efficiency/
1_4b4k-209Mx8-dclm-cos-lr0.0010-wd3.20-seed0 as closely as our framework version allows.

Diffs from our prior 8be9dtfq baseline (super-glade-5):
  - data_seed: 0 (was 42)
  - optimizer.min_lr_ratio: 0.0 (was 0.1)
  - Eval set: full Paloma (16 subdomains) added, weight=0, to match konwoo's PPL eval

Same as our 8be9dtfq baseline:
  - Model: 1_4b4k (1.4B Llama, 16 layers, 2048 hidden)
  - Train data: konwoo/dclm-164k-docs-train (164k DCLM docs ≈ 209M tokens)
  - 8 epochs (6400 steps × 64 batch × 4096 seq = 1.67B tokens)
  - LR=1e-3, WD=3.2, cosine, β1=0.9, β2=0.95, warmup=0.01, max_grad_norm=1
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

DCLM_TRAIN = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_train-d321eb"
DCLM_VAL = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_val-415aea"

# Paloma val subsets — populated after `python -m experiments.paloma` runs.
# Each path is the tokenized cache dir written by Marin's executor.
PALOMA_SUBSETS = [
    "4chan",
    "c4_100_domains",
    "c4_en",
    "dolma-v1_5",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "falcon-refinedweb",
    "gab",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep",
    "mc4",
    "ptb",
    "redpajama",
    "twitterAAE_HELM_fixed",
    "wikitext_103",
]


def _paloma_components() -> dict[str, DatasetComponent]:
    """Build val-only components for each Paloma subset.

    Marin's executor writes paloma subsets under tokenized/paloma/<name>-<hash>/. We resolve
    the hash by listing the directory; if a subset is missing (download/tokenize hasn't
    finished), fail loudly rather than silently skipping it.
    """
    paloma_dir = Path(f"{BASE_TOKENIZED}/paloma")
    found = {}
    for sub in PALOMA_SUBSETS:
        matches = sorted(paloma_dir.glob(f"{sub}-*"))
        if not matches:
            raise FileNotFoundError(
                f"Paloma subset '{sub}' not tokenized yet. Run "
                f"`.venv/bin/python -m experiments.paloma` first."
            )
        found[f"paloma_{sub}"] = DatasetComponent(cache_dir=str(matches[-1]))
    return found


paloma_components = _paloma_components()

data_config = LmDataConfig(
    components={
        "dclm_200m": DatasetComponent(cache_dir=DCLM_TRAIN),
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights={
        "dclm_200m": 1.0,
        "dclm_200m_val": 0.0,
        **{k: 0.0 for k in paloma_components},
    },
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    shuffle=True,
    block_cross_document_attention=True,
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
            tags=["data-efficiency", "1.4b", "konwoo-replication", "dclm_200m", "min_lr_0", "data_seed_0"],
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=64,
        num_train_steps=6400,
        steps_per_eval=1600,
        per_device_parallelism=8,
        per_device_eval_parallelism=8,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_konwoo_match/",
            save_interval=timedelta(minutes=10),
            keep=[{"every": 10000}],
        ),
        ray=RayConfig(auto_start_cluster=False),
    ),
    model=model_config,
    train_seq_len=4096,
    optimizer=AdamConfig(
        learning_rate=1e-3,
        weight_decay=3.2,
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
