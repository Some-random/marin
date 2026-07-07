# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Small-scale code phase 1 (mirror of C5-v3 phase 1) at Chinchilla-optimal ½ budget.
#
# Env vars:
#   MODEL_KEY     — "300m4k" or "600m4k"
#   TARGET_TOKENS — phase 1 tokens (e.g. 3e9 = ½ of 300M Chinchilla budget)
#   RUN_TAG       — wandb tag + checkpoint dir suffix
#
# Data: 80% curated code + 20% Stack-Edu Markdown — same caches/ratios as C5-v3 phase 1.

import os
from datetime import timedelta
from pathlib import Path

_secrets = Path(__file__).resolve().parents[4] / ".secrets"
if _secrets.exists():
    for line in _secrets.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig, DatasetComponent
from levanter.distributed import DistributedConfig, RayConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.reasoning_pretraining.code_ladder.models.models import model_dict

BASE_TOKENIZED = "/fsx/users/dongweij/marin/outputs/tokenized"
_TOKENIZED_BASE = Path(BASE_TOKENIZED)


def _resolve_cache(prefix: str) -> str:
    matches = sorted(_TOKENIZED_BASE.glob(f"{prefix}-*"))
    if not matches:
        raise FileNotFoundError(f"No tokenized cache for prefix '{prefix}'.")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple matches for '{prefix}': {[m.name for m in matches]}")
    return str(matches[0])


MODEL_KEY     = os.environ["MODEL_KEY"]
TARGET_TOKENS = int(float(os.environ["TARGET_TOKENS"]))
RUN_TAG       = os.environ["RUN_TAG"]


SE_PYTHON_CACHE   = _resolve_cache("c5v2_stack_edu_python_clean")
SE_MARKDOWN_CACHE = _resolve_cache("c5v2_stack_edu_markdown_clean")
NEMOTRON_CC_CACHE = _resolve_cache("c5v2_nemotron_code_concepts")
NEMOTRON_UA_CACHE = _resolve_cache("c5v2_nemotron_unconditional_algorithmic")

DCLM_VAL = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_val-415aea"


PALOMA_SUBSETS = [
    "4chan", "c4_100_domains", "c4_en", "dolma-v1_5",
    "dolma_100_programing_languages", "dolma_100_subreddits",
    "falcon-refinedweb", "gab", "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit", "manosphere_meta_sep", "mc4",
    "ptb", "redpajama", "twitterAAE_HELM_fixed", "wikitext_103",
]


def _paloma_components() -> dict[str, DatasetComponent]:
    paloma_dir = Path(f"{BASE_TOKENIZED}/paloma")
    found = {}
    for sub in PALOMA_SUBSETS:
        matches = sorted(paloma_dir.glob(f"{sub}-*"))
        if not matches:
            raise FileNotFoundError(f"Paloma subset '{sub}' not tokenized.")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple matches for {sub!r}")
        found[f"paloma_{sub}"] = DatasetComponent(cache_dir=str(matches[0]))
    return found


paloma_components = _paloma_components()


def _distributed_from_env() -> DistributedConfig:
    num_proc = os.environ.get("JAX_DIST_NUM_PROCESSES")
    if num_proc is None:
        return DistributedConfig()
    return DistributedConfig(
        num_processes=int(num_proc),
        process_id=int(os.environ["JAX_DIST_PROCESS_ID"]),
        coordinator_address=os.environ["JAX_DIST_COORDINATOR"],
        local_device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    )


CODE_RATIOS = {
    "se_python":   8.8 / 16.3,
    "nemotron_cc": 7.3 / 16.3,
    "nemotron_ua": 0.2 / 16.3,
}
MARKUP_RATIOS = {"se_markdown": 1.0}


NUM_PROC = int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1"))
TOTAL_GPUS = NUM_PROC * 8
PER_DEVICE_PARALLELISM = 8
TRAIN_BATCH_SIZE = TOTAL_GPUS * PER_DEVICE_PARALLELISM
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096
NUM_TRAIN_STEPS = TARGET_TOKENS // TOKENS_PER_STEP


_code_components = {
    "code_se_python":   DatasetComponent(cache_dir=SE_PYTHON_CACHE),
    "code_nemotron_cc": DatasetComponent(cache_dir=NEMOTRON_CC_CACHE),
    "code_nemotron_ua": DatasetComponent(cache_dir=NEMOTRON_UA_CACHE),
}
_markup_components = {"markup_se_markdown": DatasetComponent(cache_dir=SE_MARKDOWN_CACHE)}

_train_weights: dict[str, float] = {}
for k, r in CODE_RATIOS.items():
    _train_weights[f"code_{k}"] = 0.80 * r
for k, r in MARKUP_RATIOS.items():
    _train_weights[f"markup_{k}"] = 0.20 * r
_train_weights["dclm_200m_val"] = 0.0
for s in PALOMA_SUBSETS:
    _train_weights[f"paloma_{s}"] = 0.0

data_config = LmDataConfig(
    components={
        **_code_components,
        **_markup_components,
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights=_train_weights,
    num_validation_sequences={k: 0 for k in {**_code_components, **_markup_components}},
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    shuffle=True,
    block_cross_document_attention=False,
    shuffle_before_trainval_split=False,
    stop_strategy="restart",
    enforce_eos=True,
)

model_config = model_dict[MODEL_KEY]

train_config = TrainLmConfig(
    data=data_config,
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency-smallscale",
            entity="dongwei_jiang",
            tags=["smallscale", MODEL_KEY, "code-p1", RUN_TAG, f"nodes-{NUM_PROC}"],
            save_code=False,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path=f"checkpoints/smallscale_{MODEL_KEY}_code_p1/",
            save_interval=timedelta(minutes=30),
            keep=[{"every": NUM_TRAIN_STEPS // 4}],
        ),
        ray=RayConfig(auto_start_cluster=False),
        distributed=_distributed_from_env(),
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
    print(f"=== smallscale code phase 1: {MODEL_KEY} ({TARGET_TOKENS/1e9:.2f} B tokens, 80%code+20%markup) ===")
    print(f"  nodes={NUM_PROC} batch={TRAIN_BATCH_SIZE} steps={NUM_TRAIN_STEPS:,}")
    from levanter.main import train_lm
    train_lm.main(train_config)
