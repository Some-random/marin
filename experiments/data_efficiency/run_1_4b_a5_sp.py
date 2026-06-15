# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# 1.4B A5-SP: A5 recipe (single-phase, 100% NL pretraining) but with
# SlimPajama-NL instead of DCLM-baseline as the text source.
#
# Why: A5's "Mean Open-book = 0.636 / Mean Closed-book NL = 0.449" is the
# strongest NL ceiling in our §3 table; the gap from C5/C5-v2/C5-v3 to A5 is
# a recipe + data confound. Aryabumi et al §2.1 trained on
# SlimPajama-minus-Github-and-StackExchange (CommonCrawl + C4 + Books + ArXiv
# + Wikipedia, 503B); we used DCLM-baseline (CommonCrawl-only). A5-SP
# isolates the DATA axis: same hparams, same compute, same recipe — only
# text source differs (DCLM → SlimPajama-NL).
#
# All hparams locked to A5 (run_1_4b_1ep_dclm.py): batch=256×seq=4096,
# 29,343 steps, LR=3e-4 cosine, wd=0.1, β=(0.9, 0.95), warmup=1%,
# max_grad_norm=1.0, fresh init (no checkpoint load).

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
from levanter.distributed import DistributedConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from levanter.trainer import TrainerConfig

from experiments.data_efficiency.models import model_dict

BASE_TOKENIZED = "/fsx/users/dongweij/marin/outputs/tokenized"
_TOKENIZED_BASE = Path(BASE_TOKENIZED)


def _resolve_cache(prefix: str) -> str:
    matches = sorted(_TOKENIZED_BASE.glob(f"{prefix}-*"))
    if not matches:
        raise FileNotFoundError(
            f"No tokenized cache for prefix '{prefix}'. "
            "Run the slimpajama_tokenize step first."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple tokenized caches match prefix '{prefix}': {[m.name for m in matches]}. "
            f"Pin one explicitly by including the hash, e.g. '{matches[0].name}' instead of bare prefix."
        )
    return str(matches[0])


# === SlimPajama-NL shards ===
# Includes both chunk_1 (128 shards) and chunk_2 (~50 shards, after the
# chunk_2 tokenize completes). Use glob over ALL known caches.
def _collect_sp_nl_shards() -> list[str]:
    """Collect SlimPajama-NL English-only shards across all available caches."""
    shards: list[str] = []
    for prefix in ("slimpajama_nl_en", "slimpajama_nl_chunk2_en"):
        try:
            root = Path(_resolve_cache(prefix))
        except FileNotFoundError:
            continue
        shards.extend(sorted(str(p) for p in (root / "train").glob("part-*-of-*")))
    return shards


SP_NL_SHARDS = _collect_sp_nl_shards()


# === DCLM_200M_VAL — kept for cross-comparison ===
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
            raise RuntimeError(
                f"Multiple caches match {sub!r}: " + str([p.name for p in matches]) + ". "
                "Pin one explicitly (include hash in prefix)."
            )
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


# Match A5's batching:  batch = TOTAL_GPUS * 8.
# 4-node = 256 × 4096 = 1.05M tokens/step. 30.77 B / 1.05 M = 29,343 steps.
NUM_PROC = int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1"))
TOTAL_GPUS = NUM_PROC * 8
PER_DEVICE_PARALLELISM = 8
TRAIN_BATCH_SIZE = TOTAL_GPUS * PER_DEVICE_PARALLELISM
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096
TARGET_TOKENS = 30_770_000_000
NUM_TRAIN_STEPS = TARGET_TOKENS // TOKENS_PER_STEP


_sp_nl_components = {
    f"sp_nl_shard{i:03d}": DatasetComponent(cache_dir=p)
    for i, p in enumerate(SP_NL_SHARDS)
}
_sp_nl_weights = {k: 1.0 / max(1, len(_sp_nl_components)) for k in _sp_nl_components}
_shard_val_sizes = {k: 0 for k in _sp_nl_components}

data_config = LmDataConfig(
    components={
        **_sp_nl_components,
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights={
        **_sp_nl_weights,
        "dclm_200m_val": 0.0,
        **{k: 0.0 for k in paloma_components},
    },
    num_validation_sequences=_shard_val_sizes,
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
            tags=["1.4b", "1ep", "a5-sp", "slimpajama-nl", "wd-0.1", f"nodes-{NUM_PROC}"],
            save_code=False,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_a5_sp/",
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
    print("=== 1.4B A5-SP (1 epoch, SlimPajama-NL only) ===")
    print(f"  num_processes (nodes): {NUM_PROC}  total GPUs: {TOTAL_GPUS}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}  per-device: {PER_DEVICE_PARALLELISM}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B")
    print(f"  LR=3e-4, WD=0.1, cosine to 0 (warmup 1% ≈ {int(NUM_TRAIN_STEPS * 0.01)} steps)")
    print(f"  data: 100% SlimPajama-NL ({len(SP_NL_SHARDS)} shards)")
    if not SP_NL_SHARDS:
        print("\nERROR: SlimPajama-NL shards missing.")
        raise SystemExit(1)
    from levanter.main import train_lm
    train_lm.main(train_config)
