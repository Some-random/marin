# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# 1.4B C5-v5: C5-v2 recipe (single continuous cosine across both stages) but with
# SlimPajama-NL (English-only Wiki filtered) replacing DCLM in the 90% text slot.
#
# Hypothesis: at the data axis (SP-NL beats DCLM, established by C5-v4) AND at the
# LR-schedule axis (continuous cosine beats separate-cosine-per-phase for code retention,
# established by C5/C5-v2 vs C5-v3 comparison), the combination should:
#   - Beat C5-v2 on NL Means (data fix)
#   - Beat C5-v4 on Code Mean (no LR reset → less code forgetting)
#   - Possibly beat C5-v4 on NL Means too (no LR reset at start of phase 2 = phase 1
#     code circuits have less time to "drift" before NL exposure begins)
#
# What changes from C5-v2:
#   - 7 DCLM shard components → 228 SP-NL shard components (chunk_1 + chunk_2 en-filtered)
#   - Stage 2 weights: 90% spread across the 228 SP-NL shards instead of 90% across 7 DCLM shards
# What stays the same as C5-v2:
#   - Code/markup mix (Stack-Edu Python + Nemotron CC + Nemotron UA + Stack-Edu Markdown)
#   - Stage 1 = 100% code+markup at 80/20
#   - Stage 2 = 90% text + 10% (80% code + 20% markup)
#   - SINGLE continuous cosine LR schedule across both stages (mid-cosine LR at the stage boundary)
#   - 14,672 steps for stage 1, 14,672 for stage 2 = 29,344 total (≈ 30.77 B tokens)
#   - All hparams: wd=0.1, LR=3e-4, warmup 1%, AdamW β=(0.9, 0.95), max_grad_norm=1.0, batch=256

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
        raise FileNotFoundError(f"No tokenized cache for prefix '{prefix}'.")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple tokenized caches match prefix '{prefix}': {[m.name for m in matches]}. "
            f"Pin one explicitly by including the hash, e.g. '{matches[0].name}' instead of bare prefix."
        )
    return str(matches[0])


# === Clean code + markup caches (same as C5-v2 / C5-v3 / C5-v4) ===
try:
    SE_PYTHON_CACHE = _resolve_cache("c5v2_stack_edu_python_clean")
    SE_MARKDOWN_CACHE = _resolve_cache("c5v2_stack_edu_markdown_clean")
    NEMOTRON_CC_CACHE = _resolve_cache("c5v2_nemotron_code_concepts")
    NEMOTRON_UA_CACHE = _resolve_cache("c5v2_nemotron_unconditional_algorithmic")
except FileNotFoundError as _e:
    SE_PYTHON_CACHE = SE_MARKDOWN_CACHE = NEMOTRON_CC_CACHE = NEMOTRON_UA_CACHE = ""
    print(f"[c5v5] WARN: {_e}")


# === SlimPajama-NL English-only shards (chunk_1 + chunk_2) ===
def _collect_sp_nl_shards() -> list[str]:
    shards: list[str] = []
    for prefix in ("slimpajama_nl_en", "slimpajama_nl_chunk2_en"):
        try:
            root = Path(_resolve_cache(prefix))
        except FileNotFoundError:
            continue
        shards.extend(sorted(str(p) for p in (root / "train").glob("part-*-of-*")))
    return shards


SP_NL_SHARDS = _collect_sp_nl_shards()


# === In-domain held-out val + paloma per-subset for eval signal ===
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


NUM_PROC = int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1"))
TOTAL_GPUS = NUM_PROC * 8
TRAIN_BATCH_SIZE = 256
PER_DEVICE_PARALLELISM = max(1, TRAIN_BATCH_SIZE // TOTAL_GPUS)
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096
TARGET_TOKENS = 30_770_000_000
NUM_TRAIN_STEPS = TARGET_TOKENS // TOKENS_PER_STEP  # 29,343

STAGE2_START_STEP = NUM_TRAIN_STEPS // 2  # 14,671
STAGE2_START_SEQ = STAGE2_START_STEP * TRAIN_BATCH_SIZE


CODE_RATIOS = {
    "se_python": 8.8 / 16.3,
    "nemotron_cc": 7.3 / 16.3,
    "nemotron_ua": 0.2 / 16.3,
}
MARKUP_RATIOS = {"se_markdown": 1.0}

_code_key_to_cache = {
    "se_python": SE_PYTHON_CACHE,
    "nemotron_cc": NEMOTRON_CC_CACHE,
    "nemotron_ua": NEMOTRON_UA_CACHE,
}
_markup_key_to_cache = {"se_markdown": SE_MARKDOWN_CACHE}


def _stage1_weights() -> dict[str, float]:
    """100% code+markup at 80/20."""
    w: dict[str, float] = {}
    for k, r in CODE_RATIOS.items():
        w[f"code_{k}"] = 0.80 * r
    for k, r in MARKUP_RATIOS.items():
        w[f"markup_{k}"] = 0.20 * r
    for k in [
        *[f"sp_nl_shard{i:03d}" for i in range(len(SP_NL_SHARDS))],
        "dclm_200m_val",
        *[f"paloma_{s}" for s in PALOMA_SUBSETS],
    ]:
        w[k] = 0.0
    return w


def _stage2_weights() -> dict[str, float]:
    """90% SP-NL (uniform across shards) + 10% (80% code + 20% markup)."""
    w: dict[str, float] = {}
    if SP_NL_SHARDS:
        sp_share = 0.90 / len(SP_NL_SHARDS)
        for i in range(len(SP_NL_SHARDS)):
            w[f"sp_nl_shard{i:03d}"] = sp_share
    for k, r in CODE_RATIOS.items():
        w[f"code_{k}"] = 0.10 * 0.80 * r
    for k, r in MARKUP_RATIOS.items():
        w[f"markup_{k}"] = 0.10 * 0.20 * r
    for k in ["dclm_200m_val", *[f"paloma_{s}" for s in PALOMA_SUBSETS]]:
        w[k] = 0.0
    return w


_sp_nl_components = {
    f"sp_nl_shard{i:03d}": DatasetComponent(cache_dir=p)
    for i, p in enumerate(SP_NL_SHARDS)
}
_code_components = {f"code_{k}": DatasetComponent(cache_dir=v) for k, v in _code_key_to_cache.items() if v}
_markup_components = {f"markup_{k}": DatasetComponent(cache_dir=v) for k, v in _markup_key_to_cache.items() if v}

if not _code_components or not _markup_components or not _sp_nl_components:
    import warnings
    warnings.warn("Required caches are empty. Check SP-NL and code/markup caches.", stacklevel=2)

_shard_val_sizes = {k: 0 for k in {**_sp_nl_components, **_code_components, **_markup_components}}

data_config = LmDataConfig(
    components={
        **_sp_nl_components,
        **_code_components,
        **_markup_components,
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights=[
        (0, _stage1_weights()),
        (STAGE2_START_SEQ, _stage2_weights()),
    ],
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
            tags=["1.4b", "c5v5", "continuous-cosine", "slimpajama-nl", "wd-0.1", f"nodes-{NUM_PROC}"],
            save_code=False,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_c5v5/",
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
    print("=== 1.4B C5-v5 (single continuous cosine, SP-NL) ===")
    print(f"  num_processes (nodes): {NUM_PROC}  total GPUs: {TOTAL_GPUS}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}  per-device: {PER_DEVICE_PARALLELISM}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B")
    print(f"  stage 2 starts at step {STAGE2_START_STEP} / seq {STAGE2_START_SEQ:,}")
    print(f"  LR=3e-4 cosine to 0, warmup 1% (continuous across both stages)")
    print(f"  data: stage1 = 100% code+markup, stage2 = 90% SP-NL ({len(SP_NL_SHARDS)} shards) + 10% (80% code + 20% markup)")
    if not _code_components or not _markup_components or not _sp_nl_components:
        print("\nERROR: required caches missing.")
        raise SystemExit(1)
    from levanter.main import train_lm
    train_lm.main(train_config)
