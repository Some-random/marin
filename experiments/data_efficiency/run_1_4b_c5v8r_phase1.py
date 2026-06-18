# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# 1.4B C5-v8r PHASE 1: RANDOM-code-LM from scratch with separate cosine.
#
# Purpose: provide a FULLY-COOLED random-code phase-1 endpoint for
# C5-v8r phase 2. The current C5-v8r initializes from C5's continuous-
# cosine step-14672, which is mid-decay (LR ~1.5e-4) — confounds the
# code-data-axis test against C5-v4 (which inits from C5-v3 phase 1,
# fully cooled at step-14671). This re-run eliminates that confound.
#
# Structure matches C5-v3 phase 1 EXACTLY, only swapping the code+markup
# data sources:
#   - C5-v3 phase 1: curated code (Stack-Edu Python ≥3.0 + Nemotron-CC +
#     Nemotron-UA + Stack-Edu Markdown ≥3.0)
#   - C5-v8r phase 1: random code at Aryabumi Table 3/4 ratios (10 Stack
#     langs + 5 markup langs, same as the original C5 run's stage 1)
#
# Identical recipe to C5-v3 phase 1: separate cosine 3e-4 → 0, batch
# 256×4096, 14,672 steps = 15.39 B tokens, fresh init, wd=0.1.
#
# Phase 2 (run_1_4b_c5v8r_phase2.py) gets a one-line edit after this
# completes: PHASE1_INIT_FROM → checkpoints/1_4b_c5v8r_phase1/<run_id>/step-14671.

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
            f"Run `MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python "
            f"-m experiments.data_efficiency.code_data_c5v2` first."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple tokenized caches match prefix '{prefix}': {[m.name for m in matches]}. "
            f"Pin one explicitly by including the hash, e.g. '{matches[0].name}' instead of bare prefix."
        )
    return str(matches[0])


# === RAW Stack code + Markup caches (same as C5's original stage 1) ===
_STACK_LANGS = ["java", "javascript", "php", "python", "c-sharp",
                "typescript", "c", "cpp", "go", "ruby"]
_MARKUP_LANGS = ["markdown", "yaml", "json", "html", "css"]


def _stack_caches() -> dict[str, str]:
    return {lang: _resolve_cache(f"stack_{lang.replace('-', '_')}")
            for lang in _STACK_LANGS}


def _markup_caches() -> dict[str, str]:
    return {lang: _resolve_cache(f"markup_{lang}") for lang in _MARKUP_LANGS}


try:
    STACK_LANG_CACHES: dict[str, str] = _stack_caches()
    MARKUP_LANG_CACHES: dict[str, str] = _markup_caches()
except FileNotFoundError as _e:
    STACK_LANG_CACHES = {}
    MARKUP_LANG_CACHES = {}
    print(f"[c5v8r-p1] WARN: {_e}")


# === Eval-only components (no training weight) — required for Levanter eval ===
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


# === Aryabumi Table 3 (Stack top-10) + Table 4 (Markup top-5) ratios ===
STACK_RATIOS = {
    "java": 15.54 / 86.83,
    "javascript": 15.29 / 86.83,
    "php": 12.46 / 86.83,
    "python": 9.60 / 86.83,
    "c-sharp": 8.30 / 86.83,
    "typescript": 7.92 / 86.83,
    "c": 6.63 / 86.83,
    "cpp": 4.91 / 86.83,
    "go": 3.49 / 86.83,
    "ruby": 2.69 / 86.83,
}
MARKUP_RATIOS = {
    "markdown": 54.23 / 90.40,
    "yaml": 10.77 / 90.40,
    "json": 9.97 / 90.40,
    "html": 8.57 / 90.40,
    "css": 6.86 / 90.40,
}


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


# === Batch and step layout — same as C5-v2; HALF the total step count ===
NUM_PROC = int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1"))
TOTAL_GPUS = NUM_PROC * 8
TRAIN_BATCH_SIZE = 256  # same as A5/B4/C5/C5-v2
PER_DEVICE_PARALLELISM = max(1, TRAIN_BATCH_SIZE // TOTAL_GPUS)
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096  # 1,048,576
# Phase 1 = half of A5/B4/C5/C5-v2 total = 14,672 steps = 15.39 B trained
NUM_TRAIN_STEPS = 14_672


_code_key_to_cache = {lang.replace('-', '_'): STACK_LANG_CACHES.get(lang, "") for lang in STACK_RATIOS}
_markup_key_to_cache = {lang: MARKUP_LANG_CACHES.get(lang, "") for lang in MARKUP_RATIOS}


def _phase1_weights() -> dict[str, float]:
    """100% code + markup, 80/20 split. Eval-only components get weight 0."""
    w: dict[str, float] = {}
    for lang, r in STACK_RATIOS.items():
        w[f"stack_{lang.replace('-', '_')}"] = 0.80 * r
    for lang, r in MARKUP_RATIOS.items():
        w[f"markup_{lang}"] = 0.20 * r
    for k in ["dclm_200m_val", *[f"paloma_{s}" for s in PALOMA_SUBSETS]]:
        w[k] = 0.0
    return w


_code_components = {f"stack_{k}": DatasetComponent(cache_dir=v) for k, v in _code_key_to_cache.items() if v}
_markup_components = {f"markup_{k}": DatasetComponent(cache_dir=v) for k, v in _markup_key_to_cache.items() if v}

_shard_val_sizes = {k: 0 for k in {**_code_components, **_markup_components}}

data_config = LmDataConfig(
    components={
        **_code_components,
        **_markup_components,
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights=_phase1_weights(),
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
            tags=["1.4b", "c5v8r", "phase1-code-only", "random-code", "separate-cosine", "wd-0.1", f"nodes-{NUM_PROC}"],
            save_code=False,  # wandb code-artifact upload triggers gql.CreateArtifact nil-ctx SIGSEGV on non-zero ranks
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_c5v8r_phase1/",
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
    print("=== 1.4B C5-v8r PHASE 1 (RANDOM code-LM from scratch, separate cosine) ===")
    print(f"  num_processes (nodes): {NUM_PROC}  total GPUs: {TOTAL_GPUS}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}  per-device: {PER_DEVICE_PARALLELISM}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B")
    print(f"  LR=3e-4, WD=0.1, cosine to 0 (warmup 1% = ~147 steps)")
    print(f"  data: 80% raw multi-lang Stack + 20% raw multi-lang markup (Aryabumi Table 3/4 ratios)")
    print()
    print(f"  stack caches ({len(STACK_LANG_CACHES)}):")
    for lang in STACK_RATIOS:
        cache_path = STACK_LANG_CACHES.get(lang, "(MISSING)")
        cache_short = cache_path.replace(BASE_TOKENIZED + "/", "") if cache_path != "(MISSING)" else cache_path
        print(f"    stack_{lang:14s} (w={0.80 * STACK_RATIOS[lang] * 100:5.2f}%)  ← {cache_short}")
    print(f"  markup caches ({len(MARKUP_LANG_CACHES)}):")
    for lang in MARKUP_RATIOS:
        cache_path = MARKUP_LANG_CACHES.get(lang, "(MISSING)")
        cache_short = cache_path.replace(BASE_TOKENIZED + "/", "") if cache_path != "(MISSING)" else cache_path
        print(f"    markup_{lang:13s} (w={0.20 * MARKUP_RATIOS[lang] * 100:5.2f}%)  ← {cache_short}")
    if not STACK_LANG_CACHES or not MARKUP_LANG_CACHES:
        print("\nERROR: caches missing.")
        raise SystemExit(1)
    from levanter.main import train_lm
    train_lm.main(train_config)
