# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# 1.4B Code-Only Base: 24.6B Llama tokens of curated code, single-phase from scratch.
#
# Why: builds the "code-as-reasoning-prior" base for the experiment Dongwei
# proposed 2026-06-15 night — train pretty long on high-quality code first,
# then continue on very little NL, compare reasoning to NL-only training of
# same total compute.
#
# Recipe is A5-style (single-phase, fresh init, cosine to 0):
#   batch=256 × seq=4096 → 1.05 M tokens/step
#   target ≈ 24.6 B → 23,430 steps  (one epoch of available curated code)
#   LR 3e-4 cosine to 0, wd 0.1, β=(0.9, 0.95), warmup 1%, max_grad_norm 1.0
#   fresh init (no checkpoint load)
#
# Data composition (Llama-token-proportional weighting):
#   Stack-Edu Python ≥3.0          (clean)     8.8 B   ≈ 35.7 %
#   Stack-Edu Python [2.8, 3.0)    (low)       2.9 B   ≈ 11.8 %
#   Stack-Edu Python [2.7, 2.8)    (mid)       1.9 B   ≈  7.7 %
#   Stack-Edu Python [2.5, 2.7)    (low2)      3.5 B   ≈ 14.2 %  (lowest quality, still Stack-Edu filtered)
#   Nemotron Code-Concepts                     7.3 B   ≈ 29.7 %
#   Nemotron Unconditional-Algorithmic         0.2 B   ≈  0.8 %
#   TOTAL                                     24.6 B
#
# Lessons baked in from 2026-06-15 audit:
#   - Hamilton/largest-remainder rounding in MixtureDataset (lib/levanter/.../mixture.py)
#     keeps per-block apportionment within 1 sample of intended even with 6
#     small-weight components. SP-NL bucket distortion (81/17/2 instead of
#     90/8/2) bug is fixed; same fix protects this script.
#   - `_resolve_cache` asserts single-match (no `matches[-1]` silent drift).
#   - `log_jaxprs=False, log_xla_hlo=False` is the new TrainerConfig default
#     (lib/levanter/.../trainer.py) → avoids the wandb-core nil-ctx panic +
#     md5 SIGBUS that haunted the C5-v6-NEW retry chain.
#
# Llama token counts above are nominal estimates from the fetcher logs;
# actual cache rows × ~683 tok/row are looked up at script load.

import os
from datetime import timedelta
from pathlib import Path

_secrets = Path(__file__).resolve().parents[2] / ".secrets"
if _secrets.exists():
    for line in _secrets.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

import json
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
    """Strict cache resolver — asserts exactly one match.

    Pinned 2026-06-15 to refuse `matches[-1]` silent-pick when a retokenize
    creates a second cache with a different hash.
    """
    matches = sorted(_TOKENIZED_BASE.glob(f"{prefix}-*"))
    if not matches:
        raise FileNotFoundError(
            f"No tokenized cache for prefix '{prefix}'. Run the tokenize step first."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple tokenized caches match prefix '{prefix}': "
            + str([m.name for m in matches])
            + ". Pin one explicitly (include hash)."
        )
    return str(matches[0])


def _shard_rows(cache_dir: str) -> int:
    """Read the cache's top-level shard_ledger.json and return total_num_rows."""
    p = Path(cache_dir) / "train" / "shard_ledger.json"
    return int(json.loads(p.read_text())["total_num_rows"])


def _shard_tokens(cache_dir: str) -> int:
    """Read the cache's top-level .stats.json and return total_tokens.

    The .stats.json is written by the tokenizer alongside shard_ledger.json
    and contains the EXACT token count (no heuristic).
    """
    p = Path(cache_dir) / "train" / ".stats.json"
    return int(json.loads(p.read_text())["total_tokens"])


# === Resolve each CODE source (threshold ≥ 2.7) — 1 epoch at ~19.73 B ===
_CODE_SOURCES = [
    ("se_python_clean",  "c5v2_stack_edu_python_clean"),       # score ≥ 3.0
    ("se_python_low",    "c5v6new_stack_edu_python_low"),      # score [2.8, 3.0)
    ("se_python_mid",    "c5_25b_se_python_mid"),              # score [2.7, 2.8)
    ("nemotron_cc",      "c5v2_nemotron_code_concepts"),
    ("nemotron_ua",      "c5v2_nemotron_unconditional_algorithmic"),
]
# === Resolve each MARKUP source — separate 20% slot ===
_MARKUP_SOURCES = [
    ("se_markdown",      "c5v2_stack_edu_markdown_clean"),     # Stack-Edu Markdown ≥ 3.0
]

# (name, cache_dir, rows, measured_tokens)
_code_components_with_rows: list[tuple[str, str, int, int]] = []
for name, prefix in _CODE_SOURCES:
    cache = _resolve_cache(prefix)
    rows = _shard_rows(cache)
    tokens = _shard_tokens(cache)
    _code_components_with_rows.append((name, cache, rows, tokens))

_markup_components_with_rows: list[tuple[str, str, int, int]] = []
for name, prefix in _MARKUP_SOURCES:
    cache = _resolve_cache(prefix)
    rows = _shard_rows(cache)
    tokens = _shard_tokens(cache)
    _markup_components_with_rows.append((name, cache, rows, tokens))

TOTAL_CODE_ROWS = sum(rows for _, _, rows, _ in _code_components_with_rows) or 1
TOTAL_CODE_TOKENS = sum(tokens for _, _, _, tokens in _code_components_with_rows) or 1
TOTAL_MARKUP_ROWS = sum(rows for _, _, rows, _ in _markup_components_with_rows) or 1
TOTAL_MARKUP_TOKENS = sum(tokens for _, _, _, tokens in _markup_components_with_rows) or 1

# Combined for backwards-compat prints
_components_with_rows = _code_components_with_rows + _markup_components_with_rows
TOTAL_TOKENS_MEASURED = TOTAL_CODE_TOKENS + TOTAL_MARKUP_TOKENS  # informational only


# === Held-out evals ===
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


# Match A5-style batching: total batch = TOTAL_GPUS * 8.
# 4-node = 256 × 4096 = 1.05M tokens/step.
# num_train_steps = MEASURED tokens / tokens_per_step → cosine LR ends EXACTLY
# at one epoch of the corpus (no over-trained tail at tiny LR, no wasted compute).
NUM_PROC = int(os.environ.get("JAX_DIST_NUM_PROCESSES", "1"))
TOTAL_GPUS = NUM_PROC * 8
PER_DEVICE_PARALLELISM = 8
TRAIN_BATCH_SIZE = TOTAL_GPUS * PER_DEVICE_PARALLELISM
TOKENS_PER_STEP = TRAIN_BATCH_SIZE * 4096
# Target: ~24.9 B at 80% code (≥2.7 threshold = 19.73 B available, ~1.0 epoch)
# + 20% markup (Stack-Edu Markdown 12.6 B available, ~0.4 epoch).
# Set N = code_tokens / 0.80 so code slot sees exactly 1 epoch.
NUM_TRAIN_STEPS = max(1, int(TOTAL_CODE_TOKENS / 0.80) // TOKENS_PER_STEP)
TOKENS_AVAILABLE = NUM_TRAIN_STEPS * TOKENS_PER_STEP


# === Build components dict + row-proportional weights ===
# Row-proportional weights yield TOKEN-proportional sampling: when sampling
# w_i × N rows from component i, expected tokens contributed are
# w_i × N × tok_per_row_i = (rows_i / total_rows) × N × tok_per_row_i
#                        = (tokens_i / total_rows) × N,
# which is proportional to tokens_i (i.e., each component's tokens appear in
# proportion to its share of the total token budget).
_code_components = {
    name: DatasetComponent(cache_dir=cache)
    for name, cache, _, _ in _code_components_with_rows
}
_markup_components = {
    name: DatasetComponent(cache_dir=cache)
    for name, cache, _, _ in _markup_components_with_rows
}
# 80% code, row-proportional within (yields token-proportional sampling)
_code_weights = {
    name: 0.80 * rows / TOTAL_CODE_ROWS
    for name, _, rows, _ in _code_components_with_rows
}
# 20% markup, row-proportional within
_markup_weights = {
    name: 0.20 * rows / TOTAL_MARKUP_ROWS
    for name, _, rows, _ in _markup_components_with_rows
}
_shard_val_sizes = {
    name: 0
    for name, _, _, _ in (_code_components_with_rows + _markup_components_with_rows)
}

train_weights = {
    **_code_weights,
    **_markup_weights,
    "dclm_200m_val": 0.0,
    **{f"paloma_{s}": 0.0 for s in PALOMA_SUBSETS},
}

data_config = LmDataConfig(
    components={
        **_code_components,
        **_markup_components,
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights=train_weights,
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
            tags=["1.4b", "code25b-clean", "single-phase", "curated-code-2.7threshold", "code+markup-80-20", "wd-0.1", f"nodes-{NUM_PROC}"],
            save_code=False,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_code25b_clean/",
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
    print("=== 1.4B Code-Only Base CLEAN (single-phase, 80% curated code ≥2.7 + 20% Stack-Edu Markdown) ===")
    print(f"  num_processes (nodes): {NUM_PROC}  total GPUs: {TOTAL_GPUS}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}  per-device: {PER_DEVICE_PARALLELISM}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}  tokens trained: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f} B")
    print(f"  LR=3e-4, WD=0.1, cosine to 0 (warmup 1%)")
    print()
    print(f"  CODE slot (80%, target {NUM_TRAIN_STEPS * TOKENS_PER_STEP * 0.80 / 1e9:.2f} B):")
    for name, cache, rows, tokens in _code_components_with_rows:
        cache_short = cache.replace(BASE_TOKENIZED + "/", "")
        weight = _code_weights[name]
        tok_b = tokens / 1e9
        tpr = tokens / rows if rows else 0
        print(f"    {name:<22} ({tok_b:>4.2f} B, {tpr:>5.0f} tok/row, w={weight*100:>5.2f}%)  ← {cache_short}")
    print(f"  CODE total: {TOTAL_CODE_TOKENS / 1e9:.2f} B (1.0 epoch at code-slot target)")
    print()
    print(f"  MARKUP slot (20%, target {NUM_TRAIN_STEPS * TOKENS_PER_STEP * 0.20 / 1e9:.2f} B):")
    for name, cache, rows, tokens in _markup_components_with_rows:
        cache_short = cache.replace(BASE_TOKENIZED + "/", "")
        weight = _markup_weights[name]
        tok_b = tokens / 1e9
        tpr = tokens / rows if rows else 0
        print(f"    {name:<22} ({tok_b:>4.2f} B, {tpr:>5.0f} tok/row, w={weight*100:>5.2f}%)  ← {cache_short}")
    markup_target = NUM_TRAIN_STEPS * TOKENS_PER_STEP * 0.20
    print(f"  MARKUP total available: {TOTAL_MARKUP_TOKENS / 1e9:.2f} B ({markup_target / TOTAL_MARKUP_TOKENS:.2f} epoch at markup-slot target)")
    print()
    print(f"  total tokens to train: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f} B")
    from levanter.main import train_lm
    train_lm.main(train_config)
