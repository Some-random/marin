# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# 1.4B C5-v8r PHASE 2: continued text pretraining on SlimPajama-NL,
# initialized from a RANDOM-code phase-1 endpoint instead of C5-v3's
# curated-code stage 1.
#
# Hypothesis (proposed 2026-06-16):
# C5 vs C5-v2 in single-cosine + DCLM stage 2 showed curated code helps
# Code itself (+215%) but not general reasoning/NL. We suspect the failed
# transfer is masked because DCLM is the wrong continuation diet for a
# code-prior (C5-v3 vs C5-v4 already showed DCLM→SP-NL gives +12–56%
# across the board when phase 1 has curated code). C5-v8r tests this:
# if we pair RANDOM code phase 1 with SP-NL phase 2 (separate cosine),
# does it match C5-v4 (curated code phase 1 + SP-NL phase 2)?
#   - C5-v8r ≈ C5-v4 → code-data axis is null at this scale, the
#     "curated code only helps Code" finding stands.
#   - C5-v8r < C5-v4 → curated code IS contributing latent signal
#     that needed SP-NL to surface.
#
# Initialization: fresh cosine 3e-4 → 0 from C5's phase-1 step-14672
# (RANDOM StarCoderData + raw markup), fresh optimizer state. Code mix
# proportions inside the 10% code+markup slot: SAME as C5-v4 (curated),
# so the only difference vs C5-v4 is the INIT checkpoint.
#
# Caveat: C5's step-14672 was the midpoint of a CONTINUOUS-cosine 29343-
# step run, so its LR at the ckpt was ~1.5e-4 (mid-decay), not 0.
# C5-v3 phase 1's step-14671 finished at LR≈0. So the two init points
# have different "lr-schedule-so-far" histories — not perfectly clean,
# but the cleanest comparison without a self-contained random-code
# separate-cosine phase 1 redo (~8h compute).

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
from levanter.distributed import DistributedConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker import NoopConfig
from levanter.tracker.wandb import WandbConfig
from levanter.distributed import RayConfig
from levanter.trainer import TrainerConfig

from experiments.reasoning_pretraining.code_ladder.models.models import model_dict

BASE_TOKENIZED = "/fsx/users/dongweij/marin/outputs/tokenized"
_TOKENIZED_BASE = Path(BASE_TOKENIZED)


# === Phase 1 final checkpoint to initialize from ===
# RANDOM-code endpoint: C5's continuous-cosine run, step 14672 (stage-1
# transition point). Differs from C5-v4 which used C5-v3's curated-code
# step-14671 endpoint.
PHASE1_INIT_FROM = "checkpoints/1_4b_c5v8r_phase1/ll26tgto/step-14671"


def _resolve_cache(prefix: str) -> str:
    matches = sorted(_TOKENIZED_BASE.glob(f"{prefix}-*"))
    if not matches:
        raise FileNotFoundError(
            f"No tokenized cache for prefix '{prefix}'. "
            f"Run `MARIN_PREFIX=/fsx/users/dongweij/marin/outputs .venv/bin/python "
            f"-m experiments.reasoning_pretraining.code_ladder.data.code_data_c5v2` first."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple tokenized caches match prefix '{prefix}': {[m.name for m in matches]}. "
            f"Pin one explicitly by including the hash, e.g. '{matches[0].name}' instead of bare prefix."
        )
    return str(matches[0])


try:
    SE_PYTHON_CACHE = _resolve_cache("c5v2_stack_edu_python_clean")
    SE_MARKDOWN_CACHE = _resolve_cache("c5v2_stack_edu_markdown_clean")
    NEMOTRON_CC_CACHE = _resolve_cache("c5v2_nemotron_code_concepts")
    NEMOTRON_UA_CACHE = _resolve_cache("c5v2_nemotron_unconditional_algorithmic")
except FileNotFoundError as _e:
    SE_PYTHON_CACHE = ""
    SE_MARKDOWN_CACHE = ""
    NEMOTRON_CC_CACHE = ""
    NEMOTRON_UA_CACHE = ""
    print(f"[c5v8r-p2] WARN: {_e}")


# === TEXT: SlimPajama-NL (replaces DCLM) ===
# Tokenized cache at outputs/tokenized/slimpajama_nl-<hash>/train/part-<N>-of-128/
# After filtering Github + StackExchange: 13.23 B Llama-3.1 tokens.
# Per-source share (vs Aryabumi target): CC 58.0% (57.0%), C4 30.7% (29.2%),
# ArXiv 3.7% (5.0%), Books 4.3% (4.6%), Wikipedia 3.3% (4.2%).
# (Add chunk_2 shards here after the second tokenize finishes if you want
# >13.23 B unique-text budget — current setting epochs the cache ~1.05× to
# hit phase 2's 13.85 B text target.)
def _collect_sp_nl_shards_with_rows() -> list[tuple[str, int]]:
    """Per-part SP-NL components with row counts for ROW-PROPORTIONAL weighting.

    chunk1: 128 parts × ~100 M tokens (~99,892 rows/part); chunk2: 100 parts ×
    ~519 M tokens (~518,293 rows/part). Each part is its own Levanter cache
    (with train/+validation/ sub-caches written by zephyr tokenize), so
    per-part components work natively with `build_caches("validation")` even
    with `num_validation_sequences=0`.

    Weighting is ROW-proportional (rows[i] / total_rows), which is
    approximately token-proportional because chunk1 and chunk2 have similar
    avg-tokens-per-row (chunk-level row/token ratios are within ~5%). It is
    NOT exactly token-proportional — if the per-part token counts ever
    diverge meaningfully, we'd need a separate per-part token-count column.

    Replaces the original per-part-uniform bug where chunk1's 128 parts
    received 56% of the SP-NL weight despite holding only 19.8% of the rows.
    Combined with the Hamilton/largest-remainder rounding fix in
    `mixture.py:144` (2026-06-15), the actual per-block mix matches the
    intended 90/8/2 within 0.04 pp.

    Audit history (2026-06-15):
      - pre-fix per-part-uniform: chunk1/chunk2 = 56/44 (intended 19.8/80.2)
      - first attempt "row-proportional + 228 components + dump-to-largest":
        SP/code/markup = 81/17/2 (intended 90/8/2). Caught by Dongwei review;
        the 189-sample remainder pooled into code_se_python, jumping it from
        floor 88 → actual 277 per 2048-block.
      - now per-part + Hamilton + row-proportional: SP/code/markup =
        90.04/7.96/2.00 (within 0.04 pp), chunk1/chunk2 = 20.82/79.18
        (within 1 pp of intended 19.79/80.21).
    """
    import json
    out: list[tuple[str, int]] = []
    for prefix in ("slimpajama_nl_en", "slimpajama_nl_chunk2_en"):
        try:
            root = Path(_resolve_cache(prefix))
        except FileNotFoundError:
            continue
        ledger = json.loads((root / "train" / "shard_ledger.json").read_text())
        for part_name, rows in sorted(ledger["shard_rows"].items()):
            part_path = root / "train" / part_name
            if not part_path.exists():
                continue
            out.append((str(part_path), int(rows)))
    return out


SP_NL_SHARDS_WITH_ROWS = _collect_sp_nl_shards_with_rows()
SP_NL_SHARDS = [p for p, _ in SP_NL_SHARDS_WITH_ROWS]
SP_NL_TOTAL_ROWS = sum(r for _, r in SP_NL_SHARDS_WITH_ROWS) or 1


# === DCLM_200M_VAL — kept for in-domain held-out tracking ===
DCLM_VAL = f"{BASE_TOKENIZED}/data_efficiency/dclm_200m_val-415aea"


# === Code mix proportions (UNCHANGED from C5-v3 phase 2) ===
CODE_RATIOS = {
    "se_python": 8.8 / 16.3,
    "nemotron_cc": 7.3 / 16.3,
    "nemotron_ua": 0.2 / 16.3,
}
MARKUP_RATIOS = {"se_markdown": 1.0}


# === Paloma eval components (UNCHANGED) ===
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
NUM_TRAIN_STEPS = 14_672  # UNCHANGED: same phase-2 budget as C5-v3 = 15.39 B tokens.


_code_key_to_cache = {
    "se_python": SE_PYTHON_CACHE,
    "nemotron_cc": NEMOTRON_CC_CACHE,
    "nemotron_ua": NEMOTRON_UA_CACHE,
}
_markup_key_to_cache = {
    "se_markdown": SE_MARKDOWN_CACHE,
}


def _phase2_weights() -> dict[str, float]:
    """90% SlimPajama-NL (ROW-PROPORTIONAL across parts, ≈ token-proportional) + 10% (80% code + 20% markup)."""
    w: dict[str, float] = {}
    if SP_NL_SHARDS_WITH_ROWS:
        for i, (_, rows) in enumerate(SP_NL_SHARDS_WITH_ROWS):
            w[f"sp_nl_shard{i:03d}"] = 0.90 * rows / SP_NL_TOTAL_ROWS
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

_shard_val_sizes = {k: 0 for k in {**_sp_nl_components, **_code_components, **_markup_components}}

data_config = LmDataConfig(
    components={
        **_sp_nl_components,
        **_code_components,
        **_markup_components,
        "dclm_200m_val": DatasetComponent(cache_dir=DCLM_VAL),
        **paloma_components,
    },
    train_weights=_phase2_weights(),
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
    initialize_from_checkpoint_path=os.environ.get("C5V3_PHASE1_CKPT", PHASE1_INIT_FROM),
    trainer=TrainerConfig(
        seed=0,
        tracker=WandbConfig(
            project="dongwei-data-efficiency",
            entity="dongwei_jiang",
            tags=["1.4b", "c5v8r", "phase2-text-continued", "slimpajama-nl", "random-code-init", "wd-0.1", f"nodes-{NUM_PROC}"],
            save_code=False,
        ),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=TRAIN_BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        steps_per_eval=NUM_TRAIN_STEPS // 8,
        per_device_parallelism=PER_DEVICE_PARALLELISM,
        per_device_eval_parallelism=PER_DEVICE_PARALLELISM,
        checkpointer=CheckpointerConfig(
            base_path="checkpoints/1_4b_c5v8r_phase2/",
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
    print("=== 1.4B C5-v4 PHASE 2 (continued text from code-LM, SlimPajama-NL) ===")
    print(f"  num_processes (nodes): {NUM_PROC}  total GPUs: {TOTAL_GPUS}")
    print(f"  initialize_from_checkpoint_path: {train_config.initialize_from_checkpoint_path}")
    print(f"  train_batch_size: {TRAIN_BATCH_SIZE}  per-device: {PER_DEVICE_PARALLELISM}")
    print(f"  num_train_steps: {NUM_TRAIN_STEPS:,}  total trained tokens: {NUM_TRAIN_STEPS * TOKENS_PER_STEP / 1e9:.2f}B")
    print(f"  LR=3e-4 (FRESH), WD=0.1, cosine to 0 (warmup 1% ≈ 147 steps)")
    print(f"  data: 90% SlimPajama-NL ({len(SP_NL_SHARDS)} parts, row-proportional ≈ token-proportional) + 10% (80% code + 20% markup)")
    if not SE_PYTHON_CACHE or not SE_MARKDOWN_CACHE or not NEMOTRON_CC_CACHE or not NEMOTRON_UA_CACHE:
        print("\nERROR: code caches missing.")
        raise SystemExit(1)
    if not SP_NL_SHARDS:
        print("\nERROR: SlimPajama-NL parts missing.")
        raise SystemExit(1)
    from levanter.main import train_lm
    train_lm.main(train_config)
