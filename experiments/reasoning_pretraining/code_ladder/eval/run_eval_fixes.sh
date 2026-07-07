#!/usr/bin/env bash
# Re-run evals with correct settings:
#   - math at proper n-shot (gsm8k_cot 8, mbpp 3, minerva_math 4)
#   - openbookqa with the fact1 field (open-book MC, custom yaml)
#   - paloma subsets via lm-eval, so phi-1/phi-1.5 cells get filled
# Across all 4 models.

set -euo pipefail
cd /fsx/users/dongweij/marin
export HF_ALLOW_CODE_EVAL=1

OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/fixes_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"

# 16-subset Paloma list per Paloma v1.
PALOMA_TASKS="paloma_4chan_meta_sep,paloma_c4_100_domains,paloma_c4_en,paloma_dolma-v1_5,paloma_dolma_100_programing_languages,paloma_dolma_100_subreddits,paloma_falcon-refinedweb,paloma_gab,paloma_m2d2_s2orc_unsplit,paloma_m2d2_wikipedia_unsplit,paloma_manosphere_meta_sep,paloma_mc4,paloma_ptb,paloma_redpajama,paloma_twitterAAE_HELM_fixed,paloma_wikitext_103"

declare -A MODELS=(
  [baseline_nocross]=/fsx/users/dongweij/marin/checkpoints/1_4b_wd1_6_x16_nocrossblock_hf
  [code25_alg]=/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg_hf
  [phi-1]=microsoft/phi-1
  [phi-1.5]=microsoft/phi-1_5
)

run_one() {
  local LABEL="$1" MODEL_ID="$2" TASKS="$3" NSHOT="$4" BATCH="$5" EXTRA="$6"
  local OUT="$OUT_ROOT/${LABEL}__${NSHOT}shot__$(echo "$TASKS" | tr ',' '_' | cut -c1-30)"
  local LOG="$OUT.log"
  mkdir -p "$OUT"
  echo "=== [$LABEL] tasks=$TASKS n-shot=$NSHOT start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_ID,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" \
    --num_fewshot "$NSHOT" \
    --batch_size "$BATCH" \
    --log_samples \
    --output_path "$OUT" \
    --include_path /fsx/users/dongweij/marin/experiments/reasoning_pretraining/code_ladder/eval \
    $EXTRA \
    2>&1 | tee "$LOG"
  echo "=== [$LABEL] tasks=$TASKS n-shot=$NSHOT DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
}

for LABEL in "${!MODELS[@]}"; do
  M="${MODELS[$LABEL]}"
  # Math gen at proper n-shots, batched together via 8-shot (since these are gens
  # and lm-eval applies --num_fewshot to all tasks in the invocation, run separately)
  run_one "$LABEL" "$M" "gsm8k_cot" 8 8 "--confirm_run_unsafe_code"
  run_one "$LABEL" "$M" "mbpp" 3 8 "--confirm_run_unsafe_code"
  run_one "$LABEL" "$M" "minerva_math" 4 8 ""
  # openbookqa with-fact (custom yaml)
  run_one "$LABEL" "$M" "openbookqa_fact" 0 16 ""
  # NOTE: paloma_* lm-eval tasks require gated `EleutherAI/paloma` dataset
  # we don't have access to. For our 1.4B models we already have Levanter
  # Paloma loss from training. For phi-1/phi-1.5, run through Levanter
  # separately (TODO).
done

echo "All eval fixes complete. Results under $OUT_ROOT/"
