#!/usr/bin/env bash
# Run lm-eval's paloma_* tasks on phi-1 and phi-1.5, one subset per invocation.
# paloma_ptb has only 1 test doc — falls back to single-GPU (DP=1) to avoid
# the "no docs on rank N" crash. Failures on one subset don't kill the chain.

cd /fsx/users/dongweij/marin

SUBSETS=(
  paloma_4chan_meta_sep paloma_c4_100_domains paloma_c4_en paloma_dolma-v1_5
  paloma_dolma_100_programing_languages paloma_dolma_100_subreddits
  paloma_falcon-refinedweb paloma_gab paloma_m2d2_s2orc_unsplit
  paloma_m2d2_wikipedia_unsplit paloma_manosphere_meta_sep paloma_mc4
  paloma_ptb paloma_redpajama paloma_twitterAAE_HELM_fixed paloma_wikitext_103
)

# Subsets with < 8 test docs — can't shard across 8 GPUs.
SMALL_SUBSETS=(paloma_ptb)

is_small() {
  local x="$1"
  for s in "${SMALL_SUBSETS[@]}"; do [[ "$s" == "$x" ]] && return 0; done
  return 1
}

OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/paloma_phi_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"

for L in phi-1 phi-1.5; do
  CKPT=microsoft/${L/./_}
  for T in "${SUBSETS[@]}"; do
    OUT="$OUT_ROOT/${L}__${T}"
    LOG="$OUT.log"
    mkdir -p "$OUT"
    if is_small "$T"; then
      NPROC=1
      LAUNCH=(.venv/bin/python -m lm_eval)
    else
      NPROC=8
      LAUNCH=(.venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 -m lm_eval)
    fi
    echo "=== [$L] $T (DP=$NPROC) start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
    "${LAUNCH[@]}" --model hf \
      --model_args "pretrained=$CKPT,dtype=bfloat16,trust_remote_code=True" \
      --tasks "$T" \
      --batch_size 16 \
      --output_path "$OUT" \
      > "$LOG" 2>&1 && \
      echo "=== [$L] $T DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ===" || \
      echo "=== [$L] $T FAILED-CONTINUE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') (see $LOG) ==="
  done
done
echo "All paloma phi runs complete."
