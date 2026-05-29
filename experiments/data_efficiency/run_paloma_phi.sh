#!/usr/bin/env bash
# Run lm-eval's paloma_* tasks on phi-1 and phi-1.5 (allenai/paloma path
# patched in lm-eval's task yamls). Our 1.4B models already have Paloma
# from Levanter eval during training.
set -euo pipefail
cd /fsx/users/dongweij/marin

PALOMA_TASKS="paloma_4chan_meta_sep,paloma_c4_100_domains,paloma_c4_en,paloma_dolma-v1_5,paloma_dolma_100_programing_languages,paloma_dolma_100_subreddits,paloma_falcon-refinedweb,paloma_gab,paloma_m2d2_s2orc_unsplit,paloma_m2d2_wikipedia_unsplit,paloma_manosphere_meta_sep,paloma_mc4,paloma_ptb,paloma_redpajama,paloma_twitterAAE_HELM_fixed,paloma_wikitext_103"

OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/paloma_phi_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"

for L in phi-1 phi-1.5; do
  CKPT=microsoft/${L/./_}
  OUT="$OUT_ROOT/${L}__paloma"
  LOG="$OUT.log"
  mkdir -p "$OUT"
  echo "=== [$L] start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    -m lm_eval --model hf \
    --model_args "pretrained=$CKPT,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$PALOMA_TASKS" \
    --batch_size 16 \
    --log_samples --output_path "$OUT" \
    2>&1 | tee "$LOG"
  echo "=== [$L] DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
done
echo "Paloma phi runs complete."
