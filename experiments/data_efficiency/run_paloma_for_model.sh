#!/usr/bin/env bash
# Run paloma per-subset bpb eval for one model on one node.
# Usage: run_paloma_for_model.sh <LABEL> <HF_DIR>
# Designed to run inside an ssh session on the eval node.

set -uo pipefail
cd /fsx/users/dongweij/marin
export HF_TOKEN=$(cat /fsx/users/dongweij/.cache/huggingface/token)
export HF_DATASETS_CACHE=/fsx/users/dongweij/marin/outputs/hf_cache/datasets
# allenai/paloma is a Hub-hosted gated dataset; offline mode breaks it.
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
# Fix the NCCL gather_object P2P/CUMEM IPC-buffer OOM (root-caused 2026-06-22; see run_eval_v2.sh).
# This is what killed the large paloma subsets (dolma/m2d2/manosphere/redpajama/mc4) under
# multi-GPU. Disabling P2P fixes it at full 8-GPU speed — no more single-GPU fallback needed.
export NCCL_P2P_DISABLE=1

LABEL="${1:?LABEL required}"
HF_DST="${2:?HF_DST required}"
# Smaller batch for larger models (4B OOMs at 16 on 8x40GB).
BATCH_SIZE="${BATCH_SIZE:-16}"

OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/paloma_${LABEL}_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL paloma → $OUT_ROOT"

PALOMA_SUBSETS=(
  paloma_4chan_meta_sep paloma_c4_100_domains paloma_c4_en paloma_dolma-v1_5
  paloma_dolma_100_programing_languages paloma_dolma_100_subreddits
  paloma_falcon-refinedweb paloma_gab paloma_m2d2_s2orc_unsplit
  paloma_m2d2_wikipedia_unsplit paloma_manosphere_meta_sep paloma_mc4
  paloma_ptb paloma_redpajama paloma_twitterAAE_HELM_fixed paloma_wikitext_103
)

for T in "${PALOMA_SUBSETS[@]}"; do
  OUT="$OUT_ROOT/${T}"
  mkdir -p "$OUT"
  # Single-GPU for memory-heavy subsets:
  #   paloma_ptb              — known long contexts, OOMs on 8-way
  #   paloma_falcon-refinedweb — same, observed 2026-06-11 (req 15.66 GiB / 15.17 free)
  if [[ "$T" == "paloma_ptb" || "$T" == "paloma_falcon-refinedweb" ]]; then
    LAUNCH=(.venv/bin/python -m lm_eval)
  else
    LAUNCH=(.venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 -m lm_eval)
  fi
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL $T start"
  "${LAUNCH[@]}" --model hf \
    --model_args "pretrained=$HF_DST,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$T" --batch_size "$BATCH_SIZE" --output_path "$OUT" \
    --trust_remote_code > "$OUT.log" 2>&1 \
    && echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL $T DONE" \
    || echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL $T FAILED"
done

echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL paloma ALL DONE → $OUT_ROOT"
