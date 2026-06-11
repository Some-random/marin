#!/usr/bin/env bash
# Phase 6 only: paloma_bpb for 6 missing models. Sets HF_HOME to the shared
# cache so it doesn't re-fetch (avoiding remote-node auth + rate-limit issues).
cd /fsx/users/dongweij/marin
export HF_HOME=/fsx/users/ymcho/helper/hf_cache
export HF_TOKEN=$(cat /fsx/users/dongweij/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1

PALOMA_SUBSETS=(
  paloma_4chan_meta_sep paloma_c4_100_domains paloma_c4_en paloma_dolma-v1_5
  paloma_dolma_100_programing_languages paloma_dolma_100_subreddits
  paloma_falcon-refinedweb paloma_gab paloma_m2d2_s2orc_unsplit
  paloma_m2d2_wikipedia_unsplit paloma_manosphere_meta_sep paloma_mc4
  paloma_ptb paloma_redpajama paloma_twitterAAE_HELM_fixed paloma_wikitext_103
)
SMALL=(paloma_ptb)
is_small() { local x="$1"; for s in "${SMALL[@]}"; do [[ "$s" == "$x" ]] && return 0; done; return 1; }

declare -A PAL_CKPTS=(
  [baseline_s10000]=/fsx/users/dongweij/marin/checkpoints/baseline_step10000_hf
  [v1_s10000]=/fsx/users/dongweij/marin/checkpoints/v1_step10000_hf
  [v2_s10000]=/fsx/users/dongweij/marin/checkpoints/v2_step10000_hf
  [v2_final]=/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg_v2_hf
  [wd3_2_x16_s10000]=/fsx/users/dongweij/marin/checkpoints/wd3_2_x16_step10000_hf
  [wd3_2_x16_final]=/fsx/users/dongweij/marin/checkpoints/1_4b_wd3_2_x16_nocrossblock_hf
)

OUT_ROOT_PAL=/fsx/users/dongweij/marin/outputs/eval_results/paloma_missing_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT_PAL"
echo "=== Phase 6: paloma_bpb start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
for LABEL in baseline_s10000 v1_s10000 v2_s10000 v2_final wd3_2_x16_s10000 wd3_2_x16_final; do
  CKPT="${PAL_CKPTS[$LABEL]}"
  for T in "${PALOMA_SUBSETS[@]}"; do
    OUT="$OUT_ROOT_PAL/${LABEL}__${T}"
    mkdir -p "$OUT"
    if is_small "$T"; then
      LAUNCH=(.venv/bin/python -m lm_eval)
    else
      LAUNCH=(.venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 -m lm_eval)
    fi
    echo "  [$LABEL/$T] start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')"
    "${LAUNCH[@]}" --model hf \
      --model_args "pretrained=$CKPT,dtype=bfloat16,trust_remote_code=True" \
      --tasks "$T" --batch_size 16 --output_path "$OUT" \
      --trust_remote_code > "$OUT.log" 2>&1 && \
      echo "  [$LABEL/$T] DONE" || echo "  [$LABEL/$T] FAILED-CONTINUE"
  done
done
echo "=== Phase 6: paloma_bpb DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
echo "=== PHASE6 STANDALONE COMPLETE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
