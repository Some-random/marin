#!/usr/bin/env bash
# Phase 5-7 of the eval chain. Picks up after queued_chain.sh finishes the
# wd=1.6/x8 training. Adds:
#   5. v2 FINAL downstream eval (full benchmark suite at proper n-shots)
#   6. paloma_bpb on the 6 checkpoints currently missing it:
#      baseline_s10000, v1_s10000, v2_s10000, v2_final, wd=3.2/x16_s10000, wd=3.2/x16_final
#   7. wd=1.6/x8 final-step downstream + paloma_bpb + looping smoke test

cd /fsx/users/dongweij/marin
export HF_ALLOW_CODE_EVAL=1

PREV_LOG="${PREV_LOG:-/fsx/users/dongweij/marin/logs/queued_chain_20260531_171650.log}"

echo "=== waiting for previous chain to log ALL CHAINED JOBS COMPLETE ==="
while ! grep -q "ALL CHAINED JOBS COMPLETE" "$PREV_LOG" 2>/dev/null; do sleep 30; done
echo "=== previous chain finished at $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="

# === Phase 5: v2 final downstream eval ===
OUT_ROOT_V2=/fsx/users/dongweij/marin/outputs/eval_results/v2_final_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT_V2"
run_lm_eval() {
  local LABEL="$1" CKPT="$2" TASKS="$3" NSHOT="$4" BATCH="$5" EXTRA="$6" OUT_ROOT="$7"
  local OUT="$OUT_ROOT/${LABEL}__${NSHOT}shot__$(echo "$TASKS" | tr ',' '_' | cut -c1-30)"
  mkdir -p "$OUT"
  echo "=== [$LABEL] tasks=$TASKS n-shot=$NSHOT start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    -m lm_eval --model hf \
    --model_args "pretrained=$CKPT,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" --num_fewshot "$NSHOT" --batch_size "$BATCH" \
    --log_samples --output_path "$OUT" \
    --include_path /fsx/users/dongweij/marin/experiments/data_efficiency \
    --trust_remote_code \
    $EXTRA > "$OUT.log" 2>&1 && \
    echo "=== [$LABEL] tasks=$TASKS DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ===" || \
    echo "=== [$LABEL] tasks=$TASKS FAILED-CONTINUE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
}

echo "=== Phase 5: v2 final downstream eval start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
V2_FINAL_HF=/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg_v2_hf
# Convert v2 final to HF if not already
if [ ! -d "$V2_FINAL_HF" ]; then
  echo "  converting v2 final to HF"
  .venv/bin/python << 'PYV2'
import os, sys
SECRETS = "/fsx/users/dongweij/marin/.secrets"
if os.path.exists(SECRETS):
    for line in open(SECRETS):
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, "/fsx/users/dongweij/marin")
import equinox as eqx, haliax as hax, jax
from experiments.data_efficiency.models import model_dict
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import load_tokenizer
SRC = "/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg_v2/joqfahkl/step-12799"
DST = "/fsx/users/dongweij/marin/checkpoints/1_4b_25code_alg_v2_hf"
mc = model_dict["1_4b4k"]
tok = load_tokenizer("meta-llama/Meta-Llama-3.1-8B")
Vocab = hax.Axis("vocab", len(tok))
mesh = jax.sharding.Mesh(jax.devices("cpu")[:1], ("data",))
with hax.partitioning.set_mesh(mesh):
    model = eqx.filter_eval_shape(mc.build, Vocab, key=jax.random.PRNGKey(0))
    model = load_checkpoint(model, SRC, subpath="model")
    cv = mc.hf_checkpoint_converter().replaced(tokenizer=tok)
    cv.save_pretrained(model, DST)
    tok.save_pretrained(DST)
print("v2 final HF conversion done")
PYV2
fi
LABEL=v2_final
for ENTRY in "arc_easy,arc_challenge:25:16:" "hellaswag:10:16:" \
             "winogrande,mmlu,gsm8k:5:16:" \
             "piqa,boolq,sciq,openbookqa,commonsense_qa,social_iqa,logiqa:0:16:" \
             "openbookqa_fact:0:16:" \
             "gsm8k_cot:8:8:--confirm_run_unsafe_code" \
             "mbpp:3:8:--confirm_run_unsafe_code" \
             "humaneval:0:8:--confirm_run_unsafe_code" \
             "minerva_math:4:8:"; do
  IFS=":" read -r TASKS NSHOT BATCH EXTRA <<< "$ENTRY"
  run_lm_eval "$LABEL" "$V2_FINAL_HF" "$TASKS" "$NSHOT" "$BATCH" "$EXTRA" "$OUT_ROOT_V2"
done
echo "=== Phase 5: v2 final downstream DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="

# === Phase 6: paloma_bpb on the 6 missing checkpoints ===
echo "=== Phase 6: paloma_bpb missing-models start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
OUT_ROOT_PAL=/fsx/users/dongweij/marin/outputs/eval_results/paloma_missing_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT_PAL"

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

run_paloma() {
  local LABEL="$1" CKPT="$2" T="$3"
  local OUT="$OUT_ROOT_PAL/${LABEL}__${T}"
  mkdir -p "$OUT"
  if is_small "$T"; then
    NPROC=1; LAUNCH=(.venv/bin/python -m lm_eval)
  else
    NPROC=8; LAUNCH=(.venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 -m lm_eval)
  fi
  echo "  [$LABEL/$T] (DP=$NPROC) start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')"
  "${LAUNCH[@]}" --model hf \
    --model_args "pretrained=$CKPT,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$T" --batch_size 16 --output_path "$OUT" \
    --trust_remote_code > "$OUT.log" 2>&1 && \
    echo "  [$LABEL/$T] DONE" || echo "  [$LABEL/$T] FAILED-CONTINUE"
}

for LABEL in baseline_s10000 v1_s10000 v2_s10000 v2_final wd3_2_x16_s10000 wd3_2_x16_final; do
  CKPT="${PAL_CKPTS[$LABEL]}"
  for T in "${PALOMA_SUBSETS[@]}"; do
    run_paloma "$LABEL" "$CKPT" "$T"
  done
done
echo "=== Phase 6: paloma_bpb missing-models DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="

# === Phase 7: wd=1.6/x8 final downstream + paloma + looping ===
echo "=== Phase 7: wd=1.6/x8 final eval start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
# Find the run dir (most recent under 1_4b_wd1_6_x8_nocrossblock/)
WD1_6_X8_RUNDIR=$(ls -d /fsx/users/dongweij/marin/checkpoints/1_4b_wd1_6_x8_nocrossblock/*/ 2>/dev/null | sort | tail -1)
WD1_6_X8_RUNDIR=${WD1_6_X8_RUNDIR%/}
WD1_6_X8_SRC=$WD1_6_X8_RUNDIR/step-6400
WD1_6_X8_HF=/fsx/users/dongweij/marin/checkpoints/wd1_6_x8_final_hf
echo "  rundir=$WD1_6_X8_RUNDIR src=$WD1_6_X8_SRC dst=$WD1_6_X8_HF"

if [ ! -d "$WD1_6_X8_HF" ]; then
  echo "  converting wd=1.6/x8 final to HF"
  .venv/bin/python << PYWX8
import os, sys
SECRETS = "/fsx/users/dongweij/marin/.secrets"
if os.path.exists(SECRETS):
    for line in open(SECRETS):
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, "/fsx/users/dongweij/marin")
import equinox as eqx, haliax as hax, jax
from experiments.data_efficiency.models import model_dict
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import load_tokenizer
SRC = "$WD1_6_X8_SRC"
DST = "$WD1_6_X8_HF"
mc = model_dict["1_4b4k"]
tok = load_tokenizer("meta-llama/Meta-Llama-3.1-8B")
Vocab = hax.Axis("vocab", len(tok))
mesh = jax.sharding.Mesh(jax.devices("cpu")[:1], ("data",))
with hax.partitioning.set_mesh(mesh):
    model = eqx.filter_eval_shape(mc.build, Vocab, key=jax.random.PRNGKey(0))
    model = load_checkpoint(model, SRC, subpath="model")
    cv = mc.hf_checkpoint_converter().replaced(tokenizer=tok)
    cv.save_pretrained(model, DST)
    tok.save_pretrained(DST)
print("wd=1.6/x8 final HF conversion done")
PYWX8
fi

OUT_ROOT_WX8=/fsx/users/dongweij/marin/outputs/eval_results/wd1_6_x8_final_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT_WX8"
LABEL=wd1_6_x8_final
for ENTRY in "arc_easy,arc_challenge:25:16:" "hellaswag:10:16:" \
             "winogrande,mmlu,gsm8k:5:16:" \
             "piqa,boolq,sciq,openbookqa,commonsense_qa,social_iqa,logiqa:0:16:" \
             "openbookqa_fact:0:16:" \
             "gsm8k_cot:8:8:--confirm_run_unsafe_code" \
             "mbpp:3:8:--confirm_run_unsafe_code" \
             "humaneval:0:8:--confirm_run_unsafe_code" \
             "minerva_math:4:8:"; do
  IFS=":" read -r TASKS NSHOT BATCH EXTRA <<< "$ENTRY"
  run_lm_eval "$LABEL" "$WD1_6_X8_HF" "$TASKS" "$NSHOT" "$BATCH" "$EXTRA" "$OUT_ROOT_WX8"
done

# Paloma for wd=1.6/x8 final
OUT_ROOT_WX8_PAL=/fsx/users/dongweij/marin/outputs/eval_results/wd1_6_x8_paloma_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT_WX8_PAL"
for T in "${PALOMA_SUBSETS[@]}"; do
  OUT="$OUT_ROOT_WX8_PAL/${LABEL}__${T}"
  mkdir -p "$OUT"
  if is_small "$T"; then
    NPROC=1; LAUNCH=(.venv/bin/python -m lm_eval)
  else
    NPROC=8; LAUNCH=(.venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 -m lm_eval)
  fi
  echo "  [$LABEL/$T] (DP=$NPROC) start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')"
  "${LAUNCH[@]}" --model hf \
    --model_args "pretrained=$WD1_6_X8_HF,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$T" --batch_size 16 --output_path "$OUT" \
    --trust_remote_code > "$OUT.log" 2>&1 && \
    echo "  [$LABEL/$T] DONE" || echo "  [$LABEL/$T] FAILED-CONTINUE"
done

echo "=== Phase 7: wd=1.6/x8 final eval DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
echo "=== ALL CHAIN-2 JOBS COMPLETE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
