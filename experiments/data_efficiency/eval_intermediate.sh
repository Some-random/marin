#!/usr/bin/env bash
# Convert a Levanter checkpoint to HF and run lm-eval-harness on it.
# Runs on a single 8-GPU node (default gpu-st-p4d24xlarge-4).
#
# Usage:
#   ./eval_intermediate.sh --label A5-step14672 \
#                          --src /path/to/levanter/checkpoint/step-14672 \
#                          --hf-dst /path/to/hf/output \
#                          --node gpu-st-p4d24xlarge-4
#
# Pipeline:
#   1. SSH to node, run convert (Levanter -> HF) on CPU mesh
#   2. SSH to node, run lm-eval-harness (full downstream suite) on 8 GPUs
#   3. Write results under outputs/eval_results/intermediate_<LABEL>_<TS>/

set -euo pipefail

LABEL=""
SRC=""
HF_DST=""
NODE="gpu-st-p4d24xlarge-4"

while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --src) SRC="$2"; shift 2 ;;
    --hf-dst) HF_DST="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$LABEL" ] || [ -z "$SRC" ] || [ -z "$HF_DST" ]; then
  echo "Usage: $0 --label NAME --src LEVANTER_DIR --hf-dst HF_DIR [--node NODENAME]" >&2
  exit 1
fi

TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/intermediate_${LABEL}_${TS}
mkdir -p "$OUT_ROOT"

echo "=== Intermediate eval ==="
echo "  Label:  $LABEL"
echo "  Source: $SRC"
echo "  HF dst: $HF_DST"
echo "  Node:   $NODE"
echo "  Out:    $OUT_ROOT"
echo "  Start:  $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')"
echo ""

# Step 1: convert Levanter -> HF if not already
if [ ! -d "$HF_DST" ]; then
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] Converting Levanter -> HF on $NODE..."
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$NODE" "
    cd /fsx/users/dongweij/marin
    .venv/bin/python << PYEOF
import os, sys
SECRETS = '/fsx/users/dongweij/marin/.secrets'
if os.path.exists(SECRETS):
    for line in open(SECRETS):
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1); os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, '/fsx/users/dongweij/marin')
import equinox as eqx, haliax as hax, jax
from experiments.data_efficiency.models import model_dict
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import load_tokenizer
mc = model_dict['1_4b4k']
tok = load_tokenizer('meta-llama/Meta-Llama-3.1-8B')
Vocab = hax.Axis('vocab', len(tok))
mesh = jax.sharding.Mesh(jax.devices('cpu')[:1], ('data',))
with hax.partitioning.set_mesh(mesh):
    model = eqx.filter_eval_shape(mc.build, Vocab, key=jax.random.PRNGKey(0))
    model = load_checkpoint(model, '$SRC', subpath='model')
    cv = mc.hf_checkpoint_converter().replaced(tokenizer=tok)
    cv.save_pretrained(model, '$HF_DST')
    tok.save_pretrained('$HF_DST')
print('done conversion', flush=True)
PYEOF
  " 2>&1 | tee "$OUT_ROOT/convert.log" | tail -3
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] Conversion done -> $HF_DST"
else
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] HF dst already exists, skipping conversion"
fi
echo ""

# Step 2: run lm-eval-harness for the full suite
# Mimics chain_phase7.sh's task entries
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$NODE" "
  cd /fsx/users/dongweij/marin
  export HF_TOKEN=\$(cat /fsx/users/dongweij/.cache/huggingface/token)
  export HF_ALLOW_CODE_EVAL=1

  run_lm_eval() {
    local TASKS=\"\$1\" NSHOT=\"\$2\" BATCH=\"\$3\" EXTRA=\"\$4\" NPROC=\"\${5:-8}\"
    local OUT=$OUT_ROOT/\${NSHOT}shot__\$(echo \"\$TASKS\" | tr ',' '_' | cut -c1-30)
    mkdir -p \"\$OUT\"
    echo \"[\$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] tasks=\$TASKS n-shot=\$NSHOT batch=\$BATCH nproc=\$NPROC\"
    if [ \"\$NPROC\" -eq 1 ]; then
      # single-GPU path — required for code_eval (mbpp/humaneval) because HF evaluate's
      # metric cache file collides across torchrun ranks
      .venv/bin/python -m lm_eval --model hf \
        --model_args 'pretrained=$HF_DST,dtype=bfloat16,trust_remote_code=True' \
        --tasks \"\$TASKS\" --num_fewshot \"\$NSHOT\" --batch_size \"\$BATCH\" \
        --log_samples --output_path \"\$OUT\" \
        --include_path /fsx/users/dongweij/marin/experiments/data_efficiency \
        --trust_remote_code \
        \$EXTRA > \"\$OUT.log\" 2>&1 && echo '  DONE' || echo '  FAILED-CONTINUE'
    else
      .venv/bin/accelerate launch --multi_gpu --num_processes \$NPROC --num_machines 1 \
        -m lm_eval --model hf \
        --model_args 'pretrained=$HF_DST,dtype=bfloat16,trust_remote_code=True' \
        --tasks \"\$TASKS\" --num_fewshot \"\$NSHOT\" --batch_size \"\$BATCH\" \
        --log_samples --output_path \"\$OUT\" \
        --include_path /fsx/users/dongweij/marin/experiments/data_efficiency \
        --trust_remote_code \
        \$EXTRA > \"\$OUT.log\" 2>&1 && echo '  DONE' || echo '  FAILED-CONTINUE'
    fi
  }

  run_lm_eval 'arc_easy,arc_challenge' 25 16 ''
  run_lm_eval 'hellaswag' 10 16 ''
  run_lm_eval 'winogrande,mmlu,gsm8k' 5 16 ''
  run_lm_eval 'piqa,boolq,sciq,openbookqa,commonsense_qa,social_iqa,logiqa' 0 16 ''
  run_lm_eval 'openbookqa_fact' 0 16 ''
  run_lm_eval 'gsm8k_cot' 8 8 '--confirm_run_unsafe_code'
  run_lm_eval 'mbpp' 3 8 '--confirm_run_unsafe_code' 1
  run_lm_eval 'humaneval' 0 8 '--confirm_run_unsafe_code' 1
  run_lm_eval 'minerva_math' 4 8 ''
" 2>&1 | tee "$OUT_ROOT/eval.log" | grep -E "(tasks=|DONE|FAILED|Error|Traceback)" | head -30

echo ""
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] Done. Results: $OUT_ROOT"
