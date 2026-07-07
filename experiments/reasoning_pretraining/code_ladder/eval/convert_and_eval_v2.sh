#!/usr/bin/env bash
# Drop-in replacement for eval_intermediate.sh that uses the canonical
# run_eval_v2.sh for the eval step.
#
# Usage:
#   convert_and_eval_v2.sh --label NAME --src LEVANTER_DIR --hf-dst HF_DIR [--node NODENAME]
#
# Pipeline:
#   1. SSH to NODE, run Levanter -> HF conversion on CPU mesh.
#   2. SSH to NODE, run run_eval_v2.sh which executes the FULL v2 suite.

set -euo pipefail

LABEL=""
SRC=""
HF_DST=""
NODE="gpu-st-p4d24xlarge-4"
MODEL_KEY="1_4b4k"

while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --src) SRC="$2"; shift 2 ;;
    --hf-dst) HF_DST="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --model-key) MODEL_KEY="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$LABEL" ] || [ -z "$SRC" ] || [ -z "$HF_DST" ]; then
  echo "Usage: $0 --label NAME --src LEVANTER_DIR --hf-dst HF_DIR [--node NODENAME] [--model-key 1_4b4k|300m4k|600m4k|...]" >&2
  exit 1
fi

echo "=== convert + v2 eval ==="
echo "  Label:  $LABEL"
echo "  Source: $SRC"
echo "  HF dst: $HF_DST"
echo "  Node:   $NODE"
echo "  Start:  $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')"
echo ""

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
from experiments.reasoning_pretraining.code_ladder.models.models import model_dict
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import load_tokenizer
mc = model_dict['$MODEL_KEY']
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
  "
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] Conversion done -> $HF_DST"
else
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] HF dst already exists, skipping conversion"
fi
echo ""

echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] Running FULL v2 suite on $NODE..."
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$NODE" "
  cd /fsx/users/dongweij/marin
  bash experiments/reasoning_pretraining/code_ladder/eval/run_eval_v2.sh $LABEL $HF_DST
"
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] v2 eval done for $LABEL."
