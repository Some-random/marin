#!/usr/bin/env bash
# Sharded version of convert_and_eval_v2.sh.
#
# Pipeline:
#   1. SSH to first shard node, run Levanter -> HF conversion on CPU mesh.
#   2. SSH to 4 shard nodes in parallel, each runs one quarter of the v2
#      task groups (run_eval_v2.sh SHARD={A,B,C,D}) into a SHARED OUT_ROOT.
#   3. Wait for all 4 shards. Print ALL DONE.
#
# Result: ~67 min serial → ~16-19 min parallel (max of 4 shards).
#
# Usage:
#   convert_and_eval_v2_sharded.sh --label NAME --src LEVANTER_DIR \
#     --hf-dst HF_DIR --shard-nodes "n1,n2,n3,n4"
#
# Falls back to single-node sequential mode if --shard-nodes is omitted.

set -uo pipefail

LABEL=""
SRC=""
HF_DST=""
SHARD_NODES=""

while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --src) SRC="$2"; shift 2 ;;
    --hf-dst) HF_DST="$2"; shift 2 ;;
    --shard-nodes) SHARD_NODES="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$LABEL" ] || [ -z "$SRC" ] || [ -z "$HF_DST" ] || [ -z "$SHARD_NODES" ]; then
  echo "Usage: $0 --label NAME --src LEVANTER_DIR --hf-dst HF_DIR --shard-nodes 'n1,n2,n3,n4'" >&2
  exit 1
fi

IFS=',' read -r -a NODES <<< "$SHARD_NODES"
if [ "${#NODES[@]}" -ne 4 ]; then
  echo "Expected exactly 4 shard nodes, got ${#NODES[@]}: $SHARD_NODES" >&2
  exit 1
fi

CONVERT_NODE="${NODES[0]}"
OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/v2_${LABEL}_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"

echo "=== convert + v2 eval (sharded) ==="
echo "  Label:    $LABEL"
echo "  Source:   $SRC"
echo "  HF dst:   $HF_DST"
echo "  Shards:   A=${NODES[0]}  B=${NODES[1]}  C=${NODES[2]}  D=${NODES[3]}"
echo "  OUT_ROOT: $OUT_ROOT"
echo "  Start:    $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')"
echo ""

if [ ! -d "$HF_DST" ]; then
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] Converting Levanter -> HF on $CONVERT_NODE..."
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$CONVERT_NODE" "
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
  "
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] Conversion done -> $HF_DST"
else
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] HF dst already exists, skipping conversion"
fi
echo ""

echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] Dispatching 4 shards in parallel..."
SHARD_LETTERS=(A B C D)
PIDS=()
for i in 0 1 2 3; do
  NODE="${NODES[$i]}"
  SHARD="${SHARD_LETTERS[$i]}"
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] -> shard $SHARD on $NODE"
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$NODE" "
    cd /fsx/users/dongweij/marin
    OUT_ROOT='$OUT_ROOT' bash experiments/data_efficiency/run_eval_v2.sh $LABEL $HF_DST $SHARD
  " &
  PIDS+=($!)
done

# Wait for all 4 shards. Capture exit codes.
FAILED=0
for i in 0 1 2 3; do
  SHARD="${SHARD_LETTERS[$i]}"
  if wait "${PIDS[$i]}"; then
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] shard $SHARD exited 0"
  else
    rc=$?
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] shard $SHARD exited $rc — partial results in $OUT_ROOT"
    FAILED=1
  fi
done

if [ "$FAILED" -eq 0 ]; then
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] ALL DONE → $OUT_ROOT"
else
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] ALL DONE WITH AT LEAST ONE SHARD FAILURE → $OUT_ROOT"
fi
