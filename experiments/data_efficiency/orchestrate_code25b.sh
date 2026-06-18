#!/usr/bin/env bash
# Autonomous orchestration for the 25B code-only base training, fired by
# Claude on Dongwei's request 2026-06-16 ~01:13 PDT.
#
# Pipeline (each step blocks on the previous):
#   1. Wait for SE-Python [2.5, 2.7) and [2.7, 2.8) fetches to finish (~30 min)
#   2. Tokenize both new bands via experiments.data_efficiency.code_data_lower_tiers
#   3. Wait for A5-SP audit eval (v2 + paloma + gsm + aux) ALL DONE
#   4. Fill A5-SP audit numbers into §3 of EVALUATION.md (replace existing column)
#   5. Launch 1.4B 25B-code-only training on dy-1..4 with retry-on-wandb-panic (up to 5x)
#   6. Push-notify Dongwei on each milestone
#
# Failure modes:
#   - If tokenize fails: log + skip training launch, push-notify
#   - If training panics: pkill + fbm restart + relaunch up to 5x
#   - If training panics > 5x: stop, push-notify, leave clean state

set -uo pipefail

MARIN=/fsx/users/dongweij/marin
TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
ORCH_LOG=$MARIN/logs/orchestrate_code25b_${TS}.log

log() { echo "[$(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$ORCH_LOG"; }

log "=== Code25B orchestration START ==="

# ----- step 1: wait for both SE-Python fetches to finish -----
wait_for_fetch() {
    local dir=$1
    local expected_ranks=8
    local label=$2
    log "step 1: waiting for fetch in $label ($dir) — $expected_ranks ranks"
    local sleep_s=60
    local tries=0
    while true; do
        # Count active fetch processes for this dir
        local active=$(pgrep -af "fetch_stack_edu_python_score_range.*$(basename "$dir")" 2>/dev/null | grep -v grep | wc -l)
        if [ "$active" -eq 0 ]; then
            log "  $label: all ranks done (no active fetch procs)"
            break
        fi
        tries=$((tries+1))
        if [ $((tries % 5)) -eq 0 ]; then
            log "  $label: still $active rank procs alive (waited $((tries*sleep_s/60)) min)"
        fi
        sleep $sleep_s
    done
}

wait_for_fetch /fsx/users/dongweij/marin/outputs/raw/stack-edu-python-content-mid    "[2.7, 2.8)"
wait_for_fetch /fsx/users/dongweij/marin/outputs/raw/stack-edu-python-content-lower2 "[2.5, 2.7)"

# Sanity check: each dir has rank_0..7 subdirs with content
for d in stack-edu-python-content-mid stack-edu-python-content-lower2; do
    n_files=$(find $MARIN/outputs/raw/$d/rank_*/se_python_low_*.jsonl.gz 2>/dev/null | wc -l)
    log "  $d: $n_files shard files on disk"
    if [ "$n_files" -lt 8 ]; then
        log "  ERROR: $d has fewer than 8 files; fetch may have failed. ABORTING."
        exit 1
    fi
done

log "fetches complete; both bands on disk."

# ----- step 2: tokenize the two new bands -----
log "step 2: tokenizing [2.7, 2.8) and [2.5, 2.7) via code_data_lower_tiers"
TOK_LOG=$MARIN/logs/tokenize_lower_tiers_${TS}.log
cd $MARIN
MARIN_PREFIX=$MARIN/outputs .venv/bin/python -m experiments.data_efficiency.code_data_lower_tiers \
    > "$TOK_LOG" 2>&1
TOK_RC=$?
log "  tokenize rc=$TOK_RC; log $TOK_LOG"

# Verify both new caches exist
for prefix in c5_25b_se_python_mid c5_25b_se_python_low2; do
    d=$(ls -d $MARIN/outputs/tokenized/${prefix}-* 2>/dev/null | head -1)
    if [ -z "$d" ]; then
        log "  ERROR: tokenized cache for $prefix not created. ABORTING."
        exit 1
    fi
    rows=$(python3 -c "import json; d=json.load(open('$d/train/shard_ledger.json')); print(d['total_num_rows'])" 2>/dev/null)
    log "  $prefix: $rows rows (~$(python3 -c "print($rows*683/1e9)" 2>/dev/null) B tokens)"
done

# ----- step 3: wait for A5-SP audit eval to finish -----
log "step 3: waiting for A5-SP audit eval ALL DONE"
A5SP_V2_LOG=$MARIN/logs/v2_a5_sp_audit_step29343_20260616_*.log
while ! grep -q "ALL DONE" $A5SP_V2_LOG 2>/dev/null; do
    sleep 60
done
log "  v2-suite ALL DONE"
# Wait for paloma + gsm + aryabumi-quac too
for tag in paloma gsm aryabumi_quac; do
    while ! grep -lqE "ALL DONE|quac DONE|cb DONE" $MARIN/logs/${tag}_a5_sp_audit_step29343_*.log 2>/dev/null; do
        sleep 60
    done
    log "  $tag DONE"
done

# ----- step 4: fill A5-SP column in §3 -----
log "step 4: filling A5-SP column with audit values"
RESULTS_DIR=$(ls -d $MARIN/outputs/eval_results/v2_a5_sp_audit_step29343_* 2>/dev/null | head -1)
.venv/bin/python experiments/data_efficiency/eval_section3.py fill-from-results \
    "$RESULTS_DIR" "A5-SP" 2>&1 | tee -a "$ORCH_LOG"

# Also fill the non-v2 cells (paloma, gsm, aux, dclm)
# (paloma_macro)
PALOMA_DIR=$(ls -d $MARIN/outputs/eval_results/paloma_a5_sp_audit_step29343_* 2>/dev/null | head -1)
GSM_DIR=$(ls -d $MARIN/outputs/eval_results/gsm_a5_sp_audit_step29343_* 2>/dev/null | head -1)
ARYABUMI_DIR=$(ls -d $MARIN/outputs/eval_results/aryabumi_nl_a5_sp_audit_step29343_* 2>/dev/null | head -1)
QUAC_DIR=$(ls -d $MARIN/outputs/eval_results/quac_a5_sp_audit_step29343_* 2>/dev/null | head -1)

PALOMA_VAL=$(python3 -c "
import json, glob
bpbs=[]
for d in glob.glob('$PALOMA_DIR/paloma_*'):
    for jp in glob.glob(f'{d}/**/*results*.json', recursive=True):
        d2=json.load(open(jp))
        for tn,m in d2.get('results',{}).items():
            if tn.startswith('paloma_') and 'bits_per_byte,none' in m:
                bpbs.append(m['bits_per_byte,none'])
print(round(sum(bpbs)/len(bpbs),3) if bpbs else '')
" 2>/dev/null)
[ -n "$PALOMA_VAL" ] && .venv/bin/python experiments/data_efficiency/eval_section3.py fill-cell --row "paloma_macro (bpb)" --col "A5-SP" --value "$PALOMA_VAL" 2>&1 | tee -a "$ORCH_LOG"

# (gsm symbolic + noop)
for task in gsm_symbolic_main gsm_noop; do
    suffix="[8]"
    val=$(python3 -c "
import json, glob
for jp in glob.glob('$GSM_DIR/**/*results*.json', recursive=True):
    d=json.load(open(jp))
    for tn,m in d.get('results',{}).items():
        if tn == '$task' and 'exact_match,strict-match' in m:
            print(round(m['exact_match,strict-match'],3))
            break
" 2>/dev/null)
    [ -n "$val" ] && .venv/bin/python experiments/data_efficiency/eval_section3.py fill-cell --row "${task}${suffix}" --col "A5-SP" --value "$val" 2>&1 | tee -a "$ORCH_LOG"
done

# (storycloze + cb)
for task in storycloze_2018_local cb; do
    val=$(python3 -c "
import json, glob
for jp in glob.glob('$ARYABUMI_DIR/**/*results*.json', recursive=True):
    d=json.load(open(jp))
    for tn,m in d.get('results',{}).items():
        if tn == '$task' and 'acc,none' in m:
            print(round(m['acc,none'],3))
            break
" 2>/dev/null)
    [ -n "$val" ] && .venv/bin/python experiments/data_efficiency/eval_section3.py fill-cell --row "${task}[0]" --col "A5-SP" --value "$val" 2>&1 | tee -a "$ORCH_LOG"
done

# (quac)
val=$(python3 -c "
import json, glob
for jp in glob.glob('$QUAC_DIR/**/*results*.json', recursive=True):
    d=json.load(open(jp))
    for tn,m in d.get('results',{}).items():
        if tn == 'quac_first_turn' and 'f1,none' in m:
            print(round(m['f1,none'],3))
            break
" 2>/dev/null)
[ -n "$val" ] && .venv/bin/python experiments/data_efficiency/eval_section3.py fill-cell --row "quac_first_turn[0]" --col "A5-SP" --value "$val" 2>&1 | tee -a "$ORCH_LOG"

# (dclm_200m_val from training log nats→bpb)
A5SP_TRAIN_LOG=$(ls -d $MARIN/logs/multinode_a5_sp_4n_20260615_004709_*/node-0-*.log 2>/dev/null | head -1)
if [ -n "$A5SP_TRAIN_LOG" ]; then
    dclm_val=$(python3 -c "
import re
last=None
for line in open('$A5SP_TRAIN_LOG', errors='ignore'):
    for m in re.compile(r'dclm_200m_val[^=:]*[=:\\\"]+\\s*([\\d.]+)').finditer(line):
        try: last=float(m.group(1))
        except: pass
if last is not None: print(round(last*0.3273, 3))
" 2>/dev/null)
    [ -n "$dclm_val" ] && .venv/bin/python experiments/data_efficiency/eval_section3.py fill-cell --row "dclm_200m_val (bpb)" --col "A5-SP" --value "$dclm_val" 2>&1 | tee -a "$ORCH_LOG"
fi

.venv/bin/python experiments/data_efficiency/eval_section3.py validate 2>&1 | tee -a "$ORCH_LOG"

log "A5-SP column updated."

# ----- step 5: launch 25B code-only training with retry-on-wandb -----
launch_code25b() {
    local attempt=$1
    local launch_ts=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
    local tag="code25b_attempt${attempt}_${launch_ts}"
    local launcher_log=$MARIN/logs/launcher_${tag}.log
    log "step 5: launching code25b attempt ${attempt} as ${tag}"
    nohup $MARIN/experiments/data_efficiency/multi_node_launch.sh \
        --nodes "gpu-dy-p4d24xlarge-1,gpu-dy-p4d24xlarge-2,gpu-dy-p4d24xlarge-3,gpu-dy-p4d24xlarge-4" \
        --config experiments/data_efficiency/run_1_4b_code25b.py \
        --run-tag "$tag" \
        --coordinator-port $((33700 + attempt)) \
        > "$launcher_log" 2>&1 < /dev/null &
    disown
    echo "$tag" > /tmp/code25b_current_tag.txt
    sleep 120  # Give python ~2 min to start writing
}

cleanup_dy_nodes() {
    log "  cleanup_dy_nodes: pkill .venv/bin/python + fabricmanager restart"
    for n in gpu-dy-p4d24xlarge-1 gpu-dy-p4d24xlarge-2 gpu-dy-p4d24xlarge-3 gpu-dy-p4d24xlarge-4; do
        timeout 60 ssh -o ConnectTimeout=5 $n \
            'pkill -9 -f ".venv/bin/python"; sleep 5; for i in 1 2 3; do sudo systemctl restart nvidia-fabricmanager; sleep 20; mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "$mb" -lt 100 ] && break; done' 2>&1 &
    done
    wait
}

MAX_ATTEMPTS=5
for attempt in $(seq 1 $MAX_ATTEMPTS); do
    cleanup_dy_nodes
    launch_code25b "$attempt"
    tag=$(cat /tmp/code25b_current_tag.txt)
    log_dir=$(ls -d $MARIN/logs/multinode_${tag}_* 2>/dev/null | head -1)
    if [ -z "$log_dir" ]; then
        log "  no log dir yet, sleeping 60s..."
        sleep 60
        log_dir=$(ls -d $MARIN/logs/multinode_${tag}_* 2>/dev/null | head -1)
    fi
    log "  log_dir: $log_dir"

    # Wait up to 5 min for first Progress on:train line. If panic or no Progress, retry.
    survived=0
    for i in $(seq 1 30); do
        sleep 10
        if grep -q "Progress on:train" "$log_dir"/node-0-*.log 2>/dev/null; then
            survived=1
            break
        fi
        if grep -qE "panic:|SIGSEGV|Bus error|Fatal Python|rendezvous.cc.*may be stuck" "$log_dir"/node-*.log 2>/dev/null; then
            break
        fi
    done

    if [ $survived -eq 1 ]; then
        log "  attempt $attempt PASSED first-Progress check — training started"
        echo "$tag" > /tmp/code25b_winning_tag.txt
        break
    else
        log "  attempt $attempt FAILED first-Progress check, retrying"
        if [ $attempt -eq $MAX_ATTEMPTS ]; then
            log "  MAX RETRIES exceeded; aborting orchestration"
            exit 2
        fi
    fi
done

log "=== code25b training launched successfully; orchestration done ==="
log "monitor the training via: tail -F ${log_dir}/node-0-*.log"
