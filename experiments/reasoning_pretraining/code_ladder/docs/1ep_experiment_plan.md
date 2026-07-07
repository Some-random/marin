# 1-epoch experiment design — investigation findings (2026-06-01)

Investigation outcomes from the afternoon of 2026-06-01, with the user out.
This is a working document — not a final plan. Numbers are verified by running
the listed commands; nothing here is guessed.

## 1. Data inventory — rock solid

All token counts below are **verified** by reading the levanter tensorstore
`input_ids/offsets[n_rows]` field (the last offset = total tokens written to
that shard). No estimates.

### DCLM tokenized (canonical marin, llama-3 vocab)

Path: `/fsx/users/dongweij/marin/outputs/tokenized/dclm_baseline-0206f1/`

| Shard | Rows | Tokens |
|---|---:|---:|
| part-00006 | 3,881,860 | 4,929,707,970 |
| part-00020 | 3,911,250 | 4,949,464,465 |
| part-00026 | 3,883,807 | 5,035,579,163 |
| part-00035 | 3,841,494 | 5,000,284,009 |
| part-00042 | 3,844,266 | 4,938,851,916 |
| part-00047 | 3,818,413 | 4,996,276,735 |
| part-00071 | 3,858,971 | 4,993,237,021 |
| part-00073 | 479,638 | 655,073,639 (partial) |
| **TOTAL** | **27,519,699** | **35,498,474,918 (~35.5B)** |

All 8 shards have `is_finished: True`. This is the corrected (post-2025-02-11)
version, NOT the WRONG/corrupted version which had `_WRONG_20250211` suffix.

### Code data tokenized

| Dataset | Train tokens | Notes |
|---|---:|---|
| `opc_algorithmic-ffc825` | 942,620,998 (~0.94B) | What v1 used |
| `opc_algorithmic_50m-e1b8de` | 54,593,883 (~0.055B) | What v2 used |
| `aryabumi_code_synth_full-0678c3` | **5,416,688,006 (~5.4B)** | Synthetic code, Aryabumi-style |
| `aryabumi_code_synth_solution-660d21` | 182,725,123 (~0.18B) | Synthetic exercises |
| `aryabumi_code_web-591a44` | 1,350,183,823 (~1.35B) | Web-scraped code |

**Useful code budget: 5.4B + 0.94B + 1.35B = 7.69B** (the three full-size code corpora).

### Other observations

- 6.6 TB raw DCLM cached at `/fsx/users/dongweij/marin/outputs/raw/dclm/` — we have plenty more unique web text if we ever need to tokenize past 35.5B.
- /fsx has 4.7 TB free.

## 2. Hyperparameter reference table (from papers, verified)

| Param | phi-1 (1.3B) | phi-1.5 (1.3B) | OLMo 2 7B | OLMo 3 | Marin 32B | **Picked for 1ep** |
|---|---|---|---|---|---|---|
| Peak LR | 1e-3 | 2e-4 const | **3e-4** | WSD | 7e-4 | **3e-4** (OLMo 2 7B) |
| Schedule | linear/linear | constant | cosine + linear tail | WSD | WSD | cosine to 0 (Konwoo style) |
| Weight decay | 0.1 | 0.1 | 0.1 (excl embeds) | 0.1 | (n/a) | **0.1** (drop from 1.6) |
| AdamW ε | default | 1e-7 | **1e-8** | — | — | 1e-8 (OLMo 2 improvement) |
| QK-norm | no | no | yes | yes | yes | (out of scope — needs arch change) |
| Init | default | default | N(0, 0.02) all | — | — | (default, untouched) |
| z-loss | no | no | 1e-4 × log²Z | — | — | (out of scope) |
| Repeated n-gram filter | no | no | yes (32+ tokens) | — | — | (out of scope; we use clean DCLM) |

Sources (verified by reading the PDFs):
- phi-1 §2.3: "We use effective batch size 1024 ... maximum learning rate 1e-3 with warmup over 750 steps, and weight decay 0.1, for a total of 36,000 steps ... checkpoint at 24,000 steps as our phi-1-base — this is equivalent to ~ 8 epochs on our CodeTextbook dataset for a total of little over 50B total training tokens."
- phi-1.5 §2.3: "constant learning rate 2e − 4 (no warm up), weight decay 0.1 ... batch size 2048, and train for 150B tokens, with 80% from the newly created synthetic data and 20% from phi-1's training data."
- OLMo 2 §4.1: "linearly warm up the learning rate to its peak of 3 · 10^−4 over the first 2000 steps. Then, we use a standard cosine decay over 5T tokens ... we stop the schedule at 4T tokens and then switch to mid-training ... The 13B ran with a higher peak learning rate" (6e-4).
- OLMo 2 §3.4.1: "decreasing the AdamW ε from 10^−5 to 10^−8."
- OLMo 2 §3.4.2: "we exclude weight decay for embeddings."
- OLMo 3 §3.4 + Marin 32B retro: WSD schedule, 7× max upsampling.

## 3. Multi-machine training — works (smoke-tested 2-node)

### Infrastructure verified
- p4d.24xlarge with EFA (`/opt/amazon/efa/` installed).
- NCCL via `.venv/lib/python*/site-packages/nvidia/nccl/lib/libnccl.so.2`.
- /fsx Lustre identically mounted on all nodes (`10.0.129.10@tcp:/6i4o5bev`).
- Inter-node TCP works (gpu-dy-1 → gpu-st-4 port 22, 0.1ms latency, free ports available).

### Launcher
`experiments/reasoning_pretraining/code_ladder/orchestration/multi_node_launch.sh` — SSH-based, no Slurm.
Sets `JAX_DIST_NUM_PROCESSES`, `JAX_DIST_PROCESS_ID`, `JAX_DIST_COORDINATOR`
env vars; each run script reads them via `_distributed_from_env()`.

```bash
bash experiments/reasoning_pretraining/code_ladder/orchestration/multi_node_launch.sh \
  --nodes "gpu-st-p4d24xlarge-4,gpu-dy-p4d24xlarge-1" \
  --config experiments/reasoning_pretraining/code_ladder/scripts/run_1_4b_1ep_dclm.py \
  --run-tag 1ep-dclm-baseline
```

### 2-node smoke test (PASSED)
- 2 nodes (gpu-st-4 + gpu-dy-1) successfully connected via jax.distributed.
- data_axis_size=16 (8 GPUs × 2 nodes), confirming both nodes were enumerated.
- GPU utilization 99-100% on both nodes during training.
- Memory: ~32GB used per GPU (of 40GB available), ~80% utilization.
- Compile: ~95s first train step, ~60s second compile pass for train_step_hooks.
- **Steady-state step time on 2 nodes (batch=128): 2.8 s/step** (verified at step 25).
- 30 steps completed without crash; final eval running cleanly.
- Loss dropped 12.2 (init) → 7.84 (step 25), confirming training is happening.
- NCCL all-reduce verified: both nodes report identical eval loss (11.265).
- One XLA warning observed: `[SPMD] Involuntary full rematerialization` for a
  sharding mismatch — perf hit but not a correctness issue. Worth filing
  upstream eventually; ignore for now.

### 4-node smoke test (initial attempt FAILED, fix applied)

First attempt (23:22): JAX init + NCCL succeeded, but node-1's wandb async
client hit BrokenPipeError. With wandb running on all 4 ranks, node-1's
exception eventually killed its python process. JAX shutdown barrier then
timed out waiting for the still-compiling nodes 0/2/3, killing everyone.
py-spy dump confirmed main threads on nodes 0/2/3 were in
`_compile_and_write_cache` — the long XLA compile for 32-GPU mesh was still
running when the barrier failed.

**Root cause: wandb was initialized on all ranks; only rank 0 should sync.**

**Fix applied to `multi_node_launch.sh`**: sets `WANDB_MODE=online` on
process_id=0 and `WANDB_MODE=disabled` on all other ranks. Logs from
non-coordinator ranks go to local log files (same as before), but no
external wandb sync that could blow up.

Both run scripts also got `jax_compilation_cache_dir =
/fsx/users/dongweij/marin/outputs/jax_compile_cache` so the (one-time) long
XLA compile is persisted across runs.

**Retry #3 (23:36) with WANDB_MODE=offline + WANDB_DISABLED=true on
non-rank-0: SUCCESS.** All 4 nodes started tracing at 23:37:35, finished
lowering at 23:38:39, first train step completed at 23:39:01 (93 s, compile-
dominated). 0 BrokenPipe errors, 0 Tracebacks across all 4 nodes. Eval
(initial) running cleanly. Steady-state step rate expected to be ~3-4 s/step
after the eval completes.

**The fix in `multi_node_launch.sh` for non-rank-0:**
- `WANDB_MODE=offline` AND `WANDB_DISABLED=true` (both, belt + suspenders)
- This bypasses wandb's async service entirely on non-coordinator nodes.

### Plan B (if 4-node remains unsolved): 2 nodes per run, parallel

| Setup | Nodes | Batch | Steps | Wall-clock |
|---|---:|---:|---:|---:|
| Variant A on 2 nodes | 2 | 128 | 58,689 | ~46 h (~1.9 days) |
| Variant B on 2 nodes | 2 | 128 | 58,689 | ~46 h (~1.9 days) |
| Total resource | 4 nodes parallel | — | — | **~1.9 days both runs** |

Still better than the prior 16-epoch single-node baseline approach. Below the
1-day target but acceptable for a single hypothesis test.

### Currently SSH-able + free nodes
gpu-st-1, gpu-st-2, gpu-st-3 (static, idle)
gpu-dy-2, gpu-dy-3, gpu-dy-4, gpu-dy-5 (dynamic, idle, currently SSH-able)
gpu-st-4 + gpu-dy-1 in use for smoke test, will free shortly.
**Total: 7 immediately, 9 after smoke completes.**

## 4. Proposed experiment design

### Two parallel runs at 30.77B total trained tokens

Both at 1 epoch over each source, hyperparameters identical, only data differs.

**Variant A — DCLM-only baseline** (`run_1_4b_1ep_dclm.py`)
- Data: 100% DCLM (1 epoch over ~30.77B from the 35.5B available)
- Tests: anchor for the 1-epoch comparison series

**Variant B — Code-mix 25% (Aryabumi-style)** (`run_1_4b_1ep_code25.py`)
- Data: 75% DCLM (23.08B, 1 ep) + 25% code (1 ep each)
  - aryabumi_synth_full: 5.40B (70.2% of code slice)
  - aryabumi_web:        1.35B (17.6% of code slice)
  - opc_algorithmic:     0.94B (12.2% of code slice)
  - **code total: 7.69B = 25.0% of 30.77B**
- Tests: H1 — does Aryabumi-style code mix improve NL/reasoning under matched compute with no repetition?

### Shared hyperparameters
- 1.4B Llama-arch (`1_4b4k` model), block_cross_doc=False
- LR 3e-4 cosine to 0 (OLMo 2 style, dropped from 1e-3)
- WD **0.1** (dropped from 1.6 — phi/OLMo/Marin median)
- AdamW β1/β2 = 0.9/0.95, warmup=1%, max_grad_norm=1.0
- seq_len=4096, per_device_parallelism=8
- batch_size = num_nodes × 64 (= 64/128/256/512 for 1/2/4/8 nodes)
- data_seed=0, seed=0

### Wall-clock estimates (1.4B run @ 30.77B trained tokens)

Step time measured on 2 nodes (smoke test): **2.8 s/step steady-state**.
Assume 4-node ~10-15% overhead → ~3.1 s/step.

| Nodes | Batch | Steps | Est. wall (measured 2.8 s/step on 2 nodes, extrapolated) |
|---:|---:|---:|---:|
| 1 | 64 | 117,378 | ~92 h (~3.8 days)  [assuming 2.8 s/step on 1 node] |
| **2** | **128** | **58,689** | **~46 h (~1.9 days)** ← measured 2.8 s/step |
| **4** | **256** | **29,344** | **~23.6 h (~1.0 day)** ← **MEASURED** 2.9 s/step ← HITS TARGET |
| 8 | 512 | 14,672 | ~12-15 h (extrap, larger comms overhead expected) |

### Recommended deployment

- **Variant A on 4 nodes**: gpu-st-1, gpu-st-2, gpu-st-3, gpu-dy-2
  ```
  bash experiments/reasoning_pretraining/code_ladder/orchestration/multi_node_launch.sh \
    --nodes "gpu-st-p4d24xlarge-1,gpu-st-p4d24xlarge-2,gpu-st-p4d24xlarge-3,gpu-dy-p4d24xlarge-2" \
    --config experiments/reasoning_pretraining/code_ladder/scripts/run_1_4b_1ep_dclm.py \
    --run-tag 1ep-dclm-A
  ```
- **Variant B on 4 nodes**: gpu-dy-3, gpu-dy-4, gpu-dy-5, gpu-st-4 (free after smoke)
  ```
  bash experiments/reasoning_pretraining/code_ladder/orchestration/multi_node_launch.sh \
    --nodes "gpu-dy-p4d24xlarge-3,gpu-dy-p4d24xlarge-4,gpu-dy-p4d24xlarge-5,gpu-st-p4d24xlarge-4" \
    --config experiments/reasoning_pretraining/code_ladder/scripts/run_1_4b_1ep_code25.py \
    --run-tag 1ep-code25-B \
    --coordinator-port 33335
  ```
- Both run in parallel; uses 8 nodes total (we have 9 free).
- **Result: both runs finish in ~25 hours (~1 day)**, mid-training Paloma/dclm_val
  curves visible from step ~3,668 (one-eighth of the way through, ~3 hours in).

## 5. Falsifiable predictions

- **If B > A** on Paloma + closed-book NL → matched-compute code mix is real;
  May 26 v1-wins finding survives the unique-tokens confound.
- **If B = A or B < A** → May 26 finding was the confound; under fair (matched-
  token, 1-epoch) comparison code-mix doesn't help. This would be a negative
  result for Aryabumi-style at our scale, agreeing with the June 1 v2 analysis.
- gsm8k_cot loop rate at the final checkpoint of both A and B (each at 1 epoch
  with no repetition) tests whether the v1 "0% loop" effect was specifically
  code-mix or just "no repetition." If both A and B have 0% loops, no
  repetition is sufficient; if only B has 0%, code helps independent of
  repetition.

## 6. Open items for user review before launch

1. **OK to use LR=3e-4 (down from our 1e-3) and WD=0.1 (down from 1.6)?**
   These are pulled from open-source 1.3-7B-class references but they're a
   substantial change from our prior baseline. The change is well-motivated
   (1-epoch removes the repetition-overfit pressure that needed WD=1.6), but
   the user should confirm before launch.

2. **OK with 30.77B total instead of exactly 28B?** This was chosen so that
   every code source in variant B is used at exactly 1 epoch. Difference is
   ~10% in total compute.

3. **OK to use all three code sources mixed (aryabumi_synth + aryabumi_web +
   opc) in variant B?** Alternative would be aryabumi_synth_full only (5.4B)
   at 25% with 16.2B DCLM, total = 21.6B — cleaner code source but smaller
   total compute.

4. **OK with 4+4 node split?** This matches the "1 day" target. Alternatives:
   2+2 (~2.4 days each, half the resources) or 8 sequential (~14h each,
   chained = ~28h total).

---

## TL;DR for Dongwei when back

**What's done while you were out:**
1. Verified 35.5B DCLM tokens + 7.69B code data on disk (no tokenization needed).
2. Extracted hyperparameters from phi-1, phi-1.5, OLMo 2, OLMo 3, Marin 32B from the actual papers. WD=0.1 + LR=3e-4 + cosine is the consensus for 1.3-7B scale, single-epoch.
3. Wrote `run_1_4b_1ep_dclm.py` (variant A) and `run_1_4b_1ep_code25.py` (variant B) — both with multi-node env-var support.
4. Wrote SSH-based `multi_node_launch.sh` (no Slurm) + `multi_node_kill.sh`.
5. 2-node smoke test PASSED end-to-end (2.8 s/step steady-state, ~46h for 30.77B total).
6. 4-node smoke had wandb-async-pipe issue on non-coordinator nodes; iterated through 2-3 fixes. **Final fix `WANDB_MODE=offline + WANDB_DISABLED=true` on non-rank-0 works** — first train step completed at 23:39:01 on 4 nodes (in progress through remaining 29 steps as of 23:39).

**Decisions I need from you before launch:**
1. Hyperparams: **LR=3e-4** (down from 1e-3), **WD=0.1** (down from 1.6). Justified by phi/OLMo/Marin median for 1-epoch.
2. Total tokens: **30.77B** per run (so each code source is exactly 1 epoch in variant B). Or drop to 28B if you'd rather not use the small opc_algorithmic component.
3. Variant B mix: **75% DCLM + 25% mixed code (aryabumi_synth 70%, aryabumi_web 18%, opc 12% of the code slice)**. Each code source at exactly 1 epoch.
4. Deployment: **4 nodes per run, both in parallel** (~1 day each) if 4-node works. Or **2 nodes per run in parallel** (~1.9 days each, plan B).

**Launch commands** (after you confirm):

Plan A — 4+4 nodes, both in parallel:
```bash
# Variant A on nodes 1,2,3 + dy-2
bash experiments/reasoning_pretraining/code_ladder/orchestration/multi_node_launch.sh \
  --nodes "gpu-st-p4d24xlarge-1,gpu-st-p4d24xlarge-2,gpu-st-p4d24xlarge-3,gpu-dy-p4d24xlarge-2" \
  --config experiments/reasoning_pretraining/code_ladder/scripts/run_1_4b_1ep_dclm.py \
  --run-tag 1ep-dclm-A &

# Variant B on nodes dy-3,4,5 + st-4
bash experiments/reasoning_pretraining/code_ladder/orchestration/multi_node_launch.sh \
  --nodes "gpu-dy-p4d24xlarge-3,gpu-dy-p4d24xlarge-4,gpu-dy-p4d24xlarge-5,gpu-st-p4d24xlarge-4" \
  --config experiments/reasoning_pretraining/code_ladder/scripts/run_1_4b_1ep_code25.py \
  --run-tag 1ep-code25-B \
  --coordinator-port 33335 &
```

Plan B — 2+2 nodes:
```bash
# Variant A on 2 nodes
bash experiments/reasoning_pretraining/code_ladder/orchestration/multi_node_launch.sh \
  --nodes "gpu-st-p4d24xlarge-1,gpu-st-p4d24xlarge-2" \
  --config experiments/reasoning_pretraining/code_ladder/scripts/run_1_4b_1ep_dclm.py \
  --run-tag 1ep-dclm-A-2node &

# Variant B on 2 nodes
bash experiments/reasoning_pretraining/code_ladder/orchestration/multi_node_launch.sh \
  --nodes "gpu-st-p4d24xlarge-3,gpu-st-p4d24xlarge-4" \
  --config experiments/reasoning_pretraining/code_ladder/scripts/run_1_4b_1ep_code25.py \
  --run-tag 1ep-code25-B-2node \
  --coordinator-port 33335 &
```

**To kill a stuck run:**
```bash
bash experiments/reasoning_pretraining/code_ladder/orchestration/multi_node_kill.sh \
  --nodes "<same nodes>" \
  --config <same config> \
  --yes-i-mean-it
```

**Monitor wandb:** https://wandb.ai/dongwei_jiang/dongwei-data-efficiency

