# Data-efficiency experiments — infra gotchas & fixes (OPS runbook)

Operational fixes for the training + eval pipeline. These were previously kept in
`CLAUDE.local.md`; they're technical fix-logs, so they live here (greppable in-repo) and in
agent memory, while `CLAUDE.local.md` stays focused on behavioral rules. Each fix is also
applied in the code/script it concerns — this doc is the index.

---

## Eval: multi-GPU `gather_object` NCCL crash → `NCCL_P2P_DISABLE=1`

**Symptom:** `RuntimeError: NCCL Error 1: unhandled cuda error` at
`torch.distributed.gather_object` (lm_eval `evaluator.py:677/691`), killing mmlu and the
large paloma subsets under `accelerate launch --num_processes 8`. Intermittent (it succeeded
the majority of historical runs); for the current 300M/600M + NCCL 2.27.5 stack it became
near-always.

**Root cause (captured under `NCCL_DEBUG=INFO`, 2026-06-22):** an **OOM in NCCL's P2P/CUMEM
IPC-buffer allocator** on A100-**40GB**. At task end `gather_object` builds a per-peer
P2P/CUMEM gather channel to rank 0 and `cudaMalloc`s shareable IPC buffers
(`include/alloc.h:228 NCCL WARN Cuda failure 2 'out of memory'`). `"unhandled cuda error"` is
the misleading downstream symptom. mmlu (57 subtasks = biggest gather) and large paloma
subsets just tip memory over. **Not** a fabric fault, **not** a payload-size hard threshold.

**Fix:** `export NCCL_P2P_DISABLE=1` before `accelerate launch`. Applied in `run_eval_v2.sh`,
`run_paloma_for_model.sh`, `run_gsm_for_model.sh`, `run_aryabumi_nl_extras.sh`,
`run_quac_for_model.sh`. Full 8-GPU speed retained (eval is data-parallel inference + one
gather, so P2P perf is irrelevant). **The `--num_processes 1` single-GPU workaround is
retired.** A/B proof: P2P-on 8/8 fail · `NCCL_SHM_DISABLE=1` 6/6 fail (no help) ·
`NCCL_P2P_DISABLE=1` 6/6 pass; validated on 4×600M = 32/32 task-runs.

**Separate, do not confuse:** real CUDA OOM on **4B-model** paloma (`Tried to allocate 15.66
GiB`) is fixed by `BATCH_SIZE=4`, not by P2P_DISABLE.

---

## Eval: paloma needs HF ONLINE mode

`allenai/paloma` fetches a legacy builder-script from the Hub on every `load_dataset`, so
`HF_DATASETS_OFFLINE=1` / `HF_HUB_OFFLINE=1` kills it with
`ConnectionError: OfflineModeIsEnabled`. `run_paloma_for_model.sh` sets `OFFLINE=0`. Other
tasks (gsm, the v2-suite parquet datasets) are fine offline because their data ships as plain
files. Do not "fix" paloma back to offline.

---

## Eval: don't trust `ALL DONE` — it can be a lie

Runner scripts loop tasks and historically swallowed per-task failures with `||`, then printed
`ALL DONE` regardless. After 2026-06-22 the runners print an explicit
`ALL DONE (N ok, M FAILED: <names>)` summary so the marker can't lie. Still: before extracting
scores, verify at least one `<results_dir>/<subset>/results_*.json` exists; if empty, read the
per-task `.log` (the FIRST rank Traceback, not the torchrun `<NO_OTHER_FAILURES>` summary).

---

## Eval: failure triage (don't blind-retry) + resume

When a runner finishes `WITH FAILURES` it auto-runs `analyze_eval_failures.py "$OUT_ROOT"`, which
scans for task dirs with no `results_*.json`, pulls the first real traceback from each log,
classifies the root cause (`NCCL_GATHER_OOM`, `CUDA_OOM`, `OFFLINE_MODE`, `HUB_CONN`,
`CODE_EVAL_CACHE`, `DATASET_MISSING`, `KILLED`, `NCCL_OTHER`, `TIMEOUT`, `UNKNOWN`, `NO_LOG`),
marks each `transient` (safe to retry) or `permanent` (fix the config first), suggests a fix, and
writes `FAILURES.md`. **Read it before retrying** — retry only `transient` classes; for
`permanent` ones apply the fix (lower batch, set OFFLINE=0, …) first. Run manually on any dir:
```bash
.venv/bin/python experiments/reasoning_pretraining/code_ladder/eval/analyze_eval_failures.py <RESULTS_DIR> [--no-write]
```

**Resume, don't redo.** Every runner accepts `OUT_ROOT` via env and **skips any task that already
has a `results_*.json`**. After fixing a failure cause, re-run into the SAME dir to re-execute only
the failed tasks: `OUT_ROOT=<existing_dir> bash run_<x>_for_model.sh <LABEL> <HF_DIR>`.

---

## Training: Wandb nil-context SIGSEGV — disable the three artifact-save triggers

Multi-node Levanter + wandb crashes with a Go-side SIGSEGV in `gql.CreateArtifact` at
`pc=0xb458cb` (nil `ArtifactSaver.ctx`) on non-zero ranks; the cluster then hangs in NCCL
rendezvous waiting for the dead rank. A separate `_md5_file_hasher` SIGBUS on Lustre fires from
the same path. All come from Levanter's `_maybe_save_jaxpr` calling `wandb.log_artifact` on the
first-step compile. **A wandb version upgrade does NOT fix it** (the `ArtifactSaver.ctx`-nil
race remains in 0.27.2).

**Permanent fix — three switches, defaults all flipped True→False:**
1. `TrainerConfig.log_jaxprs = False` (`lib/levanter/src/levanter/trainer.py`, 2026-06-15) — jaxpr-as-artifact upload.
2. `TrainerConfig.log_xla_hlo = False` (`trainer.py`, 2026-06-15) — HLO-as-artifact upload.
3. `WandbConfig.save_code = False` (`lib/levanter/src/levanter/tracker/wandb.py`, 2026-06-17) — source-as-artifact upload.

New scripts are safe by default. If the SIGSEGV (`pc=0xb458cb`) / `_md5_file_hasher` SIGBUS /
`Terminating process` after `Lowering train_step_hooks to HLO` reappears, check whether the new
script re-enabled any of the three (per-script `True` overrides are still possible for debug).

---

## Training: fresh tokenized caches need a `validation -> train` symlink

Levanter opens both `train/` and `validation/` caches at startup even when
`num_validation_sequences=0` (zero-sample-validation suppresses sampling, NOT the cache
lookup). Fresh caches with only `train/` crash on first launch:
`FileNotFoundError: Cache ledger not found at .../validation/shard_ledger.json` →
`ValueError: No source and no cache found for component <name> split validation`.
Fix / bake into every tokenize script as a post-step:
```bash
for c in <cache>; do [ -e "$c/validation" ] || (cd "$c" && ln -s train validation); done
```

---

## Cluster: NEVER `sudo fuser -k /dev/nvidia*`

On A100 NVSwitch nodes (`gpu-dy/st-p4d24xlarge-*`), `nvidia-fabricmanager` holds
`/dev/nvidia-nvswitch0`. `fuser -k` SIGKILLs it without restart, and without fabricmanager CUDA
cannot initialize (`FAILED_PRECONDITION: No visible GPU devices`, even though `nvidia-smi -L`
still lists all 8 GPUs). Correct cleanup for leaked GPU memory:
1. `pkill -9 -f .venv/bin/python` (catch python stragglers).
2. If still leaked: `sudo systemctl restart nvidia-fabricmanager` on each node — gracefully
   resets the GPU fabric AND clears leaked memory back to ~8 MiB baseline.

`systemctl restart` is the safe form (brings the service back up); `fuser -k` is the dangerous
form (doesn't).
