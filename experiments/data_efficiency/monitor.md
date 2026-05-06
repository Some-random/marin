# Experiment Monitoring Runbook

## Quick Start

After kicking off an experiment, paste this into Claude Code to start monitoring:

```
/loop Monitor the training run (check PID with `ps aux | grep run_mixed`, log: outputs/mixed_dclm80_owm20_300m.log, WandB: dongwei_jiang/dongwei-data-efficiency).

Check schedule (exponential backoff):
- Start at 60s intervals
- Double each time the run looks healthy
- Cap at 270s (stay in prompt cache) during active training, then extend to 1200s+ once stable for 30+ min
- If the run crashes: debug, attempt restart, reset interval back to 60s

Each check:
1. Is the process alive? (ps aux | grep run_mixed, nvidia-smi)
2. Tail the log for errors, training step progress, loss values
3. If training has started, look for: NaN losses, sudden loss spikes, CUDA OOM, data loading errors
4. If metrics available (WandB or log), watch for suspicious drops in accuracy or unexpected loss behavior
5. Report status briefly

If the run dies: read the full error traceback, diagnose the issue, fix the script if needed, and restart. Then reset to frequent checking.
```

## Notes

- The loop dies if you close the session — just re-paste the above to restart
- Adjust the log path and grep pattern for different experiments
- The exponential backoff keeps it cheap: frequent early (catches crashes), then backs off once stable
