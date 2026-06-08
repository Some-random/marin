#!/bin/bash
#SBATCH --job-name=c5v2-prod
#SBATCH --partition=gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=900G
#SBATCH --time=48:00:00
#SBATCH --output=/fsx/users/dongweij/marin/logs/c5v2_prod/slurm_%j.out
#SBATCH --error=/fsx/users/dongweij/marin/logs/c5v2_prod/slurm_%j.err
#
# Multi-node production launch for C5-v2.
# Override --nodes via `sbatch --nodes=4 ...` if 8 not available.
#
# Each node runs one python process; JAX coordinates across 8 GPUs per node
# (data axis = local 8) and across nodes (replica_dcn = -1). Global batch=256.

set -e
mkdir -p /fsx/users/dongweij/marin/logs/c5v2_prod
cd /fsx/users/dongweij/marin

echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] Job $SLURM_JOB_ID on nodes: $SLURM_JOB_NODELIST"

# Coordinator is the first node
COORD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
COORD_IP=$(getent hosts "$COORD_NODE" | awk '{print $1}')
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] Coordinator: $COORD_NODE ($COORD_IP):33333"

# srun launches one task per node; SLURM_NODEID is the process rank
srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
  bash -c "
    cd /fsx/users/dongweij/marin
    echo \"[\$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] node rank \$SLURM_NODEID / \$SLURM_JOB_NUM_NODES on \$SLURMD_NODENAME\"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
    export JAX_DIST_NUM_PROCESSES=\$SLURM_JOB_NUM_NODES
    export JAX_DIST_PROCESS_ID=\$SLURM_NODEID
    export JAX_DIST_COORDINATOR=$COORD_IP:33333
    export NCCL_DEBUG=WARN
    export FI_PROVIDER=efa
    export FI_EFA_USE_DEVICE_RDMA=1
    # Only rank 0 logs to wandb online; others go offline so we don't double-write
    if [ \"\$SLURM_NODEID\" = \"0\" ]; then
      export WANDB_MODE=online
    else
      export WANDB_MODE=offline
    fi
    .venv/bin/python -m experiments.data_efficiency.run_1_4b_c5v2_clean_code
  "
