#!/bin/bash
#SBATCH --job-name=1_4b_wd3_2_x16
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --output=/fsx/users/dongweij/marin/logs/1_4b_wd3_2_x16_nocrossblock_slurm_%j.log
#SBATCH --error=/fsx/users/dongweij/marin/logs/1_4b_wd3_2_x16_nocrossblock_slurm_%j.err

set -e
cd /fsx/users/dongweij/marin

echo "Job: $SLURM_JOB_ID on node $SLURMD_NODENAME"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -8
echo ""

.venv/bin/python -m experiments.data_efficiency.run_1_4b_wd3_2_x16_nocrossblock

echo ""
echo "End: $(date)"
