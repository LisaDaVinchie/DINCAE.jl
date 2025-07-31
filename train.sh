#!/bin/bash
#SBATCH --job-name=train_dincae
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1            # Request 1 GPU

set -e

notify_telegram() {
    local status="$1"
    curl -s -X POST "https://api.telegram.org/bot${bot_id}/sendMessage" \
    -d chat_id=${chat_id} \
    -d text="Your Land_data_inpainting slurm job (Job ID: $SLURM_JOB_ID) has completed with status: $status"
}

trap 'notify_telegram "FAILED (job terminated or timed out)"' TERM EXIT


ulimit -a

# Create logs folder if it doesn't exist
mkdir -p logs

# Set PATH to include your local Julia
export PATH="$HOME/julia-1.11.6/bin:$PATH"

# Set Julia threads to match SLURM's CPU allocation
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

which julia
julia -e 'using CUDA; CUDA.versioninfo()'

# Run your Julia script
julia --project=./ examples/DINCAE_tutorial.jl || { echo "Error: Julia script failed to execute."; exit 1; }

notify_telegram "SUCCESS"

# Clear the EXIT trap to avoid double notifications
trap - EXIT

echo "Training completed successfully"
