#!/bin/bash
#SBATCH --job-name=train_dincae
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1            # Request 1 GPU

source bot_codes.txt
bot_id=${bot_id}
chat_id=${chat_id}

# export JULIA_LOAD_LIBCPP=false
# export LD_PRELOAD=/usr/lib64/libstdc++.so.6  # Replace if needed

set -euo pipefail

notify_telegram() {
    local status="$1"
    curl -s -X POST "https://api.telegram.org/bot${bot_id}/sendMessage" \
        -d chat_id="${chat_id}" \
        -d text="🚨 DINCAE SLURM job (Job ID: ${SLURM_JOB_ID}) completed with status: ${status}"
}

# Notify if job fails (trap must come *after* function definition)
trap 'notify_telegram "❌ FAILED (job terminated or timed out)"' TERM EXIT

ulimit -a

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up environment
export PATH="$HOME/julia-1.11.6/bin:$PATH" # Ensure Julia is in PATH

echo "🔧 Using Julia at: $(which julia)"
julia -e 'using CUDA; CUDA.versioninfo()'

source dincae_venv/bin/activate

# Run Julia script with fallback notification
if julia --project=. examples/DINCAE_tutorial.jl; then
    notify_telegram "✅ SUCCESS"
    trap - EXIT  # disable failure trap
    echo "🎉 Training completed successfully."
else
    echo "❗ Error: Julia script failed to execute."
    exit 1
fi

source deactivate
