#!/usr/bin/env bash
#SBATCH --job-name=eiffel-fl
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${PROFILE:?Set PROFILE, e.g. dc_cicids_sign_flip}"
SEED="${SEED:-2026}"
MAX_CONCURRENT_CLIENTS="${MAX_CONCURRENT_CLIENTS:-8}"

echo "Host: $(hostname)"
echo "Profile: $PROFILE"
echo "Seed: $SEED"
echo "Concurrent Ray clients: $MAX_CONCURRENT_CLIENTS"
echo "SLURM CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"

bash run-datacenter.sh "$PROFILE" \
  --seed "$SEED" \
  --max-concurrent-clients "$MAX_CONCURRENT_CLIENTS"
