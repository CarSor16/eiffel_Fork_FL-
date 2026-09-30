#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:-}"
SEEDS_CSV="${2:-2026}"
SBATCH_FILE="${SBATCH_FILE:-scripts/datacenter/eiffel_experiment.sbatch}"

case "$SUITE" in
  cicids) prefix="dc_cicids" ;;
  nb15) prefix="dc_nb15" ;;
  *) echo "Usage: bash scripts/datacenter/submit_matrix.sh cicids|nb15 2026,2027,2028" >&2; exit 2 ;;
esac

suffixes=(clean label_flip_targeted sign_flip model_scaling gaussian_noise lie gradient_mimicry colluding_sign_flip min_max min_sum adaptive_stealth heterogeneity_aware_mimicry targeted_family_poisoning)

IFS=',' read -r -a seeds <<< "$SEEDS_CSV"
for seed in "${seeds[@]}"; do
  for suffix in "${suffixes[@]}"; do
    profile="${prefix}_${suffix}"
    echo "Submitting $profile seed=$seed"
    sbatch --export=ALL,PROFILE="$profile",SEED="$seed" "$SBATCH_FILE"
  done
done
