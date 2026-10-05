#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PROFILES_DIR="$ROOT/experiments/toml"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/datacenter}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Virtual environment not found. Run: bash setup-datacenter.sh" >&2
  exit 1
fi

PROFILE=""
SUITE=""
SEED=""
MAX_CONCURRENT=0
DRY_RUN=0
CHECK_DATASETS=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) SUITE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-concurrent-clients) MAX_CONCURRENT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --check-datasets) CHECK_DATASETS=1; shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    -*) EXTRA+=("$1"); shift ;;
    *) if [[ -z "$PROFILE" ]]; then PROFILE="$1"; else EXTRA+=("$1"); fi; shift ;;
  esac
done

if [[ "$CHECK_DATASETS" == "1" ]]; then
  "$PYTHON" scripts/datacenter/check_datasets.py all
  exit 0
fi

resolve_profile() {
  local name="${1%.toml}"
  local path="$PROFILES_DIR/$name.toml"
  [[ -f "$path" ]] || { echo "Unknown profile: $name" >&2; exit 2; }
  printf '%s\n' "$path"
}

run_profile() {
  local name="$1"
  local path
  path="$(resolve_profile "$name")"
  local args=("$PYTHON" -m eiffel.toml_runner "$path")
  [[ "$DRY_RUN" == "1" ]] && args+=(--dry-run)
  [[ -n "$SEED" ]] && args+=("seed=$SEED")
  [[ "$MAX_CONCURRENT" -gt 0 ]] && args+=("++experiment.max_concurrent_clients=$MAX_CONCURRENT")

  local dataset="misc"
  [[ "$name" == dc_cicids_* ]] && dataset="cicids"
  [[ "$name" == dc_nb15_* ]] && dataset="nb15"
  local run_seed="${SEED:-2026}"
  local tag="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
  args+=("hydra.run.dir=$OUTPUT_ROOT/$dataset/$name/seed_$run_seed/$tag")
  args+=("${EXTRA[@]}")

  echo "============================================================"
  echo "Profile: $name"
  echo "============================================================"
  "${args[@]}"
}

suffixes=(clean label_flip_targeted sign_flip model_scaling gaussian_noise lie gradient_mimicry colluding_sign_flip min_max min_sum adaptive_stealth heterogeneity_aware_mimicry targeted_family_poisoning)

if [[ -n "$SUITE" ]]; then
  case "$SUITE" in
    cicids) prefixes=(dc_cicids) ;;
    nb15) prefixes=(dc_nb15) ;;
    all) prefixes=(dc_cicids dc_nb15) ;;
    *) echo "Suite must be cicids, nb15, or all." >&2; exit 2 ;;
  esac
  "$PYTHON" scripts/datacenter/check_datasets.py "$([[ "$SUITE" == "all" ]] && echo all || echo "$SUITE")"
  for prefix in "${prefixes[@]}"; do
    for suffix in "${suffixes[@]}"; do
      run_profile "${prefix}_${suffix}"
    done
  done
  exit 0
fi

[[ -n "$PROFILE" ]] || {
  echo "Usage: bash run-datacenter.sh PROFILE [--seed N] [--max-concurrent-clients N]" >&2
  echo "   or: bash run-datacenter.sh --suite cicids|nb15|all ..." >&2
  exit 2
}
run_profile "$PROFILE"
