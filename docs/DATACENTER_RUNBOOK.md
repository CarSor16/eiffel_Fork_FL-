# Data-center runbook

This runbook prepares Eiffel FL Security Lab for the full real-dataset campaign on a
Linux university data center. The local synthetic benchmark remains useful for code
validation; the profiles in this document are intended for thesis-scale experiments.

## Selected datasets

The two primary real datasets are:

- **NF-CSE-CIC-IDS2018-v2**, the standardized NetFlow V2 representation of
  CSE-CIC-IDS2018.
- **NF-UNSW-NB15-v2**, the standardized NetFlow V2 representation of UNSW-NB15.

Both use the same NF-V2 feature convention already supported by Eiffel, so the same
model and attack implementations can be applied without dataset-specific model code.

The data files are deliberately not committed to Git.

## Expected environment variables

Point the project to the files on the shared/scratch filesystem:

```bash
export EIFFEL_CICIDS_PATH=/shared/datasets/NF-CSE-CIC-IDS2018-v2.csv.gz
export EIFFEL_NB15_PATH=/shared/datasets/NF-UNSW-NB15-v2.csv.gz
```

The paths may also point to uncompressed CSV files.

Validate both files without loading the entire dataset:

```bash
bash run-datacenter.sh --check-datasets
```

The checker verifies that the file exists and that the NF-V2 columns required by Eiffel
are present.

## Environment setup

The supported research stack remains Python 3.10, TensorFlow 2.10.0, Flower 1.5.0 and
Ray 2.6.3.

```bash
git switch feature/fl-security-lab
git pull origin feature/fl-security-lab
bash setup-datacenter.sh --dev
```

If the cluster exposes Python 3.10 with a different executable:

```bash
PYTHON_BIN=/path/to/python3.10 bash setup-datacenter.sh --dev
```

The default configuration is CPU-first. TensorFlow 2.10 GPU use requires a
TensorFlow/CUDA-compatible cluster environment; do not change the pinned research stack
only to make a newer CUDA module work. A cluster-provided compatible container is a
better option if GPU execution is required.

## Dataset profiles

CSE-CIC-IDS2018 profiles are named:

```text
dc_cicids_clean
dc_cicids_label_flip_targeted
dc_cicids_sign_flip
dc_cicids_model_scaling
dc_cicids_gaussian_noise
dc_cicids_lie
dc_cicids_gradient_mimicry
dc_cicids_colluding_sign_flip
dc_cicids_min_max
dc_cicids_min_sum
dc_cicids_adaptive_stealth
dc_cicids_heterogeneity_aware_mimicry
dc_cicids_targeted_family_poisoning
```

UNSW-NB15 uses the same suffixes with the `dc_nb15_` prefix.

All committed data-center profiles use:

```text
10 total federated clients
30 communication rounds
1 local epoch
batch size 512
Popoola/Eiffel MLP
Dirichlet alpha = 0.5
FedAvg with instrumented round storage
family-aware binary evaluation
probe size 512
```

The CSE-CIC targeted experiments use the `Bot` family. UNSW-NB15 targeted experiments
use `Exploits`. These targets have materially more observations than the very rare
families and are therefore better suited to the first controlled targeted comparison.

## Dry-run validation

No training is started:

```bash
bash run-datacenter.sh dc_cicids_clean --dry-run
bash run-datacenter.sh dc_nb15_sign_flip --dry-run
```

On Windows the same TOMLs can be translated with:

```powershell
.\run.cmd -Suite datacenter-cicids -DryRun
.\run.cmd -Suite datacenter-nb15 -DryRun
```

## Single experiment

Example:

```bash
bash run-datacenter.sh dc_cicids_sign_flip \
  --seed 2026 \
  --max-concurrent-clients 8
```

`max-concurrent-clients` changes only how many Ray client jobs execute simultaneously.
The federated experiment still contains all 10 clients and FedAvg still waits for all
required client updates.

Results are grouped below:

```text
outputs/datacenter/<dataset>/<profile>/seed_<seed>/<timestamp>/
```

Each run retains `round_state.h5`, the resolved TOML profile and Eiffel metric files.

## Full sequential suite

```bash
bash run-datacenter.sh --suite cicids --seed 2026 --max-concurrent-clients 8
bash run-datacenter.sh --suite nb15 --seed 2026 --max-concurrent-clients 8
```

To run both datasets sequentially:

```bash
bash run-datacenter.sh --suite all --seed 2026 --max-concurrent-clients 8
```

For a large shared machine, set the concurrency only after checking available RAM.
Full CSE-CIC-IDS2018 is substantially larger than UNSW-NB15.

## SLURM

The repository contains a generic template:

```text
scripts/datacenter/eiffel_experiment.sbatch
```

It requests 16 CPUs, 128 GiB RAM and 24 hours by default. Those values are a starting
point, not assumptions about the university cluster. Add the local `--partition`,
`--account`, QoS and GPU directives required by the data center.

Example:

```bash
sbatch --export=ALL,PROFILE=dc_cicids_sign_flip,SEED=2026,MAX_CONCURRENT_CLIENTS=8 \
  scripts/datacenter/eiffel_experiment.sbatch
```

The dataset environment variables must be exported before submission or set by the
cluster job environment.

## Multi-seed campaign

After a small number of successful full runs, a matrix can be submitted with:

```bash
bash scripts/datacenter/submit_matrix.sh cicids 2026,2027,2028,2029,2030
bash scripts/datacenter/submit_matrix.sh nb15 2026,2027,2028,2029,2030
```

This submits one SLURM job per profile/seed. Inspect the script and cluster quota before
submitting a full matrix.

## Recommended staged campaign

Do not start with the entire 26-profile x multi-seed matrix. First validate:

```text
clean
targeted label flip
sign flip
colluding sign flip
gradient mimicry
targeted family poisoning
```

on each dataset with one seed. Once HDF5 and metrics are valid, repeat the scientifically
interesting subset with at least five seeds. Only then expand to scaling, Gaussian,
LIE, Min-Max, Min-Sum, Adaptive Stealth and Heterogeneity-Aware Mimicry.

This ordering reduces wasted cluster time while preserving the final reproducibility
plan.

## Analysis

The existing analyzers recursively discover `round_state.h5` files, so data-center
runs can be analyzed without changing the experiment code.

Linux:

```bash
.venv/bin/python -m eiffel.analysis.compare_metrics \
  --runs-root outputs/datacenter/cicids \
  --output-dir analysis-results/datacenter-cicids
```

and:

```bash
.venv/bin/python -m eiffel.analysis.compare_metrics \
  --runs-root outputs/datacenter/nb15 \
  --output-dir analysis-results/datacenter-nb15
```

Keep the two datasets in separate detailed comparisons so their clean baselines are not
pooled together.
