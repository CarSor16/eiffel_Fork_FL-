# Eiffel FL Security Lab

This repository is a research fork of [Léo Lavaur's Eiffel framework](https://github.com/leolavaur/eiffel), an evaluation framework for Federated Learning-based intrusion detection built on Flower.

The purpose of this fork is to extend Eiffel into a reusable experimental platform for the thesis/research topic:

> **Attacks to Federated Learning in Network-Related Tasks**

The current focus is Federated Intrusion Detection Systems (FL-NIDS/FIDS), with particular attention to poisoning attacks, non-IID client data, malicious-client coordination, update-space auditability, temporal attack behaviour, and comparison across different neural architectures.

The original Eiffel design is intentionally preserved where possible: Hydra remains the experiment configuration system, Flower remains the FL runtime, and TensorFlow/Keras remains the model framework. The extensions in this fork are added around that architecture rather than replacing it.

---

## Quick start: complete Windows setup

The supported local workflow uses **Python 3.10** and does **not require Poetry**.

From the repository root, the shortest setup is:

```powershell
.\setup.cmd
```

`setup.cmd` checks Python 3.10, creates `.venv`, installs Eiffel plus its runtime dependencies, and verifies imports for TensorFlow, Flower, Hydra, HDF5, and Eiffel.

Then validate the environment and TOML configuration layer:

```powershell
.\run.cmd -Doctor
```

The recommended first end-to-end run is:

```powershell
.\run.cmd synthetic_50k_quick_clean
```

followed by:

```powershell
.\run.cmd synthetic_50k_quick_sign_flip
```

Both quick profiles use the complete 50,000-sample synthetic federated training set but only 5 communication rounds.

### Prerequisites

Required:

```text
Windows 10/11
Git
Python 3.10.x
Internet access for the first dependency installation
```

Check installed Python versions with:

```powershell
py -0p
```

If Python 3.10 is missing:

```powershell
winget install -e --id Python.Python.3.10
```

Close and reopen the terminal after installation, then run `py -0p` again.

### Environment commands

Normal one-time setup:

```powershell
.\setup.cmd
```

Rebuild the environment from scratch:

```powershell
.\setup.cmd -Force
```

Install the optional test dependency as well:

```powershell
.\setup.cmd -Dev
```

The project interpreter is:

```text
.venv\Scripts\python.exe
```

For IntelliJ/PyCharm, select that interpreter for the project.

Manual equivalent:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e .
```

Activation is optional. If desired:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The `.cmd` launchers already invoke PowerShell with a process-local execution-policy bypass, so using `setup.cmd` and `run.cmd` normally avoids this problem.

### Unified experiment launcher

The simplest command is:

```powershell
.\run.cmd
```

With no arguments it displays an interactive menu of all TOML experiment profiles.

Run a named profile:

```powershell
.\run.cmd synthetic_50k_quick_clean
```

The `.toml` extension is optional.

List available profiles:

```powershell
.\run.cmd -List
```

Validate a profile without starting Flower:

```powershell
.\run.cmd synthetic_50k_quick_sign_flip -DryRun
```

Run the environment/configuration doctor:

```powershell
.\run.cmd -Doctor
```

Run the complete synthetic 50k attack suite:

```powershell
.\run.cmd -Suite synthetic50k
```

Validate the whole suite without training:

```powershell
.\run.cmd -Suite synthetic50k -DryRun
```

Extra Hydra overrides can still be appended for quick exploratory runs, for example:

```powershell
.\run.cmd synthetic_50k_quick_sign_flip storage.capture_inference=false
```

For experiments intended to be reported in the thesis, prefer editing or creating a TOML file so the complete configuration is explicit and reproducible.

### Real-dataset files

The synthetic profiles require no external dataset. Real NF-V2 profiles expect sampled files under:

```text
data/
└── nfv2/
    └── sampled/
        ├── cicids.csv.gz
        ├── nb15.csv.gz
        ├── toniot.csv.gz
        └── botiot.csv.gz
```

The dataset files themselves are not bundled in this repository.

### Where results are saved

Hydra creates one run directory under:

```text
outputs/YYYY-MM-DD/HH-MM-SS/
```

Typical run artifacts include `stats.json`, Hydra's `.hydra/` configuration files, normal Eiffel metrics, and—when instrumented storage is enabled—`round_state.h5`.

Because the default HDF5 path is relative, `round_state.h5` is written inside the corresponding Hydra run directory.

### Dependency versions used by the supported Windows setup

The setup intentionally pins the core runtime because Eiffel's original dependency ranges are too broad for a modern `pip` resolver.

The supported Python 3.10 environment uses:

```text
TensorFlow   2.10.0
Flower       1.5.0
Ray          2.6.3
cryptography 41.0.4
protobuf     3.19.6
NumPy        1.23.5
fsspec       2023.10.0
pydantic     1.10.17
```

The exact constraints are stored in:

```text
constraints-py310.txt
```

`setup.cmd` installs the project using those constraints. This prevents `pip` from selecting a much newer Flower release whose cryptography/protobuf requirements are incompatible with the Eiffel-era stack.

The repository's historical `poetry.lock` is not used by the standard Windows setup. The supported path is `setup.cmd` / `pip` plus `constraints-py310.txt`.

### Troubleshooting

If `py -3.10` reports that no suitable runtime exists, install Python 3.10 with the `winget` command above and reopen the terminal.

If `.venv\Scripts\Activate.ps1` does not exist, the virtual environment has not been created; run:

```powershell
.\setup.cmd
```

If the doctor reports `ModuleNotFoundError: No module named 'absl'`, the TensorFlow dependency set was only partially installed. The current setup installs the TensorFlow/Flower runtime explicitly before installing Eiffel, runs `pip check`, and verifies `absl-py` during the final import test.

If Python prints an error while processing `protobuf-3.19.6-...-nspkg.pth`, pull the latest branch and rebuild. TensorFlow 2.10 requires protobuf below 3.20, and protobuf 3.19.6 ships a legacy namespace `.pth` file. The setup removes that obsolete compatibility file after installation; Python 3.10 can use the `google` namespace package without it.

Recovery:

```powershell
git pull origin feature/fl-security-lab
.\setup.cmd -Force
.\run.cmd -Doctor
```

If the setup import check fails with `ModuleNotFoundError: No module named 'pkg_resources'`, the cause is an overly new Setuptools release. Setuptools removed `pkg_resources` in version 82.0.0, while Ray 2.6.3 still imports it. The supported environment therefore pins:

```text
setuptools==80.9.0
```

Pull the latest branch and rebuild the environment:

```powershell
git pull origin feature/fl-security-lab
.\setup.cmd -Force
```

The CUDA warning about `cudart64_110.dll` can be ignored on a CPU-only machine; TensorFlow explicitly continues without CUDA when the DLL is unavailable.

If installation fails with a Ray / PyArrow conflict, make sure you are on the latest branch version. The supported setup intentionally installs plain `ray==2.6.3`, **without** Ray's `data` extra: that extra requires an old PyArrow range incompatible with the dataset stack. The project pins `pyarrow==16.1.0`, which has a CPython 3.10 Windows wheel and is compatible with the NF-V2 pandas loader.

If installation fails with `ResolutionImpossible` and mentions Flower / cryptography / protobuf, first pull the latest branch and rebuild the virtual environment:

```powershell
git pull origin feature/fl-security-lab
.\setup.cmd -Force
```

Do not manually install the newest Flower release into this environment. The project currently targets Flower 1.5.0 for compatibility with the Eiffel codebase and TensorFlow 2.10 stack.

If the environment becomes inconsistent, rebuild it:

```powershell
.\setup.cmd -Force
```

If a real dataset run reports `Dataset not found`, verify the corresponding `.csv.gz` file under `data/nfv2/sampled/`. Synthetic runs do not use those files.

If you are unsure whether a TOML is valid, use `-DryRun`. If you are unsure whether the environment is healthy, use `-Doctor`.

### Tests

Install test support:

```powershell
.\setup.cmd -Dev
```

Then run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Focused tests for the new experiment layer:

```powershell
.\.venv\Scripts\python.exe -m pytest eiffel\core\tests\toml_runner_test.py
.\.venv\Scripts\python.exe -m pytest eiffel\core\tests\synthetic_stress_test.py
```

---
## Research direction

The project currently follows this progression:

1. reproduce and retain Eiffel's original label-flipping experiments;
2. add procedural model-poisoning attacks;
3. analyse how attacks behave under heterogeneous/non-IID federated data;
4. record both raw model-update information and behavioural inference information round by round;
5. compare attack effects across different NIDS models;
6. extend the evaluation toward adaptive, coordinated and stealthier poisoning scenarios;
7. later investigate more complex adversaries, including possible LLM-guided attack generation.

The main experimental objective is not only to measure final accuracy degradation, but also to study **how malicious updates evolve across rounds**, how distinguishable they are from benign updates, and whether an attack can selectively alter intrusion-detection behaviour while remaining difficult to detect in update space.

---

## What this fork adds

### Procedural model-poisoning attacks

In addition to Eiffel's original data-poisoning/label-flipping mechanisms, this fork adds model-update attacks applied immediately before server aggregation:

- **Sign Flip**
- **Model Scaling**
- **Gaussian Noise**
- **LIE**
- **Gradient Mimicry**
- **Colluding Sign Flip**

The attacks operate on the client model delta:

```text
Δ_i = W_i(local) - W_global
```

so the attack implementation is independent of the neural-network architecture as long as all clients use the same parameter layout.

This makes it possible to test the same poisoning mechanism on an MLP, CNN or Transformer without rewriting the attack itself.

---

## Temporal attack schedules

Model attacks can be configured to evolve over communication rounds.

Currently supported schedules are:

- `continuous`
- `late`
- `window`
- `on_off`
- `gradual`

Example:

```bash
model_attack=gradient_mimicry \
model_attack.schedule.type=late \
model_attack.schedule.start_round=6
```

This allows experiments where the adversary remains benign during the first rounds, attacks only inside a specific time window, alternates between benign and malicious behaviour, or gradually increases attack intensity.

---

## Models

The original Eiffel/Popoola-style MLP remains available and is kept as the main compatibility baseline.

Additional TensorFlow/Keras models have been added:

```text
model=popoola
model=p4p_mlp
model=cnn1d
model=ft_transformer
```

### Popoola MLP

Original Eiffel model:

```text
input
  ↓
Dense 128 + ReLU
  ↓
Dense 128 + ReLU
  ↓
Sigmoid output
```

This remains useful for reproducing experiments close to the original Eiffel/Lavaur setup.

### P4P-style MLP

A lightweight NIDS MLP inspired by recent FL-NIDS work:

```text
input
  ↓
Dense 128 + ReLU + Dropout
  ↓
Dense 64 + ReLU + Dropout
  ↓
Sigmoid output
```

### 1D-CNN

A convolutional model for flow-feature classification:

```text
features
  ↓
Conv1D
  ↓
Pooling
  ↓
Conv1D
  ↓
Pooling
  ↓
Conv1D
  ↓
Global Average Pooling
  ↓
Dense classifier
```

### FT-Transformer

A compact Transformer designed for numeric/tabular network-flow features.

Each feature is converted into an independent learned token, a learnable `[CLS]` token is prepended, and the sequence is processed by Transformer encoder blocks.

```text
numeric flow features
        ↓
feature tokenizer
        ↓
[CLS] + feature tokens
        ↓
multi-head self-attention
        ↓
feed-forward blocks
        ↓
[CLS] representation
        ↓
binary classifier
```

The default configuration is intentionally lightweight so it can be used in repeated federated experiments without the computational cost of a large Transformer.

---

## Round-by-round experiment storage

One of the main extensions of this fork is the ability to preserve the complete evolution of an FL run without creating thousands of individual files.

Each run can generate a single:

```text
round_state.h5
```

file.

The file is written incrementally and flushed after every completed round.

Conceptually, it contains:

```text
round_state.h5

/global/
    round_0000/
        weights/
    round_0001/
        weights/
    ...
    round_XXXX/
        weights/

/clients/
    round_0001/
        <client_id>/
            submitted_update/
            pre_attack_update/
            audit/
            inference
    ...

/probe/
    labels
```

### Global model state

The global model is stored at:

```text
round 0  -> initial global model
round 1  -> global model after aggregation round 1
round 2  -> global model after aggregation round 2
...
```

This makes it possible to reconstruct the complete trajectory of the global model.

### Submitted client updates

For every client and every communication round, the server stores the exact submitted model update:

```text
Δ_i^t
```

rather than storing another redundant complete local model.

Given the global model before the round:

```text
W_i^t = W_global^t + Δ_i^t
```

the corresponding local model can be reconstructed later.

This keeps the results useful for future analysis while reducing unnecessary duplication.

### Pre-attack updates

For model-poisoning attacks, a malicious client can have:

```text
local training
      ↓
pre-attack update
      ↓
attack transformation
      ↓
submitted update
      ↓
server aggregation
```

The pre-attack update is therefore stored only when a malicious attack actually modifies it.

This allows direct measurement of how an attack transforms an otherwise locally trained update.

---

## Update-space audit

For each submitted update, compact server-observable audit features are stored.

Current metrics include:

```text
L1 norm
L2 norm
L∞ norm
cosine similarity to round mean
distance to round mean
distance to round median
sign agreement with round mean
```

These metrics are intended for round-by-round comparison of benign and malicious clients and can later be extended with temporal, clustering or geometry-based signals.

Because the raw submitted updates are also preserved, new audit metrics can be computed retrospectively without rerunning the federated experiment.

---

## Inference capture

The project also records a compact behavioural view of every locally trained client model.

After local training, the client performs a deterministic inference over a fixed small probe slice.

The inference output is saved for each client and each round.

This makes it possible to compare:

```text
update-space behaviour
          +
model inference behaviour
```

and later compute metrics such as:

```text
prediction disagreement
confidence
entropy
class margins
KL divergence
JS divergence
target-class degradation
round-to-round logit drift
```

without rerunning training.

---

## Storage efficiency

The storage format is designed to preserve raw information while keeping experiments manageable.

Current choices are:

```text
global weights      float32
submitted updates   float32
pre-attack updates  float32
audit metrics       float32
inference outputs   float16
probe labels        int16
```

The HDF5 file uses compression and avoids repeating text labels or creating one file per client per round.

Weights and updates deliberately remain `float32` because later attack analysis may rely on small geometric differences between client updates.

Inference outputs use `float16` because they are mainly used for behavioural analysis and can be stored much more compactly.

---

## Pure model-poisoning experiments

Eiffel originally associates malicious clients with dataset poisoning.

This fork adds a clean poisoning profile so that a client can be marked as malicious while its local dataset remains unchanged.

Example:

```bash
num_clients=9 \
num_attackers=1 \
poisoning/profile=clean \
model_attack=sign_flip
```

In this case:

```text
9 benign clients
1 malicious client

local data of malicious client: clean
local training: normal
submitted update: Sign Flip attack
```

This distinction is important for separating **data poisoning** from **model poisoning**.

---

## Label-flipping experiments

The original Eiffel data-poisoning system is intentionally retained.

To reproduce an Eiffel-style label-flipping experiment, keep:

```text
model_attack=none
```

and use the existing poisoning profiles and targeted/untargeted configurations.

This allows the same repository to support both:

```text
data poisoning
and
model poisoning
```

under a common Flower/Hydra experimental framework.

---

## Example experiment runs

For framework validation without downloading a dataset:

```powershell
.\run.cmd synthetic_50k_quick_clean
.\run.cmd synthetic_50k_quick_sign_flip
```

For the full synthetic attack profiles:

```powershell
.\run.cmd synthetic_50k_gradient_mimicry
.\run.cmd synthetic_50k_colluding_sign_flip
```

Real-data profiles use the same launcher, for example:

```powershell
.\run.cmd realistic_sign_flip
.\run.cmd realistic_gradient_mimicry
```

Real-data commands require the corresponding NF-V2 file under `data/nfv2/sampled/`.

To try another model without editing the TOML:

```powershell
.\run.cmd synthetic_50k_quick_sign_flip model=cnn1d
```

For thesis runs, prefer saving the model choice directly in a dedicated TOML profile.

---

## TOML experiment workflow

The previous Flower attack lab used one TOML file per experiment. This fork keeps that workflow while Eiffel continues to use Hydra internally.

Runnable profiles are stored in:

```text
experiments/toml/
```

For normal use, prefer the unified launcher:

```powershell
.\run.cmd smoke_sign_flip
```

The lower-level compatibility entrypoints are still available:

```powershell
.\run-toml.ps1 experiments\toml\smoke_sign_flip.toml
python -m eiffel.toml_runner experiments\toml\smoke_sign_flip.toml
```

To validate the profile and inspect the generated Eiffel/Hydra command without starting the simulation:

```powershell
python -m eiffel.toml_runner experiments\toml\smoke_sign_flip.toml --dry-run
```

The compatibility layer supports the current attack set, temporal schedules, model choice, Dirichlet non-IID partitioning, HDF5 storage, the restored 50k synthetic benchmark, and NF-V2 dataset aliases. Synthetic and real-data profiles are kept separate so framework validation is not confused with real NIDS evaluation.

See [docs/TOML_EXPERIMENTS.md](docs/TOML_EXPERIMENTS.md) for the complete mapping and supported fields.

### Synthetic 50k benchmark

The controlled synthetic stress benchmark from the previous Flower attack lab is also
available in this Eiffel build.

The default setup creates:

```text
10 clients x 5,000 training samples = 50,000 federated samples
12,000 separate common test samples
32 features
6 traffic families
Dirichlet non-IID label skew (alpha = 0.5)
client-specific covariate shift
class overlap, rare attacks, label noise and outliers
```

The six traffic families remain available as metadata, while the learning target is
binary (Benign vs Attack) to stay compatible with Eiffel's NIDS models.

For a short validation run using the complete 50k training set:

```powershell
.\run-toml.ps1 experiments\toml\synthetic_50k_quick_clean.toml
```

and the corresponding Sign Flip test is:

```powershell
.\run-toml.ps1 experiments\toml\synthetic_50k_quick_sign_flip.toml
```

A complete attack suite can be started with:

```powershell
.\run-synthetic-50k-suite.ps1
```

## Configuration

Eiffel uses Hydra.

The main configuration remains:

```text
eiffel/conf/eiffel.yaml
```

Additional configuration groups introduced by this fork include:

```text
eiffel/conf/model/
eiffel/conf/model_attack/
eiffel/conf/storage/
```

The instrumented strategy is:

```text
strategy=instrumented_fedavg
```

and is currently the default strategy of this research fork.

The original Flower FedAvg configuration remains available:

```text
strategy=fedavg
```

---

## Storage configuration

Default storage configuration:

```yaml
storage:
  enabled: true
  path: round_state.h5
  compression: gzip
  compression_level: 4
  flush_each_round: true
  capture_inference: true
  probe_size: 256
```

Storage can be disabled for lightweight test runs:

```bash
storage.enabled=false
```

or inference collection can be disabled independently:

```bash
storage.capture_inference=false
```

---

## Project structure

The main research extensions are located in:

```text
eiffel/
├── analysis/
│   └── update_audit.py
│
├── attacks/
│   └── model.py
│
├── models/
│   ├── supervized.py
│   └── advanced.py
│
├── storage/
│   ├── round_store.py
│   └── serialization.py
│
├── strategy/
│   ├── fednoagg.py
│   └── instrumented.py
│
└── conf/
    ├── model/
    ├── model_attack/
    ├── poisoning/
    ├── storage/
    └── strategy/
```

More detailed implementation notes are available in:

[docs/FL_SECURITY_EXTENSIONS.md](docs/FL_SECURITY_EXTENSIONS.md)

---

## Original Eiffel usage

Eiffel can be used as an experiment engine by providing a Hydra configuration.

```bash
python -m eiffel -cd path/to/workdir/
```

By default, Eiffel looks for a Git repository in the current directory or one of its parents. The repository root is used as Hydra's working anchor for `outputs/` and `multirun/`.

---

## Current development status

This fork is currently an experimental research platform.

The current development branch introduces the attack, audit, storage and additional-model infrastructure while preserving the original Eiffel code path where possible.

The implementation has been structured so that experiments can later be expanded toward:

- real NIDS datasets such as NF-V2/CSE-CIC-IDS2018, UNSW-NB15 and TON-IoT;
- stronger non-IID partitions;
- multiple malicious-client ratios;
- temporal poisoning;
- coordinated/colluding adversaries;
- robust aggregation comparisons;
- semantic and temporal auditing;
- selective target-class degradation;
- more adaptive attack generation.

The immediate validation workflow is:

```text
clean baseline
      ↓
label flipping
      ↓
sign flip
      ↓
scaling / Gaussian noise
      ↓
LIE
      ↓
gradient mimicry
      ↓
colluding attacks
      ↓
real NIDS datasets
```

---

## Reproducibility note

The project is based on Eiffel and preserves the original framework as a reference point. New experiments should explicitly record:

```text
dataset
partitioning
number of clients
malicious-client ratio
seed
model
attack
attack schedule
aggregation strategy
number of rounds
local epochs
batch size
```

The round-state HDF5 file is intended to complement, not replace, the experiment configuration and standard metric outputs.

---

## References and acknowledgements

This repository is derived from:

- **Eiffel — Evaluation framework for FL-based intrusion detection using Flower**, Léo Lavaur  
  https://github.com/leolavaur/eiffel

The original Eiffel framework is retained as the architectural and experimental foundation of this fork.

The fork is being extended for research on adversarial Federated Learning in network-security applications, with particular emphasis on Federated Intrusion Detection Systems.


---

## FL-security validation, multiclass and metric comparison

The research launcher now also exposes the security-specific validation and analysis
workflow. To print the supported attacks, their meaning, temporal schedules and the
main TOML parameters:

    .\run.cmd -Attacks

Run the focused tests for procedural attacks, schedules, synthetic task modes, HDF5
round storage and metric plots:

    .\setup.cmd -Dev
    .\run.cmd -Tests

The complete binary/family-aware synthetic suite remains:

    .\run.cmd -Suite synthetic50k

A separate experimental true-multiclass synthetic suite is available for model
poisoning attacks:

    .\run.cmd -Suite multiclass

It trains directly on the six family IDs with a softmax classifier and sparse
categorical cross-entropy. The corresponding quick smoke profiles are:

    .\run.cmd synthetic_50k_quick_multiclass_clean
    .\run.cmd synthetic_50k_quick_multiclass_sign_flip

The binary Eiffel/Lavaur-compatible path remains the primary baseline. A TOML can use
dataset.task = "binary", or explicitly request binary training with family-level
reporting using dataset.task = "family_aware". For the synthetic stress dataset only,
dataset.task = "multiclass" with num_classes = 6 enables true multiclass training.

Model-update attacks are architecture/task agnostic and can therefore be used in
multiclass mode. Multiclass label flipping is intentionally rejected for now: Eiffel's
original poisoning operation is a binary Boolean label flip, so a scientifically valid
multiclass version must first define an explicit source-class -> destination-class
mapping instead of silently applying the binary operation to labels 0..5.

### Validate a round-state file

Every completed run can be checked for the HDF5 invariants required by the analysis:
round 0 global weights, previous/current global states, submitted-update shapes and
float32 dtypes, finite values, malicious/attack flags, pre-attack updates when required,
float16 inference capture, int16 probe labels and last_complete_round.

    .\run.cmd -ValidateHdf5 "outputs\YYYY-MM-DD\HH-MM-SS\round_state.h5"

The lower-level equivalent is:

    .\.venv\Scripts\python.exe -m eiffel.analysis.validate_round_state "outputs\YYYY-MM-DD\HH-MM-SS\round_state.h5"

### Compare metrics across rounds and attacks

To recursively analyse all Eiffel runs below outputs/:

    .\run.cmd -Analyze

Custom input/output roots can be selected with:

    .\run.cmd -Analyze -RunsRoot "outputs" -AnalysisOutput "analysis-results"

The analysis reads the metrics already persisted in round_state.h5; training does not
need to be repeated. It produces CSV tables plus line plots for each metric across
communication rounds, per-round attack deltas relative to the clean baseline, a grouped
bar plot comparing final-round metrics across attacks, a final-metric delta plot
relative to clean, final per-family recall comparisons, and per-family recall deltas
relative to clean. The per-round delta plots are especially useful for late, gradual
and on/off attacks because they show when degradation begins and how it evolves.

When several runs have the same inferred attack label, for example different seeds,
round trends and final attack comparisons aggregate them; final bars include standard
deviation error bars. For historical or manually organised runs, labels can be supplied
explicitly:

    .\.venv\Scripts\python.exe -m eiffel.analysis.compare_metrics --run clean="path\to\clean\round_state.h5" --run sign_flip="path\to\sign_flip\round_state.h5" --output-dir "analysis-results"

The default global metrics include accuracy, F1 where available, macro-F1, weighted-F1,
recall, miss rate, MCC, macro attack-family recall and minimum attack-family recall.
Per-family precision/recall/F1/miss-rate values are exported separately whenever the run
contains them.

### Hydra and TOML

Hydra has not been removed. The supported workflow remains TOML experiment profile ->
eiffel.toml_runner -> Hydra composition/overrides -> EIFFeL + Flower.

TOML is the stable user-facing experiment description, while Hydra remains useful for
configuration groups, command-line overrides, reproducibility, output directories and
future sweeps. A normal named TOML experiment can still receive a temporary Hydra
override from .\run.cmd without changing the committed profile.
