# Eiffel FL Security Lab

This repository is a research fork of [Léo Lavaur's Eiffel framework](https://github.com/leolavaur/eiffel), an evaluation framework for Federated Learning-based intrusion detection built on Flower.

The purpose of this fork is to extend Eiffel into a reusable experimental platform for the thesis/research topic:

> **Attacks to Federated Learning in Network-Related Tasks**

The current focus is Federated Intrusion Detection Systems (FL-NIDS/FIDS), with particular attention to poisoning attacks, non-IID client data, malicious-client coordination, update-space auditability, temporal attack behaviour, and comparison across different neural architectures.

The original Eiffel design is intentionally preserved where possible: Hydra remains the experiment configuration system, Flower remains the FL runtime, and TensorFlow/Keras remains the model framework. The extensions in this fork are added around that architecture rather than replacing it.

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

## Example model-poisoning runs

Continuous Sign Flip:

```bash
python -m eiffel \
num_clients=9 \
num_attackers=1 \
poisoning/profile=clean \
model_attack=sign_flip
```

Gradient Mimicry:

```bash
python -m eiffel \
num_clients=9 \
num_attackers=1 \
poisoning/profile=clean \
model_attack=gradient_mimicry
```

Two coordinated malicious clients:

```bash
python -m eiffel \
num_clients=8 \
num_attackers=2 \
poisoning/profile=clean \
model_attack=colluding_sign_flip
```

FT-Transformer with Sign Flip:

```bash
python -m eiffel \
model=ft_transformer \
num_clients=9 \
num_attackers=1 \
poisoning/profile=clean \
model_attack=sign_flip
```

---

## TOML experiment workflow

The previous Flower attack lab used one TOML file per experiment. This fork keeps that workflow while Eiffel continues to use Hydra internally.

Runnable profiles are stored in:

```text
experiments/toml/
```

For example:

```powershell
.\run-toml.ps1 experiments\toml\smoke_sign_flip.toml
```

or:

```powershell
python -m eiffel.toml_runner experiments\toml\smoke_sign_flip.toml
```

To validate the profile and inspect the generated Eiffel/Hydra command without starting the simulation:

```powershell
python -m eiffel.toml_runner experiments\toml\smoke_sign_flip.toml --dry-run
```

The compatibility layer supports the current attack set, temporal schedules, model choice, Dirichlet non-IID partitioning, HDF5 storage, and NF-V2 dataset aliases. The old synthetic-stress profiles have been adapted to Eiffel-compatible NF-V2 experiments rather than silently pretending the old synthetic results are reproducible on a different runtime.

See [docs/TOML_EXPERIMENTS.md](docs/TOML_EXPERIMENTS.md) for the complete mapping and supported fields.

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
