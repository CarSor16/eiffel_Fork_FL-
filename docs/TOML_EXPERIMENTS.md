# TOML experiment profiles

This fork supports the TOML experiment workflow used by the previous Flower attack lab.

TOML is the single experiment configuration backend. The runner reads one profile,
validates and normalizes it, constructs the corresponding Python objects directly,
and starts the normal Flower-based Eiffel experiment.

## Run

PowerShell:

```powershell
.\run-toml.ps1 experiments\toml\smoke_clean.toml
```

or directly:

```powershell
python -m eiffel.direct_runner experiments\toml\smoke_clean.toml
```

Validate/inspect the resolved experiment configuration without starting Flower:

```powershell
python -m eiffel.direct_runner experiments\toml\smoke_sign_flip.toml --dry-run
```

Temporary overrides can be appended using `dotted.path=value`:

```powershell
python -m eiffel.direct_runner experiments\toml\smoke_sign_flip.toml model=cnn1d
```

### Dataset-portable profiles

For model-poisoning experiments you do not need a separate TOML copy for every
dataset. The direct runtime has structural presets for every registered dataset and
can infer, when omitted:

- supervised task (`binary`, `family_aware`, or `multiclass`);
- model output class count;
- the required partitioner (`iid`, `dirichlet`, or `preassigned`);
- a compatible default model.

Use the committed portable profile and change only the dataset and attack when needed:

```powershell
.\run.cmd portable_model_attack dataset=cesnet
.\run.cmd portable_model_attack dataset=ciciot_family attack=gaussian_noise
.\run.cmd portable_model_attack dataset=mirage_app3 attack=min_max
```

The same works with the Python entrypoint:

```powershell
python -m eiffel.direct_runner experiments\toml\portable_model_attack.toml --set dataset=cesnet --dry-run
```

Explicit TOML fields always take precedence over inferred dataset defaults. This is
useful for controlled ablations, while the portable form avoids repeating dataset
boilerplate in normal experiments. Dataset-semantic attacks can still require a
dataset-specific target (for example `targeted_family_poisoning` needs a valid
`attack.targeted.target_family`, and multiclass label flipping needs an explicit
source/destination mapping).

## TOML sections

The direct runtime currently understands:

- `[experiment]`: seed, total number of clients, rounds
- `[dataset]`: registered dataset alias and dataset-specific parameters
- `[partition]`: iid, dumb, niid_class, dirichlet, preassigned
- `[model]`: popoola/mlp, p4p_mlp, cnn1d, ft_transformer, stress_mlp
- `[training]`: local_epochs, learning_rate, batch_size
- `[attack]`: mechanism, malicious_fraction, malicious_client_ids, strength and
  attack-specific parameters
- `[attack.schedule]`: continuous, late, window, on_off, gradual
- `[aggregation]`: currently fedavg
- `[storage]`: HDF5 storage controls

The TOML field `experiment.num_clients` means **total federated clients**. The runner
converts it to Eiffel's separate benign/malicious counts.

For example:

```toml
[experiment]
num_clients = 10

[attack]
malicious_fraction = 0.2
```

becomes 8 benign clients and 2 malicious clients.



## Live terminal monitor and round-boundary controls

Every run can emit a lightweight `events.jsonl` stream. Runtime mutation is disabled
by default and can be enabled explicitly:

```powershell
.\run.cmd portable_model_attack control.enabled=true
```

While the run is active, a second terminal can show the live state:

```powershell
python -m eiffel.tui watch <run-directory>
```

or open the interactive terminal controller:

```powershell
python -m eiffel.tui interactive <run-directory>
```

Runtime changes are published atomically and applied only at the boundary before the
requested FL round:

```powershell
python -m eiffel.tui set <run-directory> --round 8 aggregation.name=median
python -m eiffel.tui set <run-directory> --round 12 attack.strength=2.0
python -m eiffel.tui set <run-directory> --round 15 defense.name=norm_clipping defense.max_norm=5.0
```

The interactive mode accepts the same idea:

```text
set 8 aggregation.name=median
set 12 attack.strength=2.0
set 15 defense.name=norm_clipping defense.max_norm=5.0
```

Only server-side model-attack parameters, aggregation, and defenses are mutable during
an active run. Dataset, model architecture, client population, partitioning and
data-poisoning assignment are intentionally static because changing them mid-round would
invalidate experiment semantics.

The control protocol is file-based:

```text
run-directory/
  resolved_profile.json
  round_state.h5
  events.jsonl
  control.json
```

This separates the experiment engine from the frontend. A future Textual or web GUI can
reuse the same event/control protocol without changing Flower or the experiment runtime.

## Pluggable aggregation

Aggregation is selected independently from the attack, dataset and model:

```toml
[aggregation]
name = "fedavg"
```

Supported server-side backends are:

- `fedavg`: sample-count weighted Federated Averaging;
- `median`: coordinate-wise median;
- `trimmed_mean`: coordinate-wise symmetric trimmed mean;
- `krum`: single-update Krum selection;
- `multi_krum`: average of the lowest-scoring Krum candidates.

Examples:

```powershell
.\run.cmd portable_model_attack dataset=cesnet aggregation=median
.\run.cmd portable_model_attack dataset=cesnet aggregation=trimmed_mean aggregation.trim_ratio=0.2
.\run.cmd portable_model_attack dataset=cesnet aggregation=krum
.\run.cmd portable_model_attack dataset=cesnet aggregation=multi_krum aggregation.num_selected=4
```

For Krum and Multi-Krum, `aggregation.num_byzantine` defaults to the configured
number of malicious clients. It can be overridden explicitly. The runtime rejects
configurations that violate Krum's minimum client-count requirement. The selected
aggregation backend is stored in each HDF5 round metadata entry for reproducibility.

## Synthetic 50k benchmark

The previous attack lab's synthetic stress benchmark is available again through:

```toml
[dataset]
name = "synthetic_stress"
samples_per_client = 5000
central_test_size = 12000
num_features = 32
num_classes = 6
```

With 10 clients this creates exactly **50,000 federated training samples** (5,000 per
client) plus a separate **12,000-sample common test set**.

The generator preserves the main characteristics of the previous stress environment:
imbalanced traffic families, a rare attack family, overlapping and multimodal classes,
informative/redundant/noise features, hard examples, client-specific covariate shift,
small training-label noise, outliers, and a shifted common test distribution.

The synthetic benchmark supports both binary and true multiclass training. In the
default binary mode, the six traffic families are retained as metadata while the
supervised target is mapped to:

```text
Benign -> 0
any attack family -> 1
```

This preserves per-family recall/miss-rate analysis for the Eiffel-compatible baseline.
With `dataset.task = "multiclass"`, the same generator exposes the six families as the
supervised classes and compatible models use a softmax output.

For this dataset the generator itself creates the exact client shards. A
`PreassignedPartitioner` therefore preserves the 5,000 samples/client instead of
repartitioning them after generation. The TOML can still use:

```toml
[partition]
type = "dirichlet"
dirichlet_alpha = 0.5
```

because the TOML runner passes that alpha to the synthetic generator before selecting the
preassigned Eiffel partitioner.

Ready-to-run profiles include:

```text
synthetic_50k_quick_clean.toml
synthetic_50k_quick_sign_flip.toml
synthetic_50k_clean.toml
synthetic_50k_label_flip.toml
synthetic_50k_sign_flip.toml
synthetic_50k_model_scaling.toml
synthetic_50k_gaussian_noise.toml
synthetic_50k_lie.toml
synthetic_50k_gradient_mimicry.toml
synthetic_50k_colluding_sign_flip.toml
```

The two `quick` profiles keep the full 50k training samples but use only five FL rounds
for environment validation.

## Post-hoc inference storage

The `[storage]` table can make a completed run reusable for later analysis without
retraining:

```toml
[storage]
enabled = true
capture_inference = true
capture_logits = true
capture_probe_features = true
capture_global_inference = true
probe_size = 256
```

`capture_inference` is the master switch. With the defaults above, `round_state.h5`
stores the deterministic probe features, labels and family metadata, plus local-model
probabilities/logits for each round. During Flower distributed evaluation it also
stores probabilities/logits from the aggregated global model under `global_inference`.

The storage path is dataset-agnostic; the current logit extractor only requires a
supported Dense sigmoid/softmax classifier head.
## Dataset aliases

Current aliases:

```text
cicids / cse-cic-ids2018 -> nfv2/sampled/cicids
cicids_full                -> nfv2/full/cicids
cicids_datacenter          -> nfv2/datacenter/cicids
nb15 / unsw-nb15           -> nfv2/sampled/nb15
nb15_full                   -> nfv2/full/nb15
nb15_datacenter             -> nfv2/datacenter/nb15
toniot / ton-iot         -> nfv2/sampled/toniot
botiot                    -> nfv2/sampled/botiot
```

Dataset aliases are resolved by the direct Python dataset registry. To add a new dataset,
add its loader/registry entry to the runtime and expose its parameters through TOML.
The corresponding dataset files remain outside Git and must exist at the configured or
registered data path.

## Why the old TOMLs were adapted

The previous lab used a synthetic multiclass dataset and its own partition/model/attack
runtime. Eiffel has its own dataset loaders and now uses a direct TOML/Python
configuration layer.

The profiles in `experiments/toml/` therefore preserve the old experiment style and
attack parameters, but use Eiffel-compatible datasets/models.

In particular, the old `synthetic_stress` profiles were adapted to sampled NF-V2
CSE-CIC-IDS2018 by default. This is not intended to reproduce the numerical results of
the old synthetic benchmark; it reproduces the experimental *configuration pattern* on
the Eiffel runtime.

## Dirichlet non-IID

The fork adds `DirichletPartitioner` because the previous TOMLs used:

```toml
[partition]
type = "dirichlet"
dirichlet_alpha = 0.5
```

Lower alpha values create stronger class skew. The current implementation partitions on
the NF-V2 `Attack` metadata column.

## Model attacks

Supported TOML mechanisms:

```text
none
label_flip
sign_flip
model_scaling
gaussian_noise
lie
gradient_mimicry
colluding_sign_flip
min_max
min_sum
adaptive_stealth
heterogeneity_aware_mimicry
targeted_family_poisoning
```

`model_scaling` is translated to the internal `scaling` attack name.

Pure model-poisoning profiles automatically use Eiffel's clean poisoning profile so
malicious clients train on unmodified local labels before their updates are transformed.

For `label_flip`, the round-state file distinguishes two fractions. The
`data_poison_fraction` client attribute is the configured fraction of the selected
poisoning target, while `data_poison_effective_fraction` is the observed fraction of
the whole local training shard currently marked poisoned when the dataset exposes
Eiffel's `Poisoned` metadata. This distinction is important under non-IID targeted
attacks: a poisoning schedule may be enabled while a specific malicious client owns no
samples from the target family. Such a client is not counted as attack-active for that
round.

## Current limitation

The direct TOML runtime currently supports `aggregation.name = "fedavg"`.

The old `noniid_sign_flip_trimmed.toml` profile is intentionally not copied as a
runnable profile yet because mapping it silently to FedAvg would change the scientific
meaning of the experiment. A future instrumented robust-aggregation strategy should be
added before restoring that profile.


## Task modes

The dataset task is explicit in new profiles.

- binary: original Eiffel-compatible Benign-vs-Attack training.
- family_aware: binary training with metrics retained separately for each attack family.
- multiclass: experimental true family classification, currently supported by the
  synthetic_stress dataset only.

For multiclass, the synthetic labels are the family IDs 0..K-1 and supported neural
models use a K-way softmax output with sparse categorical cross-entropy. Procedural
model-update attacks remain unchanged because they operate on parameter deltas rather
than class labels.

Multiclass label_flip is intentionally rejected by the TOML resolver. Eiffel's
existing NFV2 poisoning method is Boolean label inversion; applying it to labels 0..K-1
would not define a valid class-to-class poisoning objective. A later implementation
should introduce explicit source_class and destination_class semantics before enabling
that experiment.

Ready-to-run multiclass profiles include clean, sign flip, scaling, Gaussian noise, LIE,
gradient mimicry and colluding sign flip, plus quick clean/sign-flip smoke profiles.

    .\run.cmd synthetic_50k_quick_multiclass_clean
    .\run.cmd synthetic_50k_quick_multiclass_sign_flip
    .\run.cmd -Suite multiclass -DryRun
    .\run.cmd -Suite multiclass

## Security validation and analysis commands

The unified launcher can describe attack mechanisms and TOML controls:

    .\run.cmd -Attacks

Focused regression tests:

    .\setup.cmd -Dev
    .\run.cmd -Tests

Validate one completed HDF5 run:

    .\run.cmd -ValidateHdf5 "outputs\YYYY-MM-DD\HH-MM-SS\round_state.h5"

Compare all stored runs under outputs:

    .\run.cmd -Analyze

The metric-analysis module reads client metrics from round_state.h5 and creates
round-by-round line plots, final grouped attack bar plots, clean-baseline deltas,
per-family recall comparisons, CSV exports and multi-seed mean/std aggregation.


## Data-center profiles

The thesis-scale real-data profiles use the full NF-V2 representations of
CSE-CIC-IDS2018 and UNSW-NB15. They are named `dc_cicids_*` and `dc_nb15_*`.
The data-center dataset loaders read file paths from `EIFFEL_CICIDS_PATH` and
`EIFFEL_NB15_PATH`, so large datasets can remain on shared/scratch storage instead
of inside the Git repository.

See `docs/DATACENTER_RUNBOOK.md` for Linux setup, SLURM submission, suite execution
and multi-seed campaigns.
