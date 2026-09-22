# TOML experiment profiles

This fork supports the TOML experiment workflow used by the previous Flower attack lab.

Eiffel still uses Hydra internally. The TOML runner is a compatibility layer: it reads
one experiment profile, validates the fields, translates them to Hydra overrides, and
starts the normal Eiffel entrypoint.

## Run

PowerShell:

```powershell
.\run-toml.ps1 experiments\toml\smoke_clean.toml
```

or directly:

```powershell
python -m eiffel.toml_runner experiments\toml\smoke_clean.toml
```

Validate/inspect the generated Eiffel command without starting Flower:

```powershell
python -m eiffel.toml_runner experiments\toml\smoke_sign_flip.toml --dry-run
```

Extra Hydra overrides can be appended after the TOML path:

```powershell
python -m eiffel.toml_runner experiments\toml\smoke_sign_flip.toml model=cnn1d
```

## TOML sections

The compatibility layer currently understands:

- `[experiment]`: seed, total number of clients, rounds
- `[dataset]`: dataset alias or explicit Hydra group
- `[partition]`: iid, dumb, niid_class, dirichlet
- `[model]`: popoola/mlp, p4p_mlp, cnn1d, ft_transformer
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

## Dataset aliases

Current aliases:

```text
cicids / cse-cic-ids2018 -> nfv2/sampled/cicids
nb15 / unsw-nb15         -> nfv2/sampled/nb15
toniot / ton-iot         -> nfv2/sampled/toniot
botiot                    -> nfv2/sampled/botiot
```

A profile can bypass aliases with:

```toml
[dataset]
hydra_group = "nfv2/sampled/cicids"
```

The corresponding dataset file must exist under Eiffel's expected `data/` path.

## Why the old TOMLs were adapted

The previous lab used a synthetic multiclass dataset and its own partition/model/attack
runtime. Eiffel is a binary NIDS framework with its own dataset loaders and Hydra
configuration.

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
```

`model_scaling` is translated to the internal `scaling` attack name.

Pure model-poisoning profiles automatically use Eiffel's clean poisoning profile so
malicious clients train on unmodified local labels before their updates are transformed.

## Current limitation

The TOML compatibility layer currently supports `aggregation.name = "fedavg"`.

The old `noniid_sign_flip_trimmed.toml` profile is intentionally not copied as a
runnable profile yet because mapping it silently to FedAvg would change the scientific
meaning of the experiment. A future instrumented robust-aggregation strategy should be
added before restoring that profile.
