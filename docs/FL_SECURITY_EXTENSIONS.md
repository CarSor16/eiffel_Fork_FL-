# FL security extensions

This branch keeps Eiffel's Hydra/Flower/TensorFlow architecture and adds model-poisoning,
round-level auditing, compact HDF5 persistence, and extra NIDS models.

## What is saved

Each run writes one `round_state.h5` file (when `storage.enabled=true`).

- `/global/round_XXXX/weights/layer_YYY`: global model weights after each round.
  Round 0 is the initial global model.
- `/clients/round_XXXX/<cid>/submitted_update/layer_YYY`: the exact update submitted
  to aggregation.
- `pre_attack_update`: saved only for malicious clients in rounds where a
  model-poisoning transformation changes the update.
- `audit`: L1/L2/Linf, cosine-to-mean, distance-to-mean/median and sign agreement.
- `inference`: the client's first deterministic inference on up to
  `storage.probe_size` test samples, persisted as float16.
- `/probe/labels`: labels for that probe slice.

Weights/updates stay float32 for later update-space analysis. Inference output is float16
to reduce storage. HDF5 gzip compression is applied inside the single file.

## Models

Hydra model choices:

```bash
model=popoola
model=p4p_mlp
model=cnn1d
model=ft_transformer
```

The default remains the original Popoola-style MLP, so original Eiffel experiments can
still be reproduced. The additional models are TensorFlow/Keras models to avoid adding a
second deep-learning runtime.

## Model-poisoning attacks

Available choices:

```bash
model_attack=none
model_attack=sign_flip
model_attack=scaling
model_attack=gaussian_noise
model_attack=lie
model_attack=gradient_mimicry
model_attack=colluding_sign_flip
```

Model attacks operate on the model delta immediately before FedAvg aggregation. LIE,
gradient mimicry and colluding sign flip use same-round client-update references at the
strategy boundary.

For a pure model-poisoning experiment, malicious clients still need to exist in Eiffel's
pool, but their dataset should remain clean:

```bash
num_clients=9 num_attackers=1 poisoning/profile=clean model_attack=sign_flip
```

For original Eiffel label-flipping experiments, keep `model_attack=none` and use the
existing poisoning configuration.

## Temporal schedules

Every model-attack YAML has a `schedule` section. It supports:

- `continuous`
- `late`
- `window`
- `on_off`
- `gradual`

Example command-line overrides:

```bash
model_attack=sign_flip \
model_attack.schedule.type=late \
model_attack.schedule.start_round=6
```

or

```bash
model_attack=gradient_mimicry \
model_attack.schedule.type=on_off \
model_attack.schedule.period=4 \
model_attack.schedule.active_rounds=2
```

## Storage controls

Defaults are in `eiffel/conf/storage/default.yaml`.

Examples:

```bash
storage.enabled=true
storage.probe_size=256
storage.capture_inference=true
storage.compression=gzip
storage.compression_level=4
```

The HDF5 file is flushed at the end of every round, so completed rounds remain readable
if a later round crashes.

## Important scope note

The new strategy is an experimental extension of Eiffel. It preserves Flower FedAvg for
aggregation, but inserts a model-attack/audit/storage boundary immediately before
aggregation. This makes the submitted update explicit and reproducible without changing
the original data-poisoning implementation.
