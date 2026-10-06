"""Defense and mitigation primitives for Eiffel FL-security experiments.

Defenses are deliberately separate from aggregation. Update-space defenses transform
submitted client deltas before aggregation. Probe distillation is a post-aggregation
experimental mitigation step that uses the already captured common probe predictions as
an ensemble teacher for the global student model.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

Update = Sequence[np.ndarray]
Updates = Sequence[Update]

DEFENSE_ALIASES = {
    "none": "none",
    "off": "none",
    "norm_clipping": "norm_clipping",
    "norm-clipping": "norm_clipping",
    "clip": "norm_clipping",
    "clipping": "norm_clipping",
    "probe_distillation": "probe_distillation",
    "probe-distillation": "probe_distillation",
    "distillation": "probe_distillation",
    "knowledge_distillation": "probe_distillation",
}


def canonical_defense_name(name: str) -> str:
    canonical = DEFENSE_ALIASES.get(str(name).strip().lower())
    if canonical is None:
        supported = ", ".join(sorted(set(DEFENSE_ALIASES.values())))
        raise ValueError(f"Unsupported defense {name!r}. Supported: {supported}.")
    return canonical


def _copy_updates(updates: Updates) -> list[list[np.ndarray]]:
    return [
        [np.asarray(layer, dtype=np.float32).copy() for layer in update]
        for update in updates
    ]


def update_l2_norm(update: Update) -> float:
    squared = sum(
        float(np.sum(np.square(np.asarray(layer, dtype=np.float64))))
        for layer in update
    )
    return float(np.sqrt(squared))


def norm_clip_updates(
    updates: Updates,
    *,
    max_norm: float,
) -> list[list[np.ndarray]]:
    """Clip every submitted update to a global L2 norm bound."""
    bound = float(max_norm)
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("defense.max_norm must be finite and > 0.")
    clipped = _copy_updates(updates)
    for update in clipped:
        norm = update_l2_norm(update)
        if not np.isfinite(norm):
            raise ValueError("Cannot clip an update with non-finite L2 norm.")
        if norm > bound and norm > 0:
            scale = np.float32(bound / norm)
            for index, layer in enumerate(update):
                update[index] = np.asarray(layer * scale, dtype=np.float32)
    return clipped


def apply_update_defense(
    updates: Updates,
    config: Mapping[str, Any] | None,
) -> list[list[np.ndarray]]:
    """Apply a pre-aggregation defense to submitted updates."""
    cfg = dict(config or {})
    name = canonical_defense_name(str(cfg.get("name", "none")))
    if name in {"none", "probe_distillation"}:
        return _copy_updates(updates)
    if name == "norm_clipping":
        return norm_clip_updates(
            updates,
            max_norm=float(cfg.get("max_norm", 10.0)),
        )
    raise AssertionError(name)


def _common_probe(
    probe_features: Sequence[np.ndarray | None],
) -> np.ndarray:
    available = [np.asarray(value, dtype=np.float32) for value in probe_features if value is not None]
    if not available:
        raise ValueError(
            "probe_distillation requires captured probe features from clients."
        )
    reference = available[0]
    for candidate in available[1:]:
        if candidate.shape != reference.shape or not np.array_equal(candidate, reference):
            raise ValueError(
                "probe_distillation requires the same deterministic probe on all "
                "participating clients."
            )
    return reference


def _teacher_logits(
    logits: Sequence[np.ndarray | None],
    probabilities: Sequence[np.ndarray | None],
) -> np.ndarray:
    available_logits = [
        np.asarray(value, dtype=np.float64) for value in logits if value is not None
    ]
    if available_logits:
        shapes = {value.shape for value in available_logits}
        if len(shapes) != 1:
            raise ValueError("Client teacher logits have inconsistent shapes.")
        return np.median(np.stack(available_logits, axis=0), axis=0)

    available_probs = [
        np.asarray(value, dtype=np.float64)
        for value in probabilities
        if value is not None
    ]
    if not available_probs:
        raise ValueError(
            "probe_distillation requires captured client logits or probabilities."
        )
    shapes = {value.shape for value in available_probs}
    if len(shapes) != 1:
        raise ValueError("Client teacher probabilities have inconsistent shapes.")
    probs = np.median(np.stack(available_probs, axis=0), axis=0)
    eps = 1e-6
    probs = np.clip(probs, eps, 1.0 - eps)
    if probs.shape[-1] == 1:
        return np.log(probs / (1.0 - probs))
    return np.log(probs)


def _soft_targets(logits: np.ndarray, temperature: float) -> np.ndarray:
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("defense.temperature must be finite and > 0.")
    scaled = np.asarray(logits, dtype=np.float64) / t
    if scaled.shape[-1] == 1:
        scaled = np.clip(scaled, -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(np.clip(scaled, -80.0, 80.0))
    return (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32)


def distill_global_weights(
    global_weights: Update,
    *,
    model_fn: Callable[[int], Any],
    probe_features: Sequence[np.ndarray | None],
    probabilities: Sequence[np.ndarray | None],
    logits: Sequence[np.ndarray | None],
    config: Mapping[str, Any],
    seed: int,
) -> list[np.ndarray]:
    """Refine aggregated weights using a robust teacher ensemble on a common probe.

    This is an experimental server-side probe-distillation defense, not a replacement
    for the aggregation algorithm. The teacher is the coordinate-wise median of client
    logits (or probabilities converted to logit space when logits are unavailable).
    """
    import tensorflow as tf

    cfg = dict(config)
    features = _common_probe(probe_features)
    teacher_logits = _teacher_logits(logits, probabilities)
    if len(features) != len(teacher_logits):
        raise ValueError(
            "Probe feature and teacher prediction lengths do not match."
        )
    targets = _soft_targets(
        teacher_logits,
        float(cfg.get("temperature", 2.0)),
    )
    if features.ndim != 2:
        raise ValueError("probe_distillation expects a 2D tabular probe feature matrix.")

    epochs = int(cfg.get("epochs", 1))
    if epochs < 1:
        raise ValueError("defense.epochs must be >= 1.")
    learning_rate = float(cfg.get("learning_rate", 1e-4))
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("defense.learning_rate must be finite and > 0.")
    alpha = float(cfg.get("alpha", 1.0))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("defense.alpha must be in [0, 1].")

    tf.random.set_seed(int(seed))
    model = model_fn(int(features.shape[1]))
    model.set_weights([np.asarray(value) for value in global_weights])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    x = tf.convert_to_tensor(features, dtype=tf.float32)
    y = tf.convert_to_tensor(targets, dtype=tf.float32)
    multiclass = int(targets.shape[-1]) > 1
    temperature = float(cfg.get("temperature", 2.0))

    for _ in range(epochs):
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            prediction = tf.clip_by_value(
                tf.cast(prediction, tf.float32),
                1e-6,
                1.0 - 1e-6,
            )
            if multiclass:
                loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(y, prediction)
                )
            else:
                loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(y, prediction)
                )
            loss = loss * (temperature ** 2)
        gradients = tape.gradient(loss, model.trainable_variables)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, model.trainable_variables)
            if gradient is not None
        ]
        optimizer.apply_gradients(pairs)

    distilled = model.get_weights()
    return [
        np.asarray(
            (1.0 - alpha) * np.asarray(before, dtype=np.float64)
            + alpha * np.asarray(after, dtype=np.float64),
            dtype=np.float32,
        )
        for before, after in zip(global_weights, distilled)
    ]
