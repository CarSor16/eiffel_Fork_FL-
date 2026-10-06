"""Pluggable server-side aggregation backends for Eiffel FL-security experiments.

All functions operate on client model updates (deltas) after any configured attack
has been applied. This keeps attack generation, aggregation and instrumentation
independent and makes aggregation comparisons reproducible.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

Update = Sequence[np.ndarray]
Updates = Sequence[Update]

AGGREGATION_ALIASES = {
    "fedavg": "fedavg",
    "avg": "fedavg",
    "mean": "fedavg",
    "median": "median",
    "coordinate_median": "median",
    "coordinate-median": "median",
    "trimmed_mean": "trimmed_mean",
    "trimmed-mean": "trimmed_mean",
    "trimmedmean": "trimmed_mean",
    "krum": "krum",
    "multi_krum": "multi_krum",
    "multi-krum": "multi_krum",
    "multikrum": "multi_krum",
}


def canonical_aggregation_name(name: str) -> str:
    canonical = AGGREGATION_ALIASES.get(str(name).strip().lower())
    if canonical is None:
        supported = ", ".join(sorted(set(AGGREGATION_ALIASES.values())))
        raise ValueError(f"Unsupported aggregation {name!r}. Supported: {supported}.")
    return canonical


def _validate_layout(updates: Updates) -> None:
    if not updates:
        raise ValueError("Aggregation requires at least one client update.")
    reference = [np.asarray(layer).shape for layer in updates[0]]
    if not reference:
        raise ValueError("Client updates must contain at least one tensor.")
    for client_index, update in enumerate(updates):
        if len(update) != len(reference):
            raise ValueError(
                f"Client update {client_index} has {len(update)} layers; "
                f"expected {len(reference)}."
            )
        for layer_index, (layer, expected_shape) in enumerate(zip(update, reference)):
            arr = np.asarray(layer)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Client {client_index} layer {layer_index} shape {arr.shape} "
                    f"does not match {expected_shape}."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Client {client_index} layer {layer_index} contains non-finite values."
                )


def _float32(update: Update) -> list[np.ndarray]:
    return [np.asarray(layer, dtype=np.float32) for layer in update]


def _stack_layers(updates: Updates) -> list[np.ndarray]:
    _validate_layout(updates)
    return [
        np.stack(
            [np.asarray(update[layer_index], dtype=np.float64) for update in updates],
            axis=0,
        )
        for layer_index in range(len(updates[0]))
    ]


def _flatten(update: Update) -> np.ndarray:
    return np.concatenate(
        [np.asarray(layer, dtype=np.float64).reshape(-1) for layer in update],
        axis=0,
    )


def fedavg(
    updates: Updates,
    num_examples: Sequence[int | float] | None = None,
) -> list[np.ndarray]:
    """Flower-compatible sample-count weighted mean of submitted updates."""
    stacks = _stack_layers(updates)
    if num_examples is None:
        weights = np.ones(len(updates), dtype=np.float64)
    else:
        if len(num_examples) != len(updates):
            raise ValueError("num_examples length must match client updates.")
        weights = np.asarray(num_examples, dtype=np.float64)
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("num_examples must be finite and non-negative.")
        if float(weights.sum()) <= 0:
            raise ValueError("At least one client must have a positive sample count.")
    weights = weights / weights.sum()
    return [
        np.asarray(np.tensordot(weights, stack, axes=(0, 0)), dtype=np.float32)
        for stack in stacks
    ]


def coordinate_median(updates: Updates) -> list[np.ndarray]:
    """Coordinate-wise median, independent of client sample counts."""
    return [
        np.asarray(np.median(stack, axis=0), dtype=np.float32)
        for stack in _stack_layers(updates)
    ]


def trimmed_mean(updates: Updates, *, trim_ratio: float = 0.2) -> list[np.ndarray]:
    """Coordinate-wise symmetric trimmed mean."""
    ratio = float(trim_ratio)
    if not 0.0 <= ratio < 0.5:
        raise ValueError("trim_ratio must be in [0, 0.5).")
    stacks = _stack_layers(updates)
    n_clients = len(updates)
    trim = int(np.floor(ratio * n_clients))
    if 2 * trim >= n_clients:
        raise ValueError("trim_ratio removes all available client values.")
    if trim == 0:
        return [
            np.asarray(np.mean(stack, axis=0), dtype=np.float32)
            for stack in stacks
        ]
    return [
        np.asarray(
            np.mean(np.sort(stack, axis=0)[trim:n_clients - trim], axis=0),
            dtype=np.float32,
        )
        for stack in stacks
    ]


def krum_scores(updates: Updates, *, num_byzantine: int) -> np.ndarray:
    """Return canonical Krum scores from squared Euclidean update distance."""
    _validate_layout(updates)
    n_clients = len(updates)
    f = int(num_byzantine)
    if f < 0:
        raise ValueError("num_byzantine must be >= 0.")
    if n_clients < 2 * f + 3:
        raise ValueError(
            "Krum requires n_clients >= 2 * num_byzantine + 3; "
            f"got n={n_clients}, f={f}."
        )
    vectors = np.stack([_flatten(update) for update in updates], axis=0)
    diff = vectors[:, None, :] - vectors[None, :, :]
    distances = np.einsum("ijk,ijk->ij", diff, diff)
    neighbour_count = n_clients - f - 2
    scores = np.empty(n_clients, dtype=np.float64)
    for index in range(n_clients):
        others = np.delete(distances[index], index)
        scores[index] = np.partition(
            others, neighbour_count - 1
        )[:neighbour_count].sum()
    return scores


def krum(updates: Updates, *, num_byzantine: int) -> list[np.ndarray]:
    """Select the single client update with the lowest Krum score."""
    scores = krum_scores(updates, num_byzantine=num_byzantine)
    selected = int(np.argmin(scores))
    return _float32(updates[selected])


def multi_krum(
    updates: Updates,
    *,
    num_byzantine: int,
    num_selected: int | None = None,
) -> list[np.ndarray]:
    """Average the client updates with the lowest Krum scores."""
    scores = krum_scores(updates, num_byzantine=num_byzantine)
    n_clients = len(updates)
    f = int(num_byzantine)
    max_selected = n_clients - f - 2
    selected_count = max_selected if num_selected is None else int(num_selected)
    if not 1 <= selected_count <= max_selected:
        raise ValueError(
            f"multi_krum num_selected must be in [1, {max_selected}]; "
            f"got {selected_count}."
        )
    indices = np.argsort(scores, kind="stable")[:selected_count]
    selected_updates = [updates[int(index)] for index in indices]
    return [
        np.asarray(np.mean(stack, axis=0), dtype=np.float32)
        for stack in _stack_layers(selected_updates)
    ]


def aggregate_updates(
    updates: Updates,
    *,
    num_examples: Sequence[int | float] | None,
    config: Mapping[str, Any] | None,
) -> list[np.ndarray]:
    """Aggregate submitted updates using the configured backend."""
    cfg = dict(config or {})
    name = canonical_aggregation_name(str(cfg.get("name", "fedavg")))
    if name == "fedavg":
        return fedavg(updates, num_examples)
    if name == "median":
        return coordinate_median(updates)
    if name == "trimmed_mean":
        return trimmed_mean(updates, trim_ratio=float(cfg.get("trim_ratio", 0.2)))
    if name == "krum":
        return krum(updates, num_byzantine=int(cfg.get("num_byzantine", 0)))
    if name == "multi_krum":
        raw_selected = cfg.get("num_selected")
        return multi_krum(
            updates,
            num_byzantine=int(cfg.get("num_byzantine", 0)),
            num_selected=None if raw_selected in (None, "") else int(raw_selected),
        )
    raise AssertionError(name)
