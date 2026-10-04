"""Procedural model-poisoning attacks applied to client updates.

All attacks operate on model deltas (local_weights - global_weights), not raw
network-flow features.  This keeps the attack layer independent from the dataset
schema and model architecture as long as all clients in a run share the same
parameter layout.

The targeted-family attack is the only mechanism that needs semantic metadata.  It
uses the generic probe metadata already emitted by Eiffel (labels, family names and
model inference) and never hard-codes a dataset-specific class or feature index.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

Update = list[np.ndarray]


def _cfg(config: Mapping[str, Any] | None, key: str, default: Any) -> Any:
    if config is None:
        return default
    return config.get(key, default)


def attack_strength_for_round(
    config: Mapping[str, Any] | None,
    server_round: int,
    total_rounds: int | None = None,
) -> float:
    """Return a [0, 1] schedule multiplier for the current round."""
    if not config or not bool(_cfg(config, "enabled", False)):
        return 0.0

    schedule = _cfg(config, "schedule", {}) or {}
    kind = str(schedule.get("type", "continuous")).lower()
    start = int(schedule.get("start_round", 1))
    end = int(schedule.get("end_round", total_rounds or 10**9))

    if server_round < start or server_round > end:
        return 0.0
    if kind == "continuous":
        return 1.0
    if kind == "late":
        late_start = int(
            schedule.get("start_round", max(1, (total_rounds or 1) // 2))
        )
        return 1.0 if server_round >= late_start else 0.0
    if kind == "window":
        return 1.0
    if kind == "on_off":
        period = max(1, int(schedule.get("period", 2)))
        active = max(1, int(schedule.get("active_rounds", 1)))
        return 1.0 if ((server_round - start) % period) < active else 0.0
    if kind == "gradual":
        ramp = max(
            1,
            int(
                schedule.get(
                    "ramp_rounds",
                    max(1, end - start + 1) if end < 10**9 else 5,
                )
            ),
        )
        progress = min(
            1.0,
            max(0.0, (server_round - start) / max(1, ramp - 1)),
        )
        initial = float(schedule.get("gradual_start_strength", 0.0))
        final = float(schedule.get("gradual_end_strength", 1.0))
        return float(initial + (final - initial) * progress)
    raise ValueError(f"Unsupported model-attack schedule: {kind}")


def _copy(update: Sequence[np.ndarray]) -> Update:
    return [np.asarray(layer, dtype=np.float32).copy() for layer in update]


def _mean_updates(updates: Sequence[Update]) -> Update:
    return [
        np.mean(np.stack([u[layer] for u in updates], axis=0), axis=0).astype(
            np.float32
        )
        for layer in range(len(updates[0]))
    ]


def _std_updates(updates: Sequence[Update]) -> Update:
    return [
        np.std(np.stack([u[layer] for u in updates], axis=0), axis=0).astype(
            np.float32
        )
        for layer in range(len(updates[0]))
    ]


def _flatten(update: Sequence[np.ndarray]) -> np.ndarray:
    if not update:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(
        [np.asarray(layer, dtype=np.float32).reshape(-1) for layer in update]
    )


def _unflatten(vector: np.ndarray, template: Sequence[np.ndarray]) -> Update:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    output: Update = []
    offset = 0
    for layer in template:
        layer = np.asarray(layer, dtype=np.float32)
        size = int(layer.size)
        output.append(vector[offset : offset + size].reshape(layer.shape).copy())
        offset += size
    if offset != vector.size:
        raise ValueError("Vector size does not match update parameter layout.")
    return output


def _matrix(updates: Sequence[Update]) -> np.ndarray:
    if not updates:
        return np.empty((0, 0), dtype=np.float32)
    vectors = [_flatten(update) for update in updates]
    width = vectors[0].size
    if any(vector.size != width for vector in vectors):
        raise ValueError("All client updates must share the same parameter layout.")
    return np.stack(vectors, axis=0).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _quantile(value: Any, name: str) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {numeric}.")
    return numeric


def _deviation_direction(
    benign_matrix: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Build a scale-aware deviation direction from benign updates."""
    mean = benign_matrix.mean(axis=0)
    std = benign_matrix.std(axis=0)
    mode = str(mode).lower()

    if mode in {"negative_mean", "mean"}:
        norm = float(np.linalg.norm(mean))
        scale = float(np.median(np.linalg.norm(benign_matrix, axis=1)))
        if norm <= 1e-12:
            direction = np.where(std > 0.0, std, 1.0)
            norm = float(np.linalg.norm(direction))
            return (direction / max(norm, 1e-12) * max(scale, 1.0)).astype(
                np.float32
            )
        return (mean / norm * max(scale, 1e-12)).astype(np.float32)

    if mode in {"sign", "sign_mean"}:
        direction = np.sign(mean) * np.where(std > 1e-12, std, 1.0)
        return direction.astype(np.float32)

    if mode in {"std", "standard_deviation"}:
        direction = np.where(std > 1e-12, std, 1.0)
        return direction.astype(np.float32)

    raise ValueError(
        "Unsupported deviation direction "
        f"'{mode}'. Use negative_mean, sign, or std."
    )


def _search_lambda(
    reference: np.ndarray,
    direction: np.ndarray,
    *,
    max_lambda: float,
    search_steps: int,
    feasible,
) -> np.ndarray:
    """Return the furthest feasible reference-lambda*direction candidate."""
    max_lambda = max(0.0, float(max_lambda))
    search_steps = max(1, int(search_steps))

    high_candidate = reference - max_lambda * direction
    if feasible(high_candidate):
        return high_candidate.astype(np.float32)

    low, high = 0.0, max_lambda
    for _ in range(search_steps):
        mid = (low + high) / 2.0
        candidate = reference - mid * direction
        if feasible(candidate):
            low = mid
        else:
            high = mid
    return (reference - low * direction).astype(np.float32)


def _min_max_crafted(
    benign_updates: Sequence[Update],
    config: Mapping[str, Any],
    multiplier: float,
) -> Update:
    if len(benign_updates) < 2:
        raise ValueError("min_max requires at least two benign client updates.")

    matrix = _matrix(benign_updates).astype(np.float64)
    reference = matrix.mean(axis=0)
    direction = _deviation_direction(
        matrix.astype(np.float32),
        str(_cfg(config, "direction", "negative_mean")),
    ).astype(np.float64)

    pairwise = matrix[:, None, :] - matrix[None, :, :]
    max_benign_distance = float(
        np.sqrt(np.square(pairwise).sum(axis=2)).max(initial=0.0)
    )
    margin = float(_cfg(config, "constraint_margin", 1.0))
    bound = max_benign_distance * max(0.0, margin)

    candidate = _search_lambda(
        reference,
        direction,
        max_lambda=float(_cfg(config, "max_lambda", 10.0)) * multiplier,
        search_steps=int(_cfg(config, "search_steps", 24)),
        feasible=lambda vec: float(
            np.linalg.norm(matrix - vec[None, :], axis=1).max(initial=0.0)
        )
        <= bound + 1e-9,
    )
    return _unflatten(candidate, benign_updates[0])


def _min_sum_crafted(
    benign_updates: Sequence[Update],
    config: Mapping[str, Any],
    multiplier: float,
) -> Update:
    if len(benign_updates) < 2:
        raise ValueError("min_sum requires at least two benign client updates.")

    matrix = _matrix(benign_updates).astype(np.float64)
    reference = matrix.mean(axis=0)
    direction = _deviation_direction(
        matrix.astype(np.float32),
        str(_cfg(config, "direction", "negative_mean")),
    ).astype(np.float64)

    pairwise = matrix[:, None, :] - matrix[None, :, :]
    benign_sums = np.square(pairwise).sum(axis=2).sum(axis=1)
    bound = float(benign_sums.max(initial=0.0))
    bound *= max(0.0, float(_cfg(config, "constraint_margin", 1.0)))

    candidate = _search_lambda(
        reference,
        direction,
        max_lambda=float(_cfg(config, "max_lambda", 10.0)) * multiplier,
        search_steps=int(_cfg(config, "search_steps", 24)),
        feasible=lambda vec: float(
            np.square(matrix - vec[None, :]).sum(axis=1).sum()
        )
        <= bound + 1e-9,
    )
    return _unflatten(candidate, benign_updates[0])


def _adaptive_stealth_crafted(
    benign_updates: Sequence[Update],
    config: Mapping[str, Any],
    multiplier: float,
) -> Update:
    if len(benign_updates) < 2:
        raise ValueError(
            "adaptive_stealth requires at least two benign client updates."
        )

    matrix = _matrix(benign_updates).astype(np.float64)
    reference = matrix.mean(axis=0)
    direction = _deviation_direction(
        matrix.astype(np.float32),
        str(_cfg(config, "direction", "sign")),
    ).astype(np.float64)

    l2_q = _quantile(_cfg(config, "l2_quantile", 0.95), "l2_quantile")
    distance_q = _quantile(
        _cfg(config, "distance_quantile", 0.95),
        "distance_quantile",
    )
    cosine_q = _quantile(
        _cfg(config, "min_cosine_quantile", 0.05),
        "min_cosine_quantile",
    )
    margin = max(0.0, float(_cfg(config, "stealth_margin", 1.0)))
    cosine_slack = max(0.0, float(_cfg(config, "cosine_slack", 0.02)))

    benign_l2 = np.linalg.norm(matrix, axis=1)
    benign_distance = np.linalg.norm(matrix - reference[None, :], axis=1)
    benign_cosine = np.asarray(
        [_cosine(vector, reference) for vector in matrix],
        dtype=np.float64,
    )

    # Keep the benign centroid itself feasible even for very tight quantiles.
    l2_bound = max(
        float(np.quantile(benign_l2, l2_q)) * margin,
        float(np.linalg.norm(reference)),
    )
    distance_bound = max(
        float(np.quantile(benign_distance, distance_q)) * margin,
        0.0,
    )
    cosine_bound = float(np.quantile(benign_cosine, cosine_q)) - cosine_slack

    def feasible(vec: np.ndarray) -> bool:
        return (
            float(np.linalg.norm(vec)) <= l2_bound + 1e-9
            and float(np.linalg.norm(vec - reference)) <= distance_bound + 1e-9
            and _cosine(vec, reference) >= cosine_bound - 1e-9
        )

    candidate = _search_lambda(
        reference,
        direction,
        max_lambda=float(_cfg(config, "max_strength", 8.0)) * multiplier,
        search_steps=int(_cfg(config, "search_steps", 24)),
        feasible=feasible,
    )
    return _unflatten(candidate, benign_updates[0])


def _nearest_benign_reference(
    raw_update: Update,
    benign_updates: Sequence[Update],
    *,
    neighbors: int,
    similarity: str,
) -> Update:
    matrix = _matrix(benign_updates)
    raw = _flatten(raw_update)
    neighbors = min(max(1, int(neighbors)), len(benign_updates))
    similarity = str(similarity).lower()

    if similarity == "cosine":
        scores = np.asarray([_cosine(raw, candidate) for candidate in matrix])
        indices = np.argsort(-scores)[:neighbors]
    elif similarity in {"l2", "euclidean"}:
        distances = np.linalg.norm(matrix - raw[None, :], axis=1)
        indices = np.argsort(distances)[:neighbors]
    else:
        raise ValueError(
            "heterogeneity_aware_mimicry similarity must be cosine or l2."
        )

    reference = matrix[indices].mean(axis=0)
    return _unflatten(reference, benign_updates[0])


def _target_true_confidence(
    inference: np.ndarray,
    labels: np.ndarray,
    families: Sequence[str],
    target_family: str,
) -> float | None:
    """Mean probability assigned to the true class for the requested family."""
    inference = np.asarray(inference)
    labels = np.asarray(labels).astype(int).reshape(-1)
    family_array = np.asarray([str(value) for value in families], dtype=object)

    rows = inference.shape[0] if inference.ndim > 0 else 0
    if rows != labels.size or family_array.size != labels.size:
        raise ValueError(
            "Targeted-family probe inference, labels and family metadata "
            "must have the same number of samples."
        )

    target_text = target_family.strip()
    if target_text.lower().startswith("class_id:"):
        try:
            target_id = int(target_text.split(":", 1)[1].strip())
        except ValueError as exc:
            raise ValueError(
                "target_family class_id selector must use class_id:<integer>."
            ) from exc
        mask = labels == target_id
    else:
        mask = np.asarray(
            [value.lower() == target_text.lower() for value in family_array],
            dtype=bool,
        )
    if not bool(mask.any()):
        return None

    if inference.ndim == 1 or (
        inference.ndim == 2 and inference.shape[1] == 1
    ):
        probability = inference.reshape(-1).astype(np.float64)
        selected_labels = labels[mask]
        selected_probability = probability[mask]
        if not np.all(np.isin(selected_labels, [0, 1])):
            raise ValueError(
                "Binary probe inference requires probe labels encoded as 0/1."
            )
        confidence = np.where(
            selected_labels == 1,
            selected_probability,
            1.0 - selected_probability,
        )
        return float(np.mean(confidence))

    if inference.ndim != 2:
        raise ValueError(
            "Targeted-family probe inference must be binary probabilities or "
            "a 2-D multiclass probability matrix."
        )

    selected_labels = labels[mask]
    if np.any(selected_labels < 0) or np.any(
        selected_labels >= inference.shape[1]
    ):
        raise ValueError(
            "Probe label is outside the multiclass inference dimension."
        )
    selected = inference[mask].astype(np.float64)
    confidence = selected[
        np.arange(selected_labels.size),
        selected_labels,
    ]
    return float(np.mean(confidence))


def _targeted_family_crafted(
    updates: Sequence[Update],
    malicious_idx: Sequence[int],
    benign_updates: Sequence[Update],
    config: Mapping[str, Any],
    multiplier: float,
    *,
    probe_inferences: Sequence[np.ndarray | None] | None,
    probe_labels: Sequence[np.ndarray | None] | None,
    probe_families: Sequence[Sequence[str] | None] | None,
) -> Update:
    if not benign_updates:
        raise ValueError(
            "targeted_family_poisoning requires at least one benign update."
        )

    target_family = str(_cfg(config, "target_family", "")).strip()
    if not target_family:
        raise ValueError(
            "targeted_family_poisoning requires model_attack.target_family."
        )
    if (
        probe_inferences is None
        or probe_labels is None
        or probe_families is None
    ):
        raise ValueError(
            "targeted_family_poisoning requires captured probe inference, "
            "labels and family metadata."
        )

    scored: list[tuple[float, int]] = []
    available: set[str] = set()
    for idx, families in enumerate(probe_families):
        if families:
            available.update(str(value) for value in families)

    for idx in malicious_idx:
        inference = probe_inferences[idx]
        labels = probe_labels[idx]
        families = probe_families[idx]
        if inference is None or labels is None or not families:
            continue
        score = _target_true_confidence(
            inference,
            labels,
            families,
            target_family,
        )
        if score is not None and np.isfinite(score):
            scored.append((float(score), idx))

    if not scored:
        names = ", ".join(sorted(available)) if available else "<none>"
        raise ValueError(
            "No malicious probe contains target family "
            f"'{target_family}'. Available probe families: {names}."
        )

    # Pick the malicious local model that already gives the target family the
    # lowest true-class confidence, then amplify its update direction relative
    # to the benign centroid.  This is architecture-independent because the
    # probe only selects the direction; poisoning still operates in update space.
    _, source_idx = min(scored, key=lambda item: item[0])
    benign_reference = _flatten(_mean_updates(benign_updates)).astype(np.float64)
    source = _flatten(updates[source_idx]).astype(np.float64)
    amplification = max(
        0.0,
        float(_cfg(config, "target_amplification", _cfg(config, "strength", 2.0))),
    )
    amplification *= multiplier
    amplified = benign_reference + amplification * (source - benign_reference)

    mimicry = float(
        np.clip(_cfg(config, "target_mimicry_lambda", 0.15), 0.0, 1.0)
    )
    crafted = mimicry * benign_reference + (1.0 - mimicry) * amplified
    return _unflatten(crafted.astype(np.float32), updates[source_idx])


def apply_round_attack(
    updates: Sequence[Update],
    malicious_mask: Sequence[bool],
    config: Mapping[str, Any] | None,
    *,
    server_round: int,
    total_rounds: int | None = None,
    seed: int = 0,
    probe_inferences: Sequence[np.ndarray | None] | None = None,
    probe_labels: Sequence[np.ndarray | None] | None = None,
    probe_families: Sequence[Sequence[str] | None] | None = None,
) -> tuple[list[Update], float]:
    """Apply the configured attack to one round of client updates.

    Model-poisoning mechanisms only depend on update tensors and therefore work
    across datasets and model architectures.  The targeted-family mechanism also
    consumes generic probe metadata; the target family itself is configured at
    runtime and is never hard-coded in the implementation.
    """
    result = [_copy(u) for u in updates]
    multiplier = attack_strength_for_round(
        config,
        server_round,
        total_rounds,
    )
    if multiplier <= 0.0 or not any(malicious_mask):
        return result, 0.0

    mechanism = str(_cfg(config, "mechanism", "none")).lower()
    strength = float(_cfg(config, "strength", 1.0)) * multiplier
    malicious_idx = [i for i, flag in enumerate(malicious_mask) if flag]
    benign_idx = [i for i, flag in enumerate(malicious_mask) if not flag]
    benign_updates = [updates[i] for i in benign_idx]
    rng = np.random.default_rng(seed + 1009 * int(server_round))

    if mechanism in {"none", "label_flip"}:
        return result, multiplier

    if mechanism == "sign_flip":
        for i in malicious_idx:
            result[i] = [
                -strength * np.asarray(x, dtype=np.float32)
                for x in updates[i]
            ]

    elif mechanism == "scaling":
        factor = float(_cfg(config, "scale_factor", 10.0)) * multiplier
        for i in malicious_idx:
            result[i] = [
                factor * np.asarray(x, dtype=np.float32)
                for x in updates[i]
            ]

    elif mechanism == "gaussian_noise":
        noise_std = float(_cfg(config, "noise_std", 1.0)) * multiplier
        for i in malicious_idx:
            poisoned = []
            for layer in updates[i]:
                layer = np.asarray(layer, dtype=np.float32)
                base = float(np.std(layer))
                sigma = noise_std * (base if base > 1e-12 else 1.0)
                poisoned.append(
                    layer
                    + rng.normal(0.0, sigma, size=layer.shape).astype(np.float32)
                )
            result[i] = poisoned

    elif mechanism == "lie":
        if not benign_updates:
            return result, multiplier
        mean = _mean_updates(benign_updates)
        std = _std_updates(benign_updates)
        z = float(_cfg(config, "lie_z", 1.5)) * multiplier
        crafted = [m - z * s for m, s in zip(mean, std)]
        for i in malicious_idx:
            result[i] = _copy(crafted)

    elif mechanism == "gradient_mimicry":
        if not benign_updates:
            return result, multiplier
        reference = _mean_updates(benign_updates)
        lam = float(_cfg(config, "mimicry_lambda", 0.85))
        lam = float(np.clip(lam * multiplier, 0.0, 1.0))
        for i in malicious_idx:
            result[i] = [
                lam * ref + (1.0 - lam) * np.asarray(raw, dtype=np.float32)
                for ref, raw in zip(reference, updates[i])
            ]

    elif mechanism == "colluding_sign_flip":
        malicious_updates = [updates[i] for i in malicious_idx]
        centroid = _mean_updates(malicious_updates)
        crafted = [-strength * layer for layer in centroid]
        for i in malicious_idx:
            result[i] = _copy(crafted)

    elif mechanism == "min_max":
        crafted = _min_max_crafted(benign_updates, config or {}, multiplier)
        for i in malicious_idx:
            result[i] = _copy(crafted)

    elif mechanism == "min_sum":
        crafted = _min_sum_crafted(benign_updates, config or {}, multiplier)
        for i in malicious_idx:
            result[i] = _copy(crafted)

    elif mechanism == "adaptive_stealth":
        crafted = _adaptive_stealth_crafted(
            benign_updates,
            config or {},
            multiplier,
        )
        for i in malicious_idx:
            result[i] = _copy(crafted)

    elif mechanism in {
        "heterogeneity_aware_mimicry",
        "heterogeneity_mimicry",
    }:
        if not benign_updates:
            return result, multiplier
        neighbors = int(_cfg(config, "neighbors", 3))
        similarity = str(_cfg(config, "similarity", "cosine"))
        mimicry = float(
            np.clip(_cfg(config, "mimicry_lambda", 0.75), 0.0, 1.0)
        )
        for i in malicious_idx:
            reference = _nearest_benign_reference(
                updates[i],
                benign_updates,
                neighbors=neighbors,
                similarity=similarity,
            )
            adversarial = [
                -strength * np.asarray(raw, dtype=np.float32)
                for raw in updates[i]
            ]
            result[i] = [
                mimicry * ref + (1.0 - mimicry) * adv
                for ref, adv in zip(reference, adversarial)
            ]

    elif mechanism == "targeted_family_poisoning":
        crafted = _targeted_family_crafted(
            updates,
            malicious_idx,
            benign_updates,
            config or {},
            multiplier,
            probe_inferences=probe_inferences,
            probe_labels=probe_labels,
            probe_families=probe_families,
        )
        for i in malicious_idx:
            result[i] = _copy(crafted)

    else:
        raise ValueError(f"Unsupported model attack mechanism: {mechanism}")

    return result, multiplier
