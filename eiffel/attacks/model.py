"""Procedural model-poisoning attacks applied to client updates.

The functions operate on deltas (local_weights - global_weights). This keeps the
implementation model-agnostic and therefore compatible with MLP, CNN and Transformer
models as long as all clients share the same parameter layout.
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
    config: Mapping[str, Any] | None, server_round: int, total_rounds: int | None = None
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
        late_start = int(schedule.get("start_round", max(1, (total_rounds or 1) // 2)))
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
        progress = min(1.0, max(0.0, (server_round - start) / max(1, ramp - 1)))
        initial = float(schedule.get("gradual_start_strength", 0.0))
        final = float(schedule.get("gradual_end_strength", 1.0))
        return float(initial + (final - initial) * progress)
    raise ValueError(f"Unsupported model-attack schedule: {kind}")


def _copy(update: Sequence[np.ndarray]) -> Update:
    return [np.asarray(layer, dtype=np.float32).copy() for layer in update]


def _mean_updates(updates: Sequence[Update]) -> Update:
    return [
        np.mean(np.stack([u[layer] for u in updates], axis=0), axis=0)
        for layer in range(len(updates[0]))
    ]


def _std_updates(updates: Sequence[Update]) -> Update:
    return [
        np.std(np.stack([u[layer] for u in updates], axis=0), axis=0)
        for layer in range(len(updates[0]))
    ]


def apply_round_attack(
    updates: Sequence[Update],
    malicious_mask: Sequence[bool],
    config: Mapping[str, Any] | None,
    *,
    server_round: int,
    total_rounds: int | None = None,
    seed: int = 0,
) -> tuple[list[Update], float]:
    """Apply the configured attack to one round of client updates.

    Returns the submitted updates and the schedule multiplier. LIE, mimicry and
    colluding attacks use same-round benign/malicious references and are therefore
    implemented at the strategy boundary immediately before aggregation.
    """
    result = [_copy(u) for u in updates]
    multiplier = attack_strength_for_round(config, server_round, total_rounds)
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
            result[i] = [-strength * np.asarray(x, dtype=np.float32) for x in updates[i]]

    elif mechanism == "scaling":
        factor = float(_cfg(config, "scale_factor", 10.0)) * multiplier
        for i in malicious_idx:
            result[i] = [factor * np.asarray(x, dtype=np.float32) for x in updates[i]]

    elif mechanism == "gaussian_noise":
        noise_std = float(_cfg(config, "noise_std", 1.0)) * multiplier
        for i in malicious_idx:
            poisoned = []
            for layer in updates[i]:
                layer = np.asarray(layer, dtype=np.float32)
                base = float(np.std(layer))
                sigma = noise_std * (base if base > 1e-12 else 1.0)
                poisoned.append(layer + rng.normal(0.0, sigma, size=layer.shape).astype(np.float32))
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

    else:
        raise ValueError(f"Unsupported model attack mechanism: {mechanism}")

    return result, multiplier
