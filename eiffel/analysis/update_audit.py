"""Server-observable audit metrics for client model updates."""

from __future__ import annotations

from typing import Sequence

import numpy as np

Update = Sequence[np.ndarray]


def flatten_update(update: Update) -> np.ndarray:
    if not update:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(
        [np.asarray(layer, dtype=np.float32).reshape(-1) for layer in update]
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def audit_updates(updates: Sequence[Update]) -> list[dict[str, float]]:
    """Compute compact per-client audit signals for one communication round."""
    flat = [flatten_update(u) for u in updates]
    if not flat:
        return []
    matrix = np.stack(flat, axis=0)
    mean = matrix.mean(axis=0)
    median = np.median(matrix, axis=0)
    mean_sign = np.sign(mean)

    output: list[dict[str, float]] = []
    for vec in flat:
        abs_vec = np.abs(vec)
        output.append(
            {
                "l1": float(abs_vec.sum()),
                "l2": float(np.linalg.norm(vec)),
                "linf": float(abs_vec.max(initial=0.0)),
                "cosine_to_mean": _cosine(vec, mean),
                "distance_to_mean": float(np.linalg.norm(vec - mean)),
                "distance_to_median": float(np.linalg.norm(vec - median)),
                "sign_agreement_mean": float(np.mean(np.sign(vec) == mean_sign)),
            }
        )
    return output
