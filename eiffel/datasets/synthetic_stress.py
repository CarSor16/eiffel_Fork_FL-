"""Deterministic synthetic stress dataset for fast FL-NIDS experiments.

This module adapts the 50k-sample synthetic stress environment used in the previous
Flower attack lab to Eiffel's binary NIDS API.

The federated training set is generated client-by-client (default: 10 clients x 5,000
samples = 50,000 samples) with Dirichlet label skew and client-specific covariate shift.
A separate central/common test set (default: 12,000 samples) is generated with a smaller
distribution shift.

The metadata keeps six traffic families. By default the supervised target is binary
(Benign=0, every attack family=1) for Eiffel/Lavaur compatibility. An experimental
multiclass task exposes the family id directly (0..K-1) for softmax models and
model-update attacks; binary Eiffel label flipping remains intentionally separate.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .nfv2 import NFV2Dataset

CLASS_NAMES = ("Benign", "Scan", "DDoS", "Botnet", "DoS", "Bruteforce")


def _base_probabilities(
    num_classes: int,
    rare_class_id: int,
    rare_class_probability: float,
) -> np.ndarray:
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if not 0 <= rare_class_id < num_classes:
        raise ValueError("rare_class_id is out of range")
    if not 0.0 < rare_class_probability < 1.0:
        raise ValueError("rare_class_probability must be in (0, 1)")

    template = np.array(
        [0.54, 0.14, 0.115, 0.03, 0.105, 0.07, 0.025, 0.015],
        dtype=np.float64,
    )
    if num_classes <= len(template):
        weights = template[:num_classes].copy()
    else:
        tail = np.geomspace(0.02, 0.005, num_classes - len(template))
        weights = np.concatenate([template, tail])

    weights[rare_class_id] = 0.0
    weights /= weights.sum()
    probabilities = weights * (1.0 - rare_class_probability)
    probabilities[rare_class_id] = rare_class_probability
    return probabilities / probabilities.sum()


def _class_names(num_classes: int) -> tuple[str, ...]:
    if num_classes <= len(CLASS_NAMES):
        return CLASS_NAMES[:num_classes]
    return CLASS_NAMES + tuple(
        f"Attack_{idx}" for idx in range(len(CLASS_NAMES), num_classes)
    )


def _latent_centers(
    *,
    seed: int,
    num_classes: int,
    latent_dim: int,
    class_separation: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 77)
    primary = rng.normal(
        0.0, class_separation, size=(num_classes, latent_dim)
    )

    # Deliberately overlap semantically related families.
    if num_classes >= 2:
        primary[1] = 0.58 * primary[0] + 0.42 * primary[1]
    if num_classes >= 5:
        primary[4] = 0.62 * primary[2] + 0.38 * primary[4]
    if num_classes >= 6:
        primary[5] = 0.55 * primary[0] + 0.45 * primary[5]

    secondary = primary + rng.normal(
        0.0, 0.55, size=(num_classes, latent_dim)
    )
    return primary, secondary


def _projection(
    *,
    seed: int,
    latent_dim: int,
    informative_features: int,
    redundant_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 199)
    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(max(1, latent_dim)),
        size=(latent_dim, informative_features),
    )
    redundant = (
        rng.normal(
            0.0,
            0.65,
            size=(informative_features, redundant_features),
        )
        if redundant_features
        else np.zeros((informative_features, 0))
    )
    return projection, redundant


def _client_probabilities(
    *,
    client_id: int,
    seed: int,
    base: np.ndarray,
    dirichlet_alpha: float,
    partition_mode: str,
    rare_class_id: int,
    rare_specialist_client: int,
    rare_specialist_strength: float,
) -> np.ndarray:
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be > 0")

    rng = np.random.default_rng(seed + 1_003 * client_id)
    if partition_mode == "iid":
        probabilities = base.copy()
    elif partition_mode == "dirichlet":
        concentration = np.maximum(
            base * dirichlet_alpha * len(base), 0.025
        )
        probabilities = rng.dirichlet(concentration)
    else:
        raise ValueError(
            "Synthetic stress partition_mode must be 'iid' or 'dirichlet'."
        )

    if client_id == rare_specialist_client:
        focus = np.zeros(len(base), dtype=np.float64)
        focus[rare_class_id] = 1.0
        probabilities = (
            (1.0 - rare_specialist_strength) * probabilities
            + rare_specialist_strength * focus
        )

    probabilities = np.clip(probabilities, 1e-12, None)
    return probabilities / probabilities.sum()


def _generate_features(
    labels: np.ndarray,
    *,
    seed: int,
    rng: np.random.Generator,
    num_features: int,
    num_classes: int,
    latent_dim: int,
    informative_features: int,
    redundant_features: int,
    class_separation: float,
    latent_noise: float,
    feature_noise: float,
    secondary_mode_probability: float,
    hard_example_fraction: float,
    client_id: int | None,
    client_shift_std: float,
    central: bool,
    central_shift_std: float,
    outlier_fraction: float,
) -> np.ndarray:
    primary, secondary = _latent_centers(
        seed=seed,
        num_classes=num_classes,
        latent_dim=latent_dim,
        class_separation=class_separation,
    )

    choose_secondary = rng.random(labels.size) < secondary_mode_probability
    centers = primary[labels].copy()
    centers[choose_secondary] = secondary[labels[choose_secondary]]

    hard = rng.random(labels.size) < hard_example_fraction
    competitors = (
        labels + rng.integers(1, num_classes, size=labels.size)
    ) % num_classes
    mix = rng.uniform(0.35, 0.60, size=labels.size)
    centers[hard] = (
        (1.0 - mix[hard, None]) * centers[hard]
        + mix[hard, None] * primary[competitors[hard]]
    )

    latent = centers + rng.normal(
        0.0, latent_noise, size=(labels.size, latent_dim)
    )

    projection, redundant_matrix = _projection(
        seed=seed,
        latent_dim=latent_dim,
        informative_features=informative_features,
        redundant_features=redundant_features,
    )

    informative = latent @ projection + rng.normal(
        0.0,
        feature_noise,
        size=(labels.size, informative_features),
    )
    redundant = (
        informative @ redundant_matrix
        + rng.normal(
            0.0,
            feature_noise * 0.65,
            size=(labels.size, redundant_features),
        )
        if redundant_features
        else np.empty((labels.size, 0), dtype=np.float64)
    )

    used = informative_features + redundant_features
    nuisance = rng.normal(
        0.0, 1.65, size=(labels.size, max(0, num_features - used))
    )
    features = np.concatenate(
        [informative, redundant, nuisance], axis=1
    )[:, :num_features]

    if central:
        shift_rng = np.random.default_rng(seed + 808_019)
        features = features + shift_rng.normal(
            0.0, central_shift_std, size=num_features
        )
        features *= shift_rng.lognormal(
            mean=0.0, sigma=0.035, size=num_features
        )
    elif client_id is not None:
        shift_rng = np.random.default_rng(
            seed + 17_171 * (client_id + 1)
        )
        features = features + shift_rng.normal(
            0.0, client_shift_std, size=num_features
        )
        features *= shift_rng.lognormal(
            mean=0.0, sigma=0.055, size=num_features
        )

    outliers = rng.random(labels.size) < outlier_fraction
    if np.any(outliers):
        features[outliers] += rng.normal(
            0.0,
            3.0,
            size=(int(outliers.sum()), num_features),
        )

    return features.astype(np.float32)


def load_data(
    *,
    seed: int,
    num_clients: int = 10,
    samples_per_client: int = 5000,
    central_test_size: int = 12000,
    num_features: int = 32,
    num_classes: int = 6,
    rare_class_id: int = 3,
    rare_class_probability: float = 0.025,
    rare_specialist_client: int = 0,
    rare_specialist_strength: float = 0.25,
    feature_noise: float = 0.60,
    latent_dim: int = 10,
    informative_features: int = 16,
    redundant_features: int = 8,
    class_separation: float = 1.80,
    latent_noise: float = 0.90,
    secondary_mode_probability: float = 0.35,
    hard_example_fraction: float = 0.12,
    train_label_noise: float = 0.01,
    client_shift_std: float = 0.18,
    central_shift_std: float = 0.10,
    outlier_fraction: float = 0.012,
    dirichlet_alpha: float = 0.5,
    partition_mode: str = "dirichlet",
    task: str = "binary",
    key: str = "synthetic_stress_50k",
    _default_target: Sequence[str] | None = None,
    **kwargs,
) -> NFV2Dataset:
    """Generate the old 50k federated stress benchmark in Eiffel form."""
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if samples_per_client < 2:
        raise ValueError("samples_per_client must be >= 2")
    if central_test_size < 2:
        raise ValueError("central_test_size must be >= 2")
    if num_features < 4:
        raise ValueError("num_features must be >= 4")
    if informative_features > num_features:
        raise ValueError("informative_features cannot exceed num_features")
    redundant_features = min(
        redundant_features, num_features - informative_features
    )
    latent_dim = min(latent_dim, num_features)

    task = str(task).lower()
    if task in {"family_aware", "multiclass_aware"}:
        task = "binary"
    if task not in {"binary", "multiclass"}:
        raise ValueError("task must be 'binary' or 'multiclass'")

    names = _class_names(num_classes)
    base = _base_probabilities(
        num_classes,
        rare_class_id,
        rare_class_probability,
    )

    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    metadata_blocks: list[pd.DataFrame] = []

    # Exactly num_clients * samples_per_client federated training samples.
    for client_id in range(num_clients):
        rng = np.random.default_rng(seed + 31_337 * client_id)
        probabilities = _client_probabilities(
            client_id=client_id,
            seed=seed,
            base=base,
            dirichlet_alpha=dirichlet_alpha,
            partition_mode=partition_mode,
            rare_class_id=rare_class_id,
            rare_specialist_client=rare_specialist_client,
            rare_specialist_strength=rare_specialist_strength,
        )
        family_labels = rng.choice(
            num_classes,
            size=samples_per_client,
            p=probabilities,
        )
        features = _generate_features(
            family_labels,
            seed=seed,
            rng=rng,
            num_features=num_features,
            num_classes=num_classes,
            latent_dim=latent_dim,
            informative_features=informative_features,
            redundant_features=redundant_features,
            class_separation=class_separation,
            latent_noise=latent_noise,
            feature_noise=feature_noise,
            secondary_mode_probability=secondary_mode_probability,
            hard_example_fraction=hard_example_fraction,
            client_id=client_id,
            client_shift_std=client_shift_std,
            central=False,
            central_shift_std=central_shift_std,
            outlier_fraction=outlier_fraction,
        )

        if task == "binary":
            supervised_target = (family_labels != 0).astype(np.int64)
        else:
            supervised_target = family_labels.astype(np.int64).copy()

        if train_label_noise > 0.0:
            noise_mask = rng.random(samples_per_client) < train_label_noise
            if task == "binary":
                supervised_target[noise_mask] = 1 - supervised_target[noise_mask]
            elif np.any(noise_mask):
                offsets = rng.integers(
                    1, num_classes, size=int(noise_mask.sum())
                )
                supervised_target[noise_mask] = (
                    supervised_target[noise_mask] + offsets
                ) % num_classes

        feature_blocks.append(features)
        target_blocks.append(supervised_target)
        metadata_blocks.append(
            pd.DataFrame(
                {
                    "Attack": [names[idx] for idx in family_labels],
                    "Split": "train",
                    "ClientHint": client_id,
                }
            )
        )

    # Separate common/central held-out test distribution.
    rng = np.random.default_rng(seed + 999_983)
    test_family_labels = rng.choice(
        num_classes,
        size=central_test_size,
        p=base,
    )
    test_features = _generate_features(
        test_family_labels,
        seed=seed,
        rng=rng,
        num_features=num_features,
        num_classes=num_classes,
        latent_dim=latent_dim,
        informative_features=informative_features,
        redundant_features=redundant_features,
        class_separation=class_separation,
        latent_noise=latent_noise,
        feature_noise=feature_noise,
        secondary_mode_probability=secondary_mode_probability,
        hard_example_fraction=hard_example_fraction,
        client_id=None,
        client_shift_std=client_shift_std,
        central=True,
        central_shift_std=central_shift_std,
        outlier_fraction=outlier_fraction,
    )
    test_target = (
        (test_family_labels != 0).astype(np.int64)
        if task == "binary"
        else test_family_labels.astype(np.int64)
    )

    feature_blocks.append(test_features)
    target_blocks.append(test_target)
    metadata_blocks.append(
        pd.DataFrame(
            {
                "Attack": [names[idx] for idx in test_family_labels],
                "Split": "test",
                "ClientHint": -1,
            }
        )
    )

    X_np = np.concatenate(feature_blocks, axis=0)
    y_np = np.concatenate(target_blocks, axis=0)
    metadata = pd.concat(metadata_blocks, ignore_index=True)

    # Standardize using federated-training statistics only.
    train_count = num_clients * samples_per_client
    train_mean = X_np[:train_count].mean(axis=0, dtype=np.float64)
    train_std = X_np[:train_count].std(axis=0, dtype=np.float64)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)
    X_np = ((X_np - train_mean) / train_std).astype(np.float32)

    columns = [f"f_{idx:02d}" for idx in range(num_features)]
    X = pd.DataFrame(X_np, columns=columns)
    y = pd.Series(y_np, name="Label")

    target = list(_default_target or ["Botnet"])
    return NFV2Dataset(
        X=X,
        y=y,
        m=metadata,
        key=key,
        _default_target=target,
    )
