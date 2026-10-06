"""Direct TOML configuration and execution for Eiffel.

Hydra/OmegaConf are intentionally not used here. TOML is the single public
experiment configuration format and is resolved into ordinary Python objects.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "cicids": "nfv2/sampled/cicids",
    "cse-cic-ids2018": "nfv2/sampled/cicids",
    "cse_cic_ids2018": "nfv2/sampled/cicids",
    "cicids_full": "nfv2/full/cicids",
    "nf-cse-cic-ids2018-v2": "nfv2/full/cicids",
    "cicids_datacenter": "nfv2/datacenter/cicids",
    "nb15": "nfv2/sampled/nb15",
    "unsw-nb15": "nfv2/sampled/nb15",
    "unsw_nb15": "nfv2/sampled/nb15",
    "nb15_full": "nfv2/full/nb15",
    "nf-unsw-nb15-v2": "nfv2/full/nb15",
    "nb15_datacenter": "nfv2/datacenter/nb15",
    "toniot": "nfv2/sampled/toniot",
    "ton-iot": "nfv2/sampled/toniot",
    "ton_iot": "nfv2/sampled/toniot",
    "botiot": "nfv2/sampled/botiot",
    "synthetic_stress": "synthetic/stress",
    "synthetic_50k": "synthetic/stress",
    "mirage": "mirage/app3",
    "mirage_app3": "mirage/app3",
    "mirage-app3": "mirage/app3",
    "mirage-genai-2025": "mirage/app3",
    "cesnet": "cesnet/quicext25_top50",
    "cesnet_quicext25": "cesnet/quicext25_top50",
    "cesnet-quicext-25": "cesnet/quicext25_top50",
    "ciciot": "ciciot/family",
    "ciciot_family": "ciciot/family",
    "ciciot2023_family": "ciciot/family",
    "ciciot_binary": "ciciot/binary",
    "ciciot2023_binary": "ciciot/binary",
    "ciciot_fine": "ciciot/fine",
    "ciciot2023_fine": "ciciot/fine",
}

MODELS = {
    "mlp": "popoola",
    "popoola": "popoola",
    "p4p_mlp": "p4p_mlp",
    "cnn1d": "cnn1d",
    "ft_transformer": "ft_transformer",
    "stress_mlp": "stress_mlp",
    "synthetic_mlp": "stress_mlp",
}

MODEL_ATTACKS = {
    "none": "none",
    "sign_flip": "sign_flip",
    "model_scaling": "scaling",
    "scaling": "scaling",
    "gaussian_noise": "gaussian_noise",
    "lie": "lie",
    "gradient_mimicry": "gradient_mimicry",
    "colluding_sign_flip": "colluding_sign_flip",
    "min_max": "min_max",
    "min_sum": "min_sum",
    "adaptive_stealth": "adaptive_stealth",
    "heterogeneity_aware_mimicry": "heterogeneity_aware_mimicry",
    "heterogeneity_mimicry": "heterogeneity_aware_mimicry",
    "targeted_family_poisoning": "targeted_family_poisoning",
}

ATTACK_TABLES = {
    "min_max": "min_max",
    "min_sum": "min_sum",
    "adaptive_stealth": "adaptive",
    "heterogeneity_aware_mimicry": "heterogeneity",
    "heterogeneity_mimicry": "heterogeneity",
    "targeted_family_poisoning": "targeted",
}

STORAGE_DEFAULTS = {
    "enabled": True,
    "path": "round_state.h5",
    "compression": "gzip",
    "compression_level": 4,
    "flush_each_round": True,
    "capture_inference": True,
    "capture_logits": True,
    "capture_probe_features": True,
    "capture_global_inference": True,
    "probe_size": 256,
}

ATTACK_DEFAULTS = {
    "none": {"enabled": False, "mechanism": "none", "strength": 1.0,
             "scale_factor": 10.0, "noise_std": 1.0, "lie_z": 1.5,
             "mimicry_lambda": 0.85},
    "sign_flip": {"enabled": True, "mechanism": "sign_flip", "strength": 1.0,
                  "scale_factor": 10.0, "noise_std": 1.0, "lie_z": 1.5,
                  "mimicry_lambda": 0.85},
    "scaling": {"enabled": True, "mechanism": "scaling", "strength": 1.0,
                "scale_factor": 10.0, "noise_std": 1.0, "lie_z": 1.5,
                "mimicry_lambda": 0.85},
    "gaussian_noise": {"enabled": True, "mechanism": "gaussian_noise", "strength": 1.0,
                       "scale_factor": 10.0, "noise_std": 1.0, "lie_z": 1.5,
                       "mimicry_lambda": 0.85},
    "lie": {"enabled": True, "mechanism": "lie", "strength": 1.0,
            "scale_factor": 10.0, "noise_std": 1.0, "lie_z": 1.5,
            "mimicry_lambda": 0.85},
    "gradient_mimicry": {"enabled": True, "mechanism": "gradient_mimicry",
                         "strength": 1.0, "scale_factor": 10.0, "noise_std": 1.0,
                         "lie_z": 1.5, "mimicry_lambda": 0.85},
    "colluding_sign_flip": {"enabled": True, "mechanism": "colluding_sign_flip",
                            "strength": 1.0, "scale_factor": 10.0,
                            "noise_std": 1.0, "lie_z": 1.5,
                            "mimicry_lambda": 0.85},
    "min_max": {"enabled": True, "mechanism": "min_max", "direction": "negative_mean",
                "search_steps": 24, "max_lambda": 10.0, "constraint_margin": 1.0},
    "min_sum": {"enabled": True, "mechanism": "min_sum", "direction": "negative_mean",
                "search_steps": 24, "max_lambda": 10.0, "constraint_margin": 1.0},
    "adaptive_stealth": {"enabled": True, "mechanism": "adaptive_stealth",
                         "direction": "sign", "search_steps": 24,
                         "max_strength": 8.0, "l2_quantile": 0.95,
                         "distance_quantile": 0.95, "min_cosine_quantile": 0.05,
                         "stealth_margin": 1.0, "cosine_slack": 0.02},
    "heterogeneity_aware_mimicry": {
        "enabled": True, "mechanism": "heterogeneity_aware_mimicry",
        "strength": 3.0, "neighbors": 3, "similarity": "cosine",
        "mimicry_lambda": 0.75,
    },
    "targeted_family_poisoning": {
        "enabled": True, "mechanism": "targeted_family_poisoning",
        "target_family": "", "target_amplification": 2.5,
        "target_mimicry_lambda": 0.15,
    },
}


class ExperimentConfigError(ValueError):
    pass


TomlExperimentError = ExperimentConfigError


def load_profile(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise ExperimentConfigError("Profile root must be a table/object.")
    return value


def _table(profile: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = profile.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExperimentConfigError(f"[{key}] must be a table.")
    return value


def parse_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def set_path(profile: dict[str, Any], dotted: str, value: Any) -> None:
    keys = [part for part in dotted.split(".") if part]
    if not keys:
        raise ExperimentConfigError("Override key cannot be empty.")
    node = profile
    for key in keys[:-1]:
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ExperimentConfigError(f"{dotted}: {key} is not a table.")
        node = child
    node[keys[-1]] = value


def apply_overrides(profile: Mapping[str, Any], values: list[str] | None) -> dict[str, Any]:
    result = copy.deepcopy(dict(profile))
    for raw in values or []:
        if "=" not in raw:
            raise ExperimentConfigError(f"Invalid override {raw!r}; expected path=value.")
        key, value = raw.split("=", 1)
        set_path(result, key.strip(), parse_scalar(value.strip()))
    return result


def _malicious_ids(attack: Mapping[str, Any]) -> list[int]:
    raw = attack.get("malicious_client_ids")
    if raw in (None, ""):
        return []
    values = [int(v) for v in raw] if isinstance(raw, (list, tuple)) else [
        int(v.strip()) for v in str(raw).split(",") if v.strip()
    ]
    if len(values) != len(set(values)) or any(v < 0 for v in values):
        raise ExperimentConfigError("malicious_client_ids must be unique non-negative IDs.")
    return values


def _schedule(attack: Mapping[str, Any], rounds: int) -> dict[str, Any]:
    raw = _table(attack, "schedule")
    kind = str(raw.get("type", "continuous")).lower()
    if kind not in {"continuous", "late", "window", "on_off", "gradual"}:
        raise ExperimentConfigError(f"Unsupported schedule {kind!r}.")
    start = int(raw.get("start_round", max(1, rounds // 2) if kind == "late" else 1))
    end = int(raw.get("end_round", rounds) or rounds)
    if start < 1 or start > rounds or end < start or end > rounds:
        raise ExperimentConfigError("Attack schedule is outside configured rounds.")
    out = {"type": kind, "start_round": start, "end_round": end}
    if kind == "on_off":
        on = int(raw.get("on_rounds", raw.get("active_rounds", 1)))
        off = int(raw.get("off_rounds", 1))
        if on < 1 or off < 1:
            raise ExperimentConfigError("on_off requires positive on/off lengths.")
        out.update(period=on + off, active_rounds=on)
    if kind == "gradual":
        ramp = int(raw.get("ramp_rounds", end - start + 1))
        if ramp < 1:
            raise ExperimentConfigError("gradual ramp_rounds must be >= 1.")
        out.update(
            ramp_rounds=ramp,
            gradual_start_strength=float(raw.get("gradual_start_strength", 0.0)),
            gradual_end_strength=float(raw.get("gradual_end_strength", 1.0)),
        )
    return out


def _selector(poison_rate: float, attack: Mapping[str, Any], rounds: int) -> str:
    if not 0.0 <= poison_rate <= 1.0:
        raise ExperimentConfigError("poison_rate must be in [0,1].")
    schedule = _schedule(attack, rounds)
    kind, start, end = schedule["type"], schedule["start_round"], schedule["end_round"]
    rate = f"{poison_rate:.8f}".rstrip("0").rstrip(".") or "0"
    if poison_rate == 0:
        return "0.0"
    if kind == "continuous" and start == 1 and end == rounds:
        return rate
    if kind in {"continuous", "late", "window"}:
        value = f"0.0+{rate}{{{start}}}"
        return value + (f"-{rate}{{{end + 1}}}" if end < rounds else "")
    if kind == "on_off":
        on = int(schedule["active_rounds"])
        off = int(schedule["period"]) - on
        value, current = "0.0", start
        while current <= end:
            value += f"+{rate}{{{current}}}"
            if current + on <= end:
                value += f"-{rate}{{{current + on}}}"
            current += on + off
        return value + (f"-{rate}{{{end + 1}}}" if end < rounds else "")
    ramp_end = min(end, start + int(schedule["ramp_rounds"]) - 1)
    step = poison_rate / (ramp_end - start + 1)
    inc = f"{step:.8f}".rstrip("0").rstrip(".")
    return f"0.0+{inc}[{start}:{ramp_end}]"


def resolve_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    experiment = dict(_table(profile, "experiment"))
    dataset = dict(_table(profile, "dataset"))
    partition = dict(_table(profile, "partition"))
    model = dict(_table(profile, "model"))
    training = dict(_table(profile, "training"))
    attack = dict(_table(profile, "attack"))
    storage = dict(_table(profile, "storage"))
    aggregation = dict(_table(profile, "aggregation"))

    total = int(experiment.get("num_clients", 10))
    rounds = int(experiment.get("rounds", 10))
    seed = int(experiment.get("seed", 1138))
    if total < 1 or rounds < 1:
        raise ExperimentConfigError("num_clients and rounds must be >= 1.")

    dataset_name = str(dataset.get("registry") or dataset.get("hydra_group")
                       or dataset.get("name", "cicids")).lower()
    registry = DATASETS.get(dataset_name, dataset_name)
    if registry not in set(DATASETS.values()):
        raise ExperimentConfigError(
            f"Unsupported dataset {dataset_name!r}; add it to the dataset registry."
        )
    synthetic = registry == "synthetic/stress"
    fixed = registry in {
        "mirage/app3", "cesnet/quicext25_top50",
        "ciciot/binary", "ciciot/family", "ciciot/fine",
    }

    task = str(dataset.get("task", "binary")).lower()
    if task == "multiclass_aware":
        task = "family_aware"
    if task not in {"binary", "family_aware", "multiclass"}:
        raise ExperimentConfigError("Unsupported dataset.task.")
    if task == "multiclass" and not (synthetic or fixed):
        raise ExperimentConfigError("Multiclass is supported only by synthetic/fixed tasks.")

    ids = _malicious_ids(attack)
    if ids:
        attackers = len(ids)
    else:
        fraction = float(attack.get("malicious_fraction", 0.0))
        attackers = 0 if fraction <= 0 else max(1, int(round(total * fraction)))
    if attackers >= total:
        raise ExperimentConfigError("At least one benign client is required.")
    if any(value >= total for value in ids):
        raise ExperimentConfigError("malicious_client_ids are outside client range.")
    benign = total - attackers

    model_name = MODELS.get(str(model.get("name", "popoola")).lower())
    if model_name is None:
        raise ExperimentConfigError("Unsupported model.")
    model_cfg: dict[str, Any] = {"name": model_name}
    model_cfg["learning_rate"] = float(training.get(
        "learning_rate", 0.0001 if model_name == "popoola" else 0.001
    ))
    defaults = {
        "p4p_mlp": {"dropout": 0.2},
        "cnn1d": {"dropout": 0.2},
        "ft_transformer": {"d_token": 64, "n_heads": 4, "n_blocks": 2,
                           "ff_factor": 2.0, "dropout": 0.1},
        "stress_mlp": {"hidden1": 64, "hidden2": 32, "weight_decay": 0.0001,
                       "beta1": 0.9, "beta2": 0.999},
    }
    model_cfg.update(defaults.get(model_name, {}))
    for key in ("dropout", "d_token", "n_heads", "n_blocks", "ff_factor",
                "hidden1", "hidden2", "weight_decay"):
        if key in model:
            model_cfg[key] = model[key]
    if "adam_beta1" in training:
        model_cfg["beta1"] = float(training["adam_beta1"])
    if "adam_beta2" in training:
        model_cfg["beta2"] = float(training["adam_beta2"])
    model_cfg["task"] = "multiclass" if task == "multiclass" else "binary"
    model_cfg["num_classes"] = int(dataset.get("num_classes", 6 if task == "multiclass" else 2))

    ptype = str(partition.get("type", "iid")).lower()
    if fixed and ptype != "preassigned":
        raise ExperimentConfigError("Fixed preprocessed datasets require preassigned partitioning.")
    if synthetic:
        if ptype not in {"iid", "dirichlet"}:
            raise ExperimentConfigError("synthetic_stress supports iid or dirichlet.")
        partition_cfg = {"type": "preassigned", "source_distribution": ptype,
                         "dirichlet_alpha": float(partition.get("dirichlet_alpha", 0.5))}
    else:
        if ptype not in {"iid", "dumb", "niid_class", "dirichlet", "preassigned"}:
            raise ExperimentConfigError("Unsupported partitioner.")
        partition_cfg = {"type": ptype}
        if ptype == "dirichlet":
            partition_cfg.update(
                dirichlet_alpha=float(partition.get("dirichlet_alpha", 0.5)),
                min_partition_size=int(partition.get("min_partition_size", 1)),
            )
        if ptype == "niid_class":
            partition_cfg.update(
                n_drop=int(partition.get("n_drop", 2)),
                n_keep=int(partition.get("n_keep", 0)),
                preserved_classes=list(partition.get("preserved_classes", ["Benign"])),
            )

    if str(aggregation.get("name", "fedavg")).lower() != "fedavg":
        raise ExperimentConfigError("Only fedavg is currently supported.")

    storage_cfg = dict(STORAGE_DEFAULTS)
    storage_cfg.update(storage)
    schedule = _schedule(attack, rounds)
    mechanism = str(attack.get("mechanism", "none")).lower()
    params = dict(attack)
    nested_name = ATTACK_TABLES.get(mechanism)
    if nested_name:
        nested = _table(attack, nested_name)
        params.update(nested)

    data_poison = None
    if mechanism == "label_flip":
        if attackers <= 0:
            raise ExperimentConfigError("label_flip requires a malicious client.")
        data_poison = {"profile": _selector(float(attack.get("poison_rate", 1.0)), attack, rounds)}
        objective = str(attack.get("objective", "untargeted")).lower()
        if task == "multiclass":
            if not fixed or "source_class" not in attack or "destination_class" not in attack:
                raise ExperimentConfigError("Multiclass label flip requires fixed data and source/destination.")
            source = int(attack["source_class"])
            destination = int(attack["destination_class"])
            if source == destination:
                raise ExperimentConfigError("source_class and destination_class must differ.")
            data_poison.update(type="targeted", target=None,
                               source_class=source, destination_class=destination)
        elif objective in {"untargeted", "all"}:
            data_poison["type"] = "untargeted"
        elif objective == "targeted":
            target = attack.get("target")
            if not target:
                raise ExperimentConfigError("targeted label flip requires attack.target.")
            data_poison.update(type="targeted",
                               target=list(target) if isinstance(target, (list, tuple)) else [target])
        else:
            raise ExperimentConfigError("Unsupported label-flip objective.")
        model_attack = dict(ATTACK_DEFAULTS["none"])
    else:
        canonical = MODEL_ATTACKS.get(mechanism)
        if canonical is None:
            raise ExperimentConfigError(f"Unsupported attack {mechanism!r}.")
        if canonical != "none" and attackers <= 0:
            raise ExperimentConfigError(f"{mechanism} requires a malicious client.")
        if canonical in {"min_max", "min_sum", "adaptive_stealth"} and benign < 2:
            raise ExperimentConfigError(f"{mechanism} requires at least two benign clients.")
        model_attack = copy.deepcopy(ATTACK_DEFAULTS[canonical])
        if canonical == "scaling" and "strength" in params:
            model_attack["scale_factor"] = float(params["strength"])
        elif "strength" in params:
            model_attack["strength"] = float(params["strength"])
        for key in (
            "scale_factor", "noise_std", "lie_z", "mimicry_lambda", "direction",
            "search_steps", "max_lambda", "constraint_margin", "l2_quantile",
            "distance_quantile", "min_cosine_quantile", "stealth_margin",
            "cosine_slack", "max_strength", "neighbors", "similarity",
            "target_family", "target_amplification", "target_mimicry_lambda",
        ):
            if key in params:
                model_attack[key] = params[key]
        model_attack["schedule"] = schedule
        if canonical == "targeted_family_poisoning":
            if not str(model_attack.get("target_family", "")).strip():
                raise ExperimentConfigError("targeted_family_poisoning requires target_family.")
            if not storage_cfg.get("enabled", True):
                raise ExperimentConfigError("targeted_family_poisoning requires storage.enabled=true.")
            if not storage_cfg.get("capture_inference", True):
                raise ExperimentConfigError("targeted_family_poisoning requires capture_inference=true.")
            if int(storage_cfg.get("probe_size", 256)) <= 0:
                raise ExperimentConfigError("targeted_family_poisoning requires probe_size > 0.")

    dataset_cfg = copy.deepcopy(dataset)
    dataset_cfg.pop("hydra_group", None)
    dataset_cfg["registry"] = registry
    dataset_cfg["task"] = task
    if synthetic:
        dataset_cfg.update(
            num_clients=total,
            partition_mode=ptype,
            dirichlet_alpha=float(partition.get("dirichlet_alpha", 0.5)),
        )

    return {
        "experiment": {
            "name": str(experiment.get("name", "eiffel-experiment")),
            "seed": seed, "num_clients": total, "num_benign": benign,
            "num_attackers": attackers, "malicious_client_ids": ids,
            "rounds": rounds,
            "max_concurrent_clients": (
                None if experiment.get("max_concurrent_clients") in (None, 0, "")
                else int(experiment["max_concurrent_clients"])
            ),
        },
        "dataset": dataset_cfg,
        "partition": partition_cfg,
        "model": model_cfg,
        "training": {
            "local_epochs": int(training.get("local_epochs", 10)),
            "batch_size": int(training.get("batch_size", 512)),
        },
        "attack": {
            "mechanism": mechanism, "schedule": schedule,
            "model_attack": model_attack, "data_poisoning": data_poison,
        },
        "aggregation": {"name": "fedavg", "implementation": "InstrumentedFedAvg"},
        "storage": storage_cfg,
    }


def _rooted(value: str | Path) -> Path:
    value = Path(value).expanduser()
    return value if value.is_absolute() else ROOT / value


def _load_dataset(cfg: Mapping[str, Any], seed: int):
    from eiffel.datasets.mirage import load_data as load_mirage
    from eiffel.datasets.nfv2 import load_data as load_nfv2
    from eiffel.datasets.preprocessed_network import load_preprocessed_data
    from eiffel.datasets.synthetic_stress import load_data as load_synthetic

    registry = str(cfg["registry"])
    if registry.startswith("nfv2/"):
        table = {
            "nfv2/sampled/cicids": ("data/nfv2/sampled/cicids.csv.gz", "cicids", ["DoS"]),
            "nfv2/sampled/nb15": ("data/nfv2/sampled/nb15.csv.gz", "nb15", ["Analysis"]),
            "nfv2/sampled/toniot": ("data/nfv2/sampled/toniot.csv.gz", "toniot", ["injection"]),
            "nfv2/sampled/botiot": ("data/nfv2/sampled/botiot.csv.gz", "botiot", ["Reconnaissance"]),
            "nfv2/full/cicids": ("data/nfv2/origin/NF-CSE-CIC-IDS2018-v2.csv.gz", "cicids", ["DoS"]),
            "nfv2/full/nb15": ("data/nfv2/origin/NF-UNSW-NB15-v2.csv.gz", "nb15", ["Analysis"]),
            "nfv2/full/toniot": ("data/nfv2/origin/NF-ToN-IoT-v2.csv.gz", "toniot", ["injection"]),
            "nfv2/full/botiot": ("data/nfv2/origin/NF-BoT-IoT-v2.csv.gz", "botiot", ["Reconnaissance"]),
        }
        if registry == "nfv2/datacenter/cicids":
            raw, key, target = os.environ.get("EIFFEL_CICIDS_PATH"), "cicids", ["Bot"]
        elif registry == "nfv2/datacenter/nb15":
            raw, key, target = os.environ.get("EIFFEL_NB15_PATH"), "nb15", ["Exploits"]
        else:
            raw, key, target = table[registry]
        if not raw:
            raise RuntimeError(f"Dataset path environment variable is missing for {registry}.")
        path = _rooted(cfg.get("path", raw))
        return load_nfv2(str(path), seed=seed, key=cfg.get("key", key),
                         _default_target=list(cfg.get("default_target", target)))

    if registry == "synthetic/stress":
        names = {
            "stress_latent_dim": "latent_dim",
            "stress_informative_features": "informative_features",
            "stress_redundant_features": "redundant_features",
            "stress_class_separation": "class_separation",
            "stress_latent_noise": "latent_noise",
            "stress_secondary_mode_probability": "secondary_mode_probability",
            "stress_hard_example_fraction": "hard_example_fraction",
            "stress_train_label_noise": "train_label_noise",
            "stress_client_shift_std": "client_shift_std",
            "stress_central_shift_std": "central_shift_std",
            "stress_outlier_fraction": "outlier_fraction",
        }
        allowed = {
            "num_clients", "samples_per_client", "central_test_size", "num_features",
            "num_classes", "rare_class_id", "rare_class_probability",
            "rare_specialist_client", "rare_specialist_strength", "feature_noise",
            "latent_dim", "informative_features", "redundant_features", "class_separation",
            "latent_noise", "secondary_mode_probability", "hard_example_fraction",
            "train_label_noise", "client_shift_std", "central_shift_std", "outlier_fraction",
            "dirichlet_alpha", "partition_mode", "task", "key",
        }
        kwargs = {}
        for raw_key, value in cfg.items():
            key = names.get(raw_key, raw_key)
            if key in allowed:
                kwargs[key] = value
        kwargs["_default_target"] = list(cfg.get("default_target", ["Botnet"]))
        return load_synthetic(seed=seed, **kwargs)

    if registry == "mirage/app3":
        return load_mirage(
            _rooted(cfg.get("task_dir", "data/mirage/tasks/app_3class")),
            seed=seed,
            expected_num_clients=int(cfg.get("expected_num_clients", 10)),
            expected_num_classes=int(cfg.get("expected_num_classes", cfg.get("num_classes", 3))),
            key=cfg.get("key", "mirage_app3"),
            _default_target=list(cfg.get("default_target", ["ChatGPT"])),
        )

    common = ["source_file", "binary_target", "binary_target_id", "family_target",
              "family_target_id", "fine_target", "fine_target_id"]
    presets = {
        "cesnet/quicext25_top50": dict(
            task_dir="data/cesnet", target_name_column="target_name",
            target_id_column="target_id",
            metadata_columns=["source_day", "target_name", "target_id"],
            exclude_columns=["source_day", "target_name", "target_id", "client_id"],
            expected_num_clients=10, expected_num_classes=50,
            key="cesnet_quicext25_top50", default_target=["*"],
        ),
        "ciciot/binary": dict(
            task_dir="data/ciciot", target_name_column="binary_target",
            target_id_column="binary_target_id", attack_metadata_column="family_target",
            metadata_columns=common, exclude_columns=common + ["client_id"],
            expected_num_clients=10, expected_num_classes=2,
            key="ciciot_binary", default_target=["DDoS"],
        ),
        "ciciot/family": dict(
            task_dir="data/ciciot", target_name_column="family_target",
            target_id_column="family_target_id",
            metadata_columns=common, exclude_columns=common + ["client_id"],
            expected_num_clients=10, expected_num_classes=8,
            key="ciciot_family", default_target=["DDoS"],
        ),
        "ciciot/fine": dict(
            task_dir="data/ciciot", target_name_column="fine_target",
            target_id_column="fine_target_id",
            metadata_columns=common, exclude_columns=common + ["client_id"],
            expected_num_clients=10, expected_num_classes=34,
            key="ciciot_fine", default_target=["*"],
        ),
    }
    if registry not in presets:
        raise ExperimentConfigError(f"No dataset builder for {registry}.")
    params = presets[registry]
    for key in tuple(params):
        if key in cfg:
            params[key] = cfg[key]
    params["task_dir"] = _rooted(params["task_dir"])
    target = params.pop("default_target")
    return load_preprocessed_data(seed=seed, _default_target=target, **params)


def _build_experiment(resolved: Mapping[str, Any]):
    from eiffel.core.experiment import Experiment
    from eiffel.datasets.partitioners import (
        DirichletPartitioner, DumbPartitioner, IIDPartitioner,
        NIIDClassPartitioner, PreassignedPartitioner,
    )
    from eiffel.datasets.poisoning import PoisonIns
    from eiffel.models.advanced import (
        mk_cnn1d, mk_ft_transformer, mk_p4p_mlp, mk_stress_mlp,
    )
    from eiffel.models.supervized import mk_popoola_mlp
    from eiffel.strategy.instrumented import InstrumentedFedAvg

    exp = resolved["experiment"]
    seed, rounds = int(exp["seed"]), int(exp["rounds"])
    dataset = _load_dataset(resolved["dataset"], seed)

    model_cfg = dict(resolved["model"])
    model_name = model_cfg.pop("name")
    model_fn = partial({
        "popoola": mk_popoola_mlp, "p4p_mlp": mk_p4p_mlp,
        "cnn1d": mk_cnn1d, "ft_transformer": mk_ft_transformer,
        "stress_mlp": mk_stress_mlp,
    }[model_name], **model_cfg)

    pcfg = resolved["partition"]
    ptype = pcfg["type"]
    if ptype == "dumb":
        partitioner = DumbPartitioner
    elif ptype == "iid":
        partitioner = partial(IIDPartitioner, class_column="Attack")
    elif ptype == "dirichlet":
        partitioner = partial(
            DirichletPartitioner, class_column="Attack",
            alpha=float(pcfg.get("dirichlet_alpha", 0.5)),
            min_partition_size=int(pcfg.get("min_partition_size", 1)),
        )
    elif ptype == "niid_class":
        partitioner = partial(
            NIIDClassPartitioner, class_column="Attack",
            preserved_classes=list(pcfg.get("preserved_classes", ["Benign"])),
            n_drop=int(pcfg.get("n_drop", 2)), n_keep=int(pcfg.get("n_keep", 0)),
        )
    else:
        partitioner = partial(PreassignedPartitioner, column="ClientHint", df_key="m")

    data_attack = None
    if resolved["attack"]["data_poisoning"] is not None:
        raw = dict(resolved["attack"]["data_poisoning"])
        raw["n_rounds"] = rounds
        data_attack = PoisonIns.from_dict(raw, default_target=dataset.default_target)

    storage = dict(resolved["storage"])
    strategy = partial(
        InstrumentedFedAvg, storage=storage,
        model_attack=dict(resolved["attack"]["model_attack"]),
        num_rounds=rounds, seed=seed,
    )
    training = resolved["training"]
    return Experiment(
        seed=seed, num_rounds=rounds,
        num_epochs=int(training["local_epochs"]),
        batch_size=int(training["batch_size"]),
        model_fn=model_fn,
        pools=[{
            "n_benign": int(exp["num_benign"]),
            "n_malicious": int(exp["num_attackers"]),
            "malicious_client_ids": list(exp["malicious_client_ids"]),
        }],
        datasets=[dataset], attacks=[data_attack], strategy=strategy,
        partitioner=partitioner, storage=storage,
        max_concurrent_clients=exp.get("max_concurrent_clients"),
    )


@contextmanager
def _chdir(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def default_output_dir(resolved: Mapping[str, Any]) -> Path:
    registry = str(resolved["dataset"]["registry"])
    root = {
        "mirage/app3": "mirage", "cesnet/quicext25_top50": "cesnet/top50",
        "ciciot/binary": "ciciot/binary", "ciciot/family": "ciciot/family",
        "ciciot/fine": "ciciot/fine",
    }.get(registry, registry.replace("/", "-"))
    now = datetime.now()
    return ROOT / "outputs" / root / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S-%f")


def run_profile(
    profile: Mapping[str, Any], *, output_dir: str | Path | None = None,
    overrides: list[str] | None = None, max_concurrent_clients: int | None = None,
) -> Path:
    raw = apply_overrides(profile, overrides)
    if max_concurrent_clients is not None:
        if max_concurrent_clients < 1:
            raise ExperimentConfigError("max_concurrent_clients must be >= 1.")
        raw.setdefault("experiment", {})["max_concurrent_clients"] = max_concurrent_clients
    resolved = resolve_profile(raw)
    run_dir = Path(output_dir).expanduser().resolve() if output_dir else default_output_dir(resolved).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input_profile.json").write_text(json.dumps(raw, indent=2, sort_keys=True))
    (run_dir / "resolved_profile.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))

    import tensorflow as tf
    from flwr.common.logger import logger as flwr_logger
    from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
    from eiffel.utils import set_seed

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    flwr_logger.setLevel(logging.WARNING)
    logging.getLogger("flwr").setLevel(logging.WARNING)
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.get_logger().setLevel(logging.ERROR)
    set_seed(int(resolved["experiment"]["seed"]))
    enable_tf_gpu_growth()

    log = logging.getLogger(__name__)
    log.info("Starting Eiffel: %s", resolved["experiment"]["name"])
    log.info("Configuration backend: TOML -> Python factories (Hydra removed)")
    log.info("Dataset: %s | attack: %s", resolved["dataset"]["registry"], resolved["attack"]["mechanism"])
    log.info("Output: %s", run_dir)

    with _chdir(run_dir):
        experiment = _build_experiment(resolved)
        Path("stats.json").write_text(json.dumps(experiment.data_stats(), indent=4))
        experiment.run()
        experiment.results.save("fit")
        experiment.results.save("distributed")
    return run_dir


def build_command(
    path: str | Path, *, output_dir: str | Path | None = None,
    overrides: list[str] | None = None, max_concurrent_clients: int | None = None,
) -> list[str]:
    command = [sys.executable, "-m", "eiffel.direct_runner", str(path)]
    if output_dir is not None:
        command += ["--output-dir", str(output_dir)]
    if max_concurrent_clients is not None:
        command += ["--max-concurrent-clients", str(max_concurrent_clients)]
    for override in overrides or []:
        command += ["--set", override]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Eiffel directly from a TOML profile.")
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-concurrent-clients", type=int)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile)
        raw = apply_overrides(profile, args.overrides)
        if args.max_concurrent_clients is not None:
            raw.setdefault("experiment", {})["max_concurrent_clients"] = args.max_concurrent_clients
        if args.dry_run:
            print(json.dumps(resolve_profile(raw), indent=2, sort_keys=True))
            return 0
        run_profile(
            profile, output_dir=args.output_dir, overrides=args.overrides,
            max_concurrent_clients=args.max_concurrent_clients,
        )
        return 0
    except Exception as exc:
        logging.getLogger(__name__).exception(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
