"""Compatibility runner for TOML experiment profiles.

The original attack lab used one TOML file per experiment, while Eiffel uses Hydra
configuration groups. This module keeps TOML as the user-facing experiment format and
translates it into Eiffel/Hydra command-line overrides.

Usage:
    python -m eiffel.toml_runner experiments/toml/smoke_sign_flip.toml
    python -m eiffel.toml_runner experiments/toml/smoke_sign_flip.toml --dry-run
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "TOML support on Python 3.10 requires 'tomli'. "
            "Install it with: python -m pip install tomli"
        ) from exc


DATASETS = {
    "cicids": "nfv2/sampled/cicids",
    "cse-cic-ids2018": "nfv2/sampled/cicids",
    "cse_cic_ids2018": "nfv2/sampled/cicids",
    "nb15": "nfv2/sampled/nb15",
    "unsw-nb15": "nfv2/sampled/nb15",
    "unsw_nb15": "nfv2/sampled/nb15",
    "toniot": "nfv2/sampled/toniot",
    "ton-iot": "nfv2/sampled/toniot",
    "ton_iot": "nfv2/sampled/toniot",
    "botiot": "nfv2/sampled/botiot",
}

MODELS = {
    "mlp": "popoola",
    "popoola": "popoola",
    "p4p_mlp": "p4p_mlp",
    "cnn1d": "cnn1d",
    "ft_transformer": "ft_transformer",
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
}


class TomlExperimentError(ValueError):
    """Invalid or unsupported TOML experiment profile."""


def load_profile(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise TomlExperimentError("The TOML root must be a table.")
    return data


def _table(profile: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = profile.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TomlExperimentError(f"[{key}] must be a TOML table.")
    return value


def _as_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _quote_hydra(value: Any) -> str:
    """Return a conservative Hydra scalar representation."""
    if isinstance(value, bool):
        return _as_bool(value)
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    # Quote values containing characters Hydra commonly interprets specially.
    if any(ch in text for ch in " ,[]{}:=+"):
        return "'" + text.replace("'", "\\'") + "'"
    return text


def _malicious_count(total_clients: int, attack: Mapping[str, Any]) -> int:
    ids = str(attack.get("malicious_client_ids", "")).strip()
    if ids:
        parsed = [item.strip() for item in ids.split(",") if item.strip()]
        return len(parsed)

    fraction = float(attack.get("malicious_fraction", 0.0))
    if fraction <= 0.0:
        return 0
    return max(1, int(round(total_clients * fraction)))


def _dataset_group(dataset: Mapping[str, Any]) -> str:
    explicit = dataset.get("hydra_group")
    if explicit:
        return str(explicit)

    name = str(dataset.get("name", "cicids")).lower()
    if name.startswith("nfv2/"):
        return name
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise TomlExperimentError(
            f"Unsupported dataset '{name}'. Supported aliases: "
            + ", ".join(sorted(DATASETS))
            + ". You can also set dataset.hydra_group explicitly."
        ) from exc


def profile_to_overrides(profile: Mapping[str, Any]) -> list[str]:
    """Translate a TOML profile into Eiffel/Hydra overrides."""
    experiment = _table(profile, "experiment")
    dataset = _table(profile, "dataset")
    partition = _table(profile, "partition")
    model = _table(profile, "model")
    training = _table(profile, "training")
    attack = _table(profile, "attack")
    aggregation = _table(profile, "aggregation")
    storage = _table(profile, "storage")

    total_clients = int(experiment.get("num_clients", 10))
    if total_clients < 1:
        raise TomlExperimentError("experiment.num_clients must be >= 1")

    mechanism = str(attack.get("mechanism", "none")).lower()
    attackers = _malicious_count(total_clients, attack)
    if attackers >= total_clients:
        raise TomlExperimentError(
            "The TOML compatibility layer requires at least one benign client."
        )
    benign = total_clients - attackers

    overrides = [
        f"seed={int(experiment.get('seed', 1138))}",
        f"num_rounds={int(experiment.get('rounds', 10))}",
        f"num_clients={benign}",
        f"num_attackers={attackers}",
        f"+datasets={_dataset_group(dataset)}",
    ]

    # Training.
    if "local_epochs" in training:
        overrides.append(f"num_epochs={int(training['local_epochs'])}")
    if "batch_size" in training:
        overrides.append(f"batch_size={int(training['batch_size'])}")

    # Model.
    model_name = str(model.get("name", "popoola")).lower()
    try:
        hydra_model = MODELS[model_name]
    except KeyError as exc:
        raise TomlExperimentError(
            f"Unsupported model '{model_name}'. Supported: {', '.join(MODELS)}"
        ) from exc
    overrides.append(f"model={hydra_model}")
    if "learning_rate" in training:
        overrides.append(f"++model.learning_rate={float(training['learning_rate'])}")
    for key in ("dropout", "d_token", "n_heads", "n_blocks", "ff_factor"):
        if key in model:
            overrides.append(f"++model.{key}={_quote_hydra(model[key])}")

    # Partitioning.
    partition_type = str(partition.get("type", "iid")).lower()
    if partition_type in {"iid", "dumb", "niid_class", "dirichlet"}:
        overrides.append(f"partitioner={partition_type}")
    else:
        raise TomlExperimentError(
            "Unsupported partition.type. Use iid, dumb, niid_class or dirichlet."
        )
    if partition_type == "dirichlet":
        overrides.append(
            f"partitioner.alpha={float(partition.get('dirichlet_alpha', 0.5))}"
        )
        if "min_partition_size" in partition:
            overrides.append(
                f"partitioner.min_partition_size={int(partition['min_partition_size'])}"
            )

    # Aggregation. The instrumented strategy is required to capture submitted
    # updates and to inject model attacks before FedAvg aggregation.
    aggregation_name = str(aggregation.get("name", "fedavg")).lower()
    if aggregation_name != "fedavg":
        raise TomlExperimentError(
            f"aggregation.name='{aggregation_name}' is not yet supported by the "
            "TOML compatibility layer. Use fedavg for the current Eiffel build."
        )
    overrides.append("strategy=instrumented_fedavg")

    # Storage.
    overrides.append("storage=default")
    if storage:
        for key in (
            "enabled",
            "path",
            "compression",
            "compression_level",
            "flush_each_round",
            "capture_inference",
            "probe_size",
        ):
            if key in storage:
                overrides.append(f"storage.{key}={_quote_hydra(storage[key])}")

    # Attacks.
    if mechanism == "label_flip":
        overrides.append("model_attack=none")
        if attackers <= 0:
            raise TomlExperimentError("label_flip requires at least one malicious client.")
        poison_rate = float(attack.get("poison_rate", 1.0))
        # Eiffel's existing profile selectors already model temporal label poisoning.
        # For now the compatibility layer uses a direct constant profile.
        overrides.append(f"attacks.0.profile={poison_rate}")
        objective = str(attack.get("objective", "untargeted")).lower()
        target = attack.get("target")
        if objective in {"untargeted", "all"}:
            overrides.append("attacks.0.type=untargeted")
        else:
            overrides.append("attacks.0.type=targeted")
            if target is not None:
                values = list(target) if isinstance(target, (list, tuple)) else [target]
                target_value = json.dumps(values, separators=(",", ":"))
                overrides.append(f"++attacks.0.target={target_value}")
        return overrides

    try:
        attack_group = MODEL_ATTACKS[mechanism]
    except KeyError as exc:
        raise TomlExperimentError(
            f"Unsupported attack mechanism '{mechanism}'. Supported: "
            + ", ".join(sorted(list(MODEL_ATTACKS) + ["label_flip"]))
        ) from exc

    overrides.append(f"model_attack={attack_group}")

    if attack_group != "none":
        # Keep the malicious clients' local data clean for pure model poisoning.
        overrides.append("poisoning/profile=clean")

    if "strength" in attack:
        if attack_group == "scaling":
            overrides.append(
                f"model_attack.scale_factor={float(attack['strength'])}"
            )
        else:
            overrides.append(f"model_attack.strength={float(attack['strength'])}")
    if "scale_factor" in attack:
        overrides.append(f"model_attack.scale_factor={float(attack['scale_factor'])}")
    if "noise_std" in attack:
        overrides.append(f"model_attack.noise_std={float(attack['noise_std'])}")
    if "lie_z" in attack:
        overrides.append(f"model_attack.lie_z={float(attack['lie_z'])}")
    if "mimicry_lambda" in attack:
        overrides.append(
            f"model_attack.mimicry_lambda={float(attack['mimicry_lambda'])}"
        )

    schedule = attack.get("schedule", {})
    if schedule and not isinstance(schedule, Mapping):
        raise TomlExperimentError("[attack.schedule] must be a TOML table.")
    schedule = schedule or {}
    schedule_type = str(schedule.get("type", "continuous")).lower()
    if schedule_type not in {"continuous", "late", "window", "on_off", "gradual"}:
        raise TomlExperimentError(f"Unsupported attack schedule '{schedule_type}'.")

    overrides.append(f"model_attack.schedule.type={schedule_type}")
    if "start_round" in schedule:
        overrides.append(
            f"model_attack.schedule.start_round={int(schedule['start_round'])}"
        )
    end_round = int(schedule.get("end_round", 0) or 0)
    if end_round > 0:
        overrides.append(f"++model_attack.schedule.end_round={end_round}")

    if schedule_type == "on_off":
        on_rounds = int(schedule.get("on_rounds", schedule.get("active_rounds", 1)))
        off_rounds = int(schedule.get("off_rounds", 1))
        overrides.append(f"++model_attack.schedule.period={on_rounds + off_rounds}")
        overrides.append(f"++model_attack.schedule.active_rounds={on_rounds}")
    elif schedule_type == "gradual":
        if "ramp_rounds" in schedule:
            overrides.append(
                f"++model_attack.schedule.ramp_rounds={int(schedule['ramp_rounds'])}"
            )
        if "gradual_start_strength" in schedule:
            overrides.append(
                "++model_attack.schedule.gradual_start_strength="
                f"{float(schedule['gradual_start_strength'])}"
            )
        if "gradual_end_strength" in schedule:
            overrides.append(
                "++model_attack.schedule.gradual_end_strength="
                f"{float(schedule['gradual_end_strength'])}"
            )

    return overrides


def build_command(path: str | Path, extra: list[str] | None = None) -> list[str]:
    profile = load_profile(path)
    overrides = profile_to_overrides(profile)
    return [sys.executable, "-m", "eiffel", *overrides, *(extra or [])]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an Eiffel experiment described by a TOML profile."
    )
    parser.add_argument("profile", type=Path, help="Path to the experiment TOML file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the TOML and print the translated Eiffel command only.",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs="*",
        help="Additional Hydra overrides appended after the TOML translation.",
    )
    args = parser.parse_args(argv)

    command = build_command(args.profile, args.hydra_overrides)
    print("TOML profile:", args.profile)
    print("Eiffel command:")
    print(" ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        return 0

    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
