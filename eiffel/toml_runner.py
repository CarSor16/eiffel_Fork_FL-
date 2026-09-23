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
    "synthetic_stress": "synthetic/stress",
    "synthetic_50k": "synthetic/stress",
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


def _schedule_table(attack: Mapping[str, Any]) -> Mapping[str, Any]:
    schedule = attack.get("schedule", {})
    if schedule is None:
        return {}
    if not isinstance(schedule, Mapping):
        raise TomlExperimentError("[attack.schedule] must be a TOML table.")
    return schedule


def _validate_schedule(
    schedule: Mapping[str, Any],
    *,
    rounds: int,
) -> tuple[str, int, int]:
    schedule_type = str(schedule.get("type", "continuous")).lower()
    allowed = {"continuous", "late", "window", "on_off", "gradual"}
    if schedule_type not in allowed:
        raise TomlExperimentError(
            f"Unsupported attack schedule '{schedule_type}'. "
            f"Supported: {', '.join(sorted(allowed))}."
        )

    default_start = max(1, rounds // 2) if schedule_type == "late" else 1
    start = int(schedule.get("start_round", default_start))
    end = int(schedule.get("end_round", rounds) or rounds)

    if start < 1 or start > rounds:
        raise TomlExperimentError(
            f"attack.schedule.start_round must be in [1, {rounds}], got {start}."
        )
    if end < start or end > rounds:
        raise TomlExperimentError(
            f"attack.schedule.end_round must be in [{start}, {rounds}], got {end}."
        )
    if schedule_type == "on_off":
        on_rounds = int(schedule.get("on_rounds", schedule.get("active_rounds", 1)))
        off_rounds = int(schedule.get("off_rounds", 1))
        if on_rounds < 1 or off_rounds < 1:
            raise TomlExperimentError(
                "on_off schedules require on_rounds >= 1 and off_rounds >= 1."
            )
    return schedule_type, start, end


def _fmt_fraction(value: float) -> str:
    return f"{float(value):.8f}".rstrip("0").rstrip(".") or "0"


def _label_flip_profile(
    poison_rate: float,
    schedule: Mapping[str, Any],
    *,
    rounds: int,
) -> str:
    """Translate a temporal TOML schedule into Eiffel's stateful poison selector."""
    if not 0.0 <= poison_rate <= 1.0:
        raise TomlExperimentError("attack.poison_rate must be in [0, 1].")

    kind, start, end = _validate_schedule(schedule, rounds=rounds)
    rate = _fmt_fraction(poison_rate)

    if poison_rate == 0.0:
        return "0.0"

    # Base poisoning is used only when the attack is active from round zero onward.
    if kind == "continuous" and start == 1 and end == rounds:
        return rate

    if kind in {"continuous", "late", "window"}:
        profile = f"0.0+{rate}{{{start}}}"
        if end < rounds:
            profile += f"-{rate}{{{end + 1}}}"
        return profile

    if kind == "on_off":
        on_rounds = int(schedule.get("on_rounds", schedule.get("active_rounds", 1)))
        off_rounds = int(schedule.get("off_rounds", 1))
        profile = "0.0"
        current = start
        while current <= end:
            profile += f"+{rate}{{{current}}}"
            off_at = current + on_rounds
            if off_at <= end:
                profile += f"-{rate}{{{off_at}}}"
            current += on_rounds + off_rounds
        if end < rounds:
            # Ensure no poisoned state leaks beyond the requested schedule window.
            profile += f"-{rate}{{{end + 1}}}"
        return profile

    # Gradual label flipping: increase the poisoned fraction evenly during the
    # requested ramp.  The final fraction remains active after the ramp.
    ramp_rounds = int(schedule.get("ramp_rounds", end - start + 1))
    if ramp_rounds < 1:
        raise TomlExperimentError("gradual schedule ramp_rounds must be >= 1.")
    ramp_end = min(end, start + ramp_rounds - 1)
    steps = ramp_end - start + 1
    increment = poison_rate / steps
    inc = _fmt_fraction(increment)
    return f"0.0+{inc}[{start}:{ramp_end}]"


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
    rounds = int(experiment.get("rounds", 10))
    dataset_name = str(dataset.get("name", "cicids")).lower()
    synthetic_stress = dataset_name in {"synthetic_stress", "synthetic_50k"}
    if total_clients < 1:
        raise TomlExperimentError("experiment.num_clients must be >= 1")
    if rounds < 1:
        raise TomlExperimentError("experiment.rounds must be >= 1")
    task = str(dataset.get("task", "binary")).lower()
    if task == "multiclass_aware":
        task = "family_aware"
    if task not in {"binary", "family_aware", "multiclass"}:
        raise TomlExperimentError(
            "dataset.task supports binary, family_aware, or multiclass."
        )
    if task == "multiclass" and not synthetic_stress:
        raise TomlExperimentError(
            "dataset.task=multiclass is currently supported only by synthetic_stress. "
            "Real NF-V2 datasets remain on the binary Eiffel baseline."
        )

    mechanism = str(attack.get("mechanism", "none")).lower()
    attackers = _malicious_count(total_clients, attack)
    if attackers >= total_clients:
        raise TomlExperimentError(
            "The TOML compatibility layer requires at least one benign client."
        )
    benign = total_clients - attackers

    overrides = [
        f"seed={int(experiment.get('seed', 1138))}",
        f"num_rounds={rounds}",
        f"num_clients={benign}",
        f"num_attackers={attackers}",
        f"+datasets={_dataset_group(dataset)}",
    ]

    if synthetic_stress:
        overrides.append(
            f"datasets.synthetic_stress.num_clients={total_clients}"
        )
        dataset_keys = {
            "samples_per_client": "samples_per_client",
            "central_test_size": "central_test_size",
            "num_features": "num_features",
            "num_classes": "num_classes",
            "rare_class_id": "rare_class_id",
            "rare_class_probability": "rare_class_probability",
            "rare_specialist_client": "rare_specialist_client",
            "rare_specialist_strength": "rare_specialist_strength",
            "feature_noise": "feature_noise",
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
        for toml_key, hydra_key in dataset_keys.items():
            if toml_key in dataset:
                overrides.append(
                    "datasets.synthetic_stress."
                    f"{hydra_key}={_quote_hydra(dataset[toml_key])}"
                )
        overrides.append(
            "datasets.synthetic_stress.task="
            + ("multiclass" if task == "multiclass" else "binary")
        )

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
    if task == "multiclass":
        num_classes = int(dataset.get("num_classes", 6))
        if num_classes < 2:
            raise TomlExperimentError("dataset.num_classes must be >= 2 for multiclass")
        overrides.append("++model.task=multiclass")
        overrides.append(f"++model.num_classes={num_classes}")
    if "learning_rate" in training:
        overrides.append(f"++model.learning_rate={float(training['learning_rate'])}")
    for key in (
        "dropout",
        "d_token",
        "n_heads",
        "n_blocks",
        "ff_factor",
        "hidden1",
        "hidden2",
        "weight_decay",
    ):
        if key in model:
            overrides.append(f"++model.{key}={_quote_hydra(model[key])}")
    if "adam_beta1" in training:
        overrides.append(f"++model.beta1={float(training['adam_beta1'])}")
    if "adam_beta2" in training:
        overrides.append(f"++model.beta2={float(training['adam_beta2'])}")

    # Partitioning.
    partition_type = str(partition.get("type", "iid")).lower()
    if synthetic_stress:
        if partition_type not in {"iid", "dirichlet"}:
            raise TomlExperimentError(
                "synthetic_stress supports partition.type iid or dirichlet."
            )
        # The synthetic generator creates the client shards itself, including
        # Dirichlet label skew and client-specific covariate shift. Eiffel must then
        # preserve those exact 5k/client shards.
        overrides.append("partitioner=preassigned")
        overrides.append(
            "datasets.synthetic_stress.partition_mode="
            f"{partition_type}"
        )
        overrides.append(
            "datasets.synthetic_stress.dirichlet_alpha="
            f"{float(partition.get('dirichlet_alpha', 0.5))}"
        )
    else:
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
        if task == "multiclass":
            raise TomlExperimentError(
                "Multiclass label flipping is intentionally disabled until the TOML "
                "defines an explicit source_class -> destination_class mapping. "
                "Use family_aware for Eiffel-compatible label flipping, or use a "
                "model-update attack with task=multiclass."
            )
        overrides.append("model_attack=none")
        if attackers <= 0:
            raise TomlExperimentError("label_flip requires at least one malicious client.")
        poison_rate = float(attack.get("poison_rate", 1.0))
        schedule = _schedule_table(attack)
        profile = _label_flip_profile(poison_rate, schedule, rounds=rounds)
        overrides.append(f"attacks.0.profile={profile}")
        objective = str(attack.get("objective", "untargeted")).lower()
        target = attack.get("target")
        if objective in {"untargeted", "all"}:
            overrides.append("attacks.0.type=untargeted")
        elif objective == "targeted":
            overrides.append("attacks.0.type=targeted")
            if not target:
                raise TomlExperimentError(
                    "targeted label_flip requires attack.target with at least one "
                    "attack-family name."
                )
            values = list(target) if isinstance(target, (list, tuple)) else [target]
            target_value = json.dumps(values, separators=(",", ":"))
            overrides.append(f"++attacks.0.target={target_value}")
        else:
            raise TomlExperimentError(
                "attack.objective for label_flip must be targeted or untargeted."
            )
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

    schedule = _schedule_table(attack)
    schedule_type, start_round, end_round = _validate_schedule(
        schedule, rounds=rounds
    )

    overrides.append(f"model_attack.schedule.type={schedule_type}")
    overrides.append(f"model_attack.schedule.start_round={start_round}")
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
