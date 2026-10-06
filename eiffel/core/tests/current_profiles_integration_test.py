"""Representative committed profiles through the direct TOML runtime."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import h5py
import pytest

from eiffel.analysis.validate_round_state import validate
from eiffel.direct_runner import build_command, load_profile

ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = ROOT / "experiments" / "toml"

PROFILE_CASES = (
    "synthetic_50k_quick_clean.toml",
    "synthetic_50k_label_flip.toml",
    "synthetic_50k_sign_flip.toml",
    "synthetic_50k_model_scaling.toml",
    "synthetic_50k_gaussian_noise.toml",
    "synthetic_50k_lie.toml",
    "synthetic_50k_gradient_mimicry.toml",
    "synthetic_50k_colluding_sign_flip.toml",
    "synthetic_50k_min_max.toml",
    "synthetic_50k_min_sum.toml",
    "synthetic_50k_adaptive_stealth.toml",
    "synthetic_50k_heterogeneity_aware_mimicry.toml",
    "synthetic_50k_targeted_family_poisoning.toml",
    "portable_model_attack.toml",
)


def _scaled(profile: dict) -> dict:
    profile = copy.deepcopy(profile)
    profile.setdefault("experiment", {})["num_clients"] = 4
    profile["experiment"]["rounds"] = 2
    dataset = profile.setdefault("dataset", {})
    dataset["samples_per_client"] = 48
    dataset["central_test_size"] = 96
    training = profile.setdefault("training", {})
    training["local_epochs"] = 1
    training["batch_size"] = min(int(training.get("batch_size", 32)), 32)
    storage = profile.setdefault("storage", {})
    storage["enabled"] = True
    storage["compression_level"] = 1
    storage["capture_inference"] = True
    storage["probe_size"] = min(int(storage.get("probe_size", 48)), 48)
    schedule = profile.setdefault("attack", {}).setdefault("schedule", {})
    schedule.update(type="continuous", start_round=1, end_round=2)
    return profile


def _expected_attackers(profile: dict) -> int:
    attack = profile.get("attack", {})
    ids = attack.get("malicious_client_ids") or []
    if ids:
        return len(ids)
    fraction = float(attack.get("malicious_fraction", 0.0))
    return 0 if fraction <= 0 else max(
        1, int(round(int(profile["experiment"]["num_clients"]) * fraction))
    )




@pytest.mark.parametrize(
    ("aggregation_name", "aggregation_config"),
    [
        ("fedavg", {"name": "fedavg"}),
        ("median", {"name": "median"}),
        ("trimmed_mean", {"name": "trimmed_mean", "trim_ratio": 0.2}),
        ("krum", {"name": "krum", "num_byzantine": 1}),
        (
            "multi_krum",
            {"name": "multi_krum", "num_byzantine": 1, "num_selected": 2},
        ),
    ],
)
def test_aggregation_backend_runs_end_to_end(
    tmp_path: Path,
    aggregation_name: str,
    aggregation_config: dict,
):
    source = PROFILE_DIR / "portable_model_attack.toml"
    profile = _scaled(load_profile(source))
    profile["experiment"]["num_clients"] = 5
    profile["experiment"]["rounds"] = 1
    profile["dataset"]["samples_per_client"] = 32
    profile["dataset"]["central_test_size"] = 64
    profile["training"]["batch_size"] = 16
    profile["attack"]["mechanism"] = "sign_flip"
    profile["attack"]["malicious_fraction"] = 0.2
    profile["attack"]["schedule"] = {
        "type": "continuous",
        "start_round": 1,
        "end_round": 1,
    }
    profile["aggregation"] = aggregation_config
    profile["storage"]["probe_size"] = 24

    profile_path = tmp_path / f"aggregation-{aggregation_name}.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    run_dir = tmp_path / f"aggregation-{aggregation_name}"
    completed = subprocess.run(
        build_command(profile_path, output_dir=run_dir),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout

    resolved = json.loads(
        (run_dir / "resolved_profile.json").read_text(encoding="utf-8")
    )
    assert resolved["aggregation"]["name"] == aggregation_name

    h5_path = run_dir / "round_state.h5"
    assert h5_path.exists(), completed.stdout
    assert not validate(h5_path)
    with h5py.File(h5_path, "r") as h5:
        assert int(h5["meta"].attrs["last_complete_round"]) == 1
        assert "round_0001" in h5["global"]
        assert len(h5["clients"]["round_0001"]) == 5




@pytest.mark.parametrize(
    ("task", "model_name", "model_config"),
    [
        ("binary", "p4p_mlp", {"name": "p4p_mlp", "dropout": 0.1}),
        ("binary", "cnn1d", {"name": "cnn1d", "dropout": 0.1}),
        (
            "binary",
            "ft_transformer",
            {
                "name": "ft_transformer",
                "d_token": 8,
                "n_heads": 2,
                "n_blocks": 1,
                "ff_factor": 2.0,
                "dropout": 0.1,
            },
        ),
        ("multiclass", "p4p_mlp", {"name": "p4p_mlp", "dropout": 0.1}),
        ("multiclass", "cnn1d", {"name": "cnn1d", "dropout": 0.1}),
        (
            "multiclass",
            "ft_transformer",
            {
                "name": "ft_transformer",
                "d_token": 8,
                "n_heads": 2,
                "n_blocks": 1,
                "ff_factor": 2.0,
                "dropout": 0.1,
            },
        ),
    ],
)
def test_model_family_runs_end_to_end_for_binary_and_multiclass(
    tmp_path: Path,
    task: str,
    model_name: str,
    model_config: dict,
):
    source = PROFILE_DIR / "portable_model_attack.toml"
    profile = _scaled(load_profile(source))
    profile["experiment"]["num_clients"] = 2
    profile["experiment"]["rounds"] = 1
    profile["dataset"]["task"] = task
    profile["dataset"]["samples_per_client"] = 32
    profile["dataset"]["central_test_size"] = 64
    profile["dataset"]["num_features"] = 8
    profile["dataset"]["stress_latent_dim"] = 4
    profile["dataset"]["stress_informative_features"] = 4
    profile["dataset"]["stress_redundant_features"] = 2
    profile["dataset"]["num_classes"] = 4 if task == "multiclass" else 2
    profile["model"] = model_config
    profile["training"]["batch_size"] = 16
    profile["attack"] = {
        "mechanism": "none",
        "malicious_fraction": 0.0,
        "schedule": {"type": "continuous", "start_round": 1, "end_round": 1},
    }
    profile["aggregation"] = {"name": "fedavg"}
    profile["defense"] = {"name": "none"}
    profile["storage"]["probe_size"] = 24

    profile_path = tmp_path / f"model-{task}-{model_name}.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    run_dir = tmp_path / f"model-{task}-{model_name}"
    completed = subprocess.run(
        build_command(profile_path, output_dir=run_dir),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout

    resolved = json.loads(
        (run_dir / "resolved_profile.json").read_text(encoding="utf-8")
    )
    assert resolved["model"]["name"] == model_name
    assert resolved["model"]["task"] == task
    if task == "multiclass":
        assert resolved["model"]["num_classes"] == 4

    h5_path = run_dir / "round_state.h5"
    assert h5_path.exists(), completed.stdout
    assert not validate(h5_path)
    with h5py.File(h5_path, "r") as h5:
        assert int(h5["meta"].attrs["last_complete_round"]) == 1
        assert "round_0001" in h5["global"]


@pytest.mark.parametrize(
    ("defense_name", "defense_config"),
    [
        ("norm_clipping", {"name": "norm_clipping", "max_norm": 1.0}),
        (
            "probe_distillation",
            {
                "name": "probe_distillation",
                "temperature": 2.0,
                "learning_rate": 0.001,
                "alpha": 0.5,
                "epochs": 1,
            },
        ),
    ],
)
def test_defense_runs_end_to_end(
    tmp_path: Path,
    defense_name: str,
    defense_config: dict,
):
    source = PROFILE_DIR / "portable_model_attack.toml"
    profile = _scaled(load_profile(source))
    profile["experiment"]["num_clients"] = 3
    profile["experiment"]["rounds"] = 1
    profile["dataset"]["samples_per_client"] = 32
    profile["dataset"]["central_test_size"] = 64
    profile["dataset"]["num_features"] = 8
    profile["dataset"]["stress_latent_dim"] = 4
    profile["dataset"]["stress_informative_features"] = 4
    profile["dataset"]["stress_redundant_features"] = 2
    profile["model"] = {
        "name": "stress_mlp",
        "hidden1": 8,
        "hidden2": 4,
        "weight_decay": 0.0,
    }
    profile["training"]["batch_size"] = 16
    profile["attack"] = {
        "mechanism": "sign_flip",
        "malicious_fraction": 1.0 / 3.0,
        "strength": 2.0,
        "schedule": {"type": "continuous", "start_round": 1, "end_round": 1},
    }
    profile["aggregation"] = {"name": "fedavg"}
    profile["defense"] = defense_config
    profile["storage"]["probe_size"] = 24
    profile["storage"]["capture_inference"] = True
    profile["storage"]["capture_logits"] = True
    profile["storage"]["capture_probe_features"] = True

    profile_path = tmp_path / f"defense-{defense_name}.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    run_dir = tmp_path / f"defense-{defense_name}"
    completed = subprocess.run(
        build_command(profile_path, output_dir=run_dir),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout

    h5_path = run_dir / "round_state.h5"
    assert h5_path.exists(), completed.stdout
    assert not validate(h5_path)
    with h5py.File(h5_path, "r") as h5:
        metadata = h5["rounds"]["round_0001"].attrs
        recorded = metadata["defense"]
        if isinstance(recorded, bytes):
            recorded = recorded.decode("utf-8")
        assert str(recorded) == defense_name
        clients = h5["clients"]["round_0001"]
        malicious = [
            client
            for client in clients.values()
            if bool(int(client.attrs.get("malicious", 0)))
        ]
        assert malicious
        assert "submitted_update" in malicious[0]
        assert "post_defense_update" in malicious[0]
        assert int(h5["meta"].attrs["last_complete_round"]) == 1


@pytest.mark.parametrize("profile_name", PROFILE_CASES)
def test_profile_runs_without_hydra(tmp_path: Path, profile_name: str):
    source = PROFILE_DIR / profile_name
    profile = _scaled(load_profile(source))
    expected_attackers = _expected_attackers(profile)
    mechanism = str(profile.get("attack", {}).get("mechanism", "none"))

    profile_path = tmp_path / (source.stem + ".json")
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    run_dir = tmp_path / source.stem
    completed = subprocess.run(
        build_command(profile_path, output_dir=run_dir),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout
    assert ".hydra" not in completed.stdout
    assert (run_dir / "resolved_profile.json").exists()

    h5_path = run_dir / str(profile["storage"].get("path", "round_state.h5"))
    assert h5_path.exists(), completed.stdout
    assert not validate(h5_path)

    with h5py.File(h5_path, "r") as h5:
        assert int(h5["meta"].attrs["last_complete_round"]) == 2
        clients = h5["clients"]["round_0002"]
        assert len(clients) == 4
        malicious = [
            c for c in clients.values()
            if bool(int(c.attrs.get("malicious", 0)))
        ]
        assert len(malicious) == expected_attackers
        if mechanism == "none":
            assert not malicious
        else:
            assert malicious
            assert any(bool(int(c.attrs.get("attack_active", 0))) for c in malicious)
