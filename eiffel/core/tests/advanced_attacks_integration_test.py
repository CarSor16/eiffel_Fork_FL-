"""Small end-to-end Flower checks for representative advanced attacks."""

from __future__ import annotations

import subprocess
from pathlib import Path

import h5py
import pytest
from omegaconf import OmegaConf

from eiffel.toml_runner import build_command


RAY_STARTUP_TIMEOUT_MARKERS = (
    "Timed out after 60 seconds while waiting for node to startup",
    "The current node timed out during startup",
)


def _run_eiffel_with_ray_startup_retry(
    command: list[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    """Run Eiffel, retrying once only for Ray's transient local startup timeout."""
    attempts = 2
    completed = None
    for attempt in range(attempts):
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        if completed.returncode == 0:
            return completed
        ray_startup_timeout = any(
            marker in completed.stdout
            for marker in RAY_STARTUP_TIMEOUT_MARKERS
        )
        if not ray_startup_timeout or attempt == attempts - 1:
            return completed
    assert completed is not None  # pragma: no cover
    return completed


def _profile_text(mechanism: str) -> str:
    if mechanism == "min_max":
        attack = """
[attack]
mechanism = "min_max"
malicious_fraction = 0.25

[attack.min_max]
direction = "sign"
search_steps = 12
max_lambda = 4.0
constraint_margin = 1.05
"""
    elif mechanism == "targeted_family_poisoning":
        attack = """
[attack]
mechanism = "targeted_family_poisoning"
malicious_fraction = 0.25

[attack.targeted]
target_family = "Scan"
target_amplification = 1.5
target_mimicry_lambda = 0.2
"""
    else:  # pragma: no cover - test construction guard
        raise AssertionError(mechanism)

    return f"""
[experiment]
name = "advanced-e2e-{mechanism}"
seed = 2026
num_clients = 4
rounds = 1

[dataset]
name = "synthetic_stress"
task = "binary"
samples_per_client = 64
central_test_size = 192
num_features = 8
num_classes = 2
rare_class_id = 1
rare_class_probability = 0.35
rare_specialist_client = 0
rare_specialist_strength = 0.10
feature_noise = 0.35
stress_latent_dim = 4
stress_informative_features = 6
stress_redundant_features = 2
stress_class_separation = 1.5
stress_latent_noise = 0.6
stress_secondary_mode_probability = 0.15
stress_hard_example_fraction = 0.03
stress_train_label_noise = 0.0
stress_client_shift_std = 0.04
stress_central_shift_std = 0.04
stress_outlier_fraction = 0.0

[partition]
type = "iid"

[model]
name = "stress_mlp"
hidden1 = 12
hidden2 = 6
weight_decay = 0.0

[training]
local_epochs = 1
learning_rate = 0.001
batch_size = 16

{attack}

[attack.schedule]
type = "continuous"

[aggregation]
name = "fedavg"

[storage]
enabled = true
path = "round_state.h5"
compression = "gzip"
compression_level = 1
flush_each_round = true
capture_inference = true
probe_size = 96
""".strip()



def _label_flip_profile_text() -> str:
    return """
[experiment]
name = "label-flip-e2e"
seed = 2026
num_clients = 4
rounds = 2

[dataset]
name = "synthetic_stress"
task = "binary"
samples_per_client = 48
central_test_size = 96
num_features = 8
num_classes = 2
rare_class_id = 1
rare_class_probability = 0.35
rare_specialist_client = 0
rare_specialist_strength = 0.10
feature_noise = 0.35
stress_latent_dim = 4
stress_informative_features = 6
stress_redundant_features = 2
stress_class_separation = 1.5
stress_latent_noise = 0.6
stress_secondary_mode_probability = 0.15
stress_hard_example_fraction = 0.03
stress_train_label_noise = 0.0
stress_client_shift_std = 0.04
stress_central_shift_std = 0.04
stress_outlier_fraction = 0.0

[partition]
type = "iid"

[model]
name = "stress_mlp"
hidden1 = 12
hidden2 = 6
weight_decay = 0.0

[training]
local_epochs = 1
learning_rate = 0.001
batch_size = 16

[attack]
mechanism = "label_flip"
malicious_fraction = 0.25
poison_rate = 0.5
objective = "untargeted"

[attack.schedule]
type = "late"
start_round = 2
end_round = 2

[aggregation]
name = "fedavg"

[storage]
enabled = true
path = "round_state.h5"
compression = "gzip"
compression_level = 1
flush_each_round = true
capture_inference = false
""".strip()




def _targeted_label_flip_without_local_target_profile_text() -> str:
    return """
[experiment]
name = "label-flip-no-local-target-e2e"
seed = 2026
num_clients = 2
rounds = 1

[dataset]
name = "synthetic_stress"
task = "binary"
samples_per_client = 32
central_test_size = 64
num_features = 8
num_classes = 2
rare_class_id = 1
rare_class_probability = 0.35
rare_specialist_client = 0
rare_specialist_strength = 0.10
feature_noise = 0.35
stress_latent_dim = 4
stress_informative_features = 6
stress_redundant_features = 2
stress_class_separation = 1.5
stress_latent_noise = 0.6
stress_secondary_mode_probability = 0.15
stress_hard_example_fraction = 0.03
stress_train_label_noise = 0.0
stress_client_shift_std = 0.04
stress_central_shift_std = 0.04
stress_outlier_fraction = 0.0

[partition]
type = "iid"

[model]
name = "stress_mlp"
hidden1 = 8
hidden2 = 4
weight_decay = 0.0

[training]
local_epochs = 1
learning_rate = 0.001
batch_size = 16

[attack]
mechanism = "label_flip"
malicious_fraction = 0.5
poison_rate = 1.0
objective = "targeted"
target = ["No-Such-Family"]

[attack.schedule]
type = "continuous"

[aggregation]
name = "fedavg"

[storage]
enabled = true
path = "round_state.h5"
compression = "gzip"
compression_level = 1
flush_each_round = true
capture_inference = false
""".strip()


@pytest.mark.parametrize(
    "mechanism",
    ["min_max", "targeted_family_poisoning"],
)
def test_advanced_attack_runs_through_toml_hydra_flower_and_hdf5(
    tmp_path: Path,
    mechanism: str,
):
    profile = tmp_path / f"{mechanism}.toml"
    run_dir = tmp_path / f"run-{mechanism}"
    profile.write_text(_profile_text(mechanism), encoding="utf-8")

    command = build_command(
        profile,
        extra=[
            f"hydra.run.dir={run_dir.as_posix()}",
            "hydra.output_subdir=.hydra",
        ],
    )
    completed = _run_eiffel_with_ray_startup_retry(
        command,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert completed.returncode == 0, completed.stdout

    hydra_config = run_dir / ".hydra" / "config.yaml"
    assert hydra_config.exists(), completed.stdout
    cfg = OmegaConf.load(hydra_config)
    assert int(cfg.pools[0].n_benign) == 3, OmegaConf.to_yaml(cfg.pools)
    assert int(cfg.pools[0].n_malicious) == 1, OmegaConf.to_yaml(cfg.pools)

    h5_path = run_dir / "round_state.h5"
    assert h5_path.exists(), completed.stdout
    with h5py.File(h5_path, "r") as h5:
        clients = h5["clients"]["round_0001"]
        assert len(clients) == 4
        malicious = [
            client
            for client in clients.values()
            if bool(int(client.attrs.get("malicious", 0)))
        ]
        client_debug = {
            cid: {
                "attrs": {
                    str(key): (
                        value.item()
                        if hasattr(value, "item")
                        else value
                    )
                    for key, value in client.attrs.items()
                },
                "keys": list(client.keys()),
            }
            for cid, client in clients.items()
        }
        assert len(malicious) == 1, client_debug
        assert bool(int(malicious[0].attrs.get("attack_active", 0)))
        assert "submitted_update" in malicious[0]
        assert "pre_attack_update" in malicious[0]
        assert "audit" in malicious[0]

        metadata = h5["rounds"]["round_0001"].attrs
        recorded = metadata["attack_mechanism"]
        if isinstance(recorded, bytes):
            recorded = recorded.decode("utf-8")
        assert str(recorded) == mechanism
        assert int(metadata["malicious_clients"]) == 1



def test_label_flip_schedule_is_persisted_end_to_end(tmp_path: Path):
    profile = tmp_path / "label_flip.toml"
    run_dir = tmp_path / "run-label-flip"
    profile.write_text(_label_flip_profile_text(), encoding="utf-8")

    command = build_command(
        profile,
        extra=[
            f"hydra.run.dir={run_dir.as_posix()}",
            "hydra.output_subdir=.hydra",
        ],
    )
    completed = _run_eiffel_with_ray_startup_retry(
        command,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert completed.returncode == 0, completed.stdout

    h5_path = run_dir / "round_state.h5"
    assert h5_path.exists(), completed.stdout
    with h5py.File(h5_path, "r") as h5:
        for round_number, expected_active, expected_fraction in (
            (1, False, 0.0),
            (2, True, 0.5),
        ):
            round_name = f"round_{round_number:04d}"
            clients = h5["clients"][round_name]
            malicious = [
                client
                for client in clients.values()
                if bool(int(client.attrs.get("malicious", 0)))
            ]
            assert len(malicious) == 1
            attacker = malicious[0]
            assert bool(int(attacker.attrs["attack_active"])) is expected_active
            assert float(attacker.attrs["data_poison_fraction"]) == pytest.approx(
                expected_fraction
            )
            if expected_active:
                assert float(
                    attacker.attrs["data_poison_effective_fraction"]
                ) == pytest.approx(0.5)
            else:
                assert "data_poison_effective_fraction" not in attacker.attrs
            assert "submitted_update" in attacker
            assert "pre_attack_update" not in attacker

            mechanism = attacker.attrs["mechanism"]
            if isinstance(mechanism, bytes):
                mechanism = mechanism.decode("utf-8")
            assert str(mechanism) == (
                "label_flip" if expected_active else "none"
            )

            metadata = h5["rounds"][round_name].attrs
            recorded = metadata["attack_mechanism"]
            if isinstance(recorded, bytes):
                recorded = recorded.decode("utf-8")
            assert str(recorded) == "label_flip"
            assert float(metadata["attack_multiplier"]) == pytest.approx(
                expected_fraction
            )
            assert int(metadata["malicious_clients"]) == 1

def test_targeted_label_flip_without_local_target_is_not_marked_active(
    tmp_path: Path,
):
    profile = tmp_path / "label_flip_no_target.toml"
    run_dir = tmp_path / "run-label-flip-no-target"
    profile.write_text(
        _targeted_label_flip_without_local_target_profile_text(),
        encoding="utf-8",
    )

    command = build_command(
        profile,
        extra=[
            f"hydra.run.dir={run_dir.as_posix()}",
            "hydra.output_subdir=.hydra",
        ],
    )
    completed = _run_eiffel_with_ray_startup_retry(
        command,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert completed.returncode == 0, completed.stdout

    with h5py.File(run_dir / "round_state.h5", "r") as h5:
        clients = h5["clients"]["round_0001"]
        malicious = [
            client
            for client in clients.values()
            if bool(int(client.attrs.get("malicious", 0)))
        ]
        assert len(malicious) == 1
        attacker = malicious[0]

        assert not bool(int(attacker.attrs["attack_active"]))
        assert float(attacker.attrs["data_poison_fraction"]) == pytest.approx(1.0)
        assert float(
            attacker.attrs["data_poison_effective_fraction"]
        ) == pytest.approx(0.0)

        mechanism = attacker.attrs["mechanism"]
        if isinstance(mechanism, bytes):
            mechanism = mechanism.decode("utf-8")
        assert str(mechanism) == "none"

        metadata = h5["rounds"]["round_0001"].attrs
        recorded = metadata["attack_mechanism"]
        if isinstance(recorded, bytes):
            recorded = recorded.decode("utf-8")
        assert str(recorded) == "label_flip"
        assert float(metadata["attack_multiplier"]) == pytest.approx(0.0)

