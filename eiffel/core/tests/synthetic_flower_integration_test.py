"""End-to-end smoke test for TOML -> Hydra -> Flower -> HDF5/JSON."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import h5py

from eiffel.toml_runner import build_command


def test_synthetic_flower_one_round_persists_metrics(tmp_path: Path):
    profile = tmp_path / "synthetic_e2e.toml"
    run_dir = tmp_path / "run"
    profile.write_text(
        """
[experiment]
name = "synthetic-e2e"
seed = 2026
num_clients = 1
rounds = 1

[dataset]
name = "synthetic_stress"
task = "binary"
samples_per_client = 128
central_test_size = 256
num_features = 8
num_classes = 4
rare_class_id = 3
rare_class_probability = 0.05
rare_specialist_client = 0
rare_specialist_strength = 0.10
feature_noise = 0.40
stress_latent_dim = 4
stress_informative_features = 6
stress_redundant_features = 2
stress_class_separation = 1.5
stress_latent_noise = 0.7
stress_secondary_mode_probability = 0.2
stress_hard_example_fraction = 0.05
stress_train_label_noise = 0.0
stress_client_shift_std = 0.05
stress_central_shift_std = 0.05
stress_outlier_fraction = 0.0

[partition]
type = "iid"

[model]
name = "stress_mlp"
hidden1 = 16
hidden2 = 8
weight_decay = 0.0

[training]
local_epochs = 1
learning_rate = 0.001
batch_size = 32

[attack]
mechanism = "none"
malicious_fraction = 0.0

[aggregation]
name = "fedavg"

[storage]
enabled = true
path = "round_state.h5"
compression = "gzip"
compression_level = 1
flush_each_round = true
capture_inference = true
probe_size = 32
""".strip(),
        encoding="utf-8",
    )

    command = build_command(
        profile,
        extra=[
            f"hydra.run.dir={run_dir.as_posix()}",
            "hydra.output_subdir=.hydra",
        ],
    )
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout

    h5_path = run_dir / "round_state.h5"
    assert h5_path.exists(), completed.stdout
    with h5py.File(h5_path, "r") as h5:
        assert "clients" in h5, completed.stdout
        assert "round_0001" in h5["clients"], completed.stdout
        clients = list(h5["clients"]["round_0001"].keys())
        assert clients, completed.stdout
        first_client = h5["clients"]["round_0001"][clients[0]]
        assert "submitted_update" in first_client
        assert "metrics" in first_client
        assert "fit" in first_client["metrics"]
        assert "evaluate" in first_client["metrics"]

    fit = json.loads((run_dir / "fit.json").read_text(encoding="utf-8"))
    distributed = json.loads(
        (run_dir / "distributed.json").read_text(encoding="utf-8")
    )
    assert fit, completed.stdout
    assert distributed, completed.stdout
