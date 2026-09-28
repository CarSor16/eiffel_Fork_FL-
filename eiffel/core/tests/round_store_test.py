"""Tests for round-by-round HDF5 storage and validation."""

import h5py
import numpy as np

from eiffel.analysis.validate_round_state import validate
from eiffel.storage.round_store import RoundStore


def test_round_store_persists_reconstructable_state(tmp_path):
    path = tmp_path / "round_state.h5"
    global_before = [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    ]
    pre_attack = [
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([0.05], dtype=np.float32),
    ]
    submitted = [
        np.array([-0.2, -0.4], dtype=np.float32),
        np.array([-0.1], dtype=np.float32),
    ]
    global_after = [
        global_before[0] + submitted[0] / 2.0,
        global_before[1] + submitted[1] / 2.0,
    ]

    with RoundStore(path, enabled=True, compression="gzip") as store:
        store.save_global(0, global_before)
        store.save_client(
            1,
            "malicious_0",
            submitted_update=submitted,
            pre_attack_update=pre_attack,
            audit={"l2_norm": 0.5, "cosine_to_mean": -0.2},
            probabilities=np.array([[0.2], [0.8]], dtype=np.float32),
            logits=np.array([[-1.3862944], [1.3862944]], dtype=np.float32),
            probe_features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            probe_labels=np.array([0, 1], dtype=np.int64),
            probe_families=["Benign", "Botnet"],
            malicious=True,
            attack_active=True,
            mechanism="sign_flip",
        )
        store.save_client_metrics(
            1,
            "malicious_0",
            {
                "global": {"accuracy": 0.75, "macro_f1": 0.72},
                "Botnet": {"recall": 0.60, "missrate": 0.40},
            },
            phase="fit",
        )
        store.save_round_metadata(
            1,
            attack_mechanism="sign_flip",
            attack_multiplier=1.0,
            malicious_clients=1,
        )
        store.save_global(1, global_after)
        store.save_global_inference(
            1,
            "malicious_0",
            probabilities=np.array([[0.25], [0.75]], dtype=np.float32),
            logits=np.array([[-1.0986123], [1.0986123]], dtype=np.float32),
        )
        store.mark_round_complete(1)

    assert validate(path) == []

    with h5py.File(path, "r") as h5:
        assert h5.attrs["format"] == "eiffel-round-state"
        assert int(h5["meta"].attrs["last_complete_round"]) == 1
        client = h5["clients"]["round_0001"]["malicious_0"]
        assert bool(client.attrs["malicious"])
        assert bool(client.attrs["attack_active"])
        assert client.attrs["mechanism"] == "sign_flip"
        assert client["inference"].dtype == np.float16
        assert client["probabilities"].dtype == np.float16
        assert client["logits"].dtype == np.float16
        assert client["inference"].id == client["probabilities"].id
        assert h5["probe"]["features"].dtype == np.float32
        assert h5["probe"]["labels"].dtype == np.int16
        assert "families" in h5["probe"]
        assert "global_inference" in h5
        global_inf = h5["global_inference"]["round_0001"]["malicious_0"]
        assert global_inf["probabilities"].dtype == np.float16
        assert global_inf["logits"].dtype == np.float16
        assert (
            client["metrics"]["fit"].attrs["global.accuracy"]
            == np.float32(0.75)
        )

        previous = h5["global"]["round_0000"]["weights"]["layer_000"][:]
        delta = client["submitted_update"]["layer_000"][:]
        reconstructed = previous + delta
        np.testing.assert_allclose(
            reconstructed,
            global_before[0] + submitted[0],
        )


def test_validator_detects_missing_pre_attack_update(tmp_path):
    path = tmp_path / "invalid.h5"
    weights = [np.array([1.0], dtype=np.float32)]
    update = [np.array([-1.0], dtype=np.float32)]

    with RoundStore(path) as store:
        store.save_global(0, weights)
        store.save_client(
            1,
            "malicious_0",
            submitted_update=update,
            malicious=True,
            attack_active=True,
            mechanism="sign_flip",
        )
        store.save_global(1, weights)
        store.mark_round_complete(1)

    errors = validate(path)
    assert any("pre_attack_update" in error for error in errors)



def test_global_inference_deduplicates_identical_payloads(tmp_path):
    path = tmp_path / "dedup.h5"
    probabilities = np.array([[0.2], [0.8]], dtype=np.float32)
    logits = np.array([[-1.3862944], [1.3862944]], dtype=np.float32)

    with RoundStore(path) as store:
        store.save_global_inference(
            1,
            "client_0",
            probabilities=probabilities,
            logits=logits,
        )
        store.save_global_inference(
            1,
            "client_1",
            probabilities=probabilities.copy(),
            logits=logits.copy(),
        )
        store.save_global_inference(
            1,
            "client_2",
            probabilities=np.array([[0.3], [0.7]], dtype=np.float32),
            logits=np.array([[-0.8472978], [0.8472978]], dtype=np.float32),
        )

    with h5py.File(path, "r") as h5:
        round_group = h5["global_inference"]["round_0001"]
        first = round_group["client_0"]
        same = round_group["client_1"]
        different = round_group["client_2"]

        assert first["probabilities"].id == same["probabilities"].id
        assert first["logits"].id == same["logits"].id
        assert first["probabilities"].id != different["probabilities"].id
        assert first["logits"].id != different["logits"].id
        assert same["inference"].id == same["probabilities"].id
