"""Tests for explicit probability/logit capture."""

import numpy as np
import pandas as pd
import pytest
from tensorflow import keras

from eiffel.core.client import EiffelClient, predict_probabilities_and_logits
from eiffel.models.advanced import (
    mk_cnn1d,
    mk_ft_transformer,
    mk_p4p_mlp,
    mk_stress_mlp,
)
from eiffel.models.supervized import mk_popoola_mlp
from eiffel.storage import decode_array


def test_binary_probabilities_match_sigmoid_of_exact_logits():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    kernel = np.array([[1.25], [-0.75]], dtype=np.float32)
    bias = np.array([0.20], dtype=np.float32)
    model.layers[-1].set_weights([kernel, bias])
    x = np.array([[1.0, 2.0], [-1.0, 0.5], [0.0, 0.0]], dtype=np.float32)

    probabilities, logits = predict_probabilities_and_logits(
        model,
        x,
        batch_size=2,
    )

    expected_logits = x @ kernel + bias
    expected_probabilities = 1.0 / (1.0 + np.exp(-expected_logits))
    np.testing.assert_allclose(logits, expected_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        probabilities,
        expected_probabilities,
        rtol=1e-6,
        atol=1e-6,
    )


def test_multiclass_probabilities_match_softmax_of_exact_logits():
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(3, activation="softmax"),
        ]
    )
    kernel = np.array(
        [[1.0, -0.5, 0.25], [0.2, 0.7, -1.0]],
        dtype=np.float32,
    )
    bias = np.array([0.10, -0.20, 0.30], dtype=np.float32)
    model.layers[-1].set_weights([kernel, bias])
    x = np.array([[1.0, 2.0], [-0.5, 0.25]], dtype=np.float32)

    probabilities, logits = predict_probabilities_and_logits(
        model,
        x,
        batch_size=2,
    )

    expected_logits = x @ kernel + bias
    shifted = expected_logits - expected_logits.max(axis=1, keepdims=True)
    expected_probabilities = np.exp(shifted) / np.exp(shifted).sum(
        axis=1,
        keepdims=True,
    )
    np.testing.assert_allclose(logits, expected_logits, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        probabilities,
        expected_probabilities,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(len(x)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_logit_capture_fails_fast_for_unsupported_output_layer():
    inputs = keras.Input(shape=(2,))
    outputs = keras.layers.Activation("sigmoid")(inputs)
    model = keras.Model(inputs, outputs)

    with pytest.raises(ValueError, match="final model layer"):
        predict_probabilities_and_logits(
            model,
            np.zeros((2, 2), dtype=np.float32),
            batch_size=2,
        )


@pytest.mark.parametrize("task,num_classes", [("binary", 2), ("multiclass", 4)])
def test_supported_models_expose_consistent_probabilities_and_logits(
    task,
    num_classes,
):
    builders = (
        lambda: mk_popoola_mlp(8, task=task, num_classes=num_classes),
        lambda: mk_p4p_mlp(8, task=task, num_classes=num_classes),
        lambda: mk_cnn1d(8, task=task, num_classes=num_classes),
        lambda: mk_ft_transformer(
            8,
            task=task,
            num_classes=num_classes,
            d_token=8,
            n_heads=2,
            n_blocks=1,
        ),
        lambda: mk_stress_mlp(8, task=task, num_classes=num_classes),
    )
    x = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(3, 8)

    for build in builders:
        model = build()
        probabilities, logits = predict_probabilities_and_logits(
            model,
            x,
            batch_size=3,
        )
        assert probabilities.shape == logits.shape
        assert probabilities.shape[0] == len(x)
        if task == "binary":
            assert probabilities.shape[1] == 1
            expected = 1.0 / (1.0 + np.exp(-logits))
        else:
            assert probabilities.shape[1] == num_classes
            shifted = logits - logits.max(axis=1, keepdims=True)
            expected = np.exp(shifted) / np.exp(shifted).sum(
                axis=1,
                keepdims=True,
            )
        np.testing.assert_allclose(
            probabilities,
            expected,
            rtol=2e-5,
            atol=2e-5,
        )


def test_disabling_logits_allows_non_dense_probability_head():
    inputs = keras.Input(shape=(2,))
    dense = keras.layers.Dense(1)(inputs)
    outputs = keras.layers.Activation("sigmoid")(dense)
    model = keras.Model(inputs, outputs)

    class Probe:
        X = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]])
        y = pd.Series([0, 1])
        m = pd.DataFrame({"Attack": ["Benign", "Threat-X"]})

        def __len__(self):
            return len(self.X)

    client = EiffelClient(
        "probe_benign_0",
        None,
        model,
        seed=2026,
        eval_fit=False,
    )
    payload = client._capture_probe_payload(
        Probe(),
        {
            "batch_size": 2,
            "probe_size": 2,
            "capture_logits": False,
            "capture_probe_features": True,
        },
    )

    assert "_eiffel_probabilities" in payload
    assert "_eiffel_logits" not in payload
    probabilities = decode_array(payload["_eiffel_probabilities"])
    assert probabilities.shape == (2, 1)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
