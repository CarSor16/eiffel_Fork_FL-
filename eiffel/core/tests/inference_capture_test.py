"""Tests for explicit probability/logit capture."""

import numpy as np
import pytest
from tensorflow import keras

from eiffel.core.client import predict_probabilities_and_logits


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
