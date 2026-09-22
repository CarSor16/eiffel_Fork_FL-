"""Additional TensorFlow/Keras models for FL-NIDS experiments."""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from keras.losses import BinaryCrossentropy, Loss
from keras.optimizers import Adam, Optimizer


def _compile(
    model: keras.Model,
    loss_fn: Optional[Loss],
    optimizer: Optional[Optimizer],
    learning_rate: float,
) -> keras.Model:
    model.compile(
        optimizer=optimizer or Adam(learning_rate=learning_rate),
        loss=loss_fn or BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def mk_p4p_mlp(
    n_features: int,
    loss_fn: Optional[Loss] = None,
    optimizer: Optional[Optimizer] = None,
    learning_rate: float = 0.001,
    dropout: float = 0.2,
) -> keras.Model:
    """MLP 128-64 with dropout, matching the family used in recent FL-NIDS work."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return _compile(model, loss_fn, optimizer, learning_rate)


def mk_cnn1d(
    n_features: int,
    loss_fn: Optional[Loss] = None,
    optimizer: Optional[Optimizer] = None,
    learning_rate: float = 0.001,
    dropout: float = 0.2,
) -> keras.Model:
    """Lightweight 1D CNN for flow-feature intrusion detection."""
    inputs = keras.Input(shape=(n_features,))
    x = keras.layers.Reshape((n_features, 1))(inputs)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return _compile(keras.Model(inputs, outputs, name="eiffel_cnn1d"), loss_fn, optimizer, learning_rate)


class FeatureTokenizer(keras.layers.Layer):
    """Map each scalar feature to an independent learned token."""

    def __init__(self, d_token: int, **kwargs):
        super().__init__(**kwargs)
        self.d_token = int(d_token)

    def build(self, input_shape):
        n_features = int(input_shape[-1])
        self.weight = self.add_weight(
            name="tokenizer_weight",
            shape=(n_features, self.d_token),
            initializer="he_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="tokenizer_bias",
            shape=(n_features, self.d_token),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=-1) * self.weight + self.bias


class CLSToken(keras.layers.Layer):
    """Prepend one learned CLS token to a batch of feature tokens."""

    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls_token",
            shape=(1, 1, int(input_shape[-1])),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        cls = tf.tile(self.cls, [batch, 1, 1])
        return tf.concat([cls, inputs], axis=1)


def mk_ft_transformer(
    n_features: int,
    loss_fn: Optional[Loss] = None,
    optimizer: Optional[Optimizer] = None,
    learning_rate: float = 0.001,
    d_token: int = 64,
    n_heads: int = 4,
    n_blocks: int = 2,
    ff_factor: float = 2.0,
    dropout: float = 0.1,
) -> keras.Model:
    """Compact FT-Transformer for numeric network-flow features."""
    if d_token % n_heads != 0:
        raise ValueError("d_token must be divisible by n_heads")

    inputs = keras.Input(shape=(n_features,))
    x = FeatureTokenizer(d_token)(inputs)
    x = CLSToken()(x)

    for block in range(int(n_blocks)):
        norm = keras.layers.LayerNormalization(name=f"block_{block}_ln1")(x)
        attn = keras.layers.MultiHeadAttention(
            num_heads=int(n_heads),
            key_dim=int(d_token // n_heads),
            dropout=float(dropout),
            name=f"block_{block}_mha",
        )(norm, norm)
        x = x + keras.layers.Dropout(dropout)(attn)

        norm = keras.layers.LayerNormalization(name=f"block_{block}_ln2")(x)
        ff = keras.layers.Dense(int(d_token * ff_factor), activation=tf.nn.gelu)(norm)
        ff = keras.layers.Dropout(dropout)(ff)
        ff = keras.layers.Dense(d_token)(ff)
        x = x + keras.layers.Dropout(dropout)(ff)

    cls = keras.layers.Lambda(lambda t: t[:, 0, :], name="take_cls")(x)
    cls = keras.layers.LayerNormalization(name="final_ln")(cls)
    outputs = keras.layers.Dense(1, activation="sigmoid")(cls)
    return _compile(
        keras.Model(inputs, outputs, name="eiffel_ft_transformer"),
        loss_fn,
        optimizer,
        learning_rate,
    )
