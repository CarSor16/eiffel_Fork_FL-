"""Tests for binary and experimental multiclass neural heads."""

from keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy

from eiffel.models.advanced import (
    mk_cnn1d,
    mk_ft_transformer,
    mk_p4p_mlp,
    mk_stress_mlp,
)
from eiffel.models.supervized import mk_popoola_mlp


def test_binary_heads_remain_single_sigmoid_outputs():
    builders = (
        lambda: mk_popoola_mlp(8, task="binary"),
        lambda: mk_p4p_mlp(8, task="binary"),
        lambda: mk_cnn1d(8, task="binary"),
        lambda: mk_ft_transformer(
            8,
            task="binary",
            d_token=8,
            n_heads=2,
            n_blocks=1,
        ),
        lambda: mk_stress_mlp(8, task="binary"),
    )

    for build in builders:
        model = build()
        assert model.output_shape[-1] == 1
        assert isinstance(model.loss, BinaryCrossentropy)


def test_multiclass_heads_use_softmax_width_and_sparse_cross_entropy():
    builders = (
        lambda: mk_popoola_mlp(8, task="multiclass", num_classes=4),
        lambda: mk_p4p_mlp(8, task="multiclass", num_classes=4),
        lambda: mk_cnn1d(8, task="multiclass", num_classes=4),
        lambda: mk_ft_transformer(
            8,
            task="multiclass",
            num_classes=4,
            d_token=8,
            n_heads=2,
            n_blocks=1,
        ),
        lambda: mk_stress_mlp(8, task="multiclass", num_classes=4),
    )

    for build in builders:
        model = build()
        assert model.output_shape[-1] == 4
        assert isinstance(model.loss, SparseCategoricalCrossentropy)
        assert model.layers[-1].activation.__name__ == "softmax"
