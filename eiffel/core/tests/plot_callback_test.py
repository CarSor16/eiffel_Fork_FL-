"""Tests for the repaired Hydra plotting callback."""

from eiffel.callbacks.plot import PlotCallback


def test_plot_callback_reads_nested_eiffel_global_metrics(tmp_path):
    output = tmp_path / "plot.png"
    metrics = {
        "benign_0": {
            "1": {"global": {"accuracy": 0.80}},
            "2": {"global": {"accuracy": 0.84}},
            "3": {"global": {"accuracy": 0.87}},
        },
        "malicious_0": {
            "1": {"global": {"accuracy": 0.72}},
            "2": {"global": {"accuracy": 0.68}},
            "3": {"global": {"accuracy": 0.63}},
        },
    }

    callback = PlotCallback(
        input="fit.json",
        output=str(output),
        metric="accuracy",
    )
    callback._plot(metrics, str(output))

    assert output.exists()
    assert output.stat().st_size > 0
