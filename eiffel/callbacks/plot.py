"""Plotting Hydra callbacks for Eiffel."""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig
from scipy.interpolate import BSpline, make_interp_spline


class PlotterCallback(Callback):
    """Plot callback.

    This class implements Hydra's callback mechanism to plot the FL runs.

    For reference, see: https://hydra.cc/docs/experimental/callbacks/.
    """

    def __init__(self, output: str, input: str = "fit.json") -> None:
        self.input = input
        self.output = output

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        """Call when a job ends."""
        try:
            f = Path(self.input).read_text()
            metrics = json.loads(f)
            self._plot(metrics, output=self.output)
        except FileNotFoundError:
            # Failed/interrupted jobs may end before Eiffel writes result JSON.
            # The original experiment exception is the useful diagnostic in that case.
            return

    def _plot(self, metrics: dict, output: str) -> None:
        """Generate the plot."""
        raise NotImplementedError


class PlotCallback(PlotterCallback):
    """Plot callback.

    This callback plots a selected metric for each run. By default, it plots the mean of
    the selected metric in blue and, if attackers are present, the mean of the same
    metric for the attackers in red.
    """

    def __init__(
        self,
        *,
        metric: str = "accuracy",
        smooth: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metric = metric
        self.smooth = smooth

    def _plot(self, metrics: dict, output: str) -> None:
        """Generate the plot from Eiffel fit/distributed result JSON."""
        benign_by_round: dict[int, list[float]] = defaultdict(list)
        attacker_by_round: dict[int, list[float]] = defaultdict(list)

        for cid, cmetrics in metrics.items():
            if not isinstance(cmetrics, dict):
                continue
            target = attacker_by_round if "malicious" in str(cid) else benign_by_round
            for round_key, round_metrics in cmetrics.items():
                if not isinstance(round_metrics, dict):
                    continue
                value = round_metrics.get(self.metric)
                if value is None:
                    global_metrics = round_metrics.get("global", {})
                    if isinstance(global_metrics, dict):
                        value = global_metrics.get(self.metric)
                if not isinstance(value, (int, float)):
                    continue
                try:
                    round_number = int(round_key)
                except (TypeError, ValueError):
                    continue
                target[round_number].append(float(value))

        if not benign_by_round:
            return

        rounds = sorted(benign_by_round)
        benign_metrics = [
            float(np.mean(benign_by_round[round_number]))
            for round_number in rounds
        ]
        benign_plot = (rounds, benign_metrics)

        attacker_rounds = sorted(attacker_by_round)
        attacker_metrics = [
            float(np.mean(attacker_by_round[round_number]))
            for round_number in attacker_rounds
        ]
        attacker_plot = (attacker_rounds, attacker_metrics)

        if self.smooth and len(rounds) >= 3:
            # 300 represents number of points to make between min and max.
            lin_x = np.linspace(min(rounds), max(rounds), 300)
            benign_spl = make_interp_spline(
                rounds, benign_metrics, k=2
            )  # type: BSpline
            benign_plot = (lin_x, benign_spl(lin_x))

            if len(attacker_rounds) >= 3:
                attacker_x = np.linspace(
                    min(attacker_rounds), max(attacker_rounds), 300
                )
                attacker_spl = make_interp_spline(
                    attacker_rounds, attacker_metrics, k=2
                )
                attacker_plot = (attacker_x, attacker_spl(attacker_x))

        plt.figure()
        plt.title(f"Mean {self.metric}")
        plt.xlabel("Rounds")
        plt.ylabel(self.metric.title())
        plt.plot(*benign_plot, label="Benign")
        if attacker_metrics:
            plt.plot(*attacker_plot, label="Attacker")
        plt.legend()
        plt.savefig(Path(output))
        plt.close()
