"""Eiffel client API."""

import itertools
import json
import logging
from collections import Counter
from copy import deepcopy
from functools import reduce
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd
import ray
from flwr.client import Client, NumPyClient
from flwr.common import Config, Scalar
from flwr.simulation.ray_transport.utils import enable_tf_gpu_growth
from keras.callbacks import History
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from eiffel.datasets.dataset import Dataset, DatasetHandle
from eiffel.datasets.poisoning import PoisonIns, PoisonTask
from eiffel.utils import set_seed
from eiffel.utils.logging import VerbLevel
from eiffel.utils.typing import EiffelCID, MetricsDict, NDArray
from eiffel.storage import encode_array

from .pool import Pool

logger = logging.getLogger(__name__)


def mk_client_init_fn(seed: int) -> Callable[[], None]:
    """Return a client initializer function.

    Parameters
    ----------
    seed : int
        The seed to use for random number generation.

    Returns
    -------
    Callable[[None], None]
        The client initializer function.
    """

    def init_fn() -> None:
        set_seed(seed)
        # Enable GPU growth upon actor init
        # does nothing if `num_gpus` in client_resources is 0.0
        enable_tf_gpu_growth()

    return init_fn


class EiffelClient(NumPyClient):
    """Eiffel client.

    Attributes
    ----------
    cid : EiffelCID
        The client ID.
    data_holder : DatasetHolder
        A reference to the datasets, living in the Ray object store.
    model : keras.Model
        The model to train.
    verbose : VerbLevel
        The verbosity level.
    seed : Optional[int]
        The seed to use for random number generation.
    poison_ins : Optional[PoisonIns]
        The poisoning instructions, if any.
    """

    cid: EiffelCID
    data_holder: DatasetHandle
    model: keras.Model
    poison_ins: Optional[PoisonIns]

    def __init__(
        self,
        cid: EiffelCID,
        data_holder: DatasetHandle,
        model: keras.Model,
        *,
        verbose: VerbLevel = VerbLevel.SILENT,
        seed: int,
        poison_ins: Optional[PoisonIns] = None,
        eval_fit: bool = True,
    ) -> None:
        """Initialize the EiffelClient."""
        self.cid = cid
        self.data_holder = data_holder
        self.model = model
        self.verbose = verbose
        self.seed = seed
        self.poison_ins = poison_ins
        self.eval_fit = eval_fit
        set_seed(seed)

    def get_parameters(self, config: Config) -> list[NDArray]:
        """Return the current parameters.

        Returns
        -------
        list[NDArray]
            Current model parameters.
        """
        return self.model.get_weights()

    def fit(
        self, parameters: list[NDArray], config: Config
    ) -> tuple[list[NDArray], int, dict]:
        """Fit the model to the local data set.

        Parameters
        ----------
        parameters : list[NDArray]
            The initial parameters to train on, generally those of the global model.
        config : Config
            The configuration for the training.

        Returns
        -------
        list[NDArray]
            The updated parameters.
        int
            The number of examples used for training.
        MetricsDict
            The metrics collected during training.
        """
        if self.poison_ins is not None:
            if "round" not in config:
                logger.warning(
                    f"{self.cid}: No round number provided, skipping poisoning."
                )
            elif self.poison_ins.tasks is None:
                logger.debug(
                    f"{self.cid}: No poisoning tasks provided, skipping poisoning."
                )
            elif config["round"] in self.poison_ins.tasks:
                self.poison(self.poison_ins.tasks[config["round"]])
                logger.debug(f"{self.cid}: Poisoned the dataset.")

        train_set: Dataset = ray.get(self.data_holder.get.remote("train"))
        self.model.set_weights(parameters)
        hist: History = self.model.fit(
            train_set.to_sequence(
                int(config["batch_size"]), target=1, seed=self.seed, shuffle=True
            ),
            epochs=int(config["num_epochs"]),
            verbose=0,
        )

        ret = {
            "_cid": self.cid,
        }

        # Capture a compact, deterministic first inference for round-level analysis.
        # The payload is transiently base64 encoded because Flower metrics accept
        # scalar values; the instrumented strategy decodes it and stores float16 HDF5.
        if bool(config.get("capture_inference", False)):
            test_set: Dataset = ray.get(self.data_holder.get.remote("test"))
            probe_size = min(int(config.get("probe_size", 256)), len(test_set))
            if probe_size > 0:
                probe_x = test_set.X.iloc[:probe_size].to_numpy()
                probe_y = test_set.y.iloc[:probe_size].to_numpy()
                inference = self.model.predict(
                    probe_x,
                    batch_size=int(config["batch_size"]),
                    verbose=0,
                )
                ret["_eiffel_inference"] = encode_array(
                    np.asarray(inference), dtype="float16"
                )
                ret["_eiffel_probe_labels"] = encode_array(
                    np.asarray(probe_y), dtype="int16"
                )
                if "Attack" in test_set.m.columns:
                    ret["_eiffel_probe_families"] = json.dumps(
                        test_set.m["Attack"].iloc[:probe_size].astype(str).tolist()
                    )

        if self.eval_fit:
            test_loss, _, metrics = self.evaluate(self.model.get_weights(), config)
            ret.update(metrics)
            ret["fit"] = json.dumps({
                "test_loss": test_loss,
                "fit_accuracy": hist.history["accuracy"][-1],
                "fit_loss": hist.history["loss"][-1],
            })

        return (self.model.get_weights(), len(train_set), ret)

    def evaluate(
        self, parameters: list[NDArray], config: Config
    ) -> tuple[float, int, dict]:
        """Evaluate the model on the local data set.

        Parameters
        ----------
        parameters : list[NDArray]
            The parameters of the model to evaluate.
        config : Config
            The configuration for the evaluation.

        Returns
        -------
        float
            The loss of the model during evaluation.
        int
            The number of samples used for evaluation.
        MetricsDict
            The metrics collected during evaluation.
        """
        batch_size = int(config["batch_size"])

        self.model.set_weights(parameters)

        test_set: Dataset = ray.get(self.data_holder.get.remote("test"))

        output = self.model.evaluate(
            test_set.to_sequence(batch_size, target=1, seed=self.seed, shuffle=True),
            verbose=self.verbose,
        )
        try:
            output = dict(zip(self.model.metrics_names, output))
            loss = output["loss"]
        except TypeError:
            # If `evaluate` returns a single value, it is a scalar for the loss.
            loss = output

        # Do not shuffle the test set for inference, otherwise we cannot compare y_pred
        # with y_true.
        inferences: NDArray = self.model.predict(
            test_set.to_sequence(batch_size, target=1, seed=self.seed),
            verbose=self.verbose,
        )

        y_pred = np.around(inferences).astype(int).reshape(-1)

        y_true = test_set.y.to_numpy().astype(int)

        return_data: dict[str, Any] = {}

        # Eiffel trains a binary Benign-vs-Attack detector, but the metadata retains
        # the original attack family.  Report metrics for every family so selective
        # degradation can be studied as a multiclass-aware NIDS analysis.
        class_df = test_set.m["Attack"].astype(str)
        attack_recalls: list[float] = []
        attack_missrates: list[float] = []
        for label in (c for c in class_df.unique() if c != "Benign"):
            mask = class_df == label
            y_true_attack = y_true[mask]
            y_pred_attack = y_pred[mask]
            _, _, fn, tp = confusion_matrix(
                y_true_attack, y_pred_attack, labels=(0, 1)
            ).ravel()
            denom = tp + fn
            recall = float(tp / denom) if denom else 0.0
            missrate = float(fn / denom) if denom else 0.0
            attack_recalls.append(recall)
            attack_missrates.append(missrate)
            return_data[label] = {
                "recall": recall,
                "missrate": missrate,
                "support": int(mask.sum()),
            }

        benign_mask = class_df == "Benign"
        if bool(benign_mask.any()):
            benign_pred = y_pred[benign_mask]
            false_positive_rate = float(np.mean(benign_pred == 1))
            return_data["Benign"] = {
                "false_positive_rate": false_positive_rate,
                "specificity": 1.0 - false_positive_rate,
                "support": int(benign_mask.sum()),
            }

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=(0, 1)).ravel()
        return_data["global"] = metrics_from_confmat(tn, fp, fn, tp)
        if attack_recalls:
            return_data["global"].update(
                {
                    "macro_attack_recall": float(np.mean(attack_recalls)),
                    "min_attack_recall": float(np.min(attack_recalls)),
                    "macro_attack_missrate": float(np.mean(attack_missrates)),
                }
            )

        return_data["global"]["loss"] = float(loss)
        return_data["_cid"] = self.cid

        return (loss, len(test_set), {k: json.dumps(v) for k, v in return_data.items()})

    def poison(self, task: PoisonTask) -> None:
        """Poison the dataset.

        Parameters
        ----------
        task : PoisonTask
            The poisoning task to perform.
        """
        assert self.poison_ins is not None

        self.data_holder.poison.remote(
            "train",
            task.fraction,
            task.operation,
            seed=self.seed,
            target_classes=self.poison_ins.target,
        )
        if self.poison_ins.poison_eval:
            self.data_holder.poison.remote(
                "test",
                task.fraction,
                task.operation,
                seed=self.seed,
                target_classes=self.poison_ins.target,
            )


def mk_client(
    cid: EiffelCID,
    mappings: dict[EiffelCID, tuple[ray.ObjectRef, Optional[PoisonIns], keras.Model]],
    seed: int,
) -> Client:
    """Return a client based on its CID."""
    if cid not in mappings:
        raise ValueError(f"Client `{cid}` not found in mappings.")

    handle, attack, model_fn = mappings[cid]

    return EiffelClient(
        cid,
        handle,
        model_fn(ray.get(handle.get.remote("train")).X.shape[1]),
        seed=seed,
        poison_ins=attack,
    ).to_client()


def mean_absolute_error(x_orig: pd.DataFrame, x_pred: pd.DataFrame) -> np.ndarray:
    """Mean absolute error.

    Parameters
    ----------
    x_orig : pd.DataFrame
        True labels.
    x_pred : pd.DataFrame
        Predicted labels.

    Returns
    -------
    ndarray[float]
        Mean absolute error.
    """
    return np.mean(np.abs(x_orig - x_pred), axis=1)


def mean_squared_error(x_orig: pd.DataFrame, x_pred: pd.DataFrame) -> np.ndarray:
    """Mean squared error.

    Parameters
    ----------
    x_orig : pd.DataFrame
        True labels.
    x_pred : pd.DataFrame
        Predicted labels.

    Returns
    -------
    ndarray[float]
        Mean squared error.
    """
    return np.mean((x_orig - x_pred) ** 2, axis=1)


def root_mean_squared_error(x_orig: pd.DataFrame, x_pred: pd.DataFrame) -> np.ndarray:
    """Root mean squared error.

    Parameters
    ----------
    x_orig : pd.DataFrame
        True labels.
    x_pred : pd.DataFrame
        Predicted labels.

    Returns
    -------
    ndarray[float]
        Root mean squared error.
    """
    return np.sqrt(np.mean((x_orig - x_pred) ** 2, axis=1))


def metrics_from_confmat(*conf: int) -> dict[str, float]:
    """Translate a confusion matrix into metrics.

    Parameters
    ----------
    conf : tuple[int]
        The confusion matrix, under the form (tn, fp, fn, tp).

    Returns
    -------
    dict[str, float]
        Dictionary with the evaluation metrics (accuracy, precision, recall, f1,
        missrate, fallout).
    """
    tn, fp, fn, tp = conf

    return {
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "precision": float(tp / (tp + fp)) if (tp + fp) != 0 else 0,
        "recall": float(tp / (tp + fn)) if (tp + fn) != 0 else 0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) != 0 else 0,
        "missrate": float(fn / (fn + tp)) if (fn + tp) != 0 else 0,
        "fallout": float(fp / (fp + tn)) if (fp + tn) != 0 else 0,
    }
