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
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support
from tensorflow import keras

from eiffel.datasets.dataset import Dataset, DatasetHandle
from eiffel.datasets.poisoning import (
    PoisonIns,
    PoisonTask,
    poisoning_fraction_at_round,
    poisoning_is_configured,
)
from eiffel.utils import set_seed
from eiffel.utils.logging import VerbLevel
from eiffel.utils.typing import EiffelCID, MetricsDict, NDArray
from eiffel.storage import encode_array

from .pool import Pool

logger = logging.getLogger(__name__)


def select_probe_positions(
    test_set: Dataset,
    probe_size: int,
    *,
    seed: int,
) -> np.ndarray:
    """Select a deterministic, approximately proportional stratified probe.

    Attack-family metadata is preferred when available so rare NIDS families are not
    silently omitted merely because a source dataset is ordered by class. Labels are
    the generic fallback for datasets without an Attack metadata column.
    """
    total = len(test_set)
    size = min(max(0, int(probe_size)), total)
    if size <= 0:
        return np.empty(0, dtype=np.int64)
    if size == total:
        return np.arange(total, dtype=np.int64)

    if "Attack" in test_set.m.columns:
        strata = test_set.m["Attack"].astype(str).to_numpy()
    else:
        strata = test_set.y.astype(str).to_numpy()

    _, inverse, counts = np.unique(
        strata,
        return_inverse=True,
        return_counts=True,
    )
    n_groups = len(counts)
    quotas = counts.astype(np.float64) * (float(size) / float(total))
    allocation = np.floor(quotas).astype(int)

    minimum = np.zeros(n_groups, dtype=int)
    if size >= n_groups:
        minimum[:] = 1
        allocation = np.maximum(allocation, minimum)
    allocation = np.minimum(allocation, counts)

    while int(allocation.sum()) > size:
        candidates = np.flatnonzero(allocation > minimum)
        if candidates.size == 0:
            break
        over = allocation[candidates] - quotas[candidates]
        idx = candidates[int(np.argmax(over))]
        allocation[idx] -= 1

    while int(allocation.sum()) < size:
        candidates = np.flatnonzero(allocation < counts)
        if candidates.size == 0:
            break
        deficit = quotas[candidates] - allocation[candidates]
        idx = candidates[int(np.argmax(deficit))]
        allocation[idx] += 1

    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for group_id, count in enumerate(allocation):
        if count <= 0:
            continue
        positions = np.flatnonzero(inverse == group_id)
        chosen = rng.choice(positions, size=int(count), replace=False)
        selected.append(np.asarray(chosen, dtype=np.int64))

    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(selected)).astype(np.int64)


def predict_probabilities_and_logits(
    model: keras.Model,
    x: np.ndarray,
    *,
    batch_size: int,
    verbose: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return model probabilities and exact pre-activation logits.

    Eiffel's supported classifiers end in a Dense sigmoid/softmax layer.  Building a
    small view of the model that exposes that layer's input lets us reconstruct the
    exact affine pre-activation values without changing training semantics.
    """
    if not model.layers:
        raise ValueError("Inference capture requires a model with at least one layer.")
    final_layer = model.layers[-1]
    if not isinstance(final_layer, keras.layers.Dense):
        raise ValueError(
            "Inference capture with logits requires the final model layer to be "
            "keras.layers.Dense."
        )
    activation_name = getattr(final_layer.activation, "__name__", "")
    if activation_name not in {"sigmoid", "softmax"}:
        raise ValueError(
            "Inference capture with logits requires a final sigmoid or softmax "
            f"Dense layer, got activation={activation_name!r}."
        )

    view = keras.Model(
        inputs=model.inputs,
        outputs=[final_layer.input, model.output],
    )
    hidden, probabilities = view.predict(
        x,
        batch_size=int(batch_size),
        verbose=verbose,
    )
    weights = final_layer.get_weights()
    if not weights:
        raise ValueError("Final Dense layer has no weights; cannot reconstruct logits.")
    kernel = np.asarray(weights[0], dtype=np.float32)
    logits = np.matmul(np.asarray(hidden, dtype=np.float32), kernel)
    if len(weights) > 1:
        logits = logits + np.asarray(weights[1], dtype=np.float32)

    probabilities = np.asarray(probabilities, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)
    if probabilities.shape != logits.shape:
        raise ValueError(
            "Probability/logit shape mismatch: "
            f"{probabilities.shape} != {logits.shape}."
        )
    return probabilities, logits


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
        The data-poisoning instructions, if any.
    is_malicious : bool
        Explicit security role. This is independent from data poisoning so pure
        model-poisoning clients do not need fake PoisonIns instructions.
    """

    cid: EiffelCID
    data_holder: DatasetHandle
    model: keras.Model
    poison_ins: Optional[PoisonIns]
    is_malicious: bool

    def __init__(
        self,
        cid: EiffelCID,
        data_holder: DatasetHandle,
        model: keras.Model,
        *,
        verbose: VerbLevel = VerbLevel.SILENT,
        seed: int,
        poison_ins: Optional[PoisonIns] = None,
        is_malicious: bool | None = None,
        eval_fit: bool = True,
    ) -> None:
        """Initialize the EiffelClient."""
        self.cid = cid
        self.data_holder = data_holder
        self.model = model
        self.verbose = verbose
        self.seed = seed
        self.poison_ins = poison_ins
        if is_malicious is False and poison_ins is not None:
            raise ValueError(
                "A client with data-poisoning instructions cannot be explicitly "
                "marked benign."
            )
        self.is_malicious = (
            bool(poison_ins is not None)
            if is_malicious is None
            else bool(is_malicious)
        )
        self.eval_fit = eval_fit
        set_seed(seed)

    def _capture_probe_payload(
        self,
        test_set: Dataset,
        config: Config,
    ) -> dict[str, str]:
        """Encode one deterministic probe for post-hoc analysis."""
        positions = select_probe_positions(
            test_set,
            int(config.get("probe_size", 256)),
            seed=self.seed,
        )
        if positions.size <= 0:
            return {}

        probe_x = test_set.X.iloc[positions].to_numpy()
        probe_y = test_set.y.iloc[positions].to_numpy()
        capture_logits = bool(config.get("capture_logits", True))
        if capture_logits:
            probabilities, logits = predict_probabilities_and_logits(
                self.model,
                probe_x,
                batch_size=int(config["batch_size"]),
                verbose=0,
            )
        else:
            probabilities = np.asarray(
                self.model.predict(
                    probe_x,
                    batch_size=int(config["batch_size"]),
                    verbose=0,
                ),
                dtype=np.float32,
            )
            logits = None

        payload: dict[str, str] = {
            "_eiffel_probabilities": encode_array(
                probabilities,
                dtype="float16",
            ),
            "_eiffel_probe_labels": encode_array(
                np.asarray(probe_y),
                dtype="int16",
            ),
        }
        if logits is not None:
            payload["_eiffel_logits"] = encode_array(logits, dtype="float16")
        if bool(config.get("capture_probe_features", True)):
            payload["_eiffel_probe_features"] = encode_array(
                np.asarray(probe_x),
                dtype="float32",
            )
        if "Attack" in test_set.m.columns:
            payload["_eiffel_probe_families"] = json.dumps(
                test_set.m["Attack"].iloc[positions].astype(str).tolist()
            )
        return payload

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

        round_number = (
            int(config["round"]) if "round" in config else None
        )
        data_attack = (
            "label_flip"
            if poisoning_is_configured(self.poison_ins)
            else "none"
        )
        data_poison_fraction = (
            poisoning_fraction_at_round(self.poison_ins, round_number)
            if data_attack != "none" and self.poison_ins is not None
            else 0.0
        )
        data_poison_active = bool(data_poison_fraction > 0.0)
        if data_attack != "none" and "Poisoned" in train_set.m.columns:
            # NF-V2-compatible datasets expose the actual local poisoning state.
            # This matters under non-IID partitions: a targeted attack can be
            # scheduled but have no effect on a client that owns no target samples.
            data_poison_active = bool(
                train_set.m["Poisoned"].astype(bool).any()
            )

        ret = {
            "_cid": self.cid,
            "_eiffel_malicious": self.is_malicious,
            "_eiffel_data_attack": data_attack,
            "_eiffel_data_poison_fraction": float(data_poison_fraction),
            "_eiffel_data_poison_active": data_poison_active,
        }

        # Capture a compact deterministic probe. Flower metrics only accept scalar
        # payloads, so arrays are encoded transiently and decoded by the strategy.
        if bool(config.get("capture_inference", False)):
            test_set: Dataset = ray.get(self.data_holder.get.remote("test"))
            ret.update(self._capture_probe_payload(test_set, config))

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

        y_true = test_set.y.to_numpy().astype(int)
        inference_array = np.asarray(inferences)
        multiclass = inference_array.ndim > 1 and inference_array.shape[-1] > 1
        y_pred = (
            np.argmax(inference_array, axis=1).astype(int).reshape(-1)
            if multiclass
            else (inference_array.reshape(-1) >= 0.5).astype(int)
        )

        return_data: dict[str, Any] = {}
        if "Attack" in test_set.m.columns:
            class_df = test_set.m["Attack"].astype(str)
        elif multiclass:
            class_df = pd.Series(
                [f"class_{int(value)}" for value in y_true],
                index=test_set.y.index,
                dtype="object",
            )
        else:
            class_df = pd.Series(
                np.where(y_true == 0, "Benign", "Attack"),
                index=test_set.y.index,
                dtype="object",
            )

        if multiclass:
            labels = sorted(int(v) for v in np.unique(y_true))
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=labels,
                zero_division=0,
            )
            class_names: dict[int, str] = {}
            for class_id in labels:
                names = class_df[y_true == class_id].value_counts()
                class_names[class_id] = (
                    str(names.index[0]) if len(names) else f"class_{class_id}"
                )
            attack_recalls: list[float] = []
            for idx, class_id in enumerate(labels):
                name = class_names[class_id]
                return_data[name] = {
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1": float(f1[idx]),
                    "missrate": float(1.0 - recall[idx]),
                    "support": int(support[idx]),
                }
                if name != "Benign":
                    attack_recalls.append(float(recall[idx]))
            return_data["global"] = {
                "accuracy": float(np.mean(y_pred == y_true)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "mcc": float(matthews_corrcoef(y_true, y_pred)),
                "num_classes": float(len(labels)),
                "loss": float(loss),
            }
            if attack_recalls:
                return_data["global"].update(
                    {
                        "macro_attack_recall": float(np.mean(attack_recalls)),
                        "min_attack_recall": float(np.min(attack_recalls)),
                        "macro_attack_missrate": float(
                            np.mean([1.0 - value for value in attack_recalls])
                        ),
                    }
                )
            return_data["confusion_matrix"] = confusion_matrix(
                y_true, y_pred, labels=labels
            ).tolist()
        else:
            # Binary Benign-vs-Attack training with per-family recall/miss-rate.
            attack_recalls = []
            attack_missrates = []
            for label in (name for name in class_df.unique() if name != "Benign"):
                mask = class_df == label
                y_true_attack = y_true[mask]
                y_pred_attack = y_pred[mask]
                _, _, fn, tp = confusion_matrix(
                    y_true_attack, y_pred_attack, labels=(0, 1)
                ).ravel()
                denom = tp + fn
                recall_value = float(tp / denom) if denom else 0.0
                missrate_value = float(fn / denom) if denom else 0.0
                attack_recalls.append(recall_value)
                attack_missrates.append(missrate_value)
                return_data[label] = {
                    "recall": recall_value,
                    "missrate": missrate_value,
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
            return_data["global"].update(
                {
                    "macro_f1": float(
                        f1_score(y_true, y_pred, average="macro", zero_division=0)
                    ),
                    "weighted_f1": float(
                        f1_score(y_true, y_pred, average="weighted", zero_division=0)
                    ),
                    "mcc": float(matthews_corrcoef(y_true, y_pred)),
                }
            )
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

        metrics_payload = {k: json.dumps(v) for k, v in return_data.items()}
        if bool(config.get("capture_inference", False)):
            metrics_payload.update(self._capture_probe_payload(test_set, config))
        return (loss, len(test_set), metrics_payload)

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
    mappings: dict[
        EiffelCID,
        tuple[ray.ObjectRef, Optional[PoisonIns], keras.Model]
        | tuple[ray.ObjectRef, Optional[PoisonIns], keras.Model, bool],
    ],
    seed: int,
) -> NumPyClient:
    """Return a Flower 1.5-compatible NumPyClient based on its CID.

    Flower's simulation layer accepts a ClientLike and wraps NumPyClient instances
    internally.  Eiffel pins Flower 1.5.0, where NumPyClient does not expose the
    newer instance method `to_client()`.
    """
    if cid not in mappings:
        raise ValueError(f"Client `{cid}` not found in mappings.")

    mapping = mappings[cid]
    if len(mapping) == 4:
        handle, attack, model_fn, is_malicious = mapping
    else:
        # Backward compatibility for callers that still provide the historical
        # (handle, poison_ins, model_fn) tuple.
        handle, attack, model_fn = mapping
        is_malicious = bool(attack is not None) or "malicious" in str(cid)

    return EiffelClient(
        cid,
        handle,
        model_fn(ray.get(handle.get.remote("train")).X.shape[1]),
        seed=seed,
        poison_ins=attack,
        is_malicious=is_malicious,
    )


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
