"""FedAvg with model-poisoning hooks and compact round-state persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from eiffel.analysis.update_audit import audit_updates
from eiffel.attacks.model import apply_round_attack
from eiffel.storage import RoundStore, decode_array

logger = logging.getLogger(__name__)


def _plain(value: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _failure_summary(failures: list[Any], limit: int = 5) -> str:
    """Format Flower client failures without assuming one concrete failure shape."""
    items: list[str] = []
    for failure in failures[:limit]:
        if isinstance(failure, BaseException):
            items.append(f"{type(failure).__name__}: {failure}")
            continue
        if isinstance(failure, tuple) and len(failure) == 2:
            client, result = failure
            cid = getattr(client, "cid", "<unknown>")
            status = getattr(result, "status", None)
            items.append(f"client={cid} status={status!r}")
            continue
        items.append(repr(failure))
    suffix = "" if len(failures) <= limit else f"; +{len(failures) - limit} more"
    return "; ".join(items) + suffix


class InstrumentedFedAvg(FedAvg):
    """FedAvg that preserves raw round state and supports procedural model attacks."""

    def __init__(
        self,
        *args,
        storage: Mapping[str, Any] | None = None,
        model_attack: Mapping[str, Any] | None = None,
        num_rounds: int | None = None,
        seed: int = 0,
        **kwargs,
    ) -> None:
        initial_parameters = kwargs.get("initial_parameters")
        super().__init__(*args, **kwargs)

        self.storage_cfg = _plain(storage or {})
        self.attack_cfg = _plain(model_attack or {})
        self.num_rounds = num_rounds
        self.seed = int(seed)

        enabled = bool(self.storage_cfg.get("enabled", True))
        self.store = RoundStore(
            self.storage_cfg.get("path", "round_state.h5"),
            enabled=enabled,
            compression=self.storage_cfg.get("compression", "gzip"),
            compression_level=int(self.storage_cfg.get("compression_level", 4)),
            flush_each_round=bool(self.storage_cfg.get("flush_each_round", True)),
        )

        self._global_weights: list[np.ndarray] | None = None
        if isinstance(initial_parameters, Parameters):
            self._global_weights = [
                np.asarray(x, dtype=np.float32)
                for x in parameters_to_ndarrays(initial_parameters)
            ]
            self.store.save_global(0, self._global_weights)

    @staticmethod
    def _extract_probe_payload(
        metrics: dict,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        list[str] | None,
    ]:
        probabilities = None
        logits = None
        features = None
        labels = None
        families = None

        payload = metrics.pop("_eiffel_probabilities", None)
        if payload is None:
            # Backward compatibility with runs/clients using the old transient key.
            payload = metrics.pop("_eiffel_inference", None)
        if isinstance(payload, str):
            probabilities = decode_array(payload)
        payload = metrics.pop("_eiffel_logits", None)
        if isinstance(payload, str):
            logits = decode_array(payload)
        payload = metrics.pop("_eiffel_probe_features", None)
        if isinstance(payload, str):
            features = decode_array(payload)
        payload = metrics.pop("_eiffel_probe_labels", None)
        if isinstance(payload, str):
            labels = decode_array(payload)
        payload = metrics.pop("_eiffel_probe_families", None)
        if isinstance(payload, str):
            try:
                decoded = json.loads(payload)
                if isinstance(decoded, list):
                    families = [str(v) for v in decoded]
            except json.JSONDecodeError:
                logger.warning("Unable to decode probe attack-family metadata.")
        return probabilities, logits, features, labels, families

    @staticmethod
    def _logical_cid(metrics: Mapping[str, Any], fallback: str) -> str:
        """Return Eiffel's logical CID instead of relying on Flower proxy IDs."""
        reported = metrics.get("_cid")
        if reported is None:
            return str(fallback)
        if isinstance(reported, str):
            try:
                decoded = json.loads(reported)
                if isinstance(decoded, str):
                    return decoded
            except json.JSONDecodeError:
                pass
        return str(reported)

    @staticmethod
    def _decode_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Decode Eiffel's JSON-valued Flower metrics without mutating the input."""
        decoded: dict[str, Any] = {}
        for key, value in metrics.items():
            if str(key).startswith("_eiffel_") or key == "_cid":
                continue
            if isinstance(value, str):
                try:
                    decoded[str(key)] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass
            decoded[str(key)] = value
        return decoded

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            if failures:
                details = _failure_summary(list(failures))
                raise RuntimeError(
                    f"All clients failed during fit in round {server_round}. "
                    f"Flower reported {len(failures)} failure(s). {details}"
                )
            raise RuntimeError(
                f"Round {server_round} produced no fit results and no explicit "
                "Flower failures. Refusing to create an empty experiment."
            )
        if failures:
            details = _failure_summary(list(failures))
            raise RuntimeError(
                f"Round {server_round} fit reported {len(failures)} client "
                f"failure(s). Refusing partial aggregation because it would change "
                f"the configured benign/malicious population. {details}"
            )
        if self._global_weights is None:
            logger.warning(
                "InstrumentedFedAvg has no previous global weights; raw update capture "
                "is skipped for round %s.", server_round
            )
            return super().aggregate_fit(server_round, results, failures)

        clients: list[ClientProxy] = []
        logical_cids: list[str] = []
        reported_malicious: list[bool | None] = []
        fit_results: list[FitRes] = []
        local_weights: list[list[np.ndarray]] = []
        probabilities: list[np.ndarray | None] = []
        logits: list[np.ndarray | None] = []
        probe_features: list[np.ndarray | None] = []
        probe_labels: list[np.ndarray | None] = []
        probe_families: list[list[str] | None] = []

        for client, fit_res in results:
            clients.append(client)
            logical_cids.append(
                self._logical_cid(fit_res.metrics, str(client.cid))
            )
            ground_truth = fit_res.metrics.get("_eiffel_malicious")
            reported_malicious.append(
                bool(ground_truth)
                if isinstance(ground_truth, (bool, int, np.integer))
                else None
            )
            fit_results.append(fit_res)
            local_weights.append(
                [np.asarray(x, dtype=np.float32) for x in parameters_to_ndarrays(fit_res.parameters)]
            )
            probs, logit_values, features, labels, families = (
                self._extract_probe_payload(fit_res.metrics)
            )
            probabilities.append(probs)
            logits.append(logit_values)
            probe_features.append(features)
            probe_labels.append(labels)
            probe_families.append(families)

        pre_updates = [
            [local - global_ for local, global_ in zip(weights, self._global_weights)]
            for weights in local_weights
        ]
        malicious_mask = [
            reported
            if reported is not None
            else ("malicious" in cid)
            for cid, reported in zip(logical_cids, reported_malicious)
        ]

        submitted_updates, schedule_multiplier = apply_round_attack(
            pre_updates,
            malicious_mask,
            self.attack_cfg,
            server_round=int(server_round),
            total_rounds=self.num_rounds,
            seed=self.seed,
            probe_inferences=probabilities,
            probe_labels=probe_labels,
            probe_families=probe_families,
        )

        mechanism = str(self.attack_cfg.get("mechanism", "none"))
        attack_active = schedule_multiplier > 0.0

        for idx, fit_res in enumerate(fit_results):
            submitted_weights = [
                global_ + delta
                for global_, delta in zip(self._global_weights, submitted_updates[idx])
            ]
            fit_res.parameters = ndarrays_to_parameters(submitted_weights)

        audits = audit_updates(submitted_updates)

        for idx, client in enumerate(clients):
            malicious = malicious_mask[idx]
            cid = logical_cids[idx]
            changed = malicious and attack_active and mechanism not in {"none", "label_flip"}
            self.store.save_client(
                int(server_round),
                cid,
                submitted_update=submitted_updates[idx],
                pre_attack_update=pre_updates[idx] if changed else None,
                audit=audits[idx],
                probabilities=probabilities[idx],
                logits=logits[idx],
                probe_features=probe_features[idx],
                probe_labels=probe_labels[idx],
                probe_families=probe_families[idx],
                malicious=malicious,
                attack_active=bool(changed),
                mechanism=mechanism if changed else "none",
            )
            self.store.save_client_metrics(
                int(server_round),
                cid,
                self._decode_metrics(fit_results[idx].metrics),
                phase="fit",
            )
        self.store.save_round_metadata(
            int(server_round),
            attack_mechanism=mechanism,
            attack_multiplier=float(schedule_multiplier),
            malicious_clients=int(sum(malicious_mask)),
        )

        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self._global_weights = [
                np.asarray(x, dtype=np.float32)
                for x in parameters_to_ndarrays(aggregated)
            ]
            self.store.save_global(int(server_round), self._global_weights)

        self.store.mark_round_complete(int(server_round))
        return aggregated, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures,
    ):
        """Persist distributed evaluation metrics before normal FedAvg aggregation."""
        if not results and failures:
            details = _failure_summary(list(failures))
            raise RuntimeError(
                f"All clients failed during evaluation in round {server_round}. "
                f"Flower reported {len(failures)} failure(s). {details}"
            )
        if failures:
            details = _failure_summary(list(failures))
            raise RuntimeError(
                f"Round {server_round} evaluation reported {len(failures)} client "
                f"failure(s). Refusing incomplete persisted metrics. {details}"
            )
        for client, evaluate_res in results:
            probs, logit_values, features, labels, families = (
                self._extract_probe_payload(evaluate_res.metrics)
            )
            cid = self._logical_cid(
                evaluate_res.metrics,
                str(client.cid),
            )
            self.store.save_client_metrics(
                int(server_round),
                cid,
                self._decode_metrics(evaluate_res.metrics),
                phase="evaluate",
            )
            self.store.save_probe(
                cid,
                features=features,
                labels=labels,
                families=families,
            )
            if bool(self.storage_cfg.get("capture_global_inference", True)):
                self.store.save_global_inference(
                    int(server_round),
                    cid,
                    probabilities=probs,
                    logits=logit_values,
                )
        if results and self.store.flush_each_round:
            self.store.flush()
        return super().aggregate_evaluate(server_round, results, failures)
