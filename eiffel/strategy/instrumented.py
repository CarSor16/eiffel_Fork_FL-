"""FedAvg with model-poisoning hooks and compact round-state persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from flwr.common import (
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
    def _extract_inference(metrics: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
        inference = None
        labels = None
        payload = metrics.pop("_eiffel_inference", None)
        if isinstance(payload, str):
            inference = decode_array(payload)
        payload = metrics.pop("_eiffel_probe_labels", None)
        if isinstance(payload, str):
            labels = decode_array(payload)
        return inference, labels

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return super().aggregate_fit(server_round, results, failures)
        if self._global_weights is None:
            logger.warning(
                "InstrumentedFedAvg has no previous global weights; raw update capture "
                "is skipped for round %s.", server_round
            )
            return super().aggregate_fit(server_round, results, failures)

        clients: list[ClientProxy] = []
        fit_results: list[FitRes] = []
        local_weights: list[list[np.ndarray]] = []
        inferences: list[np.ndarray | None] = []
        probe_labels: list[np.ndarray | None] = []

        for client, fit_res in results:
            clients.append(client)
            fit_results.append(fit_res)
            local_weights.append(
                [np.asarray(x, dtype=np.float32) for x in parameters_to_ndarrays(fit_res.parameters)]
            )
            inf, labels = self._extract_inference(fit_res.metrics)
            inferences.append(inf)
            probe_labels.append(labels)

        pre_updates = [
            [local - global_ for local, global_ in zip(weights, self._global_weights)]
            for weights in local_weights
        ]
        malicious_mask = ["malicious" in str(client.cid) for client in clients]

        submitted_updates, schedule_multiplier = apply_round_attack(
            pre_updates,
            malicious_mask,
            self.attack_cfg,
            server_round=int(server_round),
            total_rounds=self.num_rounds,
            seed=self.seed,
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
            changed = malicious and attack_active and mechanism not in {"none", "label_flip"}
            self.store.save_client(
                int(server_round),
                str(client.cid),
                submitted_update=submitted_updates[idx],
                pre_attack_update=pre_updates[idx] if changed else None,
                audit=audits[idx],
                inference=inferences[idx],
                probe_labels=probe_labels[idx],
                malicious=malicious,
                attack_active=bool(changed),
                mechanism=mechanism if changed else "none",
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
