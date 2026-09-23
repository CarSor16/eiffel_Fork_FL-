"""HDF5 storage for raw FL state, audit signals, and inference outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

NDArraySeq = Sequence[np.ndarray]


def _safe(value: str) -> str:
    return str(value).replace("/", "_")


class RoundStore:
    """Append-safe single-file store for one Eiffel run.

    The file is organised by round and client. Raw client updates and global model
    weights are stored as float32. Inference outputs are stored as float16. HDF5
    compression avoids the file explosion caused by one NPZ per client/round.
    """

    def __init__(
        self,
        path: str | Path = "round_state.h5",
        *,
        enabled: bool = True,
        compression: str | None = "gzip",
        compression_level: int = 4,
        flush_each_round: bool = True,
    ) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.compression = compression
        self.compression_level = int(compression_level)
        self.flush_each_round = bool(flush_each_round)
        self._h5: h5py.File | None = None

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._h5 = h5py.File(self.path, "a")
            self._h5.attrs["format"] = "eiffel-round-state"
            self._h5.attrs["format_version"] = 1

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None

    def flush(self) -> None:
        if self._h5 is not None:
            self._h5.flush()

    def _write_array(
        self,
        group: h5py.Group,
        name: str,
        value: np.ndarray,
        *,
        dtype: str,
    ) -> None:
        if name in group:
            del group[name]
        arr = np.asarray(value, dtype=np.dtype(dtype))
        kwargs = {}
        if self.compression and arr.size > 0 and arr.ndim > 0:
            kwargs["compression"] = self.compression
            if self.compression == "gzip":
                kwargs["compression_opts"] = self.compression_level
            kwargs["shuffle"] = True
        group.create_dataset(name, data=arr, **kwargs)

    def _write_layers(
        self,
        parent: h5py.Group,
        name: str,
        arrays: NDArraySeq,
        *,
        dtype: str = "float32",
    ) -> None:
        if name in parent:
            del parent[name]
        group = parent.create_group(name)
        for idx, array in enumerate(arrays):
            self._write_array(group, f"layer_{idx:03d}", np.asarray(array), dtype=dtype)

    def save_global(self, server_round: int, weights: NDArraySeq) -> None:
        if self._h5 is None:
            return
        group = self._h5.require_group("global").require_group(
            f"round_{int(server_round):04d}"
        )
        self._write_layers(group, "weights", weights, dtype="float32")
        if self.flush_each_round:
            self._h5.flush()

    @staticmethod
    def _flatten_metrics(
        metrics: Mapping[str, object], prefix: str = ""
    ) -> dict[str, float]:
        flat: dict[str, float] = {}
        for key, value in metrics.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, Mapping):
                flat.update(RoundStore._flatten_metrics(value, full_key))
            elif isinstance(value, (bool, int, float, np.integer, np.floating)):
                flat[full_key] = float(value)
        return flat

    def save_client_metrics(
        self,
        server_round: int,
        cid: str,
        metrics: Mapping[str, object],
        *,
        phase: str,
    ) -> None:
        """Persist flattened numeric client metrics for one round and phase."""
        if self._h5 is None:
            return
        group = (
            self._h5.require_group("clients")
            .require_group(f"round_{int(server_round):04d}")
            .require_group(_safe(cid))
            .require_group("metrics")
            .require_group(_safe(phase))
        )
        for key, value in self._flatten_metrics(metrics).items():
            group.attrs[_safe(key)] = np.float32(value)

    def save_probe_families(self, families: Sequence[str]) -> None:
        """Store the fixed probe attack-family labels once."""
        if self._h5 is None or not families:
            return
        probe = self._h5.require_group("probe")
        if "families" in probe:
            return
        dtype = h5py.string_dtype(encoding="utf-8")
        probe.create_dataset(
            "families",
            data=np.asarray([str(v) for v in families], dtype=object),
            dtype=dtype,
        )

    def save_round_metadata(
        self,
        server_round: int,
        *,
        attack_mechanism: str,
        attack_multiplier: float,
        malicious_clients: int,
    ) -> None:
        """Store compact round-level attack metadata."""
        if self._h5 is None:
            return
        group = self._h5.require_group("rounds").require_group(
            f"round_{int(server_round):04d}"
        )
        group.attrs["attack_mechanism"] = str(attack_mechanism)
        group.attrs["attack_multiplier"] = np.float32(attack_multiplier)
        group.attrs["malicious_clients"] = int(malicious_clients)

    def save_client(
        self,
        server_round: int,
        cid: str,
        *,
        submitted_update: NDArraySeq,
        pre_attack_update: NDArraySeq | None = None,
        audit: Mapping[str, float] | None = None,
        inference: np.ndarray | None = None,
        probe_labels: np.ndarray | None = None,
        malicious: bool = False,
        attack_active: bool = False,
        mechanism: str = "none",
    ) -> None:
        if self._h5 is None:
            return
        round_group = self._h5.require_group("clients").require_group(
            f"round_{int(server_round):04d}"
        )
        group = round_group.require_group(_safe(cid))
        group.attrs["cid"] = str(cid)
        group.attrs["malicious"] = np.uint8(bool(malicious))
        group.attrs["attack_active"] = np.uint8(bool(attack_active))
        group.attrs["mechanism"] = str(mechanism)

        self._write_layers(
            group, "submitted_update", submitted_update, dtype="float32"
        )
        if pre_attack_update is not None:
            self._write_layers(
                group, "pre_attack_update", pre_attack_update, dtype="float32"
            )

        if audit:
            audit_group = group.require_group("audit")
            for key, value in audit.items():
                audit_group.attrs[str(key)] = np.float32(value)

        if inference is not None:
            self._write_array(group, "inference", inference, dtype="float16")

        if probe_labels is not None:
            probe = self._h5.require_group("probe")
            if "labels" not in probe:
                self._write_array(probe, "labels", probe_labels, dtype="int16")

    def mark_round_complete(self, server_round: int) -> None:
        if self._h5 is None:
            return
        meta = self._h5.require_group("meta")
        meta.attrs["last_complete_round"] = int(server_round)
        if self.flush_each_round:
            self._h5.flush()

    def __enter__(self) -> "RoundStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
