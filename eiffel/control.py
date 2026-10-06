"""File-based runtime control plane for interactive Eiffel experiments.

The control plane is intentionally independent from any UI. A terminal or web frontend
can read events.jsonl and atomically replace control.json. Runtime changes are consumed
at FL round boundaries so a round always has one internally consistent configuration.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


MUTABLE_PREFIXES = ("attack.", "aggregation.", "defense.")


class RuntimeControlError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_path(target: dict[str, Any], dotted: str, value: Any) -> None:
    keys = [part for part in str(dotted).split(".") if part]
    if not keys:
        raise RuntimeControlError("Control key cannot be empty.")
    node = target
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        if not isinstance(child, dict):
            raise RuntimeControlError(
                f"Cannot set {dotted!r}: {key!r} is not a mapping."
            )
        node = child
    node[keys[-1]] = value


def validate_runtime_changes(changes: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_key, value in changes.items():
        key = str(raw_key).strip()
        if not key.startswith(MUTABLE_PREFIXES):
            raise RuntimeControlError(
                f"{key!r} is static during a run. Runtime controls may change only "
                "model-attack, aggregation, or defense fields."
            )
        if key.startswith("attack.") and key in {
            "attack.malicious_fraction",
            "attack.malicious_client_ids",
            "attack.poison_rate",
            "attack.objective",
            "attack.source_class",
            "attack.destination_class",
        }:
            raise RuntimeControlError(
                f"{key!r} changes client/data assignment and cannot be changed "
                "during an active run."
            )
        result[key] = value
    return result


class EventStream:
    def __init__(self, path: str | Path = "events.jsonl", *, enabled: bool = True):
        self.path = Path(path)
        self.enabled = bool(enabled)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        record = {"timestamp": _utc_now(), "event": str(event), **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass


class ControlPlane:
    def __init__(
        self,
        path: str | Path = "control.json",
        *,
        enabled: bool = False,
    ):
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.last_revision = -1
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def read_for_round(self, server_round: int) -> tuple[int, dict[str, Any]] | None:
        if not self.enabled or not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeControlError(
                f"Unable to read runtime control file {self.path}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise RuntimeControlError("control.json root must be an object.")
        revision = int(raw.get("revision", 0))
        if revision <= self.last_revision:
            return None
        apply_from_round = int(raw.get("apply_from_round", 1))
        if int(server_round) < apply_from_round:
            return None
        changes = raw.get("set", {})
        if not isinstance(changes, Mapping):
            raise RuntimeControlError("control.json 'set' must be an object.")
        validated = validate_runtime_changes(changes)
        self.last_revision = revision
        return revision, validated


def write_control(
    path: str | Path,
    *,
    revision: int,
    changes: Mapping[str, Any],
    apply_from_round: int = 1,
) -> Path:
    """Atomically publish one runtime-control revision."""
    if int(revision) < 0:
        raise RuntimeControlError("revision must be >= 0.")
    if int(apply_from_round) < 1:
        raise RuntimeControlError("apply_from_round must be >= 1.")
    validated = validate_runtime_changes(changes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "revision": int(revision),
        "apply_from_round": int(apply_from_round),
        "set": validated,
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return path


def apply_changes(
    *,
    attack: dict[str, Any],
    aggregation: dict[str, Any],
    defense: dict[str, Any],
    changes: Mapping[str, Any],
) -> None:
    """Apply validated dotted changes to live strategy dictionaries."""
    grouped = {
        "attack": attack,
        "aggregation": aggregation,
        "defense": defense,
    }
    for dotted, value in validate_runtime_changes(changes).items():
        root, remainder = dotted.split(".", 1)
        _set_path(grouped[root], remainder, value)
