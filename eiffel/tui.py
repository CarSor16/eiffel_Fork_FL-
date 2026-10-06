"""Terminal monitor and runtime controller for Eiffel experiments.

This first UI intentionally uses only the Python standard library. It talks to the
file-based control plane, so a later Textual/web frontend can reuse exactly the same
runtime protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from eiffel.control import RuntimeControlError, write_control


def _parse_scalar(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _latest(events: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("event") == kind:
            return event
    return {}


def status_text(run_dir: str | Path) -> str:
    root = Path(run_dir)
    resolved = _read_json(root / "resolved_profile.json")
    events = read_events(root / "events.jsonl")
    latest = _latest(events, "round_complete")
    started = _latest(events, "round_start")

    experiment = resolved.get("experiment", {})
    dataset = resolved.get("dataset", {})
    model = resolved.get("model", {})
    aggregation = resolved.get("aggregation", {})
    defense = resolved.get("defense", {})
    attack = resolved.get("attack", {})
    total_rounds = int(experiment.get("rounds", started.get("total_rounds", 0)) or 0)
    current_round = int(
        latest.get("round", started.get("round", 0)) or 0
    )
    metrics = latest.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    lines = [
        "EIFFEL FL SECURITY LAB",
        "=" * 64,
        f"Run          : {root}",
        f"Dataset      : {dataset.get('registry', dataset.get('name', '-'))}",
        f"Model        : {model.get('name', '-')}",
        f"Task         : {model.get('task', dataset.get('task', '-'))}",
        f"Round        : {current_round}/{total_rounds}",
        f"Attack       : {started.get('attack', attack.get('mechanism', '-'))}",
        f"Aggregation  : {started.get('aggregation', aggregation.get('name', '-'))}",
        f"Defense      : {started.get('defense', defense.get('name', '-'))}",
    ]
    if latest:
        if latest.get("loss") is not None:
            lines.append(f"Loss         : {float(latest['loss']):.6f}")
        for key, label in (
            ("accuracy", "Accuracy"),
            ("macro_f1", "Macro F1"),
            ("min_class_recall", "Min recall"),
            ("min_attack_recall", "Min attack recall"),
        ):
            if key in metrics:
                lines.append(f"{label:<13}: {float(metrics[key]):.6f}")
        lines.append(
            f"Malicious    : {int(latest.get('malicious_clients', 0))}"
        )
        lines.append(
            f"Attack scale : {float(latest.get('attack_multiplier', 0.0)):.3f}"
        )
    control = _read_json(root / "control.json")
    if control:
        lines.append("-" * 64)
        lines.append(
            f"Control rev  : {control.get('revision', 0)} "
            f"(from round {control.get('apply_from_round', 1)})"
        )
        changes = control.get("set", {})
        if isinstance(changes, dict):
            for key, value in sorted(changes.items()):
                lines.append(f"  {key} = {value}")
    lines.append("=" * 64)
    return "\n".join(lines)


def publish_changes(
    run_dir: str | Path,
    assignments: list[str],
    *,
    apply_from_round: int,
) -> Path:
    root = Path(run_dir)
    current = _read_json(root / "control.json")
    revision = int(current.get("revision", -1)) + 1
    changes: dict[str, Any] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise RuntimeControlError(
                f"Invalid control assignment {assignment!r}; expected key=value."
            )
        key, raw = assignment.split("=", 1)
        changes[key.strip()] = _parse_scalar(raw.strip())
    return write_control(
        root / "control.json",
        revision=revision,
        changes=changes,
        apply_from_round=int(apply_from_round),
    )


def watch(run_dir: str | Path, *, interval: float = 1.0, clear: bool = True) -> None:
    root = Path(run_dir)
    while True:
        if clear:
            os.system("cls" if os.name == "nt" else "clear")
        print(status_text(root), flush=True)
        time.sleep(max(0.2, float(interval)))


def interactive(run_dir: str | Path) -> None:
    """Simple cross-platform interactive terminal controller."""
    root = Path(run_dir)
    stop = threading.Event()
    print(status_text(root))
    print(
        "\nCommands: status | set ROUND key=value [key=value ...] | watch | help | quit"
    )

    def event_printer() -> None:
        seen = 0
        events_path = root / "events.jsonl"
        while not stop.is_set():
            events = read_events(events_path)
            if len(events) > seen:
                for event in events[seen:]:
                    kind = event.get("event", "event")
                    round_number = event.get("round", "-")
                    if kind == "round_complete":
                        metrics = event.get("metrics", {})
                        acc = (
                            f" acc={float(metrics['accuracy']):.4f}"
                            if isinstance(metrics, dict) and "accuracy" in metrics
                            else ""
                        )
                        print(
                            f"\n[event] round {round_number} complete | "
                            f"agg={event.get('aggregation')} "
                            f"defense={event.get('defense')}{acc}"
                        )
                    elif kind == "control_applied":
                        print(
                            f"\n[event] control revision {event.get('revision')} "
                            f"applied at round {round_number}"
                        )
                seen = len(events)
            stop.wait(0.5)

    thread = threading.Thread(target=event_printer, daemon=True)
    thread.start()
    try:
        while True:
            raw = input("eiffel> ").strip()
            if not raw:
                continue
            if raw in {"quit", "exit", "q"}:
                return
            if raw in {"help", "?"}:
                print(
                    "status\n"
                    "set ROUND attack.strength=2.0 aggregation.name=median\n"
                    "set ROUND defense.name=norm_clipping defense.max_norm=5.0\n"
                    "watch\nquit"
                )
                continue
            if raw == "status":
                print(status_text(root))
                continue
            if raw == "watch":
                print(status_text(root))
                continue
            if raw.startswith("set "):
                parts = raw.split()
                if len(parts) < 3:
                    print("Usage: set ROUND key=value [key=value ...]")
                    continue
                try:
                    round_number = int(parts[1])
                    path = publish_changes(
                        root,
                        parts[2:],
                        apply_from_round=round_number,
                    )
                    print(f"Published {path}")
                except (ValueError, RuntimeControlError) as exc:
                    print(f"Control rejected: {exc}")
                continue
            print("Unknown command. Type 'help'.")
    finally:
        stop.set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Monitor and control a running Eiffel experiment."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status_parser = sub.add_parser("status")
    status_parser.add_argument("run_dir", type=Path)

    watch_parser = sub.add_parser("watch")
    watch_parser.add_argument("run_dir", type=Path)
    watch_parser.add_argument("--interval", type=float, default=1.0)
    watch_parser.add_argument("--no-clear", action="store_true")

    set_parser = sub.add_parser("set")
    set_parser.add_argument("run_dir", type=Path)
    set_parser.add_argument("--round", type=int, required=True)
    set_parser.add_argument("assignments", nargs="+")

    interactive_parser = sub.add_parser("interactive")
    interactive_parser.add_argument("run_dir", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            print(status_text(args.run_dir))
        elif args.command == "watch":
            try:
                watch(
                    args.run_dir,
                    interval=args.interval,
                    clear=not args.no_clear,
                )
            except KeyboardInterrupt:
                return 0
        elif args.command == "set":
            path = publish_changes(
                args.run_dir,
                args.assignments,
                apply_from_round=args.round,
            )
            print(path)
        elif args.command == "interactive":
            interactive(args.run_dir)
        return 0
    except RuntimeControlError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
