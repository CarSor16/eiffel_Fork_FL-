"""Validate Eiffel round_state.h5 structural and numeric invariants."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


def _layers(group: h5py.Group) -> list[np.ndarray]:
    return [np.asarray(group[name]) for name in sorted(group.keys())]


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    with h5py.File(path, "r") as h5:
        fmt = h5.attrs.get("format", "")
        if isinstance(fmt, bytes):
            fmt = fmt.decode("utf-8", errors="replace")
        if fmt != "eiffel-round-state":
            errors.append(f"unexpected format attribute: {fmt!r}")

        if "global" not in h5:
            errors.append("missing /global")
            return errors
        if "round_0000" not in h5["global"]:
            errors.append("missing /global/round_0000")
        if "clients" not in h5:
            errors.append("missing /clients")
            return errors

        complete = None
        if "meta" in h5 and "last_complete_round" in h5["meta"].attrs:
            complete = int(h5["meta"].attrs["last_complete_round"])
        else:
            errors.append("missing /meta:last_complete_round")

        client_rounds = sorted(h5["clients"].keys())
        for round_name in client_rounds:
            try:
                round_number = int(round_name.rsplit("_", 1)[-1])
            except ValueError:
                errors.append(f"invalid round group name: /clients/{round_name}")
                continue

            previous_name = f"round_{round_number - 1:04d}"
            current_name = f"round_{round_number:04d}"
            if previous_name not in h5["global"]:
                errors.append(
                    f"{round_name}: missing previous global {previous_name}"
                )
                continue
            if current_name not in h5["global"]:
                errors.append(
                    f"{round_name}: missing aggregated global {current_name}"
                )

            previous = h5["global"][previous_name].get("weights")
            if previous is None:
                errors.append(f"{round_name}: previous global has no weights")
                continue
            global_layers = _layers(previous)

            for cid, client in h5["clients"][round_name].items():
                submitted = client.get("submitted_update")
                if submitted is None:
                    errors.append(
                        f"{round_name}/{cid}: missing submitted_update"
                    )
                    continue
                update_layers = _layers(submitted)
                if len(update_layers) != len(global_layers):
                    errors.append(
                        f"{round_name}/{cid}: submitted layer count "
                        f"{len(update_layers)} != global {len(global_layers)}"
                    )
                    continue

                for layer_idx, (global_layer, update_layer) in enumerate(
                    zip(global_layers, update_layers)
                ):
                    if global_layer.shape != update_layer.shape:
                        errors.append(
                            f"{round_name}/{cid}: layer {layer_idx} shape "
                            f"{update_layer.shape} != global {global_layer.shape}"
                        )
                    if update_layer.dtype != np.float32:
                        errors.append(
                            f"{round_name}/{cid}: layer {layer_idx} dtype "
                            f"{update_layer.dtype} != float32"
                        )
                    if not np.all(np.isfinite(update_layer)):
                        errors.append(
                            f"{round_name}/{cid}: layer {layer_idx} contains NaN/inf"
                        )
                    if global_layer.shape == update_layer.shape:
                        reconstructed = (
                            global_layer.astype(np.float32)
                            + update_layer.astype(np.float32)
                        )
                        if not np.all(np.isfinite(reconstructed)):
                            errors.append(
                                f"{round_name}/{cid}: reconstructed local "
                                f"layer {layer_idx} is not finite"
                            )

                malicious = bool(client.attrs.get("malicious", 0))
                active = bool(client.attrs.get("attack_active", 0))
                mechanism = client.attrs.get("mechanism", "none")
                if isinstance(mechanism, bytes):
                    mechanism = mechanism.decode("utf-8", errors="replace")
                if active and not malicious:
                    errors.append(
                        f"{round_name}/{cid}: attack_active set on benign client"
                    )
                if (
                    active
                    and mechanism not in {"none", "label_flip"}
                    and "pre_attack_update" not in client
                ):
                    errors.append(
                        f"{round_name}/{cid}: active model attack without "
                        "pre_attack_update"
                    )

                if "inference" in client:
                    inference = np.asarray(client["inference"])
                    if inference.dtype != np.float16:
                        errors.append(
                            f"{round_name}/{cid}: inference dtype "
                            f"{inference.dtype} != float16"
                        )
                    if not np.all(np.isfinite(inference)):
                        errors.append(
                            f"{round_name}/{cid}: inference contains NaN/inf"
                        )

                if "audit" in client:
                    for key, value in client["audit"].attrs.items():
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            errors.append(
                                f"{round_name}/{cid}: audit {key} is not numeric"
                            )
                            continue
                        if not math.isfinite(numeric):
                            errors.append(
                                f"{round_name}/{cid}: audit {key} is NaN/inf"
                            )

        if complete is not None and client_rounds:
            highest = max(
                int(name.rsplit("_", 1)[-1]) for name in client_rounds
            )
            if complete != highest:
                errors.append(
                    f"last_complete_round={complete} but highest client "
                    f"round is {highest}"
                )

        if "probe" in h5 and "labels" in h5["probe"]:
            labels = np.asarray(h5["probe"]["labels"])
            if labels.dtype != np.int16:
                errors.append(
                    f"probe labels dtype {labels.dtype} != int16"
                )

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an Eiffel round_state.h5 file."
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if not args.path.exists():
        parser.error(f"file not found: {args.path}")

    errors = validate(args.path)
    if errors:
        print(f"FAILED: {args.path}")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"OK: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
