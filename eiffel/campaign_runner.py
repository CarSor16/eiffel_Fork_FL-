"""Batch campaign runner for reproducible Eiffel experiment sweeps.

The runner mutates user-facing TOML profiles in memory and executes every variant
through the same direct TOML runtime used by run.cmd. Each variant is isolated in
its own subprocess and output directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from eiffel.direct_runner import build_command, load_profile


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parse_csv_values(value: str) -> list[Any]:
    values = [_parse_scalar(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Sweep value list cannot be empty.")
    return values


def _parse_seeds(value: str) -> list[int]:
    return [int(item) for item in _parse_csv_values(value)]


def _set_path(profile: dict[str, Any], dotted: str, value: Any) -> None:
    keys = [part.strip() for part in dotted.split(".") if part.strip()]
    if not keys:
        raise ValueError("Sweep key cannot be empty.")
    node: dict[str, Any] = profile
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        if not isinstance(child, dict):
            raise ValueError(
                f"Cannot set {dotted}: {'.'.join(keys[:-1])} is not a table."
            )
        node = child
    node[keys[-1]] = value


def _get_total_clients(profile: dict[str, Any]) -> int:
    experiment = profile.get("experiment", {})
    return int(experiment.get("num_clients", 10))


def _sweep_specs(raw_specs: Iterable[str]) -> list[tuple[str, list[Any]]]:
    specs: list[tuple[str, list[Any]]] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(
                "--set expects dotted.path=value1,value2 (for example "
                "attack.strength=1.5,3.0)."
            )
        key, raw_values = raw.split("=", 1)
        specs.append((key.strip(), _parse_csv_values(raw_values)))
    return specs


def build_variants(
    base_profile: dict[str, Any],
    *,
    seeds: list[int] | None = None,
    sweep_specs: list[tuple[str, list[Any]]] | None = None,
    malicious_combination_size: int | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic campaign profile variants."""
    seeds = list(seeds or [int(base_profile.get("experiment", {}).get("seed", 1138))])
    sweep_specs = list(sweep_specs or [])

    total_clients = _get_total_clients(base_profile)
    if malicious_combination_size is None:
        malicious_sets: list[tuple[int, ...] | None] = [None]
    else:
        size = int(malicious_combination_size)
        if size < 1 or size >= total_clients:
            raise ValueError(
                "malicious combination size must be in "
                f"[1, {total_clients - 1}]."
            )
        malicious_sets = list(itertools.combinations(range(total_clients), size))

    keys = [key for key, _ in sweep_specs]
    value_products = (
        list(itertools.product(*(values for _, values in sweep_specs)))
        if sweep_specs
        else [()]
    )

    variants: list[dict[str, Any]] = []
    for seed, malicious_ids, values in itertools.product(
        seeds,
        malicious_sets,
        value_products,
    ):
        profile = copy.deepcopy(base_profile)
        profile.setdefault("experiment", {})["seed"] = int(seed)
        if malicious_ids is not None:
            profile.setdefault("attack", {})["malicious_client_ids"] = list(
                malicious_ids
            )
            profile["attack"].pop("malicious_fraction", None)
        for key, value in zip(keys, values):
            _set_path(profile, key, value)
        variants.append(profile)
    return variants


def _variant_fields(profile: dict[str, Any], index: int) -> dict[str, str]:
    attack = profile.get("attack", {}) or {}
    schedule = attack.get("schedule", {}) or {}
    ids = attack.get("malicious_client_ids", [])
    return {
        "variant": f"{index:04d}",
        "seed": str(profile.get("experiment", {}).get("seed", "")),
        "mechanism": str(attack.get("mechanism", "none")),
        "malicious_client_ids": ",".join(str(value) for value in ids),
        "malicious_fraction": str(attack.get("malicious_fraction", "")),
        "schedule": str(schedule.get("type", "continuous")),
    }


def _command(
    profile: dict[str, Any],
    *,
    output_dir: Path,
    max_concurrent_clients: int | None,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "campaign_profile.json"
    profile_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return build_command(
        profile_path,
        output_dir=output_dir,
        max_concurrent_clients=max_concurrent_clients,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic parameter sweeps from an Eiffel TOML profile."
    )
    parser.add_argument("profile", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per campaign variant.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        help="Comma-separated seeds, e.g. 2026,2027,2028.",
    )
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        help=(
            "Sweep a TOML field: dotted.path=value1,value2. Repeat for a "
            "Cartesian product."
        ),
    )
    parser.add_argument(
        "--all-malicious-combinations",
        type=int,
        metavar="K",
        help="Try every K-client malicious subset of experiment.num_clients.",
    )
    parser.add_argument(
        "--max-concurrent-clients",
        type=int,
        help=(
            "Optional local safety cap. Omit in Work/server mode to let Ray use "
            "the maximum safe concurrency available."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    if args.max_concurrent_clients is not None and args.max_concurrent_clients < 1:
        parser.error("--max-concurrent-clients must be >= 1")

    base = load_profile(args.profile)
    specs = _sweep_specs(args.sets)
    variants = build_variants(
        base,
        seeds=args.seeds,
        sweep_specs=specs,
        malicious_combination_size=args.all_malicious_combinations,
    )

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "campaign_manifest.csv"

    rows: list[dict[str, str]] = []
    failed = 0
    print(f"Campaign variants: {len(variants)}")
    for index, profile in enumerate(variants, 1):
        fields = _variant_fields(profile, index)
        variant_dir = output_root / fields["variant"]
        command = _command(
            profile,
            output_dir=variant_dir,
            max_concurrent_clients=args.max_concurrent_clients,
        )
        fields["output_dir"] = str(variant_dir)
        fields["command"] = " ".join(shlex.quote(part) for part in command)

        print("")
        print(
            f"[{index}/{len(variants)}] "
            f"seed={fields['seed']} "
            f"attack={fields['mechanism']} "
            f"malicious=[{fields['malicious_client_ids']}]"
        )
        if args.dry_run:
            print(fields["command"])
            fields["return_code"] = "dry-run"
        else:
            completed = subprocess.run(command, check=False)
            fields["return_code"] = str(completed.returncode)
            if completed.returncode != 0:
                failed += 1
                rows.append(fields)
                if not args.continue_on_error:
                    break
                continue
        rows.append(fields)

    fieldnames = [
        "variant",
        "seed",
        "mechanism",
        "malicious_client_ids",
        "malicious_fraction",
        "schedule",
        "output_dir",
        "return_code",
        "command",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("")
    print(f"Campaign manifest: {manifest_path}")
    if failed:
        print(f"Failed variants: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
