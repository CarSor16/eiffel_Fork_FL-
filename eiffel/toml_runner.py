"""Backward-compatible import surface for the direct TOML runner.

The previous implementation translated TOML into Hydra overrides.  Hydra is no
longer part of the runtime; callers should use the same public functions, now
implemented by eiffel.direct_runner.
"""

from eiffel.direct_runner import (
    ExperimentConfigError,
    TomlExperimentError,
    apply_overrides,
    build_command,
    load_profile,
    main,
    resolve_profile,
    run_profile,
    set_path,
)

__all__ = [
    "ExperimentConfigError",
    "TomlExperimentError",
    "apply_overrides",
    "build_command",
    "load_profile",
    "main",
    "resolve_profile",
    "run_profile",
    "set_path",
]

if __name__ == "__main__":
    raise SystemExit(main())
