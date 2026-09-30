#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
DEV=0
if [[ "${1:-}" == "--dev" ]]; then DEV=1; fi

"$PYTHON_BIN" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 10), f"Python 3.10 required, got {sys.version}"
print("Python:", sys.version.split()[0])
PY

if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip wheel "setuptools==80.9.0"
.venv/bin/python -m pip install -c constraints-py310.txt -e .
if [[ "$DEV" == "1" ]]; then
  .venv/bin/python -m pip install -c constraints-py310.txt "pytest>=7.3,<8"
fi

.venv/bin/python -m pip check
.venv/bin/python - <<'PY'
import tensorflow as tf
import flwr, ray, h5py, numpy, pandas, pyarrow
print("TensorFlow:", tf.__version__)
print("Flower:", flwr.__version__)
print("Ray:", ray.__version__)
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("PyArrow:", pyarrow.__version__)
print("Environment: OK")
PY
