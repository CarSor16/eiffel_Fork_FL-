"""Small helpers for transporting NumPy arrays through Flower metrics."""

from __future__ import annotations

import base64
import io
import zlib

import numpy as np


def encode_array(array: np.ndarray, dtype: str = "float16") -> str:
    """Encode an array for transient transport in a Flower Scalar string.

    The persisted representation is HDF5, not this base64 string. Base64 is used only
    between client and server because Flower metrics accept scalar values.
    """
    arr = np.asarray(array, dtype=np.dtype(dtype))
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(zlib.compress(buf.getvalue(), level=6)).decode("ascii")


def decode_array(payload: str) -> np.ndarray:
    """Decode an array produced by :func:`encode_array`."""
    raw = zlib.decompress(base64.b64decode(payload.encode("ascii")))
    return np.load(io.BytesIO(raw), allow_pickle=False)
