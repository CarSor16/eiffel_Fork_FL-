"""Compact round-by-round storage for Eiffel experiments."""

from .round_store import RoundStore
from .serialization import decode_array, encode_array

__all__ = ["RoundStore", "encode_array", "decode_array"]
