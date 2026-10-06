"""Eiffel: FL-security experimentation framework built on Flower."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")
os.environ.setdefault("OMP_NUM_THREADS", "64")
