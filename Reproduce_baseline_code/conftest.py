"""Test-session setup.

Anaconda's MKL-linked NumPy and the pip-installed PyTorch each bring their own
copy of the Intel OpenMP runtime (``libiomp5md.dll``). Importing both in one
process aborts inside LAPACK on Windows. The environment variable below must be
set before either library is imported, which is why it lives in ``conftest.py``
rather than in a test module.

This is a property of the local Anaconda + PyTorch install, not of the
reproduced algorithms. The durable fix is a single OpenMP runtime in the
environment (e.g. ``conda install -c conda-forge pytorch``, or a venv with
pip-installed NumPy so both come from the same wheel family); until then this
keeps the suite runnable.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Keep BLAS single-threaded during tests so timings and the closed-form fits
# are reproducible across machines.
os.environ.setdefault("OMP_NUM_THREADS", "1")
