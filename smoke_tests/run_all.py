"""Runs every smoke test in this directory and exits non-zero on first failure.

These are shape/gradient-flow checks on random tensors and synthetic arrays --

they verify the paper-aligned code paths execute and differentiate correctly,

not that they reproduce the paper's reported numbers (which requires the real

dataset and a full training run).

"""

import _pathsetup  # noqa: F401

import runpy

import sys

from pathlib import Path


TEST_MODULES = [

    "test_tokenizer",

    "test_site_dia",

    "test_recon_loss",

    "test_psf_kernels",

    "test_baselines",

    "test_perturbations",

]


def main():

    here = Path(__file__).parent

    failures = []

    for name in TEST_MODULES:

        path = here / f"{name}.py"

        print(f"--- running {name} ---")

        try:

            runpy.run_path(str(path), run_name="__main__")

        except Exception as exc:

            failures.append((name, exc))

            print(f"FAIL: {name}: {exc}")


    print()

    if failures:

        print(f"{len(failures)}/{len(TEST_MODULES)} smoke tests failed:")

        for name, exc in failures:

            print(f"  - {name}: {exc}")

        sys.exit(1)


    print(f"All {len(TEST_MODULES)} smoke tests passed.")


if __name__ == "__main__":

    main()

