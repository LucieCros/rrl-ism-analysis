# -*- coding: utf-8 -*-
"""
migrate_alphagamma_pickle.py
=============================

One-off conversion of ``files/alphagamma.pickle`` (pickled ``scipy``
``CubicSpline`` objects) into ``files/alphagamma.npz`` (plain numpy arrays).

Why
---
``CubicSpline`` is a subclass of ``PPoly``, and scipy has changed the
internal pickle state format of ``PPoly`` (adding an array-namespace field
as part of Array API support). A pickle written by one scipy version can
fail to unpickle on another with e.g.::

    AttributeError: 'tuple' object has no attribute 'pop'

Storing the raw breakpoints/coefficients instead removes the dependency on
scipy's pickle format entirely: ``spectral_tools.modeling`` rebuilds the
splines at import time with ``scipy.interpolate.PPoly(c, x, extrapolate,
axis)``, which is a stable public constructor and reproduces the original
``CubicSpline`` evaluation exactly (same ``x``/``c`` arrays, same
``__call__`` code path).

Only ``alpha_f`` and ``gamma_f`` are converted: the pickle also contains
``grad_alpha_f``/``grad_gamma_f``, but ``modeling._load_interpolators``
discards those immediately and nothing else in the codebase uses them.

Usage
-----
Run this once, in an environment where the *existing* pickle can still be
loaded (i.e. before upgrading scipy on that machine)::

    python pipelines/migrate_alphagamma_pickle.py

The original ``alphagamma.pickle`` is left untouched.
"""

import argparse
import os
import pickle

import numpy as np

DEFAULT_FILES_DIR = os.path.join(os.path.dirname(__file__), "..", "files")


def convert(src: str, dst: str) -> None:
    """Extract raw PPoly arrays from the pickled splines and save as .npz."""
    with open(src, "rb") as fh:
        alpha_f, _grad_alpha_f, gamma_f, _grad_gamma_f = pickle.load(fh)

    np.savez(
        dst,
        alpha_x=alpha_f.x,
        alpha_c=alpha_f.c,
        alpha_extrapolate=np.array(alpha_f.extrapolate),
        alpha_axis=np.array(alpha_f.axis),
        gamma_x=gamma_f.x,
        gamma_c=gamma_f.c,
        gamma_extrapolate=np.array(gamma_f.extrapolate),
        gamma_axis=np.array(gamma_f.axis),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default=os.path.join(DEFAULT_FILES_DIR, "alphagamma.pickle"),
        help="Existing pickle file to read (default: files/alphagamma.pickle)",
    )
    parser.add_argument(
        "--dst",
        default=os.path.join(DEFAULT_FILES_DIR, "alphagamma.npz"),
        help="Output .npz file to write (default: files/alphagamma.npz)",
    )
    args = parser.parse_args()
    convert(args.src, args.dst)
    print(f"Wrote {args.dst}")
