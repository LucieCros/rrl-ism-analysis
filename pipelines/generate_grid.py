#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_grid.py
================
Generate parameter-space grids of CRRL line widths and integrated areas,
used as model lookups for chi² fitting in :mod:`spectral_tools.grid_exploration`.

For each principal quantum number ``n`` in the configured subset, a NetCDF
file is written containing two 6-D arrays (n, Te, Ne, T0, L, vt):

- ``deltaf`` : total Voigt FWHM of the line [Hz]
- ``area``   : integrated optical depth area [arbitrary unit]

Two grid resolutions are available via the ``--mode`` flag:

``coarse`` (default)
    Wide parameter ranges for initial exploration.

``fine``
    Narrow, well-resolved ranges for refinement around the best-fit solution.

Grid parameters (ranges, step sizes) and quantum numbers are read from an
external YAML configuration file, making them fully user-configurable without
touching the code. The default config file is ``files/grid_config.yaml``.

Usage
-----
::

    # Coarse grid with default config
    python3 generate_grid.py --mode coarse

    # Fine grid with custom config
    python3 generate_grid.py --mode fine --config my_grid_config.yaml

    # Override output directory
    python3 generate_grid.py --mode coarse --outdir /data/grids

Command-line arguments
----------------------
--mode      Grid resolution: ``coarse`` or ``fine`` (default: ``coarse``).
--config    Path to the YAML configuration file
            (default: ``files/grid_config.yaml``).
--outdir    Override the output directory defined in the config file.
--datadir   Directory containing ``B1B2.pickle`` and ``alphagamma.pickle``
            (default: value of ``spectral_tools.modeling.FILES_PATH``).

Configuration file format
--------------------------
See ``files/grid_config.yaml`` for a fully documented example. The file
defines two top-level keys (``coarse`` and ``fine``), each containing:

- ``outdir``          : output directory for NetCDF files
- ``quantum_numbers`` : list of integer principal quantum numbers
- ``parameters``      : dict of ``{name: [start, stop, step]}`` ranges for
                        Te, Ne, T0, L, vt

Dependencies
------------
numpy, xarray, astropy, tqdm, pickle, pyyaml
Internal: spectral_tools.atoms, spectral_tools.modeling
"""

import argparse
import os
import time
import warnings

import numpy as np
import tqdm
import xarray as xr
import yaml

from spectral_tools.atoms import line_freq
from spectral_tools.modeling import (
    FILES_PATH,
    FUNCB1B2,
    harmsuminfini,
    doppler_broadening,
    pressure_broadening,
    radiation_broadening,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: str = "files/grid_config.yaml"
ALPHA: float        = -2.6


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str, mode: str) -> dict:
    """
    Load and validate grid configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    mode : str
        Grid mode: ``'coarse'`` or ``'fine'``.

    Returns
    -------
    dict with keys:
        - ``outdir``          : str — output directory
        - ``quantum_numbers`` : numpy.ndarray of int
        - ``Te, Ne, T0, L, vt`` : numpy.ndarray — parameter axes

    Raises
    ------
    KeyError
        If ``mode`` is not found in the YAML file.
    ValueError
        If a required key is missing from the mode section.
    """
    with open(config_path, "r") as fh:
        full_cfg = yaml.safe_load(fh)

    if mode not in full_cfg:
        raise KeyError(
            f"Mode '{mode}' not found in '{config_path}'. "
            f"Available modes: {list(full_cfg.keys())}"
        )

    cfg = full_cfg[mode]

    # Validate required keys
    for key in ("outdir", "quantum_numbers", "parameters"):
        if key not in cfg:
            raise ValueError(
                f"Missing required key '{key}' under mode '{mode}' "
                f"in '{config_path}'."
            )
    for param in ("Te", "Ne", "T0", "L", "vt"):
        if param not in cfg["parameters"]:
            raise ValueError(
                f"Missing parameter '{param}' under "
                f"'{mode}.parameters' in '{config_path}'."
            )

    result = {
        "outdir":          cfg["outdir"],
        "quantum_numbers": np.array(cfg["quantum_numbers"], dtype=int),
    }
    for param, (start, stop, step) in cfg["parameters"].items():
        result[param] = np.arange(start, stop, step)

    return result


# ---------------------------------------------------------------------------
# Vectorised bn·βn block computation
# ---------------------------------------------------------------------------

def bnbetan_block(n_block: np.ndarray, Te: np.ndarray,
                  Ne: np.ndarray) -> np.ndarray:
    """
    Compute the non-LTE product bn·βn for a block of quantum numbers.

    Evaluates the pre-computed interpolators from ``B1B2.pickle`` on the
    full (Te, Ne) grid for each n in the block. Vectorised over n but not
    over (Te, Ne) to keep memory usage bounded.

    Parameters
    ----------
    n_block : numpy.ndarray, shape (N,)
        Principal quantum numbers.
    Te : numpy.ndarray, shape (nTe,)
        Electron temperatures [K].
    Ne : numpy.ndarray, shape (nNe,)
        Electron densities [cm⁻³].

    Returns
    -------
    numpy.ndarray, shape (N, nTe, nNe), dtype float32
        bn·βn values on the (n, Te, Ne) grid.
    """
    out = np.empty((len(n_block), len(Te), len(Ne)), dtype=np.float32)
    for i, n in enumerate(n_block):
        f1, f2 = FUNCB1B2[str(int(n))]
        out[i] = (f1(Te[:, None], Ne[None, :])
                  * f2(Te[:, None], Ne[None, :]))
    return out


# ---------------------------------------------------------------------------
# Vectorised total broadening (grid-optimised)
# ---------------------------------------------------------------------------

def total_broadening_grid(n_g, Ne, Te, somme_inf, T0, vt,
                           nu0_arr, logn_arr, n2_arr,
                           n_offset, alpha=ALPHA) -> np.ndarray:
    """
    Total Voigt FWHM [MHz] on a 6-D (n, Te, Ne, T0, L, vt) grid.

    Uses the Thompson (1987) approximation:
    FWHM_V ≈ 0.5346·dL + sqrt(0.2166·dL² + dG²)

    Pre-computed arrays (``nu0_arr``, ``logn_arr``, ``n2_arr``) are passed
    in to avoid redundant calls to :func:`~spectral_tools.atoms.line_freq`
    inside the loop.

    Parameters
    ----------
    n_g : numpy.ndarray, shape (n, 1, 1, 1, 1, 1)
        Quantum numbers broadcast over the grid axes.
    Ne, Te, T0, vt : numpy.ndarray
        Physical parameter grids, shape (1, nTe, nNe, nT0, nL, nvt).
    somme_inf : float
        Pre-computed harmonic sum for radiation broadening.
    nu0_arr : numpy.ndarray
        Central frequencies for each n in the block [MHz].
    logn_arr : numpy.ndarray
        log(n) for each n in the block.
    n2_arr : numpy.ndarray
        n² for each n in the block.
    n_offset : int
        Index of the first n in the block (for relative indexing).
    alpha : float, optional
        Galactic background spectral index. Default -2.6.

    Returns
    -------
    numpy.ndarray
        Total Voigt FWHM [MHz], same shape as the broadcast grid.
    """
    idx   = n_g.ravel().astype(int) - n_offset
    nu0_g = nu0_arr[idx, None, None, None, None, None]

    # Gaussian (Doppler) broadening
    dG = doppler_broadening(n_g, Te, vt, nu0_g)

    # Natural broadening (Lorentzian component)
    dL_nat = (1.2e-6
              * logn_arr[idx, None, None, None, None, None]
              / n2_arr[idx,  None, None, None, None, None]
              * nu0_g)

    # Total Lorentzian: pressure + radiation + natural
    dL = (pressure_broadening(n_g, Ne, Te)
          + radiation_broadening(n_g, T0, somme_inf, nu0_g, alpha)
          + dL_nat)

    # Thompson (1987) Voigt FWHM approximation
    return 0.5346 * dL + np.sqrt(0.2166 * dL**2 + dG**2)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate CRRL parameter-space grids (coarse or fine)."
    )
    parser.add_argument(
        "--mode", choices=["coarse", "fine"], default="coarse",
        help="Grid resolution mode (default: coarse).",
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help=f"Path to the YAML config file (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Override the output directory defined in the config file.",
    )
    parser.add_argument(
        "--datadir", type=str, default=FILES_PATH,
        help=f"Directory with B1B2.pickle and alphagamma.pickle "
             f"(default: {FILES_PATH}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Load configuration ──────────────────────────────────────────────────
    cfg = load_config(args.config, args.mode)

    # CLI --outdir overrides config value
    outdir = args.outdir or cfg["outdir"]
    os.makedirs(outdir, exist_ok=True)

    quantum_numbers = cfg["quantum_numbers"]
    Te = cfg["Te"]
    Ne = cfg["Ne"]
    T0 = cfg["T0"]
    L  = cfg["L"]
    vt = cfg["vt"]

    print(f"Grid mode      : {args.mode}")
    print(f"Config file    : {args.config}")
    print(f"Output dir     : {outdir}")
    print(f"Grid shape     : Te={len(Te)} × Ne={len(Ne)} × "
          f"T0={len(T0)} × L={len(L)} × vt={len(vt)}")
    print(f"Quantum nbs ({len(quantum_numbers)}): {quantum_numbers.tolist()}")
    print()

    # ── Pre-compute quantities independent of n ─────────────────────────────
    SOMME_INF = harmsuminfini(ALPHA - 2, precision=3)

    Te_g, Ne_g, T0_g, L_g, vt_g = np.meshgrid(
        Te, Ne, T0, L, vt, indexing="ij"
    )

    # Area contributions independent of n
    contrib_temp    = (Te_g / 100.0) ** (-2.5)
    contrib_density = (Ne_g / 0.1) ** 2

    # ── Main loop over quantum numbers ──────────────────────────────────────
    time_0 = time.time()

    for nq in tqdm.tqdm(quantum_numbers,
                        desc=f"Generating {args.mode} grid"):

        n_block  = np.array([nq])
        nu0_arr  = np.array([line_freq(nq).value])  # central frequency [MHz]
        logn_arr = np.log(n_block)
        n2_arr   = n_block ** 2

        n_g   = n_block[:, None, None, None, None, None]
        Te_g2 = Te_g[None, ...]
        Ne_g2 = Ne_g[None, ...]
        T0_g2 = T0_g[None, ...]
        vt_g2 = vt_g[None, ...]

        # ── deltaf : total Voigt FWHM [Hz] ──────────────────────────────────
        deltaf_block = total_broadening_grid(
            n_g,
            Ne=Ne_g2, Te=Te_g2, T0=T0_g2, vt=vt_g2,
            somme_inf=SOMME_INF,
            nu0_arr=nu0_arr, logn_arr=logn_arr, n2_arr=n2_arr,
            n_offset=int(n_block[0]),
            alpha=ALPHA,
        ).astype(np.float32) * 1e6   # MHz → Hz

        # ── area : integrated optical depth ─────────────────────────────────
        contrib_lte = bnbetan_block(n_block, Te, Ne)
        contrib_lte = contrib_lte[:, :, :, None, None, None]

        area_block = (
            0.2
            * contrib_lte
            * contrib_temp[None, ...]
            * contrib_density[None, ...]
            * L_g[None, ...]
        ).astype(np.float32)

        # ── Write compressed NetCDF ──────────────────────────────────────────
        ds = xr.Dataset(
            data_vars={
                "deltaf": (("n", "Te", "Ne", "T0", "L", "vt"), deltaf_block),
                "area":   (("n", "Te", "Ne", "T0", "L", "vt"), area_block),
            },
            coords={"n": n_block, "Te": Te, "Ne": Ne,
                    "T0": T0, "L": L, "vt": vt},
            attrs={
                "mode":        args.mode,
                "config":      args.config,
                "alpha":       ALPHA,
                "description": "CRRL parameter-space grid — deltaf [Hz], area [arb]",
            },
        )
        encoding = {
            "deltaf": {"zlib": True, "complevel": 4, "dtype": "float32"},
            "area":   {"zlib": True, "complevel": 4, "dtype": "float32"},
        }
        ds.to_netcdf(
            os.path.join(outdir, f"grid-{nq}.nc"),
            mode="w", engine="netcdf4", encoding=encoding,
        )

    elapsed = int(time.time() - time_0)
    print(
        f"\nDone. Grid {len(Te)}×{len(Ne)}×{len(T0)}×{len(L)}×{len(vt)} "
        f"({len(quantum_numbers)} transitions) computed in {elapsed} s"
    )


if __name__ == "__main__":
    main()
