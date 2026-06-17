#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_time_average.py
===============
Weighted temporal averaging pipeline for NenuFAR spectral observations.

Reads all cleaned ``temp_*.fits`` files for a given source and RRL series,
applies optional LSR Doppler corrections to align them on a common frequency
grid, computes the weighted average optical depth and flux density spectra,
estimates a local RMS map, and writes a single stacked FITS file.

Usage
-----
::

    python3 run_time_average.py -source CASA -l Calph

    python3 run_time_average.py -source CYGA -gal -l Calph

    python3 run_time_average.py -source TAUA -off -l Calph

    # Skip Nov 2023 observations and stack the rest:
    python3 run_time_average.py -source CASA -l Calph -passnov

Command-line arguments
----------------------
-source     Source identifier (required), e.g. ``CASA``, ``TAUA``.
-l          RRL series (default: ``Calph``).
-nocorr     Skip LSR Doppler correction (grid on rest frequencies instead).
-gal        Galaxy mode: use galactic velocity and gal output path.
-passnov    Exclude Oct/Nov 2023 observations from the stack.
-off        Stack OFF-beam observations.
-onoff      Stack ON+OFF observations together.

Output
------
A single FITS file with three extensions:

- HDU 0 : averaged optical depth τ(ν)
- HDU 1 : averaged flux density S(ν) [Jy]
- HDU 2 : local RMS σ(ν) computed over a ±1024-channel sliding window

The output filename follows the convention::

    alltime_{source}_{line}.fits
    alltime_{source}_{line}_OFF.fits    (if -off)
    alltime_{source}_{line}_ONOFF.fits  (if -onoff)
    no_nov-alltime_{source}.fits        (if -passnov)

Dependencies
------------
Standard library: argparse, os, time, warnings
Third-party: numpy, astropy (fits, units), tqdm
Internal: spectral_tools.tools (doppler_corrections, doppler_correction,
          get_linefree), spectral_tools.io
"""

import argparse
import os
import time
import warnings

import numpy as np
from astropy.io import fits
import astropy.units as u
from tqdm import tqdm

import spectral_tools.tools as tools
from spectral_tools.io import (
    read_source_velocity,
    normalise_source_name,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Root directory for reduced per-observation data.
DATA_ROOT: str = "/home/lcros"

#: Source info table (tab-separated: name, ra, dec, velo, coeff).
SOURCE_INFO: str = "../files/source_info.txt"

#: RRL catalogue CSV.
RRLS_PATH: str = "../files/rrlines.csv"

#: Half-window [channels] for the sliding RMS computation.
RMS_HALF_WINDOW: int = 1024

#: Galactic source default velocity [km/s].
GALAXY_VELO: float = 16811.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Weighted time-averaging of cleaned NenuFAR observations."
    )
    parser.add_argument("-source",  "--source",  type=str, required=True,
                        help="Source identifier, e.g. CASA.")
    parser.add_argument("-nocorr",  "--no_corr", action="store_true",
                        help="Skip LSR Doppler correction.")
    parser.add_argument("-gal",     "--galaxy",  action="store_true",
                        help="Galaxy mode.")
    parser.add_argument("-l",       "--line",    type=str, default="Calph",
                        help="RRL series (default: Calph).")
    parser.add_argument("-passnov", "--passnov", action="store_true",
                        help="Exclude Oct/Nov 2023 observations.")
    parser.add_argument("-off",     "--off",     action="store_true",
                        help="Stack OFF-beam observations.")
    parser.add_argument("-onoff",   "--onoff",   action="store_true",
                        help="Stack ON+OFF observations together.")
    return parser.parse_args()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args       = parse_args()
    sourcename = args.source
    source     = normalise_source_name(sourcename)
    my_line    = args.line
    exclude_nov = args.passnov
    start      = time.time()

    # ── Build path to temp observations ────────────────────────────────────
    if args.galaxy:
        obs_path = os.path.join(DATA_ROOT, f"DATA-{sourcename}",
                                f"reduced-gal/{my_line}/")
    else:
        obs_path = os.path.join(DATA_ROOT, f"DATA-{sourcename}", f"{my_line}/")

    temp_path = os.path.join(obs_path, "temp")

    # ── Collect temp FITS files ─────────────────────────────────────────────
    all_obs = []
    for fname in os.listdir(temp_path):
        # Optionally skip Nov 2023 data
        if exclude_nov and (("202311" in fname) or ("202310" in fname)):
            continue
        if "temp" not in fname or "fits" not in fname:
            continue
        # Filter by beam polarity
        if args.off and "OFF" not in fname:
            continue
        if not args.onoff and not args.off and "OFF" in fname:
            continue
        all_obs.append(fname)

    print(f"Stacking {len(all_obs)} observations for source '{sourcename}'")

    # ── Accumulate per-observation arrays ───────────────────────────────────
    spectra:    list[np.ndarray] = []
    jy_spectra: list[np.ndarray] = []
    weights:    list[np.ndarray] = []
    fdopps:     list[np.ndarray] = []
    fmins:      list[float]      = []
    fmaxs:      list[float]      = []
    df:         float            = 0.0
    fmini:      float            = 0.0
    fmaxi:      float            = 0.0
    last_hdr                     = None   # kept for FITS header template

    for ite, fname in enumerate(tqdm(all_obs, desc="Loading observations")):

        # Read source velocity from info table
        velo = GALAXY_VELO if args.galaxy else read_source_velocity(
            source, is_off=args.off
        )
        if isinstance(velo, list):
            velo = velo[0]

        # Open temp FITS
        reduc = fits.open(os.path.join(temp_path, fname))
        hdr   = reduc[0].header
        ra    = hdr["CRVAL1"]
        dec   = hdr["CRVAL2"]
        date  = hdr["TIME"]

        # Build frequency axis and compute Doppler shift
        whole_freq = np.arange(hdr["FMIN"],
                               hdr["FMAX"] + hdr["CDELT3"],
                               hdr["CDELT3"])
        delta_v = tools.doppler_corrections(
            date, ra=ra, dec=dec
        )[-1].to(u.km / u.s).value

        fdopp = tools.doppler_correction(whole_freq, delta_v * 1000.0)

        # Guard: check frequency axis length matches spectrum
        spectrum    = reduc[0].data[:, 0, 0]
        jy_spectrum = reduc[1].data[:, 0, 0]
        weight      = 1.0 / reduc[-1].data[:, 0, 0]**2

        if len(spectrum) != len(fdopp):
            print(f"[SKIP] {fname}: length mismatch "
                  f"(spectrum={len(spectrum)}, fdopp={len(fdopp)})")
            reduc.close()
            continue

        fdopps.append(fdopp)
        fmins.append(fdopp[0])
        fmaxs.append(fdopp[-1])
        spectra.append(spectrum)
        jy_spectra.append(jy_spectrum)
        weights.append(weight)

        df    = hdr["CDELT3"]
        fmini = hdr["FMIN"]
        fmaxi = hdr["FMAX"] + df

        last_hdr = reduc[0].header.copy()
        reduc.close()

    if not spectra:
        print("No valid observations found — aborting.")
        return

    # ── Align on common frequency grid ─────────────────────────────────────
    if args.no_corr:
        # No Doppler correction: grid on rest frequencies
        spectra, jy_spectra, weights, fref = tools.align_on_rest_grid(
            spectra, jy_spectra, weights, fmini, fmaxi, df
        )
    else:
        # Doppler-corrected: grid on barycentric frequencies
        spectra, jy_spectra, weights, fref = tools.align_on_doppler_grid(
            spectra, jy_spectra, weights, fdopps, fmins, fmaxs, df
        )

    weights_arr = np.array(weights)

    # ── Weighted average ────────────────────────────────────────────────────
    weight_sum = np.nansum(weights_arr, axis=0)
    avg_tau    = np.nansum(spectra,    axis=0) / weight_sum
    avg_jy     = np.nansum(jy_spectra, axis=0) / weight_sum

    # ── Local RMS on line-free channels ────────────────────────────────────
    linefree = tools.get_linefree(fref, avg_tau, myLine=my_line,
                                   path_rrls=RRLS_PATH)
    rms_arr  = tools.sliding_rms(linefree, RMS_HALF_WINDOW)

    # ── Format as (N, 1, 1) cubes for FITS compatibility ────────────────────
    N = len(avg_tau)
    def _cube(arr):
        out = np.zeros((N, 1, 1))
        out[:, 0, 0] = arr
        return out

    # ── Build FITS HDUList ──────────────────────────────────────────────────
    hdr_tau = last_hdr.copy()
    hdr_tau.set("STACKNUMBER", len(spectra),
                "Number of observations in the stack")
    hdr_tau["FMIN"]   = min(fmins)
    hdr_tau["FMAX"]   = max(fmaxs)
    hdr_tau["CRVAL3"] = fref[-1]
    hdr_tau["CRPIX3"] = len(fref) - 1
    hdr_tau["NAXIS3"] = N
    hdr_tau["NAXIS1"] = 1
    hdr_tau["NAXIS2"] = 1
    hdr_tau["OBJECT"] = source
    hdr_tau["NAME"]   = "Optical depth"

    hdr_jy       = hdr_tau.copy()
    hdr_jy["NAME"] = "Intensity (Jy)"

    hdr_rms        = hdr_tau.copy()
    hdr_rms["NAME"] = "RMS"

    hdul = fits.HDUList([
        fits.PrimaryHDU(_cube(avg_tau), header=hdr_tau),
        fits.ImageHDU(_cube(avg_jy),    header=hdr_jy),
        fits.ImageHDU(_cube(rms_arr),   header=hdr_rms),
    ])

    # ── Write output ────────────────────────────────────────────────────────
    if exclude_nov:
        out_name = os.path.join(obs_path, f"no_nov-alltime_{source}.fits")
    elif args.off:
        out_name = os.path.join(obs_path, f"alltime_{source}_{my_line}_OFF.fits")
    elif args.onoff:
        out_name = os.path.join(obs_path, f"alltime_{source}_{my_line}_ONOFF.fits")
    else:
        out_name = os.path.join(obs_path, f"alltime_{source}_{my_line}.fits")

    hdul.writeto(out_name, overwrite=True)
    print(f"\nStacked output written to: {out_name}")
    print(f"Processing time: {(time.time() - start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
