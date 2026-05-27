#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_observation.py
====================
Single-observation spectral cleaning pipeline for NenuFAR RRL data.

This script processes one raw FITS observation bloc (two polarisation lanes)
through a multi-order baseline subtraction and RFI mitigation sequence,
and writes a cleaned FITS file ready for time-averaging.

It is normally called as a subprocess by :mod:`pipelines.run_cleaning`,
but can also be run standalone for testing or manual inspection.

Cleaning sequence (per sub-band)
---------------------------------
0. Global large-scale flattening (rebinning + smooth division)
1. Global sigma-clipping on the full band
2. Per sub-band 0th-order mitigation : narrow spikes (gradient threshold)
3. Per sub-band 1st-order flattening : S-shape from median of neighbours
4. Per sub-band 1st-order RFI mitigation : broader spikes
5. Per sub-band 2nd-order flattening : residual faint baseline
6. Per sub-band 3rd-order mitigation : intra-line filter (Savitzky-Golay)
7. Per sub-band 4th-order mitigation : remaining positive spikes
8. Final global sliding sigma-clipping

Output FITS structure
---------------------
- HDU 0 : optical depth τ(ν),    shape (N, 1, 1)
- HDU 1 : flux density S(ν) [Jy], shape (N, 1, 1)
- HDU 2 : RMS on line-free channels, shape (N, 1, 1)
- HDU 3 : total RMS,               shape (N, 1, 1)

Usage
-----
::

    python3 clean_observation.py \\
        -path /home/lcros/DATA-CASA/Calph/ \\
        -name lines_CASA_20210609_120000 \\
        -ext .spectra.0.fits \\
        -v -47 -ra 23h23m24s -dec 58d48m54 \\
        -coeff [1,1] -line Calph -mask 1.8 -cw 200

    # Special mode for bright sources (relaxed sigma-clipping):
    python3 clean_observation.py ... -spec

    # Diagnostic plots for sub-bands 134 and 135:
    python3 clean_observation.py ... -plot -diag [134,135]

Command-line arguments
----------------------
-path       Output directory for this source/line.
-name       Observation name stem (without lane index and extension).
-ext        File extension including polarisation suffix (e.g. ``.spectra.0``).
-v          Source velocity [km/s], scalar or Python list for multi-component.
-ra         Right ascension of the target (J2000), e.g. ``23h23m24s``.
-dec        Declination of the target (J2000), e.g. ``58d48m54``.
-line       RRL series to process (default: ``Calph``).
-coeff      Continuum model coefficients as a Python list, e.g. ``[1.2, -0.7]``.
-fct        Continuum model function name in ``tools`` (default:
            ``continuum_polyn``).
-mask       Mask half-width in units of the expected line Voigt FWHM.
            Pass 0 to disable. Default 1.8.
-Nsig       Sigma-clipping threshold in units of local RMS. Default 3.
-sv         Savitzky-Golay window divisor for 1st-order flattening. Default 10.
-spec       Special mode: skip global sigma-clipping, use wider intra-line
            filter. For bright sources where normal clipping destroys the line.
-off        Process OFF-beam lanes (2 and 3 instead of 0 and 1).
-plot       Save diagnostic PNG plots for the sub-bands listed in -diag.
-diag       Python list of sub-band indices to plot, e.g. ``[134,135]``.
-test       Test mode: read files from the local directory instead of
            the cluster data root.

Dependencies
------------
Standard library: sys, os, time, argparse, warnings
Third-party: numpy, astropy, scipy, pandas, tqdm
Internal: spectral_tools.tools, spectral_tools.io, spectral_tools.modeling,
          spectral_tools.graphics
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm

import spectral_tools.tools as tools
import spectral_tools.modeling as mdl
import spectral_tools.graphics as graphics
from spectral_tools.io import load_bloc, build_fits_header

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: Root path for raw input FITS blocs on the cluster.
DATA_ROOT: str = "/juliette2/datartemix2/lt10"

#: Path to the RRL catalogue CSV.
RRLS_PATH: str = "files/rrlines.csv"

#: Path to the template FITS header text file.
FITS_HEADER_TEMPLATE: str = "files/fitsheader.txt"

#: Reference CNM physical parameters for estimating expected line widths.
#: Used to set the mask width in units of the Voigt FWHM.
CNM_REF = dict(Te=100.0, Ne=2e-2, T0=2000.0, L=10.0, vt=4.0)

#: Valid quantum number range for bn·βn tables.
N_MIN_VALID: int = 426
N_MAX_VALID: int = 850

#: Maximum iterations for sigma-clipping loops (safety cap).
MAX_CLIP_ITER: int = 120

#: Minimum fraction of valid (non-NaN) channels required to keep a sub-band.
MIN_VALID_FRAC: float = 0.30   # i.e. at most 70 % NaN allowed

#: Minimum number of valid neighbouring sub-bands for 1st-order flattening.
MIN_GOOD_NEIGHBOURS: int = 4

#: Number of neighbouring sub-bands on each side used for 1st-order flattening.
N_NEIGHBOURS: int = 11

#: Edge protection: fraction of channels on each side left unmasked.
EDGE_FRAC: int = 70   # NCHAN // EDGE_FRAC channels protected on each edge


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Single-observation spectral cleaning for NenuFAR RRL data."
    )
    parser.add_argument("-path",  "--path",               type=str)
    parser.add_argument("-name",  "--name",               type=str)
    parser.add_argument("-ext",   "--extension",          type=str)
    parser.add_argument("-v",     "--velo")
    parser.add_argument("-line",  "--myLine",             type=str,
                        default="Calph")
    parser.add_argument("-ra",    "--ra",                 type=str)
    parser.add_argument("-dec",   "--dec",                type=str)
    parser.add_argument("-Nsig",  "--Nsigma",             type=int,  default=3)
    parser.add_argument("-coeff", "--coeffs",             default=[1, 1])
    parser.add_argument("-fct",   "--continuum_function", type=str,
                        default="continuum_polyn")
    parser.add_argument("-mask",  "--mask",               type=float,
                        default=1.8)
    parser.add_argument("-plot",  "--plot",               action="store_true")
    parser.add_argument("-diag",  "--diagnostic",         type=str,
                        default="[]")
    parser.add_argument("-spec",  "--special",            action="store_true")
    parser.add_argument("-off",   "--off",                action="store_true")
    parser.add_argument("-test",  "--test",               action="store_true")
    parser.add_argument("-sv",    "--savgol_window",      type=int,  default=10)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sub-band reliability check
# ---------------------------------------------------------------------------

def _subband_is_unreliable(SUBREF: np.ndarray, SUBS2: np.ndarray,
                            count_good_sb: int, NCHAN: int) -> bool:
    """
    Return ``True`` if a sub-band should be skipped (filled with NaN).

    A sub-band is considered unreliable if:
    - All neighbouring sub-bands are NaN
    - More than 70 % of its own channels are NaN
    - Fewer than :data:`MIN_GOOD_NEIGHBOURS` valid neighbours were found

    Parameters
    ----------
    SUBREF : numpy.ndarray
        Current sub-band spectrum.
    SUBS2 : numpy.ndarray, shape (N_NEIGHBOURS, NCHAN)
        Stack of neighbouring sub-band spectra.
    count_good_sb : int
        Number of valid neighbouring sub-bands used.
    NCHAN : int
        Number of channels per sub-band.

    Returns
    -------
    bool
    """
    all_nan       = len(SUBS2[~np.isnan(SUBS2)]) == 0
    too_many_nan  = np.sum(np.isnan(SUBREF)) > (1 - MIN_VALID_FRAC) * NCHAN
    too_few_neigh = count_good_sb < MIN_GOOD_NEIGHBOURS
    return all_nan or too_many_nan or too_few_neigh


# ---------------------------------------------------------------------------
# Sub-band edge detection
# ---------------------------------------------------------------------------

def _find_clean_edges(meansmooth: np.ndarray, NCHAN: int,
                       Nsigma: float) -> tuple[int, int]:
    """
    Detect the channel indices of the clean interior of a sub-band.

    Uses the second derivative of the smoothed baseline to locate sharp
    sub-band edges, then returns conservative interior bounds.

    Parameters
    ----------
    meansmooth : numpy.ndarray
        Smoothed baseline estimate for this sub-band.
    NCHAN : int
        Number of channels per sub-band.
    Nsigma : float
        Detection threshold in units of the local gradient std.

    Returns
    -------
    idx_lo : int
        Index of the left clean interior boundary.
    idx_hi : int
        Index of the right clean interior boundary.
    """
    wind_len = len(meansmooth) // 10
    wind_len += 1 - wind_len % 2

    grad        = np.abs(np.diff(meansmooth, 2))
    grad        = np.where(np.isnan(grad), 0.0, grad)
    grad_smooth = savgol_filter(grad, wind_len, 5, mode="interp")

    # Evaluate threshold on the central half only (avoid edge artefacts)
    mid     = grad_smooth[NCHAN // 4: NCHAN * 3 // 4]
    thresh  = np.nanmean(np.abs(mid)) + Nsigma * np.nanstd(mid)
    indexes = np.where(np.abs(grad_smooth) > thresh)[0]

    if len(indexes) <= 2:
        return NCHAN // 10, NCHAN - NCHAN // 10

    consec = max(np.diff(indexes, 1)) == 1
    if consec and np.mean(indexes) <= NCHAN / 2:
        return max(indexes), NCHAN - NCHAN // 10
    if consec and np.mean(indexes) >= NCHAN / 2:
        return NCHAN // 10, min(indexes)

    jump = np.argmax(np.abs(np.diff(indexes, 1)))
    lo   = np.clip(indexes[jump],     NCHAN // 10,       NCHAN // 2 - NCHAN // 10)
    hi   = np.clip(indexes[jump + 1], NCHAN // 2 + NCHAN // 10, NCHAN - NCHAN // 10)
    return lo, hi


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_observation(args: argparse.Namespace) -> None:
    """
    Run the full cleaning pipeline for one observation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (see :func:`parse_args`).
    """
    start = time.time()

    # ── Parse scalar/list velocity ──────────────────────────────────────────
    velo = eval(args.velo)
    if not isinstance(velo, list):
        velo = [float(velo)]
    else:
        velo = list(np.array(velo, dtype=float))

    # ── Continuum model ─────────────────────────────────────────────────────
    coeffs        = np.array(eval(args.coeffs), dtype=float)
    continuum_fct = getattr(tools, args.continuum_function)

    # ── Diagnostic sub-band list ────────────────────────────────────────────
    diag_bands = eval(args.diagnostic)

    # ── Mode-dependent parameters ───────────────────────────────────────────
    if args.special:
        pval      = 0.5    # looser chi² threshold for misshapen-subband detection
        Nsiglarge = 7      # wider gradient threshold for spike detection
    else:
        pval      = 0.05
        Nsiglarge = 5

    # Fraction of a NDATA pixel below which it is treated as fully flagged
    epsi = 0.2

    # ── Log file ────────────────────────────────────────────────────────────
    is_galactic = velo[0] >= 1000
    if is_galactic:
        logdir = os.path.join(args.path, f"reduced-gal/{args.myLine}")
        suffix = "OFF" if args.off else ""
        logfilename = os.path.join(logdir, f"LOG_temp_{args.name}{suffix}.txt")
    else:
        logdir = os.path.join(args.path, f"{args.myLine}/temp")
        suffix = "OFF" if args.off else ""
        logfilename = os.path.join(logdir, f"LOG_temp_{args.name}{suffix}.txt")

    logfile = open(logfilename, "w")
    logfile.write(f"Log for reduction of file {args.name}\n")
    logfile.write(f"created : {time.strftime('%Y/%m/%d-%H:%M:%S')}\n")
    logfile.write(f"Command line : {sys.argv}\n\n")

    # ── Lane selection (ON beam: 0,1 — OFF beam: 2,3) ───────────────────────
    LANES = [2, 3] if args.off else [0, 1]

    # ── Output accumulators ─────────────────────────────────────────────────
    ITOTAL_flat = np.array([])
    ITOTAL_Jy   = np.array([])
    Ilinefree   = np.array([])
    FTOTAL      = np.array([])

    # ===========================================================================
    # MAIN LOOP — one iteration per polarisation lane
    # ===========================================================================
    for Lane_ in LANES:

        Lane = Lane_ % 2   # logical lane index within the pair (0 or 1)

        # ── Load observation bloc ───────────────────────────────────────────
        if args.test:
            bloc_path = f"{args.path.split('/')[-2]}/{args.name}{Lane_}{args.extension}"
        else:
            bloc_path = os.path.join(
                DATA_ROOT, args.path.split("/")[-2],
                f"{args.name}{Lane_}{args.extension}"
            )

        HDU   = load_bloc(bloc_path, path_rrls=RRLS_PATH)
        NCHAN = HDU.NCHANNELS[0]
        df    = HDU.DF[0] / 1000.0   # channel width [MHz]

        # ── LSR Doppler correction ──────────────────────────────────────────
        DELTAV = tools.doppler_corrections(
            HDU.date, ra=args.ra, dec=args.dec
        )[-1].to(u.km / u.s).value

        # ── Expected line positions and widths ──────────────────────────────
        # Used to set the line mask width in units of the Voigt FWHM.
        v0         = velo[0]
        lines_full = tools.get_line(10, 85, v0 - DELTAV,
                                    args.myLine, path=RRLS_PATH)
        every_line0 = np.array(lines_full)
        every_n0    = np.array(lines_full.index)

        # Restrict to the valid bn·βn range [N_MIN_VALID, N_MAX_VALID]
        valid      = (every_n0 < N_MAX_VALID) & (every_n0 > N_MIN_VALID)
        every_line = every_line0[valid]
        every_n    = every_n0[valid]

        # Expected Voigt FWHM for each transition at reference CNM parameters
        wV  = np.array([
            mdl.total_broadening(n, **CNM_REF) for n in every_n
        ])
        wVs = pd.DataFrame(wV, index=every_n)

        # The mask half-width is always expressed as a multiple of the
        # expected Voigt FWHM at CNM reference parameters (args.mask × wV).
        # Pass -mask 0 on the CLI to disable masking entirely.
        mask = args.mask

        # ── Time integration ────────────────────────────────────────────────
        NDATA = HDU.NDATA.copy()
        NDATA[NDATA < 1 - epsi] = 0.0   # flag pixels below validity threshold
        DATA        = HDU.DATA
        GRID        = NDATA * DATA
        WEIGHTSSUM  = np.sum(NDATA, axis=1)
        NSUB        = HDU.NBEAMLETS[0]

        integ    = np.sum(GRID, axis=1) / WEIGHTSSUM
        I_noflat = np.copy(integ)
        freq_all = np.copy(HDU.freq)

        # ── Step 0 : global large-scale flattening ──────────────────────────
        # Rebin to one value per sub-band, interpolate back, divide.
        # Removes the broad spectral envelope (bandpass shape).
        I_rebin, f_rebin  = tools.rebinning(I_noflat, freq_all, NCHAN)
        I_smooth_large    = interp1d(f_rebin, I_rebin,
                                      fill_value="extrapolate",
                                      bounds_error=False)(freq_all)
        large_correction  = I_noflat / I_smooth_large - 1.0

        # ── Step 1 : global sigma-clipping ──────────────────────────────────
        # sliding_rms gives a local noise estimate per channel without
        # requiring an interpolation step, consistent with the rest of
        # the pipeline.
        ystd_interp = tools.sliding_rms(large_correction, NCHAN // 2)
        lenwrongs   = int(np.sum(np.abs(large_correction) > args.Nsigma * ystd_interp))

        if args.special:
            logfile.write("No global sigma clipping (special mode)\n")
        else:
            logfile.write("Global sigma clipping loop\n")
            large_correction, n_iter = tools.adaptive_sigma_clip_loop(
                large_correction, args.Nsigma, max_iter=MAX_CLIP_ITER
            )
            logfile.write(f"  completed in {n_iter} iterations\n")
            if n_iter == MAX_CLIP_ITER:
                logfile.write("/!\\ max iterations reached in global clipping\n")

        # ── Optional overview diagnostic plot ───────────────────────────────
        if args.plot:
            savename = os.path.join(
                args.path, f"{args.myLine}/temp/lane-{Lane}-temp_{args.name}.png"
            )
            graphics.plot_overview(
                DATA, NDATA, freq_all, integ, large_correction,
                savename, n_chan=NCHAN
            )

        # ====================================================================
        # SUB-BAND LOOP
        # ====================================================================
        for band in range(NSUB):

            FREF   = np.copy(freq_all[NCHAN * band: NCHAN * (band + 1)])
            SUBREF = np.copy(large_correction[NCHAN * band: NCHAN * (band + 1)])
            in_diag = args.plot and (int(band + Lane * NSUB) in diag_bands)

            if in_diag:
                ARR00 = np.copy(SUBREF)   # panel 0 — raw sub-band

            # ── Step 2 : 0th-order mitigation — narrow spikes ───────────────
            # Flag channels where the first derivative exceeds Nsiglarge×σ.
            grad   = np.diff(SUBREF, 1)
            thresh = Nsiglarge * np.nanstd(np.abs(grad)) + np.nanmean(np.abs(grad))
            spike  = np.concatenate(([0], np.abs(grad)))
            SUBREF[spike > thresh] = np.nan

            if in_diag:
                ARR01 = np.copy(SUBREF)   # panel 0 — after spike removal

            # ── RRL positions within this sub-band ──────────────────────────
            lines = []
            for v in velo:
                lines += list(tools.get_line(
                    FREF[0], FREF[-1], v - DELTAV,
                    args.myLine, path=RRLS_PATH
                ))
            n_transitions = list(tools.get_line(
                FREF[0], FREF[-1], velo[0] - DELTAV,
                args.myLine, path=RRLS_PATH
            ).index)

            # ── Step 3 : 1st-order flattening — S-shape from neighbours ─────
            # Build a window of N_NEIGHBOURS sub-bands centred on current band.
            half = N_NEIGHBOURS // 2
            Dinf = max(0, band - half)
            Dsup = min(NSUB, band + half + 1)
            # Extend window at the edges to always have N_NEIGHBOURS bands
            if Dinf == 0:
                Dsup = min(NSUB, Dsup + (half - band))
            elif Dsup == NSUB:
                Dinf = max(0, Dinf - (band + half + 1 - NSUB))
            interval = np.arange(Dinf, Dsup, dtype=int)

            SUBS2        = np.full((N_NEIGHBOURS, len(FREF)), np.nan)
            count_good_sb = 0

            # Rejection threshold: neighbour too different from current band
            thresh_shape = pval * np.sqrt(np.nansum(SUBREF**2))

            for i, k in enumerate(interval):
                subband  = np.copy(large_correction[NCHAN * k: NCHAN * (k + 1)])
                freqband = np.copy(freq_all[NCHAN * k: NCHAN * (k + 1)])

                # Skip fully flagged sub-bands
                if np.all(np.isnan(subband)):
                    continue
                # Skip misshapen sub-bands (chi²-like test)
                if np.nansum((SUBREF - subband)**2) > thresh_shape:
                    continue

                # Mask RRL channels in each neighbour before averaging
                linesk  = []
                ntransk = []
                for v in velo:
                    linesk  += list(tools.get_line(
                        freqband[0], freqband[-1], v - DELTAV,
                        args.myLine, path=RRLS_PATH
                    ))
                    ntransk += list(tools.get_line(
                        freqband[0], freqband[-1], v - DELTAV,
                        args.myLine, path=RRLS_PATH
                    ).index)

                ntransk = np.array(ntransk)
                valid_n = ntransk < N_MAX_VALID
                linesk  = np.array(linesk)[valid_n]
                ntransk = ntransk[valid_n]

                if len(linesk) == 0:
                    SUBS2[i] = subband
                else:
                    masked = np.copy(subband)
                    for ii, f_raie in enumerate(linesk):
                        if mask > 0 and int(ntransk[ii]) in wVs.index:
                            # Mask width = args.mask × expected Voigt FWHM
                            DeltaF = mask * wVs.loc[int(ntransk[ii])].values
                        else:
                            DeltaF = 0.0
                        masked[np.abs(freqband - f_raie) < DeltaF] = np.nan
                    # Restore edge channels (avoid masking sub-band boundaries)
                    edge = NCHAN // EDGE_FRAC
                    masked[-edge:] = subband[-edge:]
                    masked[:edge]  = subband[:edge]
                    SUBS2[i]       = masked

                count_good_sb += 1

            # ── Reliability check ────────────────────────────────────────────
            if _subband_is_unreliable(SUBREF, SUBS2, count_good_sb, NCHAN):
                logfile.write(f"  Unreliable sub-band {band} — skipped\n")
                ITOTAL_flat = np.concatenate(
                    [ITOTAL_flat, np.full(NCHAN, np.nan)]
                )
                ITOTAL_Jy = np.concatenate(
                    [ITOTAL_Jy, np.full(NCHAN, np.nan)]
                )
                FTOTAL    = np.concatenate([FTOTAL, FREF])
                Ilinefree = np.concatenate(
                    [Ilinefree, np.full(NCHAN, np.nan)]
                )
                continue

            # ── Build smoothed median baseline from neighbours ───────────────
            # Replace NaNs at edges with nearest valid value before smoothing.
            mean = np.nanmedian(SUBS2, axis=0)
            tools.fill_edges(mean)

            if in_diag:
                ARR20 = np.copy(mean)   # panel 2 — median of neighbours

            mean2 = interp1d(
                FREF[~np.isnan(mean)], mean[~np.isnan(mean)],
                kind="linear", bounds_error=False, fill_value="nearest"
            )(FREF)

            wind_len  = max(len(mean2) // args.savgol_window, 5)
            wind_len += 1 - wind_len % 2    # ensure odd window length
            meansmooth = savgol_filter(mean2, wind_len, 3)

            if in_diag:
                ARR21 = np.copy(meansmooth)   # panel 2 — smoothed median

            # ── 1st-order flat sub-band ──────────────────────────────────────
            flat_sub2 = SUBREF - meansmooth

            if in_diag:
                ARR30 = np.copy(flat_sub2)   # panel 3 — 1st-order flat

            # ── Detect clean interior edges ──────────────────────────────────
            indexes_lo, indexes_hi = _find_clean_edges(
                meansmooth, NCHAN, args.Nsigma
            )

            # ── Step 4 : 1st-order RFI mitigation — broader spikes ──────────
            y01  = np.copy(meansmooth)
            clean = np.copy(flat_sub2)

            if in_diag:
                ARR40 = np.copy(clean)   # panel 4 — before 1st RFI mitigation

            # Protect known RRL positions from clipping
            for ii, f_raie in enumerate(lines):
                n_idx = n_transitions[ii] if ii < len(n_transitions) else n_transitions[-1]
                if mask > 0 and int(n_idx) in wVs.index:
                    # Mask width = args.mask × expected Voigt FWHM at this n
                    DeltaF = mask * wVs.loc[int(n_idx)].values
                else:
                    DeltaF = 0.0
                clean[np.abs(FREF - f_raie) < DeltaF] = np.nan

            if not args.special:
                clean, n_iter = tools.adaptive_sigma_clip_loop(
                    clean, args.Nsigma, max_iter=MAX_CLIP_ITER
                )
                # Propagate flagged channels back to SUBREF and flat_sub2
                flagged = np.where(np.isnan(clean) & ~np.isnan(flat_sub2))[0]
                SUBREF[flagged]    = np.nan
                flat_sub2[flagged] = np.nan
                if n_iter == MAX_CLIP_ITER:
                    logfile.write(
                        f"  /!\\ max iter in 1st RFI mitigation, sb {band}\n"
                    )

            if in_diag:
                ARR41 = np.copy(flat_sub2)   # panel 4 — after 1st RFI mitigation

            # ── Step 5 : 2nd-order flattening — residual faint baseline ──────
            # Interpolate NaNs in clean spectrum, separate into edge / middle
            # regions, then fit a Savitzky-Golay baseline.
            if np.isnan(clean[0]):
                clean[0] = flat_sub2[~np.isnan(flat_sub2)][0]
            if np.isnan(clean[-1]):
                clean[-1] = flat_sub2[~np.isnan(flat_sub2)][-1]

            clean1     = np.copy(clean)
            cleanmid   = clean1[indexes_lo:indexes_hi]
            cleanleft  = clean1[:indexes_lo]
            cleanright = clean1[indexes_hi:]

            # Middle: replace NaN with sub-band mean
            cleanmid[np.isnan(cleanmid)] = np.nanmean(cleanmid)

            # Edges: linear interpolation; seed boundary with middle mean
            if np.isnan(cleanleft[-1]):
                cleanleft[-1] = np.nanmean(cleanmid)
            if np.isnan(cleanright[0]):
                cleanright[0] = np.nanmean(cleanmid)

            f_left  = FREF[:indexes_lo]
            f_right = FREF[indexes_hi:]
            cleanleft = interp1d(
                f_left[~np.isnan(cleanleft)],
                cleanleft[~np.isnan(cleanleft)],
                kind="linear", bounds_error=False
            )(f_left)
            cleanright = interp1d(
                f_right[~np.isnan(cleanright)],
                cleanright[~np.isnan(cleanright)],
                kind="linear", bounds_error=False
            )(f_right)

            clean1   = np.concatenate([cleanleft, cleanmid, cleanright])
            linefree = np.copy(clean1)

            wind_len2  = max(len(clean1) // 20, 5)
            wind_len2 += 1 - wind_len2 % 2
            if args.special:
                y11 = savgol_filter(clean1, 2 * max(int(mask), 15) + 1, 5)
            else:
                y11 = savgol_filter(clean1, wind_len2, 3)

            # 2nd-order flat sub-band
            y2 = SUBREF - y01 - y11

            if in_diag:
                ARR50 = np.copy(y01 + y11)      # panel 5 — total baseline
                ARR60 = np.copy(SUBREF - y01)   # panel 6 — 2nd-order flat before
                ARR61 = np.copy(y2)             # panel 6 — 2nd-order flat after

            # ── Step 6 : 3rd-order mitigation — intra-line filter ────────────
            # Zeros NaN positions, applies a short Savitzky-Golay to capture
            # any residual smooth variation hiding under/around the lines.
            y22 = np.copy(y2)
            linefree[np.isnan(y2)] = 0.0
            y22[np.isnan(y2)]      = 0.0

            if args.special:
                win3 = max(int(mask) + 1 - int(mask) % 2, 15)
            else:
                win3 = NCHAN // 60 + 1 - (NCHAN // 60) % 2  # empirical

            y2_filtre = savgol_filter(y22, win3, 3)
            clean     = y2 - y2_filtre

            if in_diag:
                ARR70 = np.copy(y2)           # panel 7 — before intra-line filter
                ARR71 = np.copy(y2_filtre)    # panel 7 — intra-line filter

            # Sigma-clipping after intra-line filter
            clean, n_iter = tools.adaptive_sigma_clip_loop(
                clean, args.Nsigma, max_iter=MAX_CLIP_ITER
            )
            flagged = np.where(np.isnan(clean) & ~np.isnan(y2))[0]
            y2[flagged]       = np.nan
            linefree[flagged] = np.nan
            if n_iter == MAX_CLIP_ITER:
                logfile.write(
                    f"  /!\\ max iter in intra-line mitigation, sb {band}\n"
                )

            if in_diag:
                ARR72 = np.copy(y2)   # panel 7 — after intra-line filter
                ARR80 = np.copy(y2)   # panel 8 — before final clipping

            # ── Step 7 : 4th-order mitigation — remaining positive spikes ────
            y2[y2 > args.Nsigma * np.nanstd(y2)] = np.nan

            if in_diag:
                ARR81 = np.copy(y2)   # panel 8 — final cleaned sub-band

            # ── Accumulate results ───────────────────────────────────────────
            tau          = np.copy(y2)
            continuum    = (1.0 + tau) * continuum_fct(FREF, coeffs)
            tau_linefree = linefree - y11

            ITOTAL_flat = np.concatenate([ITOTAL_flat, tau])
            ITOTAL_Jy   = np.concatenate([ITOTAL_Jy,  continuum])
            Ilinefree   = np.concatenate([Ilinefree,   tau_linefree])
            FTOTAL      = np.concatenate([FTOTAL,      FREF])

            # ── Optional sub-band diagnostic plot ────────────────────────────
            if in_diag:
                savename = os.path.join(
                    args.path, f"{args.myLine}/temp/temp_{args.name}{band}.png"
                )
                graphics.plot_subband(
                    FREF,
                    raw=ARR00,            after_spike=ARR01,
                    neighbors_mean=ARR20, neighbors_smooth=ARR21,
                    flat1=ARR30,
                    pre_rfi=ARR40,        post_rfi=ARR41,
                    baseline=ARR50,
                    pre_flat2=ARR60,      flat2=ARR61,
                    pre_intra=ARR70,      intra_filter=ARR71,
                    post_intra=ARR72,
                    pre_final=ARR80,      final=ARR81,
                    SUBS2=SUBS2,
                    n_transitions=n_transitions,
                    indexes_lo=indexes_lo, indexes_hi=indexes_hi,
                    lines=lines,
                    savename=savename,
                )

    # ===========================================================================
    # FINAL GLOBAL MITIGATION
    # ===========================================================================
    logfile.write("Final global mitigation loop\n")

    clean_final = np.copy(Ilinefree)
    clean_final[np.isnan(clean_final)] = 0.0
    kstop     = 1
    lenwrongs = 1

    while lenwrongs > 0 and kstop < MAX_CLIP_ITER:
        ystd      = tools.std_glissant(clean_final, NCHAN)
        bad       = np.where(np.abs(clean_final) > args.Nsigma * ystd)[0]
        lenwrongs = len(bad)
        ITOTAL_flat[bad] = np.nan
        ITOTAL_Jy[bad]   = np.nan
        Ilinefree[bad]   = np.nan
        logfile.write(f"  iter {kstop:3d} — bad channels: {lenwrongs}\n")
        kstop += 1

    if kstop == MAX_CLIP_ITER:
        logfile.write("/!\\ max iterations reached in final clipping\n")

    # ===========================================================================
    # RMS COMPUTATION AND FITS OUTPUT
    # ===========================================================================
    N               = len(Ilinefree)
    RMS             = np.zeros((N, 1, 1))
    RMS_linefree    = np.zeros((N, 1, 1))
    RMS[:, 0, 0]          = tools.std_glissant(ITOTAL_flat, NCHAN)
    RMS_linefree[:, 0, 0] = tools.std_glissant(Ilinefree,   NCHAN)

    def _cube(arr: np.ndarray) -> np.ndarray:
        """Reshape 1-D array to (N, 1, 1) for FITS ImageHDU."""
        out         = np.zeros((N, 1, 1))
        out[:, 0, 0] = arr
        return out

    # ── Build FITS header from template ─────────────────────────────────────
    hdr_prim = build_fits_header(FITS_HEADER_TEMPLATE)
    hdr_prim["NCHAN"]  = NCHAN
    hdr_prim["TIME"]   = HDU.date.value
    hdr_prim["FREQ"]   = float(np.nanmean(FTOTAL))
    coord              = SkyCoord(args.ra, args.dec)
    hdr_prim["CRVAL1"] = float(coord.ra.value)
    hdr_prim["CRVAL2"] = float(coord.dec.value)
    hdr_prim["NAXIS3"] = N
    hdr_prim["NAXIS1"] = 1
    hdr_prim.set("FMIN",  float(FTOTAL[0]),    "lowest frequency")
    hdr_prim.set("FMAX",  float(FTOTAL[-1]),   "highest frequency")
    hdr_prim.set("CRPIX3", N,                  "Pixel coordinate of reference point")
    hdr_prim.set("CDELT3", HDU.DF[0] / 1e3,   "frequency increment at reference point")
    hdr_prim.set("CUNIT3", HDU.FREFUNIT[0],    "Units of coordinate increment and value")
    hdr_prim.set("CTYPE3", "FREQ",             "")
    hdr_prim.set("CRVAL3", float(FTOTAL[N-1]), "Coordinate value at reference point")
    hdr_prim.set("NAME",   "Optical depth")

    hdr_jy              = hdr_prim.copy()
    hdr_rms             = hdr_prim.copy()
    hdr_rms_linefree    = hdr_prim.copy()
    hdr_jy["NAME"]           = "Intensity (Jy)"
    hdr_rms["NAME"]          = "RMS"
    hdr_rms_linefree["NAME"] = "RMS on line free"

    hdul = fits.HDUList([
        fits.PrimaryHDU(_cube(ITOTAL_flat), header=hdr_prim),
        fits.ImageHDU(_cube(ITOTAL_Jy),     header=hdr_jy),
        fits.ImageHDU(RMS_linefree,          header=hdr_rms_linefree),
        fits.ImageHDU(RMS,                   header=hdr_rms),
    ])

    # ── Write output ─────────────────────────────────────────────────────────
    if is_galactic:
        outdir = os.path.join(args.path, f"reduced-gal/{args.myLine}")
        suffix = "OFF" if args.off else ""
        outpath = os.path.join(outdir, f"temp_{args.name}{suffix}.fits")
    else:
        outdir  = os.path.join(args.path, f"{args.myLine}/temp")
        suffix  = "OFF" if args.off else ""
        outpath = os.path.join(outdir, f"temp_{args.name}{suffix}.fits")

    hdul.writeto(outpath, overwrite=True)
    print(f"Written to {outpath}")

    elapsed = (time.time() - start) / 60.0
    logfile.write(f"total processing time : {elapsed:.2f} min\n")
    logfile.close()
    print(f"Processing time = {elapsed:.2f} min  [{args.name}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    clean_observation(parse_args())
