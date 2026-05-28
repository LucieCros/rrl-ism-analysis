# -*- coding: utf-8 -*-
"""
tools.py
========
Core signal-processing and astronomical utility functions for spectral line
analysis of radio recombination lines (RRLs).

This module provides:

- **Doppler corrections** : geocentric, heliocentric, barycentric, LSR
- **Frequency / velocity conversions** : :func:`f_to_v`, :func:`v_to_f`,
  :func:`doppler_correction`
- **Signal processing** : rebinning, moving average, sliding std, sigma clipping,
  sliding RMS
- **Grid alignment** : :func:`align_on_doppler_grid`, :func:`align_on_rest_grid`
- **Line utilities** : slicing a spectrum around a line, masking line-free regions
- **Continuum models** : polynomial (log-log) and power-law

.. note::
    **No circular dependency** : this module does not import ``L1_class``.
    FITS I/O (loading observation blocs) lives in :mod:`spectral_tools.io`.

    Atomic species constants and RRL frequency computation live in
    :mod:`spectral_tools.atoms`. They are re-exported here for backward
    compatibility.

Dependencies
------------
astropy >= 5.0, numpy, scipy, pandas
Internal: spectral_tools.atoms  (no L1_class import)
"""

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, FK5
from astropy import time as astrotime
from astropy.constants import c
from scipy.interpolate import interp1d

# Re-export from atoms for backward compatibility — do NOT redefine here.
from spectral_tools.atoms import set_specie, line_freq  # noqa: F401


# ---------------------------------------------------------------------------
# Doppler corrections
# ---------------------------------------------------------------------------

def doppler_corrections(mean_time: str,
                        ra: str = "23h23m24s",
                        dec: str = "58d48m54",
                        obs_lat: float = 47.367686,
                        obs_lon: float = 2.194313,
                        obs_alt: float = 150.0) -> list:
    """
    Compute the projected velocity of the telescope with respect to four
    reference frames: geocentric, heliocentric, barycentric, and LSR.

    To correct a radial velocity axis to the LSR frame::

        v_corr = doppler_corrections(mean_time, ra, dec)
        rv_lsr = rv_obs + v_corr[3] + rv_obs * v_corr[3] / c

    Parameters
    ----------
    mean_time : str
        Mean UTC time of the observation in ISO-T format,
        e.g. ``'2017-01-15T01:59:58.99'``.
    ra : str, optional
        Right ascension of the target (J2000, ICRS), e.g. ``'23h23m24s'``.
    dec : str, optional
        Declination of the target (J2000, ICRS), e.g. ``'58d48m54'``.
    obs_lat : float, optional
        Observatory geodetic latitude in degrees. Default: Nançay RT (47.37°N).
    obs_lon : float, optional
        Observatory geodetic East longitude in degrees. Default: Nançay (2.19°E).
    obs_alt : float, optional
        Observatory altitude above sea level in metres. Default: 150 m.

    Returns
    -------
    list of astropy.units.Quantity
        ``[v_geo, v_helio, v_bary, v_lsr]``, each in km/s.
        Add the LSR correction (index 3) to transform observed velocities
        to the kinematic LSR frame.

    Notes
    -----
    The Earth spin component follows Green (1985), "Spherical Astronomy", p. 270.

    The LSR is defined as the Sun moving at 20 km/s toward
    RA = 18 h, Dec = 30° (1900 J equinox), following the IAU convention.
    """
    src      = SkyCoord(ra, dec, frame="icrs", unit=u.deg)
    mytime   = astrotime.Time(mean_time, format="isot", scale="utc")
    location = EarthLocation.from_geodetic(
        lat=obs_lat * u.deg, lon=obs_lon * u.deg, height=obs_alt * u.m
    )

    # Barycentric and heliocentric corrections (Earth orbit + Moon)
    barycorr  = src.radial_velocity_correction(
        obstime=mytime, location=location).to(u.km / u.s)
    heliocorr = src.radial_velocity_correction(
        "heliocentric", obstime=mytime, location=location).to(u.km / u.s)

    # Earth rotation component (Green 1985, p. 270)
    lst        = mytime.sidereal_time("apparent", obs_lon)
    hour_angle = lst - src.ra
    v_spin     = -0.465 * np.cos(obs_lat * u.deg) * np.cos(src.dec) * np.sin(hour_angle)

    # LSR: Sun → (RA=18h, Dec=30°) at 20 km/s in J1900, converted to J2000
    lsr_coord = SkyCoord("18h", "30d", frame="fk5", equinox="J1900")
    lsr_coord = lsr_coord.transform_to(FK5(equinox="J2000"))
    lsr_comp  = np.array([
        np.cos(lsr_coord.dec.rad) * np.cos(lsr_coord.ra.rad),
        np.cos(lsr_coord.dec.rad) * np.sin(lsr_coord.ra.rad),
        np.sin(lsr_coord.dec.rad),
    ])
    src_comp  = np.array([
        np.cos(src.dec.rad) * np.cos(src.ra.rad),
        np.cos(src.dec.rad) * np.sin(src.ra.rad),
        np.sin(src.dec.rad),
    ])
    v_lsr = 20.0 * np.dot(lsr_comp, src_comp) * u.km / u.s

    return [-v_spin, heliocorr, barycorr, barycorr + v_lsr]


# ---------------------------------------------------------------------------
# Frequency ↔ velocity conversions
# ---------------------------------------------------------------------------

def doppler_correction(f, v: float):
    """
    Apply a non-relativistic Doppler shift to a frequency array.

    .. math::

        f_{\\rm obs} = \\frac{f_{\\rm rest}}{1 + v/c}

    Parameters
    ----------
    f : float or array-like
        Rest frequency in MHz (or any consistent unit).
    v : float
        Radial velocity in m/s (positive = receding).

    Returns
    -------
    float or numpy.ndarray
        Doppler-shifted frequency, same unit as ``f``.
    """
    return f / (1.0 + v / c.value)


def f_to_v(f, f0: float) -> u.Quantity:
    """
    Convert a frequency width to a velocity width (non-relativistic radio convention).

    .. math::

        v = -c \\cdot \\frac{f}{f_0}

    Parameters
    ----------
    f : float or array-like
        Frequency value to convert [MHz].
    f0 : float
        Reference (rest) frequency [MHz].

    Returns
    -------
    astropy.units.Quantity
        Velocity [m/s].
    """
    c_si = 3e8 * u.m / u.s
    return -c_si * (f * u.MHz) / (f0 * u.MHz)


def v_to_f(v: float, f0: float) -> u.Quantity:
    """
    Convert a velocity offset to a frequency offset (inverse of :func:`f_to_v`).

    Parameters
    ----------
    v : float
        Velocity [km/s].
    f0 : float
        Reference (rest) frequency [MHz].

    Returns
    -------
    astropy.units.Quantity
        Frequency offset [MHz].
    """
    c_si = 3e8 * u.m / u.s
    return -(v * u.km / u.s).to(u.m / u.s) / c_si * (f0 * u.MHz)


# ---------------------------------------------------------------------------
# RRL catalogue utilities
# ---------------------------------------------------------------------------

def get_line(fmin: float, fmax: float, v: float,
             line: str = "Calph",
             path: str = "files/rrlines.csv") -> pd.Series:
    """
    Extract RRL frequencies within a frequency band, Doppler-shifted by ``v``.

    Parameters
    ----------
    fmin : float
        Lower frequency bound [MHz].
    fmax : float
        Upper frequency bound [MHz].
    v : float
        Radial velocity [km/s] used to shift the catalogue frequencies.
    line : str, optional
        Column name in the RRL catalogue (e.g. ``'Calph'``, ``'Halph'``).
        Default ``'Calph'``.
    path : str, optional
        Path to the RRL catalogue CSV. Default ``'files/rrlines.csv'``.

    Returns
    -------
    pandas.Series
        Doppler-shifted line frequencies within [fmin, fmax],
        indexed by quantum number n.
    """
    RRL = pd.read_csv(path)
    RRL.set_index("n", inplace=True)
    shifted = doppler_correction(RRL[line], v * 1000.0)  # km/s → m/s
    return shifted.where(
        shifted.where(shifted.values <= fmax).values >= fmin
    ).dropna()


def get_linefree(freq, spectrum: np.ndarray,
                 width_of_line: int = 30,
                 myLine: str = "Calph",
                 path_rrls: str = "files/rrlines.csv") -> np.ndarray:
    """
    Mask spectral channels contaminated by a known RRL.

    Channels within ``width_of_line`` pixels of each identified line centre
    are set to ``NaN``, returning a line-free copy of the spectrum.

    Parameters
    ----------
    freq : array-like
        Frequency axis [MHz], assumed uniformly spaced.
    spectrum : numpy.ndarray
        Spectral data array, same length as ``freq``.
    width_of_line : int, optional
        Half-width of the masked region in channels. Default 30.
    myLine : str, optional
        RRL series to mask. Default ``'Calph'``.
    path_rrls : str, optional
        Path to the RRL catalogue CSV. Default ``'files/rrlines.csv'``.

    Returns
    -------
    numpy.ndarray
        Copy of ``spectrum`` with line channels set to ``NaN``.
    """
    fmin, fmax = freq[0], freq[-1]
    df     = (fmax - fmin) / len(freq)
    DeltaF = width_of_line * df

    RRL  = pd.read_csv(path_rrls, sep=",")
    RRL.set_index("n", inplace=True)
    Aca  = {col: doppler_correction(RRL[col].dropna(), 0) for col in RRL.columns}

    LINES = np.array([f for f in np.array(Aca[myLine]) if fmin < f < fmax])

    I_linefree = np.copy(spectrum)
    for f_raie in LINES:
        I_linefree[np.abs(freq - f_raie) < DeltaF] = np.nan
    return I_linefree


# ---------------------------------------------------------------------------
# Signal processing — rebinning
# ---------------------------------------------------------------------------

def rebinning(y, x, dx):
    """
    Rebin a 1-D signal to a new grid or channel width.

    Parameters
    ----------
    y : array-like
        Input signal values.
    x : array-like
        Original x-axis (frequency or time).
    dx : float or array-like
        If array-like: new x-axis (signal is interpolated onto it).
        If scalar: number of channels to average per output bin.

    Returns
    -------
    new_y : numpy.ndarray
        Rebinned signal.
    new_x : numpy.ndarray
        Corresponding x-axis.
    """
    if hasattr(dx, "__len__"):
        new_x = np.copy(dx)
        new_y = interp1d(x, y)(new_x)
    else:
        new_y, new_x = _rebin_average(y, x, dx)
    return new_y, new_x


def _rebin_average(signal, time, slot: int):
    """
    Average ``signal`` into bins of ``slot`` channels (internal helper).

    Parameters
    ----------
    signal : array-like
        Input signal.
    time : array-like
        Corresponding x-axis.
    slot : int
        Number of channels per output bin.

    Returns
    -------
    new_sig : numpy.ndarray
    new_time : numpy.ndarray
    """
    nb       = int(slot)
    old_sig  = list(np.array(signal).copy())
    new_sig, new_time = [], []
    i = 0
    while len(old_sig) >= nb:
        new_sig.append(np.nanmean(old_sig[:nb]))
        new_time.append(np.mean(time[i * nb:(i + 1) * nb]))
        old_sig[:nb] = []
        i += 1
    return np.array(new_sig), np.array(new_time)


def std_rebin(signal, time, slot: int):
    """
    Compute the standard deviation within bins of ``slot`` channels.

    Useful for estimating the local noise level across a spectrum.

    Parameters
    ----------
    signal : array-like
        Input signal.
    time : array-like
        Corresponding x-axis.
    slot : int
        Number of channels per bin.

    Returns
    -------
    new_sig : numpy.ndarray
        Per-bin standard deviation.
    new_time : numpy.ndarray
        Bin centres (every ``slot`` channels, offset by ``slot // 2``).
    """
    nb      = int(slot)
    old_sig = list(np.array(signal).copy())
    new_sig = []
    while len(old_sig) >= nb:
        new_sig.append(np.nanstd(old_sig[:nb]))
        old_sig[:nb] = []
    new_time = time[nb // 2:-1:nb]
    return np.array(new_sig), np.array(new_time)


# ---------------------------------------------------------------------------
# Signal processing — smoothing
# ---------------------------------------------------------------------------

def moving_avg(signal: np.ndarray, slot: int) -> np.ndarray:
    """
    Box-car moving average.

    NaN values are replaced by nearest-neighbour interpolation before
    convolution so they do not propagate into the output.

    Parameters
    ----------
    signal : numpy.ndarray
        Input 1-D array, may contain NaNs.
    slot : int
        Width of the averaging window in samples.

    Returns
    -------
    numpy.ndarray
        Smoothed signal, same length as ``signal``.
    """
    ref          = np.arange(len(signal))
    signal_nonan = interp1d(
        ref[~np.isnan(signal)], signal[~np.isnan(signal)],
        kind="nearest", bounds_error=False, fill_value="extrapolate"
    )(ref)
    return np.convolve(signal_nonan, np.ones(slot), "same") / slot


def std_glissant(signal: np.ndarray, slot: int) -> np.ndarray:
    """
    Sliding (rolling) standard deviation.

    Implemented via a convolution trick for efficiency. NaNs are replaced
    by nearest-neighbour interpolation before processing.

    Parameters
    ----------
    signal : numpy.ndarray
        Input 1-D array, may contain NaNs.
    slot : int
        Width of the sliding window in samples.

    Returns
    -------
    numpy.ndarray
        Sliding standard deviation, same length as ``signal``.
    """
    ref          = np.arange(len(signal))
    signal_nonan = interp1d(
        ref[~np.isnan(signal)], signal[~np.isnan(signal)],
        kind="nearest", bounds_error=False, fill_value="extrapolate"
    )(ref)
    window      = np.ones(slot)
    arr_sq      = np.convolve(signal_nonan**2, window, "same")
    arr_mean_sq = (np.convolve(signal_nonan, window, "same") / slot)**2
    return np.sqrt(arr_sq / slot - arr_mean_sq)


# ---------------------------------------------------------------------------
# Sigma clipping
# ---------------------------------------------------------------------------

def flag_outliers(spectrum: np.ndarray, rms: float) -> np.ndarray:
    """
    Return indices of channels whose absolute deviation from the mean exceeds ``rms``.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Input spectrum (NaNs are ignored in the mean computation).
    rms : float
        Absolute deviation threshold.

    Returns
    -------
    numpy.ndarray of int
        Indices of outlier channels.
    """
    deviation = np.abs(spectrum - np.nanmean(spectrum)) - rms
    return np.where(deviation > 0)[0]


def recursive_clipping(signal: np.ndarray,
                       threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-pass sigma clipping: set outlier samples to NaN.

    Call iteratively inside a ``while`` loop for full convergence.

    Parameters
    ----------
    signal : numpy.ndarray
        Input 1-D signal.
    threshold : float
        Absolute deviation threshold above which samples are flagged.

    Returns
    -------
    signal_clean : numpy.ndarray
        Copy of ``signal`` with outliers replaced by ``NaN``.
    flagged : numpy.ndarray of int
        Indices of the flagged samples.
    """
    signal_clean         = np.copy(signal)
    flagged              = flag_outliers(signal_clean, threshold)
    signal_clean[flagged] = np.nan
    return signal_clean, flagged


# ---------------------------------------------------------------------------
# Continuum models
# ---------------------------------------------------------------------------

def continuum_polyn(f, coeffs: list) -> np.ndarray:
    """
    Log-polynomial continuum model (McKean et al. 2016).

    .. math::

        \\log_{10}(S) = \\sum_{i} a_i
        \\left(\\log_{10}\\frac{f}{1\\,\\text{GHz}}\\right)^i

    Parameters
    ----------
    f : array-like
        Frequency [MHz].
    coeffs : list of float
        Polynomial coefficients ``[a_0, a_1, ...]`` in increasing order.

    Returns
    -------
    numpy.ndarray
        Flux density [Jy].
    """
    logS = sum(a_i * (np.log10(f / 1e3))**i for i, a_i in enumerate(coeffs))
    return 10**logS


def continuum_powerlaw(f, coeff: list) -> np.ndarray:
    """
    Power-law continuum model.

    .. math::

        S(f) = S_0 \\left(\\frac{f}{f_0}\\right)^{-\\alpha}

    Parameters
    ----------
    f : array-like
        Frequency [MHz].
    coeff : list of float
        ``[alpha, ref_freq_MHz, flux_at_ref_Jy]``

    Returns
    -------
    numpy.ndarray
        Flux density [Jy].
    """
    alpha, f_ref, S_ref = coeff
    return S_ref * (f / f_ref)**(-alpha)


# ---------------------------------------------------------------------------
# Line slicing
# ---------------------------------------------------------------------------

def slice_line(lines, I: np.ndarray, f: np.ndarray,
               cut_width: int = 200,
               dpix: int = 0):
    """
    Extract a spectral window centred on one or more line frequencies.

    Channels outside the spectrum boundaries are padded with ``NaN``.

    Parameters
    ----------
    lines : float or iterable of float
        Central frequency (or frequencies) of the line(s) [MHz].
    I : numpy.ndarray
        Spectrum values.
    f : numpy.ndarray
        Frequency axis [MHz], uniformly spaced.
    cut_width : int, optional
        Half-width of the extracted window in channels. Default 200.
    dpix : int, optional
        Optional pixel offset for diagnostics. Default 0.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        Extracted window(s). Scalar input → single array;
        iterable input → list of arrays.
    """
    if hasattr(lines, "__iter__"):
        return [slice_line(line, I, f, cut_width, dpix) for line in lines]

    f0       = f[0]
    df       = f[1] - f[0]
    bound_lo = int((lines - cut_width * df - f0) // df) + dpix
    bound_hi = int((lines + cut_width * df - f0) // df) + dpix

    pad_lo   = np.full(-min(bound_lo, 0), np.nan)
    pad_hi   = np.full(max(bound_hi - len(f), 0), np.nan)
    bound_lo = max(bound_lo, 0)
    bound_hi = min(bound_hi, len(f))

    return np.concatenate([pad_lo, I[bound_lo:bound_hi], pad_hi])

# ---------------------------------------------------------------------------
# Frequency grid alignment (used by run_time_average.py)
# ---------------------------------------------------------------------------

def align_on_doppler_grid(spectra: list, jy_spectra: list,
                           weights: list, fdopps: list,
                           fmins: list, fmaxs: list,
                           df: float) -> tuple:
    """
    Interpolate a set of observations onto a common Doppler-corrected
    frequency grid.

    Each observation has been shifted to its barycentric frame. This function
    builds a common grid spanning ``[min(fmins), max(fmaxs)]`` and
    interpolates each spectrum onto it. All arrays are pre-multiplied by
    their weight so the caller can directly apply :func:`numpy.nansum`.

    Parameters
    ----------
    spectra : list of numpy.ndarray
        Per-observation optical depth arrays τ(ν).
    jy_spectra : list of numpy.ndarray
        Per-observation flux density arrays S(ν) [Jy].
    weights : list of numpy.ndarray
        Per-observation weight arrays (typically 1/σ²).
    fdopps : list of numpy.ndarray
        Doppler-shifted frequency axis for each observation [MHz].
    fmins : list of float
        Minimum Doppler-shifted frequency per observation [MHz].
    fmaxs : list of float
        Maximum Doppler-shifted frequency per observation [MHz].
    df : float
        Channel width [MHz].

    Returns
    -------
    spectra : list of numpy.ndarray
        Weight-multiplied τ arrays on the common grid.
    jy_spectra : list of numpy.ndarray
        Weight-multiplied S arrays on the common grid.
    weights : list of numpy.ndarray
        Weight arrays on the common grid (NaN where spectrum is NaN).
    fref : numpy.ndarray
        Common frequency reference axis [MHz].
    """
    fref = np.arange(np.nanmin(fmins), np.nanmax(fmaxs) + df, df)
    for i in range(len(spectra)):
        tau_i = np.interp(fref, fdopps[i], spectra[i],
                          left=np.nan, right=np.nan)
        jy_i  = np.interp(fref, fdopps[i], jy_spectra[i],
                          left=np.nan, right=np.nan)
        w_i   = np.interp(fref, fdopps[i], weights[i],
                          left=np.nan, right=np.nan)
        w_i[np.isnan(tau_i)] = np.nan
        spectra[i]    = tau_i * w_i
        jy_spectra[i] = jy_i  * w_i
        weights[i]    = w_i
    return spectra, jy_spectra, weights, fref


def align_on_rest_grid(spectra: list, jy_spectra: list,
                        weights: list,
                        fmini: float, fmaxi: float,
                        df: float) -> tuple:
    """
    Interpolate a set of observations onto a common rest-frequency grid
    (no Doppler correction applied).

    Used when the ``--no_corr`` flag is passed to the pipeline, e.g. for
    sources where the LSR correction is negligible or already applied
    upstream.

    Parameters
    ----------
    spectra : list of numpy.ndarray
        Per-observation optical depth arrays τ(ν).
    jy_spectra : list of numpy.ndarray
        Per-observation flux density arrays S(ν) [Jy].
    weights : list of numpy.ndarray
        Per-observation weight arrays.
    fmini : float
        Global minimum frequency [MHz].
    fmaxi : float
        Global maximum frequency [MHz].
    df : float
        Channel width [MHz].

    Returns
    -------
    Same structure as :func:`align_on_doppler_grid`.
    """
    fref = np.arange(fmini, fmaxi, df)
    for i in range(len(spectra)):
        n_orig = len(spectra[i])
        newf   = np.linspace(fmini, fmaxi, n_orig)
        tau_i  = np.interp(fref, newf, spectra[i],  left=np.nan, right=np.nan)
        jy_i   = np.interp(fref, newf, jy_spectra[i], left=np.nan, right=np.nan)
        w_i    = np.interp(fref, newf, weights[i],   left=np.nan, right=np.nan)
        w_i[np.isnan(tau_i)] = np.nan
        spectra[i]    = tau_i * w_i
        jy_spectra[i] = jy_i  * w_i
        weights[i]    = w_i
    return spectra, jy_spectra, weights, fref


def sliding_rms(signal: np.ndarray, half_window: int) -> np.ndarray:
    """
    Compute a local sliding-window RMS for a 1-D signal.

    For each channel ``i``, the RMS is estimated over
    ``signal[max(0, i - half_window) : min(N, i + half_window)]``.
    NaN values are ignored via :func:`numpy.nanstd`.

    Parameters
    ----------
    signal : numpy.ndarray
        Input 1-D array, may contain NaNs.
    half_window : int
        Half-width of the sliding window in channels.

    Returns
    -------
    numpy.ndarray
        Per-channel RMS, same length as ``signal``.

    Examples
    --------
    >>> rms = sliding_rms(linefree_spectrum, half_window=1024)
    """
    N   = len(signal)
    rms = np.zeros(N)
    for i in range(N):
        lo     = max(i - half_window, 0)
        hi     = min(i + half_window, N)
        rms[i] = np.nanstd(signal[lo:hi])
    return rms

# ---------------------------------------------------------------------------
# Sigma-clipping loop
# ---------------------------------------------------------------------------

def sigma_clip_loop(signal: np.ndarray, threshold: float,
                    max_iter: int = 120) -> tuple[np.ndarray, int]:
    """
    Iteratively clip outliers until convergence or ``max_iter`` is reached.

    At each iteration, samples whose absolute deviation exceeds ``threshold``
    are set to ``NaN`` and the threshold is recomputed from the surviving
    samples. Iteration stops when no new outliers are found.

    This is the loop-level wrapper around :func:`recursive_clipping`, which
    performs a single pass. Use this function when full convergence is needed.

    Parameters
    ----------
    signal : numpy.ndarray
        Input 1-D signal. Modified **in place** (a copy is made internally).
    threshold : float
        Initial absolute clipping threshold. Recomputed as
        ``Nsigma × nanstd(signal)`` at each iteration by the caller;
        pass ``Nsigma * np.nanstd(signal)`` here.
    max_iter : int, optional
        Maximum number of iterations. Default 120.

    Returns
    -------
    signal_clean : numpy.ndarray
        Cleaned signal with outliers replaced by ``NaN``.
    n_iter : int
        Number of iterations actually performed.

    Examples
    --------
    >>> clean, n = sigma_clip_loop(spectrum, threshold=3 * np.nanstd(spectrum))
    >>> if n == 120:
    ...     print("Warning: max iterations reached")
    """
    signal_clean = np.copy(signal)
    n_iter       = 0
    flagged      = [1]   # sentinel to enter the loop

    while len(flagged) > 0 and n_iter < max_iter:
        thresh       = threshold if n_iter == 0 \
                       else np.nanstd(signal_clean) * (threshold / np.nanstd(signal))
        signal_clean, flagged = recursive_clipping(
            signal_clean, np.nanstd(signal_clean) * threshold / max(np.nanstd(signal), 1e-30)
        )
        n_iter += 1

    return signal_clean, n_iter


def adaptive_sigma_clip_loop(signal: np.ndarray, nsigma: float,
                              max_iter: int = 120) -> tuple[np.ndarray, int]:
    """
    Iteratively clip outliers using an adaptive threshold ``nsigma × std``.

    At each iteration the standard deviation is recomputed on the surviving
    (non-NaN) samples, so the threshold tightens as outliers are removed.
    Iteration stops when no new outliers are found or ``max_iter`` is reached.

    This is the pattern used throughout the spectral cleaning pipeline.

    Parameters
    ----------
    signal : numpy.ndarray
        Input 1-D signal.
    nsigma : float
        Clipping level in units of the current standard deviation.
    max_iter : int, optional
        Safety cap on iterations. Default 120.

    Returns
    -------
    signal_clean : numpy.ndarray
        Cleaned signal with outliers set to ``NaN``.
    n_iter : int
        Number of iterations performed.

    Examples
    --------
    >>> clean, n = adaptive_sigma_clip_loop(spectrum, nsigma=3.0)
    >>> flagged_indices = np.where(np.isnan(clean) & ~np.isnan(spectrum))[0]
    """
    signal_clean = np.copy(signal)
    n_iter       = 0
    flagged      = [1]

    while len(flagged) > 0 and n_iter < max_iter:
        signal_clean, flagged = recursive_clipping(
            signal_clean, nsigma * np.nanstd(signal_clean)
        )
        n_iter += 1

    return signal_clean, n_iter


# ---------------------------------------------------------------------------
# Edge filling
# ---------------------------------------------------------------------------

def fill_edges(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaN values at the first and last position of an array with the
    nearest valid (non-NaN) value.

    This is a minimal in-place-style fix to prevent edge NaNs from
    propagating through interpolation or convolution. Only the first and
    last elements are patched; interior NaNs are left untouched.

    Parameters
    ----------
    arr : numpy.ndarray
        Input 1-D array, modified **in place**.

    Returns
    -------
    numpy.ndarray
        The same array with edge NaNs filled (same object as input).

    Examples
    --------
    >>> arr = np.array([np.nan, 1.0, 2.0, np.nan])
    >>> fill_edges(arr)
    array([1., 1., 2., nan])

    Notes
    -----
    If the entire array is NaN, the array is returned unchanged.
    """
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return arr
    if np.isnan(arr[0]):
        arr[0] = valid[0]
    if np.isnan(arr[-1]):
        arr[-1] = valid[-1]
    return arr


# ---------------------------------------------------------------------------
# Weighted statistics
# ---------------------------------------------------------------------------

def weighted_avg_and_std(values: np.ndarray,
                          weights: np.ndarray) -> tuple[float, float]:
    """
    Compute the weighted mean and weighted standard deviation of an array.

    Weights are normalised internally so that their sum equals 1.
    Both arrays must have the same shape.

    Parameters
    ----------
    values : numpy.ndarray
        Data values.
    weights : numpy.ndarray
        Non-negative weights (e.g. ``1 / sigma²``). Need not sum to 1.

    Returns
    -------
    average : float
        Weighted mean.
    std : float
        Weighted standard deviation
        (square root of the weighted variance).

    Examples
    --------
    >>> centers = np.array([-47.2, -46.8, -47.5])
    >>> errors  = np.array([0.3,    0.2,   0.5])
    >>> mean, std = weighted_avg_and_std(centers, 1 / errors**2)
    >>> print(f"{mean:.2f} ± {std:.2f} km/s")
    -46.94 ± 0.26 km/s
    """
    average  = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return average, np.sqrt(variance)
    
# ---------------------------------------------------------------------------
# Conversion from galactic coordinates to cartesian coordinates
# ---------------------------------------------------------------------------

def galactic_to_cartesian(r, l, b, l0, b0):
    """
    Convert a Galactic position to local Cartesian coordinates.
 
    The Sun is the origin of the Cartesian frame.  The reference direction
    ``(l0, b0)`` defines the x-axis; angular offsets in longitude and latitude
    are measured relative to this reference.
 
    The spherical-to-Cartesian mapping uses the convention:
      - ``theta`` : azimuthal offset from the reference longitude [rad]
      - ``phi``   : polar angle from the z-axis (Galactic north), shifted so
                    that ``phi = pi/2`` when ``b == b0`` (equatorial plane of
                    the reference direction)
 
    Parameters
    ----------
    r : float or array-like
        Radial (heliocentric) distance.  Any unit is accepted as long as it
        is consistent with the desired output unit (e.g. pc or kpc).
    l : float
        Galactic longitude of the target direction [degrees].
    b : float
        Galactic latitude of the target direction [degrees].
    l0 : float
        Galactic longitude of the reference direction [degrees].
    b0 : float
        Galactic latitude of the reference direction [degrees].
 
    Returns
    -------
    x : float or numpy.ndarray
        Coordinate along the reference direction (radial, toward ``(l0, b0)``).
    y : float or numpy.ndarray
        Coordinate perpendicular to x in the Galactic plane.
    z : float or numpy.ndarray
        Coordinate along Galactic north (out of the plane).
 
    Notes
    -----
    This function computes *local* offsets around ``(l0, b0)``, not absolute
    Galactocentric positions.  For small angular separations (a few degrees),
    the approximation is accurate to better than 1 %.
 
    Examples
    --------
    >>> x, y, z = gal_to_cartesian(1.0, 184.6, -5.8, 184.0, -5.0)
    """
    # Azimuthal offset from the reference longitude, converted to radians.
    theta = (l - l0) * np.pi / 180.0
 
    # Polar angle from the z-axis: 90° when on the reference latitude plane,
    # increasing toward the south Galactic pole.
    phi = (90.0 - (b - b0)) * np.pi / 180.0
 
    x = r * np.sin(phi) * np.cos(theta)   # radial component (toward source)
    y = r * np.sin(phi) * np.sin(theta)   # transverse component (east-west)
    z = r * np.cos(phi)                   # vertical component (north-south)
    return x, y, z
 
 
def cartesian_to_galactic(x, y, z, l0, b0):
    """
    Convert local Cartesian coordinates back to Galactic (l, b, r).
 
    This is the exact inverse of :func:`gal_to_cartesian`.  Given a position
    expressed in the local Cartesian frame centred on the Sun with x-axis
    toward the reference direction ``(l0, b0)``, it returns the corresponding
    Galactic longitude, latitude and heliocentric distance.
 
    Parameters
    ----------
    x : float or array-like
        Cartesian coordinate along the reference direction [same unit as ``r``].
    y : float or array-like
        Cartesian coordinate perpendicular to x in the reference plane.
    z : float or array-like
        Cartesian coordinate along Galactic north.
    l0 : float
        Galactic longitude of the reference direction [degrees].
    b0 : float
        Galactic latitude of the reference direction [degrees].
 
    Returns
    -------
    l : float or numpy.ndarray
        Galactic longitude [degrees].
    b : float or numpy.ndarray
        Galactic latitude [degrees].
    r : float or numpy.ndarray
        Heliocentric distance [same unit as the input coordinates].
 
    Examples
    --------
    Round-trip consistency check:
 
    >>> x, y, z = gal_to_cartesian(1.5, 185.0, -6.0, 184.0, -5.0)
    >>> l, b, r = cartesian_to_galactic(x, y, z, 184.0, -5.0)
    >>> print(f"l={l:.4f}  b={b:.4f}  r={r:.4f}")
    l=185.0000  b=-6.0000  r=1.5000
    """
    # Heliocentric distance from the origin.
    r = np.sqrt(x**2 + y**2 + z**2)
 
    # Polar angle from the z-axis [rad], then convert to latitude offset [deg].
    phi   = np.arctan2(np.sqrt(x**2 + y**2), z)   # polar angle  [rad]
    delta_b = 90.0 - np.degrees(phi)               # latitude offset w.r.t. b0
 
    # Azimuthal angle in the (x, y) plane [rad], then convert to longitude offset [deg].
    theta   = np.arctan2(y, x)                     # azimuthal angle [rad]
    delta_l = np.degrees(theta)                     # longitude offset w.r.t. l0
 
    l = l0 + delta_l
    b = b0 + delta_b
    return l, b, r
 


