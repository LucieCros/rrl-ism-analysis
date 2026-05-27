# -*- coding: utf-8 -*-
"""
line_fitting.py
===============
Voigt profile fitting functions for radio recombination line (RRL) analysis.

This module provides:

- **Profile evaluation** : single and multi-component Voigt profiles
- **Profile geometry** : FWHM, area ↔ peak amplitude conversions
- **Uncertainty propagation** : analytical error formulas for fitted parameters
- **Curve fitting** : high-level fitter for stacked multi-cloud spectra

All width parameters (``fwhm_L``, ``fwhm_G``) are in the same unit as the
x-axis (typically km/s for velocity spectra, or MHz for frequency spectra).

Removed from original version
------------------------------
- ``voigt_area_old`` and ``voigt_area_inv_old`` : superseded by
  :func:`voigt_area` and :func:`voigt_area_inv`.
- ``fwhm_fct`` lambda : duplicate of :func:`voigt_fwhm` with swapped argument
  order — use :func:`voigt_fwhm` directly.
- ``voigt_fwhm_LC`` : identical to :func:`voigt_fwhm`, removed.
- ``triplevoigt`` and ``triplevoigt_ar`` : hardcoded 3-component wrappers —
  use :func:`multiple_voigt` with a list of 3 components instead.

References
----------
- Sorochenko & Smirnov (1990) — Voigt area approximation.
- Thompson (1987) — Voigt FWHM approximation.

Dependencies
------------
astropy >= 5.0, numpy, scipy
"""

import numpy as np
from astropy.modeling.models import Voigt1D
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Basic Voigt profile
# ---------------------------------------------------------------------------

def voigt(v, x_0: float, a_L: float, fwhm_L: float, fwhm_G: float) -> np.ndarray:
    """
    Evaluate a single Voigt profile.

    Thin wrapper around :class:`astropy.modeling.models.Voigt1D`.

    Parameters
    ----------
    v : array-like
        Velocity (or frequency) axis.
    x_0 : float
        Line centre, same unit as ``v``.
    a_L : float
        Amplitude of the Lorentzian component (peak of the Lorentzian).
    fwhm_L : float
        FWHM of the Lorentzian component.
    fwhm_G : float
        FWHM of the Gaussian component.

    Returns
    -------
    numpy.ndarray
        Profile values at each point of ``v``.
    """
    return Voigt1D(x_0=x_0, amplitude_L=a_L, fwhm_L=fwhm_L, fwhm_G=fwhm_G)(v)


def multiple_voigt(v, centers: list, amplitudes: list,
                   lorentz_widths: list, gaussian_widths: list) -> np.ndarray:
    """
    Evaluate a sum of Voigt profiles (multi-component model).

    Parameters
    ----------
    v : array-like
        Velocity (or frequency) axis.
    centers : list of float
        Line centres for each component.
    amplitudes : list of float
        Lorentzian amplitudes for each component.
    lorentz_widths : list of float
        Lorentzian FWHMs for each component.
    gaussian_widths : list of float
        Gaussian FWHMs for each component.

    Returns
    -------
    numpy.ndarray
        Summed profile values at each point of ``v``.

    Notes
    -----
    All four parameter lists must have the same length (number of components).

    Examples
    --------
    Fit two velocity components at -5 and +5 km/s::

        result = multiple_voigt(v,
                                centers=[-5.0, 5.0],
                                amplitudes=[-0.1, -0.05],
                                lorentz_widths=[2.0, 2.0],
                                gaussian_widths=[3.0, 3.0])
    """
    return sum(
        voigt(v, centers[i], amplitudes[i], lorentz_widths[i], gaussian_widths[i])
        for i in range(len(centers))
    )


# ---------------------------------------------------------------------------
# Profile geometry : FWHM
# ---------------------------------------------------------------------------

def voigt_fwhm(fwhm_G: float, fwhm_L: float) -> float:
    """
    Compute the total FWHM of a Voigt profile (Thompson 1987 approximation).

    .. math::

        \\text{FWHM}_V = 0.5346\\,f_L
        + \\sqrt{0.2166\\,f_L^2 + f_G^2}

    Accurate to better than 0.02 % over the full range of
    :math:`f_L / f_G` ratios.

    Parameters
    ----------
    fwhm_G : float or array-like
        FWHM of the Gaussian (Doppler) core.
    fwhm_L : float or array-like
        FWHM of the Lorentzian wings.

    Returns
    -------
    float or numpy.ndarray
        Total Voigt FWHM, same unit as the inputs.

    .. warning::
        Argument order is **(fwhm_G, fwhm_L)** — Gaussian first, Lorentzian
        second. This matches the physical convention (Doppler broadening is
        the dominant term for CRRLs at high n).
    """
    return (0.5346 * fwhm_L
            + np.sqrt(0.2166 * fwhm_L**2 + fwhm_G**2))


def voigt_fwhm_error(fwhm_G: float, fwhm_L: float,
                     d_fwhm_G: float, d_fwhm_L: float) -> float:
    """
    Propagate uncertainties on Lorentzian and Gaussian FWHMs to the total
    Voigt FWHM (first-order error propagation).

    Parameters
    ----------
    fwhm_G : float
        Gaussian FWHM.
    fwhm_L : float
        Lorentzian FWHM.
    d_fwhm_G : float
        Uncertainty on ``fwhm_G``.
    d_fwhm_L : float
        Uncertainty on ``fwhm_L``.

    Returns
    -------
    float
        Uncertainty on the total Voigt FWHM.
    """
    denom = np.sqrt(0.2166 * fwhm_L**2 + fwhm_G**2)
    return (0.5346 * d_fwhm_L
            + (0.2166 * fwhm_L * d_fwhm_L + fwhm_G * d_fwhm_G) / denom)


def gaussian_fwhm_from_voigt(fwhm_V: float, fwhm_L: float) -> float:
    """
    Recover the Gaussian FWHM from the total Voigt FWHM and the Lorentzian
    FWHM (inverse of the Thompson approximation).

    Parameters
    ----------
    fwhm_V : float
        Total Voigt FWHM.
    fwhm_L : float
        Lorentzian FWHM.

    Returns
    -------
    float
        Gaussian FWHM.
    """
    g = (0.5346 + np.sqrt(0.2166)) * fwhm_L
    return np.sqrt(fwhm_V - g)


# ---------------------------------------------------------------------------
# Profile geometry : area ↔ peak amplitude
# ---------------------------------------------------------------------------

def voigt_area(peak: float, fwhm_G: float, fwhm_L: float) -> float:
    """
    Compute the integrated area under a Voigt profile from its peak value.

    Uses the approximation of Sorochenko & Smirnov (1990):

    .. math::

        A = p \\cdot \\text{peak} \\cdot \\text{FWHM}_V

    where :math:`p = 1.57 - 0.507\\,\\exp(-0.85\\,f_L/f_G)`.

    Parameters
    ----------
    peak : float
        Peak value of the Voigt profile (not of the Lorentzian component).
    fwhm_G : float
        FWHM of the Gaussian component.
    fwhm_L : float
        FWHM of the Lorentzian component.

    Returns
    -------
    float
        Integrated area, in units of ``peak × fwhm_G`` (e.g. km/s if both
        peak and widths are expressed in those units).

    .. warning::
        Argument order is **(peak, fwhm_G, fwhm_L)** — consistent with
        :func:`voigt_fwhm`.
    """
    fwhm = voigt_fwhm(fwhm_G, fwhm_L)
    p    = 1.57 - 0.507 * np.exp(-0.85 * fwhm_L / fwhm_G)
    return peak * fwhm * p


def voigt_area_inv(area: float, fwhm_G: float, fwhm_L: float) -> float:
    """
    Recover the Voigt profile peak from its integrated area (inverse of
    :func:`voigt_area`).

    Parameters
    ----------
    area : float
        Integrated area of the profile.
    fwhm_G : float
        FWHM of the Gaussian component.
    fwhm_L : float
        FWHM of the Lorentzian component.

    Returns
    -------
    float
        Peak value of the Voigt profile.

    .. warning::
        Argument order is **(area, fwhm_G, fwhm_L)** — consistent with
        :func:`voigt_area` and :func:`voigt_fwhm`.
    """
    fwhm = voigt_fwhm(fwhm_G, fwhm_L)
    p    = 1.57 - 0.507 * np.exp(-0.85 * fwhm_L / fwhm_G)
    return area / fwhm / p


def lorentz_amplitude(area: float, fwhm_G: float, fwhm_L: float) -> float:
    """
    Compute the Lorentzian amplitude ``a_L`` from the integrated area.

    This is the amplitude parameter expected by
    :class:`~astropy.modeling.models.Voigt1D` and :func:`voigt`.

    Parameters
    ----------
    area : float
        Integrated area of the profile (km/s).
    fwhm_G : float
        FWHM of the Gaussian component (km/s).
    fwhm_L : float
        FWHM of the Lorentzian component (km/s).

    Returns
    -------
    float
        Lorentzian amplitude ``a_L``.
    """
    fwhm = voigt_fwhm(fwhm_G, fwhm_L)
    p    = 1.57 - 0.507 * np.exp(-0.85 * fwhm_L / fwhm_G)
    return area / p / fwhm


# ---------------------------------------------------------------------------
# Uncertainty propagation
# ---------------------------------------------------------------------------

def gaussian_fwhm_error(fwhm_V: float, fwhm_L: float,
                         d_fwhm_V: float, d_fwhm_L: float) -> float:
    """
    Propagate uncertainties to the Gaussian FWHM recovered from a Voigt fit.

    Parameters
    ----------
    fwhm_V : float
        Total Voigt FWHM.
    fwhm_L : float
        Lorentzian FWHM.
    d_fwhm_V : float
        Uncertainty on ``fwhm_V``.
    d_fwhm_L : float
        Uncertainty on ``fwhm_L``.

    Returns
    -------
    float
        Uncertainty on the Gaussian FWHM.
    """
    sigma = gaussian_fwhm_from_voigt(fwhm_V, fwhm_L)
    coeff = 0.5346 + np.sqrt(0.2166)
    ds2   = (d_fwhm_V**2 + coeff**2 * d_fwhm_L**2) / (4.0 * sigma**2)
    return np.sqrt(ds2)


def lorentz_amplitude_error(a_L: float, area: float, d_area: float,
                             fwhm_V: float, d_fwhm_V: float,
                             fwhm_G: float, fwhm_L: float,
                             d_fwhm_G: float, d_fwhm_L: float) -> float:
    """
    Propagate uncertainties to the Lorentzian amplitude ``a_L``.

    Applies the full analytical error formula:

    .. math::

        \\Delta a_L^2 =
            \\left(\\frac{a_L}{A}\\right)^2 \\Delta A^2
          + \\left(\\frac{a_L}{c}\\right)^2 \\Delta c^2
          + \\left(\\frac{a_L}{w}\\right)^2 \\Delta w^2

    where :math:`c` is the Sorochenko & Smirnov shape factor and :math:`w`
    is the total Voigt FWHM.

    Parameters
    ----------
    a_L : float
        Current Lorentzian amplitude value.
    area : float
        Integrated line area.
    d_area : float
        Uncertainty on ``area``.
    fwhm_V : float
        Total Voigt FWHM.
    d_fwhm_V : float
        Uncertainty on ``fwhm_V``.
    fwhm_G : float
        Gaussian FWHM.
    fwhm_L : float
        Lorentzian FWHM.
    d_fwhm_G : float
        Uncertainty on ``fwhm_G``.
    d_fwhm_L : float
        Uncertainty on ``fwhm_L``.

    Returns
    -------
    float
        Uncertainty on ``a_L``.
    """
    sqrt2ln2 = np.sqrt(2.0 * np.log(2.0))

    # Shape factor k and c (Sorochenko & Smirnov 1990)
    k = sqrt2ln2 * fwhm_G / (sqrt2ln2 * fwhm_G + 0.5 * fwhm_L)
    c = 1.572 + 0.05288 * k - 1.323 * k**2 + 0.7658 * k**3

    # Partial derivatives of k
    denom = (sqrt2ln2 * fwhm_G + 0.5 * fwhm_L)**2
    dk_ds = (sqrt2ln2 * (sqrt2ln2 * fwhm_G + 0.5 * fwhm_L) - 2.0 * np.log(2.0) * fwhm_G) / denom
    dk_dg = (-sqrt2ln2 * fwhm_G * 0.5) / denom

    dc_dk = 0.05288 - 2.0 * 1.323 * k + 3.0 * 0.7658 * k**2
    Dc2   = dc_dk**2 * (dk_ds**2 * d_fwhm_G**2 + dk_dg**2 * d_fwhm_L**2)

    Da2 = (
        (a_L / area)**2  * d_area**2
      + (a_L / c)**2     * Dc2
      + (a_L / fwhm_V)**2 * d_fwhm_V**2
    )
    return np.sqrt(Da2)


# ---------------------------------------------------------------------------
# High-level fitter
# ---------------------------------------------------------------------------

def fit_multi_voigt(v: np.ndarray, stack: np.ndarray,
                    velo_shift: list, turbu: list) -> tuple:
    """
    Fit a multi-component Voigt model to a stacked RRL spectrum.

    Builds a parametric model with one shared line centre (``center``) and
    per-component fractional amplitudes and Lorentzian widths. The Gaussian
    width of each component is fixed to the convolution of the turbulent
    velocity and the spectral resolution.

    The free parameters are:

    - ``center``  : common velocity centre [km/s]
    - ``a_0``     : amplitude of the first (reference) component
    - ``frac_i``  (i ≥ 1) : amplitude ratios relative to ``a_0``
    - ``lw_i``    : Lorentzian FWHM of each component [km/s]

    Parameters
    ----------
    v : numpy.ndarray
        Velocity axis in km/s.
    stack : numpy.ndarray
        Optical depth spectrum (τ) to fit.
    velo_shift : list of float
        Velocity offsets of each component from ``center`` [km/s].
        Length determines the number of components.
    turbu : list of float
        Turbulent velocity of each component [km/s]. Combined in quadrature
        with the spectral resolution to set the Gaussian width.

    Returns
    -------
    popt : numpy.ndarray
        Optimal parameters ``[center, a_0, frac_1, ..., lw_0, lw_1, ...]``.
    pcov : numpy.ndarray
        Covariance matrix of the fit.
    voigt_fct : callable
        The fitted model function, callable as ``voigt_fct(v, *popt)``.
    convos : list of float
        Gaussian half-widths used for each component [km/s].

    Notes
    -----
    Bounds are set so that amplitudes are negative (absorption lines) and
    Lorentzian widths stay within the spectral range.
    """
    nb     = len(velo_shift)
    dv     = np.abs(v[1] - v[0])            # velocity resolution [km/s]
    deltav = np.abs(v[-1] - v[0]) / 2.0     # half-range of the spectrum [km/s]

    # Gaussian widths: turbulence convolved with spectral resolution
    convos = [0.5 * np.sqrt(turbu[i]**2 + dv**2) for i in range(nb)]

    # Build the lambda string dynamically for curve_fit compatibility
    # Parameters: center, a_0, [frac_1, ..., frac_{nb-1}], lw_0, ..., lw_{nb-1}
    param_str = "center, a_0, " + ", ".join(f"frac_{i}" for i in range(1, nb)) \
                + ", " + ", ".join(f"lw_{i}" for i in range(nb))

    centers_str = "[" + ", ".join(f"center + {velo_shift[i]}" for i in range(nb)) + "]"

    ampli_parts = ["a_0"] + [f"frac_{i}*a_0" for i in range(1, nb)]
    ampli_str   = "[" + ", ".join(ampli_parts) + "]"

    lw_str = "[" + ", ".join(f"lw_{i}" for i in range(nb)) + "]"

    model_str = (
        f"lambda v, {param_str} : "
        f"multiple_voigt(v, {centers_str}, {ampli_str}, {lw_str}, {convos})"
    )
    voigt_fct = eval(model_str)  # noqa: S307  (dynamic but controlled)

    bounds = (
        [velo_shift[0] - 5.0] + [-1.0] * nb + [0.0]    * nb,
        [velo_shift[0] + 5.0] + [ 0.0] * nb + [deltav] * nb,
    )

    popt, pcov = curve_fit(voigt_fct, v, stack, bounds=bounds, maxfev=3000)
    return popt, pcov, voigt_fct, convos
