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

- profile          — Sum of N Voigt components, scaled by OG (curve_fit model)
- fit_stack        — Fit a profile on a single stacked spectrum


References
----------
- Sorochenko & Smirnov (1990) — Voigt area approximation.
- Thompson (1987) — Voigt FWHM approximation.

Dependencies
------------
External: astropy >= 5.0, numpy, scipy, typing
Local: spectral_tools.tools
"""

import numpy as np
from astropy.modeling.models import Voigt1D
from scipy.optimize import curve_fit
from typing import Callable, Sequence, Union

from spectral_tools.tools import f_to_v, v_to_f


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
# Fitting functions
# ---------------------------------------------------------------------------

def profile(f: np.ndarray[float],
          *args,
           og: float = 1e4) -> np.ndarray:
    """
    Sum of N Voigt components, scaled by the optical-depth gain factor.

    This is the model function passed to ``scipy.optimize.curve_fit``.
    Parameters for all components are packed flat in ``args`` as groups of
    four: ``[c0, a0, lw0, gw0,  c1, a1, lw1, gw1, …]``.

    The returned value is ``−Σ voigt_i × og`` so that absorption lines
    (negative optical depth) produce a positive residual for ``curve_fit``.

    Parameters
    ----------
    f : np.ndarray
        Frequency axis [MHz].
    og : float, optional
        Optical-depth gain factor applied to the sum.  Default ``1e4``.
        This is used to help the 'curve_fit' function due to low performance with smaller values.
        Use og=1. for unscaled voigt profiles.
    *args : float
        Flat sequence of Voigt parameters ``[c, a, lw, gw]`` per component.
        ``len(args)`` must be a multiple of 4.
    
    Returns
    -------
    np.ndarray
        ``−Σ voigt_i(v) × og``, same shape as ``v``.

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_tools.line_fitting import profile
    >>> v = np.linspace(-0.01, 0.01, 500)
    >>> y = multi_voigt(v, 0.0, 500.0, 1e-4, 5e-4, og=1e4)
    """
    total = np.zeros(len(f))
    for k in range(len(args) // 4):
        c0, a0, lw0, gw0 = args[4*k : 4*(k+1)]
        total += voigt(f, c0, a0, lw0, gw0)
    return -total * og

    
def fit_stack(
        spectrum: np.ndarray,
        freq_axis: np.ndarray,
        central_freq: float,
        velos: Union[float, Sequence[float]] = 0.,
        dvelos: Union[float, Sequence[float]] = 20.0,
        og: float = 1e4,
        amp_lo: float = 0.0,
        amp_hi: float = 1.0,
        width_max_kms: float = 200.0,
        width_max_kms_gw: float | None = None,
        width_max_kms_lw: float | None = None,
        wing_fraction: float = 1 / 6,
        maxfev: int = 10000,
        rms: bool = True,
        blind_fit:bool = False,
        regime: str = "Lorentz"
        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a profile on a single (rebinned) stacked RRL spectrum.

    The spectrum is assumed to contain absorption lines (negative optical
    depth).  The model :func:`profile` returns ``−Σ voigt_i × og``,
    which makes residuals positive for ``curve_fit``.

    Parameters
    ----------
    spectrum : np.ndarray, shape (M,)
        Rebinned stacked spectrum already multiplied by ``og``.
    freq_axis : np.ndarray, shape (M,)
        Frequency axis corresponding to ``spectrum`` [MHz].
    central_freq : float
        Representative line frequency for this stack interval [MHz].
        Used to convert velocity guesses and width limits to frequency
        offsets via ``v_to_f``.
    velos : sequence of float
        Initial velocity guess for each Voigt component [km/s].
        Single-component fit: ``[0.0]``.
    dvelos : float or sequence of float, optional
        Half-search window around each velocity guess [km/s].
        A scalar applies the same window to all components.
        Default ``20.0`` km/s.
    og : float, optional
        Optical-depth gain factor.  Default ``1e4``.
    amp_min_frac : float, optional
        Lower bound on amplitude.
        Default 0.0
    amp_max_frac : float, optional
        Upper bound on amplitude.
        Default 1.0
        Initial guess is set to ``1 / og``.
    width_max_kms : float, optional
        Maximum allowed Lorentzian and Gaussian HWHM [km/s].
        Converted to MHz internally via ``v_to_f_fn``.
        Default ``200.0`` km/s.  Pass ``400.0`` for high-n stacks.
    width_max_kms_gw : float, optional
        Maximum allowed Gaussian FWHM [km/s].
        Converted to MHz internally via ``v_to_f``.
        Default None.
    width_max_kms_lw : float, optional
        Maximum allowed Lorentzian FWHM [km/s].
        Converted to MHz internally via ``v_to_f``.
        Default None.
    wing_fraction : float, optional
        Fraction of the spectrum used as wings on each side to estimate
        the noise level when ``rms=True``.  The central portion
        ``[wing_fraction : 1 − wing_fraction]`` is masked.
        Default ``1/6``  →  outer sixth on each side.
    maxfev : int, optional
        Maximum number of function evaluations for ``curve_fit``.
        Default ``10000``.
    rms : bool, optional
        If ``True``, weight the fit by the inverse of the wing noise
        (passes ``sigma=noise_std`` to ``curve_fit``).  Default ``True``.
    blind_fit : bool, optional
        If True, the fit is performed without bounds to the optimisation. 
    regime : str, optional
        Accepted values are "Lorentz" and "Doppler". If the regime is lorentzian -> w_L > w_D (and resp.)
        Default "Lorentz".::

    Returns
    -------
    popt : np.ndarray, shape (4 × N_comp,)
        Fitted parameters ``[c0, a0, lw0, gw0,  c1, a1, lw1, gw1, …]``.
        All-zeros array if ``curve_fit`` fails or returns NaN.
    pcov : np.ndarray, shape (4 × N_comp, 4 × N_comp)
        Parameter covariance matrix.  All-zeros on failure.

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_tools.line_fitting import fit_stack
    >>> freq = np.linspace(-0.01, 0.01, 250)
    >>> spec = np.zeros(250)
    >>> popt, pcov = fit_stack(spec, freq, central_freq=47.0, velos=[0.0])
    """
    # ensure variable validity
    if not isinstance(velos, Sequence):
        velos = [velos]
    
    if regime not in ["Lorentz", "Doppler"]:
        raise ValueError(f"{regime} is not a valid value for regime. Supported inputs are: 'Lorentz' or 'Doppler'")
    
    n_comp = len(velos)
    n_pars = 4 * n_comp
    
    # ── Noise from wings ──────────────────────────────────────────────────
    n_pts = len(spectrum)
    i_lo  = int(np.floor(n_pts * wing_fraction))
    i_hi  = int(np.ceil(n_pts * (1.0 - wing_fraction)))
    wings = spectrum.astype(float, copy=True)
    wings[i_lo:i_hi] = np.nan
    noise = float(np.nanstd(wings)) if rms else 1.0

    # ── Amplitude bounds ──────────────────────────────────────────────────
    amp_p0 = 1/og # og represents 1/magnitude of the dip (expected)

    # ── Width bound in frequency ──────────────────────────────────────────
    w_max_lw  = -v_to_f(width_max_kms, central_freq).value if (width_max_kms_lw is None) else -v_to_f(width_max_kms_lw, central_freq).value
    w_max_gw  = -v_to_f(width_max_kms, central_freq).value if (width_max_kms_gw is None) else -v_to_f(width_max_kms_gw, central_freq).value
    
    match regime :
       case "Lorentz" :
           frac_gw = 1 / 4.0
           frac_lw = 1 / 2.0
       case "Doppler" :
           frac_gw = 1 / 2.0
           frac_lw = 1 / 4.0
    
    w_p0lw = -v_to_f(width_max_kms * frac_lw, central_freq).value if (width_max_kms_lw is None) else -v_to_f(width_max_kms_lw * frac_lw, central_freq).value
    w_p0gw = -v_to_f(width_max_kms * frac_gw, central_freq).value if (width_max_kms_gw is None) else -v_to_f(width_max_kms_gw * frac_gw, central_freq).value
    # ── Build p0 and bounds ───────────────────────────────────────────────
    bounds = np.zeros((2, n_pars))
    p0     = np.zeros(n_pars)

    for k, v0 in enumerate(velos):
        dv = dvelos[k] if hasattr(dvelos, '__len__') else float(dvelos)

        # Centre
        bounds[0][4*k] = -v_to_f(v0 - dv, central_freq).value
        bounds[1][4*k] = -v_to_f(v0 + dv, central_freq).value
        p0[4*k]        = -v_to_f(v0,       central_freq).value

        # Amplitude
        bounds[0][4*k+1] = amp_lo
        bounds[1][4*k+1] = amp_hi
        p0[4*k+1]        = amp_p0

        # lw and gw widths
        for j in (2, 3): # width is always > 0
            bounds[0][4*k+j] = 0.0
        
        bounds[1][4*k+2] = w_max_lw
        bounds[1][4*k+3] = w_max_gw
        
        p0[4*k+2] = w_p0lw
        p0[4*k+3] = w_p0gw

    # ── Model closure: binds og so curve_fit signature stays (v, *params) ─
    def _model(v, *args):
        return profile(v, *args, og=og)

    # ── Fit ───────────────────────────────────────────────────────────────
    try:
    #if True :
        if blind_fit :
            popt, pcov = curve_fit(
                _model, freq_axis, spectrum,
                p0=p0, # bounds=bounds,
                maxfev=maxfev, nan_policy='omit',
               sigma=noise,
            )
        else :
            popt, pcov = curve_fit(
                _model, freq_axis, spectrum,
                p0=p0, bounds=bounds,
                maxfev=maxfev, nan_policy='omit',
               sigma=noise,
            )
    except (RuntimeError, ValueError) as e:
    #else:
        print(e)
        return np.zeros(n_pars), np.zeros((n_pars, n_pars))

    if np.any(np.isnan(popt)):
        print("No fit found")
        return np.zeros(n_pars), np.zeros((n_pars, n_pars))

    return popt, pcov 


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
