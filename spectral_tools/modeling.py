# -*- coding: utf-8 -*-
"""
modeling.py
===========
Physical modeling of carbon radio recombination lines (CRRLs).

This module computes theoretical line profiles and integrated areas for
CRRLs, and provides cost functions for parameter-space exploration.

All physical quantities are **unitless** (SI or km/s as documented per
function), hence the original name ``modeling_unitless``. Astropy units are
not used at runtime for performance reasons on large grids.

This module provides:

- **Line broadening** : natural, Doppler, pressure, radiation, total Voigt FWHM
- **Line amplitude** : integrated area from non-LTE population factors (bn·βn)
- **Profile construction** : full Voigt line profiles in velocity space
- **Grid modeling** : synthetic (n, velocity) surfaces for single or
  multi-component ISM clouds
- **Cost functions** : χ² evaluation for optimisation or grid search

Architecture note
-----------------
Broadening functions that were duplicated in ``generate_grid_loop.py`` and
``generate_finegrid.py`` are now centralised here. Those pipeline scripts
should import from this module instead.

Dependencies
------------
astropy (constants only), numpy, scipy, pickle
Internal: spectral_tools.atoms, spectral_tools.fitfunc

External data files (loaded at import time)
-------------------------------------------
``../files/B1B2.pickle``
    Pre-computed bn·βn interpolators, dict ``{str(n): (f1, f2)}``.
``../files/alphagamma.pickle``
    Pre-computed collisional broadening interpolators
    ``(alpha_f, grad_alpha_f, gamma_f, grad_gamma_f)``.

References
----------
- Gordon & Sorochenko (1992) — integrated area formula (eq. 2.x).
- Salgado et al. (2017a, 2017b) — radiation broadening.
- Salas et al. (2017) — radiation broadening cross-check.
"""

import os
import pickle
import warnings

import numpy as np
from astropy.constants import k_B, c
from astropy.constants import u as amu
from astropy.modeling.models import Voigt1D
from scipy.interpolate import interp1d

from spectral_tools.atoms import line_freq
from spectral_tools import line_fitting
import spectral_tools.tools as tools

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Physical constants (unitless for performance on large grids)
# ---------------------------------------------------------------------------

#: Atomic mass of carbon-12 [kg]
M_C = 12 * amu.value

#: Boltzmann constant [J/K]
k_B_SI = k_B.value

#: Speed of light [km/s]
c_kms = c.value * 1e-3


# ---------------------------------------------------------------------------
# Load pre-computed interpolators from serialised files
# ---------------------------------------------------------------------------

#: Path to the directory containing the reference pickle files.
#: Override this variable before importing if your files live elsewhere.
FILES_PATH: str = os.path.join(os.path.dirname(__file__), "..", "files")


def _load_interpolators(path: str) -> tuple:
    """
    Load bn·βn and collisional broadening interpolators from pickle files.

    Parameters
    ----------
    path : str
        Directory containing ``B1B2.pickle`` and ``alphagamma.pickle``.

    Returns
    -------
    FUNCB1B2 : dict
        ``{str(n): (f1, f2)}`` where ``bn*betan = f1(Te, Ne) * f2(Te, Ne)``.
    alpha_f : callable
        Interpolator for the collisional rate exponent α(Te).
    gamma_f : callable
        Interpolator for the collisional rate exponent γ(Te).
    """
    with open(os.path.join(path, "B1B2.pickle"), "rb") as fh:
        FUNCB1B2 = pickle.load(fh)
    with open(os.path.join(path, "alphagamma.pickle"), "rb") as fh:
        alpha_f, _grad_alpha_f, gamma_f, _grad_gamma_f = pickle.load(fh)
    return FUNCB1B2, alpha_f, gamma_f


FUNCB1B2, _alpha_f, _gamma_f = _load_interpolators(FILES_PATH)


# ---------------------------------------------------------------------------
# Non-LTE departure coefficients
# ---------------------------------------------------------------------------

def bnbetan(n, Te: float, Ne: float) -> float:
    """
    Compute the non-LTE product bn·βn for principal quantum number ``n``.

    The product quantifies the departure from local thermodynamic equilibrium
    (LTE) of level ``n``. Values are obtained from pre-computed interpolators
    stored in ``B1B2.pickle``.

    Parameters
    ----------
    n : int or array-like of int
        Principal quantum number(s).
    Te : float
        Electron temperature [K].
    Ne : float
        Electron density [cm⁻³].

    Returns
    -------
    float or numpy.ndarray
        bn·βn value(s), dimensionless.
    """
    if hasattr(n, "__len__"):
        res = np.zeros(np.shape(n))
        for k, nk in enumerate(n):
            f1, f2 = FUNCB1B2[str(int(nk))]
            res[k] = f1(Te, Ne) * f2(Te, Ne)
        return res
    f1, f2 = FUNCB1B2[str(int(n))]
    return f1(Te, Ne) * f2(Te, Ne)


# ---------------------------------------------------------------------------
# Line broadening mechanisms
# ---------------------------------------------------------------------------

def natural_broadening(n) -> float:
    """
    FWHM of the natural (spontaneous emission) Lorentzian broadening [MHz].

    From Gordon & Sorochenko (1992), eq. 2.15:

    .. math::

        \\Delta\\nu_{\\rm nat} = 1.2\\times10^{-6}\\,\\frac{\\ln n}{n^2}\\,\\nu_0

    Parameters
    ----------
    n : int or array-like
        Principal quantum number.

    Returns
    -------
    float or numpy.ndarray
        Natural broadening FWHM [MHz].
    """
    nu_0    = line_freq(n).value          # central frequency [MHz]
    dnu_rel = 1.2e-6 * np.log(n) / n**2  # dimensionless relative width
    return dnu_rel * nu_0


def doppler_broadening(n, Te: float, v_turb: float) -> float:
    """
    FWHM of the Doppler (Gaussian) broadening [MHz].

    Combines thermal agitation and non-thermal turbulence in quadrature
    (Gordon & Sorochenko 1992, eq. 2.26):

    .. math::

        \\Delta V_G = \\sqrt{4\\ln 2}\\,\\sqrt{\\frac{2k_BT_e}{m_C} + v_{\\rm turb}^2}

    Parameters
    ----------
    n : int or array-like
        Principal quantum number.
    Te : float
        Electron temperature [K].
    v_turb : float
        Non-thermal (turbulent) velocity dispersion [km/s].

    Returns
    -------
    float or numpy.ndarray
        Doppler broadening FWHM [MHz].
    """
    # Thermal velocity squared [km²/s²]
    v_th2 = 2.0 * k_B_SI * Te / M_C * 1e-6
    dV_G  = np.sqrt(4.0 * np.log(2.0)) * np.sqrt(v_th2 + v_turb**2)
    nu_0  = line_freq(n).value
    return dV_G * nu_0 / c_kms


def pressure_broadening(n, Ne: float, Te: float) -> float:
    """
    FWHM of the collisional (pressure) Lorentzian broadening [MHz].

    Uses the pre-computed interpolators α_f(Te) and γ_f(Te) from
    ``alphagamma.pickle`` (calibrated on atomic structure calculations):

    .. math::

        \\Delta\\nu_{\\rm col} = \\frac{N_e\\,10^{\\alpha(T_e)}\\,n^{\\gamma(T_e)}}{\\pi}

    Parameters
    ----------
    n : int or array-like
        Principal quantum number.
    Ne : float
        Electron density [cm⁻³].
    Te : float
        Electron temperature [K].

    Returns
    -------
    float or numpy.ndarray
        Pressure broadening FWHM [MHz].
    """
    Gamma_col = Ne * 10**_alpha_f(Te) * n**_gamma_f(Te) * 1e-6  # Hz → MHz
    return Gamma_col / np.pi


def harmsuminfini(alpha: float, precision: int = 3) -> float:
    """
    Compute the generalised harmonic sum Σ n^alpha until convergence.

    Convergence is assessed by rounding to ``precision`` decimal places.
    The sum is capped at 100 terms to prevent infinite loops.

    Parameters
    ----------
    alpha : float
        Exponent of the harmonic sum. Must be negative for convergence.
    precision : int, optional
        Number of decimal places for convergence test. Default 3.

    Returns
    -------
    float
        Converged partial sum, rounded to ``precision`` decimals.
    """
    somme, n = 0.0, 0
    prev_round, curr_round = 0.0, 1.0
    while curr_round > prev_round:
        n += 1
        prev_round = np.round(somme, precision)
        somme     += n**alpha
        curr_round = np.round(somme, precision)
        if n == 100:
            break
    return curr_round


def radiation_broadening(n, T0: float, alpha: float = -2.6,
                          precision: int = 3) -> float:
    """
    FWHM of the radiation (Galactic background) Lorentzian broadening [MHz].

    Follows Salgado et al. (2017b), accounting for the power-law spectral
    index of the Galactic synchrotron background:

    .. math::

        \\Delta\\nu_{\\rm rad} = \\frac{2}{\\pi}\\,2.137\\times10^4\\,
        \\left(\\frac{2R_\\infty c}{\\nu_0}\\right)^{\\alpha+1}
        k_B T_0\\,\\nu_0\\,n^{-3\\alpha-2}\\,\\sum_{\\Delta n}(\\Delta n)^{\\alpha-2}

    Parameters
    ----------
    n : int or array-like
        Principal quantum number.
    T0 : float
        Brightness temperature of the Galactic background [K].
    alpha : float, optional
        Spectral index of the background radiation law. Default -2.6
        (reference frequency 100 MHz).
    precision : int, optional
        Precision for :func:`harmsuminfini`. Default 3.

    Returns
    -------
    float or numpy.ndarray
        Radiation broadening FWHM [MHz].

    Notes
    -----
    Reference frequency convention:
    - alpha = -2.6 → ν₀ = 100 MHz
    - otherwise   → ν₀ = 45  MHz
    """
    nu0_ref = 100e6 if alpha == -2.6 else 45e6
    coeff   = 2.0 / np.pi * 2.137e4
    ratio   = (6.5796839e15 / nu0_ref)**(alpha + 1)
    n_dep   = 1.380649e-16 * T0 * nu0_ref * n**(-3 * alpha - 2)
    somme   = harmsuminfini(alpha - 2, precision)
    return coeff * ratio * n_dep * somme * 1e-6  # Hz → MHz


def lorentzian_broadening(n, Ne: float, Te: float,
                           T0: float, alpha: float = -2.6) -> float:
    """
    Total Lorentzian FWHM [MHz]: sum of pressure, radiation, and natural terms.

    Parameters
    ----------
    n : int or array-like
        Principal quantum number.
    Ne : float
        Electron density [cm⁻³].
    Te : float
        Electron temperature [K].
    T0 : float
        Galactic background brightness temperature [K].
    alpha : float, optional
        Spectral index of the background. Default -2.6.

    Returns
    -------
    float or numpy.ndarray
        Total Lorentzian FWHM [MHz].
    """
    return (pressure_broadening(n, Ne, Te)
            + radiation_broadening(n, T0, alpha)
            + natural_broadening(n))


def total_broadening(n, Ne: float, Te: float,
                     T0: float = 1400.0, v_turb: float = 0.0,
                     alpha: float = -2.6) -> float:
    """
    Total Voigt FWHM of the line [MHz], using the Thompson (1987) approximation.

    .. math::

        \\Delta\\nu_V \\approx 0.5346\\,\\Delta\\nu_L
        + \\sqrt{0.2166\\,\\Delta\\nu_L^2 + \\Delta\\nu_G^2}

    Parameters
    ----------
    n : int or array-like
        Principal quantum number.
    Ne : float
        Electron density [cm⁻³].
    Te : float
        Electron temperature [K].
    T0 : float, optional
        Galactic background temperature [K]. Default 1400 K.
    v_turb : float, optional
        Turbulent velocity [km/s]. Default 0.
    alpha : float, optional
        Background spectral index. Default -2.6.

    Returns
    -------
    float or numpy.ndarray
        Total Voigt FWHM [MHz].
    """
    dG = doppler_broadening(n, Te, v_turb)
    dL = lorentzian_broadening(n, Ne, Te, T0, alpha)
    return 0.5346 * dL + np.sqrt(0.2166 * dL**2 + dG**2)


def total_broadening_kms(n, Ne: float, Te: float,
                          T0: float, v_turb: float,
                          alpha: float = -2.6) -> float:
    """
    Total Voigt FWHM converted to velocity units [km/s].

    Convenience wrapper around :func:`total_broadening` and
    :func:`~spectral_tools.tools.f_to_v`.

    Parameters
    ----------
    n : int or array-like
        Principal quantum number.
    Ne : float
        Electron density [cm⁻³].
    Te : float
        Electron temperature [K].
    T0 : float
        Galactic background temperature [K].
    v_turb : float
        Turbulent velocity [km/s].
    alpha : float, optional
        Background spectral index. Default -2.6.

    Returns
    -------
    float or numpy.ndarray
        Total Voigt FWHM [km/s] (positive value).
    """
    dfreq = total_broadening(n, Ne, Te, T0, v_turb, alpha)
    freq  = line_freq(n).value
    return -tools.f_to_v(dfreq, freq).value * 1e-3


# ---------------------------------------------------------------------------
# Integrated line area
# ---------------------------------------------------------------------------

def integrated_area(n, Te: float, Ne: float, L: float) -> float:
    """
    Compute the integrated optical depth area of a CRRL (Gordon & Sorochenko).

    .. math::

        \\int\\tau\\,d\\nu = -0.2\\,b_n\\beta_n
        \\left(\\frac{T_e}{100}\\right)^{-2.5}
        \\left(\\frac{N_e}{0.1}\\right)^2 L

    Parameters
    ----------
    n : int or array-like
        Principal quantum number.
    Te : float
        Electron temperature [K].
    Ne : float
        Electron density [cm⁻³].
    L : float
        Path length through the emitting region [pc].

    Returns
    -------
    float or numpy.ndarray
        Integrated area [MHz] (negative for absorption lines).
    """
    contrib_lte     = bnbetan(n, Te, Ne)
    contrib_temp    = (Te / 100.0)**(-2.5)
    contrib_density = (Ne / 0.1)**2
    return -0.2 * contrib_lte * contrib_temp * contrib_density * L * 1e-6


# ---------------------------------------------------------------------------
# Lorentzian amplitude (astropy Voigt1D convention)
# ---------------------------------------------------------------------------

def lorentz_amplitude_from_area(area: float, fwhm_L: float) -> float:
    """
    Convert an integrated area to the Lorentzian amplitude used by
    :class:`~astropy.modeling.models.Voigt1D`.

    Uses the exact astropy normalisation: the Lorentzian component of
    Voigt1D is normalised such that its integral equals
    ``amplitude_L × π × (fwhm_L / 2)``, giving:

    .. math::

        a_L = \\frac{2\\,A}{\\pi\\,\\Delta\\nu_L}

    Parameters
    ----------
    area : float
        Integrated line area (same units as fwhm_L × amplitude).
    fwhm_L : float
        Lorentzian FWHM (same units as area / amplitude).

    Returns
    -------
    float
        Lorentzian amplitude ``a_L`` for :class:`~astropy.modeling.models.Voigt1D`.

    Notes
    -----
    This replaces the old ``a_Lorentz_old`` (Sorochenko approximation) which
    was less accurate for the astropy Voigt1D normalisation convention.
    """
    return 2.0 * area / np.pi / fwhm_L


# ---------------------------------------------------------------------------
# Full Voigt line profile and surface generation
# ---------------------------------------------------------------------------

def full_voigt_line(center: float, n: int, Te: float, Ne: float,
                    v_turb: float, L: float, T0: float,
                    alpha: float = -2.6):
    """
    Build a :class:`~astropy.modeling.models.Voigt1D` instance for one CRRL.

    All physical broadening terms are computed and converted to km/s before
    being passed to Voigt1D.

    Parameters
    ----------
    center : float
        Line centre velocity [km/s].
    n : int
        Principal quantum number.
    Te : float
        Electron temperature [K].
    Ne : float
        Electron density [cm⁻³].
    v_turb : float
        Turbulent velocity [km/s].
    L : float
        Path length [pc].
    T0 : float
        Galactic background temperature [K].
    alpha : float, optional
        Background spectral index. Default -2.6.

    Returns
    -------
    astropy.modeling.models.Voigt1D
        Callable profile; evaluate on a velocity array with ``profile(v)``.
    """
    f0 = line_freq(n).value  # central frequency [MHz]

    # Convert frequency widths to velocity widths [km/s]
    fwhm_G = -tools.f_to_v(doppler_broadening(n, Te, v_turb), f0).value * 1e-3
    fwhm_L = -tools.f_to_v(lorentzian_broadening(n, Ne, Te, T0, alpha), f0).value * 1e-3

    area   = -tools.f_to_v(integrated_area(n, Te, Ne, L), f0).value * 1e-3
    a_L    = lorentz_amplitude_from_area(area, fwhm_L)

    return Voigt1D(x_0=center, amplitude_L=a_L, fwhm_L=fwhm_L, fwhm_G=fwhm_G)


def create_surface(vref: np.ndarray, quantum_numbers,
                   Te: float, Ne: float, T0: float,
                   v_turb: float, L: float, v0: float) -> np.ndarray:
    """
    Generate a synthetic (n × velocity) spectral surface for one ISM component.

    Parameters
    ----------
    vref : numpy.ndarray
        Velocity reference axis [km/s].
    quantum_numbers : array-like of int
        Principal quantum numbers to model.
    Te : float
        Electron temperature [K].
    Ne : float
        Electron density [cm⁻³].
    T0 : float
        Galactic background temperature [K].
    v_turb : float
        Turbulent velocity [km/s].
    L : float
        Path length [pc].
    v0 : float
        Velocity of the component [km/s].

    Returns
    -------
    numpy.ndarray, shape (len(quantum_numbers), len(vref))
        Synthetic optical depth surface τ(n, v).
    """
    surface = np.zeros((len(quantum_numbers), len(vref)))
    for k, n in enumerate(np.array(quantum_numbers, dtype=int)):
        surface[k] = full_voigt_line(v0, n, Te, Ne, v_turb, L, T0)(vref)
    return surface


def create_surface_multi(vref: np.ndarray, quantum_numbers,
                          Tes, Nes, T0s, v_turbs, Ls,
                          velocities) -> np.ndarray:
    """
    Generate a synthetic spectral surface for multiple ISM components.

    Each component contributes additively to the total optical depth.

    Parameters
    ----------
    vref : numpy.ndarray
        Velocity reference axis [km/s].
    quantum_numbers : array-like of int
        Principal quantum numbers.
    Tes, Nes, T0s, v_turbs, Ls : array-like of float
        Physical parameters for each component (same length as ``velocities``).
    velocities : array-like of float
        Velocity of each component [km/s].

    Returns
    -------
    numpy.ndarray, shape (len(quantum_numbers), len(vref))
        Total synthetic optical depth surface τ(n, v).
    """
    surface = np.zeros((len(quantum_numbers), len(vref)))
    for i in range(len(velocities)):
        surface += create_surface(
            vref, quantum_numbers,
            Tes[i], Nes[i], T0s[i], v_turbs[i], Ls[i], velocities[i]
        )
    return surface


# ---------------------------------------------------------------------------
# Cost functions for optimisation / grid search
# ---------------------------------------------------------------------------

def cost_function(params, quantum_numbers, observation: np.ndarray,
                  rms: np.ndarray, center: float,
                  vref: np.ndarray) -> float:
    """
    χ² cost function for a single-component model.

    .. math::

        \\chi^2 = \\sum_{n,v} \\left(\\frac{\\tau_{\\rm mod} - \\tau_{\\rm obs}}{\\sigma}\\right)^2

    Parameters
    ----------
    params : array-like of float
        ``[Te, Ne, T0, v_turb, L]``
    quantum_numbers : array-like of int
        Principal quantum numbers.
    observation : numpy.ndarray
        Observed optical depth surface, shape (n, v).
    rms : numpy.ndarray
        Per-transition noise level, shape (n,).
    center : float
        Component velocity [km/s].
    vref : numpy.ndarray
        Velocity axis [km/s].

    Returns
    -------
    float
        Total χ² (sum over all n and v channels).
    """
    Te, Ne, T0, v_turb, L = params
    model     = create_surface(vref, quantum_numbers, Te, Ne, T0, v_turb, L, center)
    residuals = np.divide((model - observation).T, rms).T
    return np.nansum(residuals**2)


def cost_function_multi(params, quantum_numbers, observation: np.ndarray,
                         rms: np.ndarray, velocities,
                         vref: np.ndarray) -> float:
    """
    χ² cost function for a multi-component model.

    Parameters are interleaved as:
    ``[Te_0, Te_1, ..., Ne_0, Ne_1, ..., T0_0, T0_1, ...,
       vt_0, vt_1, ..., L_0, L_1, ...]``

    Parameters
    ----------
    params : array-like of float
        Flattened parameter array, length ``5 × n_components``.
    quantum_numbers : array-like of int
        Principal quantum numbers.
    observation : numpy.ndarray
        Observed optical depth surface, shape (n, v).
    rms : numpy.ndarray
        Per-transition noise level, shape (n,).
    velocities : array-like of float
        Velocity of each component [km/s].
    vref : numpy.ndarray
        Velocity axis [km/s].

    Returns
    -------
    float
        Total χ² (sum over all n and v channels).
    """
    n_comp  = len(velocities)
    n_param = 5
    # Unpack interleaved parameter layout
    Tes, Nes, T0s, v_turbs, Ls = [
        params[i: i + n_comp]
        for i in range(0, n_param * n_comp, n_comp)
    ]
    model     = create_surface_multi(vref, quantum_numbers,
                                     Tes, Nes, T0s, v_turbs, Ls, velocities)
    residuals = np.divide((model - observation).T, rms).T
    return np.nansum(residuals**2)


# ---------------------------------------------------------------------------
# Observation loader (used by fitting notebooks)
# ---------------------------------------------------------------------------

def load_stacked_observations(source: str, path: str,
                               source_info_file: str = os.path.join(os.path.dirname(__file__), "..", "files", "source_info.txt"),
                               n_max: int = 849,
                               new_velocity: float = None) -> tuple:
    """
    Load pre-stacked per-transition spectra for a given source.

    Reads ``*noflag.txt`` files from ``path``, grids them to a common
    velocity axis, and returns the ordered stack.

    Parameters
    ----------
    source : str
        Source identifier string, e.g. ``'CASA_A'``.
    path : str
        Directory containing the per-transition text files.
    source_info_file : str, optional
        Path to the source info table. Default ``'files/source_info.txt'``.
    n_max : int, optional
        Maximum quantum number to include. Default 849.
    new_velocity : float, optional
        Override the source velocity from the info file [km/s].

    Returns
    -------
    stacks : numpy.ndarray, shape (N_transitions, N_channels)
        Optical depth spectra regridded to the common velocity axis.
    quantum_numbers : numpy.ndarray of int
        Corresponding principal quantum numbers.
    rms : numpy.ndarray
        Per-transition RMS noise.
    velocity : float
        Source velocity used [km/s].
    vref : numpy.ndarray
        Common velocity axis [km/s].
    """
    # Read source velocity from info file
    velocity = new_velocity
    if velocity is None:
        with open(source_info_file) as doc:
            for line in doc:
                if line.split()[0] in source:
                    velocity = eval(line.split()[3])
                    break
        if velocity is None:
            raise ValueError(f"Source '{source}' not found in {source_info_file}")

    # Collect spectra
    STACKS, RMS, quantum_numbers = [], [], []
    for fname in os.listdir(path):
        if "noflag.txt" not in fname or "png" in fname:
            continue
        n_min = int(fname.split("_")[3])
        n_max_file = int(fname.split("_")[-1].split("-")[0])
        if n_min != n_max_file or n_max_file > n_max:
            continue
        quantum_numbers.append(0.5 * (n_min + n_max_file))
        spectrum = np.loadtxt(os.path.join(path, fname))
        STACKS.append(spectrum)
        RMS.append(np.nanstd(spectrum))

    # Sort by quantum number and regrid to common velocity axis
    quantum_numbers, idx = np.unique(quantum_numbers, return_index=True)
    RMS    = np.array(RMS)[idx]
    STACKS = [STACKS[k] for k in idx]

    vref = STACKS[-1][0]
    stacks_regrid = np.zeros((len(STACKS), len(vref)))
    for i, spec in enumerate(STACKS):
        x, y = spec
        stacks_regrid[i] = interp1d(x, y, bounds_error=False)(vref)

    return stacks_regrid, np.array(quantum_numbers, dtype=int), RMS, velocity, vref
