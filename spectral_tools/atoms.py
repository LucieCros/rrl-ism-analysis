# -*- coding: utf-8 -*-
"""
atoms.py
========
Atomic constants and radio recombination line (RRL) frequency calculations.

This module is the single source of truth for:
- Atomic species data (mass, ionisation potential, abundance, charge)
- RRL central frequency computation (Gordon & Sorochenko 1992, eq. A.5)

All other modules in ``spectral_tools`` should import from here rather than
redefining these quantities locally.

References
----------
Gordon & Sorochenko (1992), "Radio Recombination Lines", Table 1 and Appendix A.

Dependencies
------------
astropy >= 5.0
"""

import astropy.units as u
from astropy.constants import c, m_e, Ryd
from astropy.constants import u as amu

# ---------------------------------------------------------------------------
# Atomic species database
# ---------------------------------------------------------------------------
# Format per entry:
#   [atomic_mass (u.m.a.), ionisation_potential (eV),
#    abundance_relative_to_HI, V_X - V_H (km/s), electric_charge Z]
#
# Sources: Gordon & Sorochenko (1992), Table 1.
# Negative values (-1) indicate "not applicable" for exotic isotopes.

SPECIES_DATA: dict[str, list[float]] = {
    "HI":   [1.0078,    13.6,  1.0,    0.0,   1.0],
    "HeI":  [4.0026,    24.6,  0.1,  122.1,   1.0],
    "CI":   [12.0000,   11.4,  3e-4, 149.5,   1.0],
    "NI":   [14.0067,    1.0,  1.0,    1.0,   1.0],
    "SI":   [37.9721,   10.3,  2e-5, 158.0,   1.0],
    "CI13": [13.00335,  -1.0, -1.0,   -1.0,   1.0],  # carbon-13
    "CI14": [14.003241, -1.0, -1.0,   -1.0,   1.0],  # carbon-14
}


def set_specie(specie: str) -> list[float]:
    """
    Return atomic constants for a given species identifier.

    Performs a substring match for backward compatibility with call sites
    that pass strings like ``'CI'`` or ``'HI'``.

    Parameters
    ----------
    specie : str
        Species identifier, e.g. ``'HI'``, ``'CI'``, ``'HeI'``,
        ``'SI'``, ``'CI13'``, ``'CI14'``.

    Returns
    -------
    list of float
        ``[atomic_mass, ionisation_potential, abundance, V_X_minus_V_H, Z]``

    Raises
    ------
    ValueError
        If ``specie`` does not match any entry in :data:`SPECIES_DATA`.

    Examples
    --------
    >>> set_specie('CI')
    [12.0, 11.4, 0.0003, 149.5, 1.0]

    >>> set_specie('HI')
    [1.0078, 13.6, 1.0, 0.0, 1.0]
    """
    for key, data in SPECIES_DATA.items():
        if key in specie:
            return data
    raise ValueError(
        f"Unknown species '{specie}'. "
        f"Available species: {list(SPECIES_DATA.keys())}"
    )


def line_freq(n, dn: int = 1, specie: str = "CI") -> u.Quantity:
    """
    Compute the central frequency of a radio recombination line (RRL).

    Uses the reduced-mass corrected Rydberg constant following
    Gordon & Sorochenko (1992), Appendix A, eq. A.5.

    The transition is n → n + dn (e.g. dn=1 for α lines, dn=2 for β, etc.).

    Parameters
    ----------
    n : int or array-like of int
        Principal quantum number of the lower level.
    dn : int, optional
        Quantum number jump. Default is 1 (α line).
    specie : str, optional
        Atomic species identifier. Default is ``'CI'``.

    Returns
    -------
    astropy.units.Quantity
        Line frequency in MHz.

    Examples
    --------
    >>> line_freq(500)                          # Cα n=500
    <Quantity ... MHz>

    >>> line_freq(np.arange(500, 510), dn=2)   # Cβ lines n=500–509
    <Quantity [...] MHz>

    Notes
    -----
    The reduced-mass corrected Rydberg constant is:

    .. math::

        R_X = \\frac{R_\\infty}{1 + m_e / M_X}

    where :math:`M_X` is the atomic mass in atomic mass units and
    :math:`m_e` is the electron mass.
    """
    X = set_specie(specie)
    M_X = X[0]   # atomic mass [u.m.a.]
    Z   = X[4]   # electric charge

    # Reduced-mass correction to the Rydberg constant
    m_e_amu = m_e / amu
    R_X = Ryd / (1.0 + m_e_amu / M_X)

    return (R_X * Z**2 * c * (1.0 / n**2 - 1.0 / (n + dn)**2)).to(u.MHz)
