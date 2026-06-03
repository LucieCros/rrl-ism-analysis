# -*- coding: utf-8 -*-
"""
io.py
=====
FITS observation I/O for the NenuFAR/LOFAR spectral pipeline.

This module is the **single entry point** for all file-based I/O in the
pipeline. It covers two distinct concerns:

**1. FITS observation blocs** (:func:`load_bloc`)
    The only module in ``spectral_tools`` that imports :mod:`L1_class`,
    keeping the dependency graph acyclic:

    .. code-block:: none

        atoms.py         (no internal imports)
            ↑
        tools.py         (imports atoms)
            ↑
        L1_class.py      (imports tools)
            ↑
        io.py            (imports L1_class)   ← you are here
            ↑
        modeling.py, gridexplo.py, pipelines…

**2. Source catalogue** (:func:`read_source_info`, :func:`read_source_velocity`,
    :func:`normalise_source_name`)
    Centralised readers for ``source_info.txt``, shared by all pipeline
    scripts to avoid duplicating the same parsing logic.

Usage
-----
::

    from spectral_tools.io import load_bloc, read_source_info

    hdu  = load_bloc("/path/to/observation.fits")
    info = read_source_info("CAS_A", "CASA")

Dependencies
------------
Internal: L1_class
External: os, astropy.io, numpy
"""

import os
from spectral_tools import L1_class as L1
from astropy.io import fits
import numpy as np

# Default path to the RRL catalogue — override before calling if needed.
DEFAULT_RRLS_PATH: str = os.path.join(os.path.dirname(__file__), "..", "files", "rrlines.csv")

def load_bloc(path_fits: str,
              path_rrls: str = DEFAULT_RRLS_PATH) -> L1.L1:
    """
    Load a FITS observation bloc and return a fully initialised :class:`L1_class.L1` object.

    This is the recommended way to open raw observations from the pipeline.
    The observation date is extracted automatically via
    :meth:`~L1_class.L1.get_date`.

    Parameters
    ----------
    path_fits : str
        Absolute path to the ``.fits`` observation bloc.
    path_rrls : str, optional
        Path to the RRL catalogue CSV file used for line identification.
        Default: ``'files/rrlines.csv'``.

    Returns
    -------
    L1_class.L1
        Parsed observation object. Key attributes include:

        - ``freq``       : frequency axis [MHz]
        - ``DATA``       : calibrated data array
        - ``NDATA``      : normalisation (weight) array
        - ``date``       : observation date (:class:`astropy.time.Time`)
        - ``NBEAMLETS``  : number of sub-bands
        - ``NCHANNELS``  : channels per sub-band

    Examples
    --------
    >>> from spectral_tools.io import load_bloc
    >>> hdu = load_bloc("/data/CasA/obs_20210609.fits")
    >>> hdu.freq.shape
    (32768,)

    Notes
    -----
    The original ``calcul.importing()`` function has been moved here to break
    the circular dependency between ``tools.py`` and ``L1_class.py``.
    """
    hdu = L1.L1(path_fits, path_rrls)
    hdu.get_date()
    return hdu


# ---------------------------------------------------------------------------
# Source catalogue I/O
# ---------------------------------------------------------------------------

#: Default path to the source info table.
DEFAULT_SOURCE_INFO: str = "files/source_info.txt"


def read_source_info(source: str, sourcename: str,
                     is_off: bool = False,
                     path: str = DEFAULT_SOURCE_INFO) -> dict | None:
    """
    Read astrometric and physical parameters for a source from the info table.

    The table is tab-separated with columns::

        name  ra  dec  velocity  spectral_coeff

    The lookup performs a substring match to handle naming variants
    (e.g. ``'CASA'`` matches ``'CAS_A'``).

    Parameters
    ----------
    source : str
        Parsed source name (may differ from ``sourcename`` for special cases
        such as cloud fields with a ``_J2000`` suffix).
    sourcename : str
        Raw source identifier from the command line.
    is_off : bool, optional
        Whether we are processing OFF-beam data. Controls which on/off entry
        is selected when both exist. Default ``False``.
    path : str, optional
        Path to the source info table. Default ``'files/source_info.txt'``.

    Returns
    -------
    dict or None
        ``{'ra', 'dec', 'velo', 'coeff'}`` if found, ``None`` if the source
        is not in the table.

    Examples
    --------
    >>> info = read_source_info("CAS_A", "CASA")
    >>> info['ra'], info['velo']
    ('23h23m24s', '-47')
    """
    with open(path) as doc:
        for line in doc:
            if not line.strip():
                continue
            sour = line.split("\t")[0]
            name_match = (
                (sour in source) or (sour in sourcename) or (source in sour)
            )
            if not name_match:
                continue
            is_neutral = ("on" not in sour) and ("off" not in sour)
            if is_neutral:
                parts = line.split()
                return {"ra": parts[1], "dec": parts[2],
                        "velo": parts[3], "coeff": parts[4]}
            if is_off and "off" in sour:
                parts = line.split()
                return {"ra": parts[1], "dec": parts[2],
                        "velo": parts[3], "coeff": parts[4]}
            if not is_off and "on" in sour:
                parts = line.split()
                return {"ra": parts[1], "dec": parts[2],
                        "velo": parts[3], "coeff": parts[4]}
    return None


def read_source_velocity(source: str, is_off: bool = False,
                         path: str = DEFAULT_SOURCE_INFO) -> float:
    """
    Read only the radial velocity of a source from the info table.

    Convenience wrapper around :func:`read_source_info` for scripts that
    only need the velocity (e.g. :mod:`pipelines.run_time_average`).

    Parameters
    ----------
    source : str
        Source name to look up.
    is_off : bool, optional
        Whether we are processing OFF-beam data. Default ``False``.
    path : str, optional
        Path to the source info table. Default ``'files/source_info.txt'``.

    Returns
    -------
    float
        Source velocity [km/s].

    Raises
    ------
    ValueError
        If the source is not found in the table.
    """
    info = read_source_info(source, source, is_off=is_off, path=path)
    if info is None:
        raise ValueError(f"Source '{source}' not found in {path}")
    return eval(info["velo"])


def normalise_source_name(sourcename: str) -> str:
    """
    Normalise a NenuFAR source identifier to a canonical lookup name.

    Handles the special naming conventions used in the NenuFAR pipeline:

    - Trailing ``A``  → ``{base}_A``  (e.g. ``'CYGA'`` → ``'CYG_A'``)
    - Trailing ``P``  → ``'CYG_LOOP'``
    - Trailing ``X``  → ``'CYG_X'``
    - All other cases → returned unchanged

    Parameters
    ----------
    sourcename : str
        Raw source identifier from the command line.

    Returns
    -------
    str
        Normalised source name for use with :func:`read_source_info`.

    Examples
    --------
    >>> normalise_source_name("CYGA")
    'CYG_A'
    >>> normalise_source_name("CASA")
    'CASA'
    """
    if sourcename.endswith("A"):
        return sourcename[:-1] + "_A"
    if sourcename.endswith("P"):
        return "CYG_LOOP"
    if sourcename.endswith("X"):
        return "CYG_X"
    return sourcename


# ---------------------------------------------------------------------------
# FITS header builder
# ---------------------------------------------------------------------------

def build_fits_header(template_path: str) -> "fits.Header":
    """
    Build an :class:`astropy.io.fits.Header` from a plain-text template.

    The template uses the standard FITS card syntax::

        KEY = VALUE / comment

    Integer casting is applied automatically to ``NAXIS*`` and ``*BIT*`` keys.
    Parsing stops at the first line containing ``END``.

    Parameters
    ----------
    template_path : str
        Path to the ``fitsheader.txt`` template file (e.g.
        ``'files/fitsheader.txt'``).

    Returns
    -------
    astropy.io.fits.Header
        Populated FITS header object ready to be attached to an HDU.

    Examples
    --------
    >>> from spectral_tools.io import build_fits_header
    >>> hdr = build_fits_header("files/fitsheader.txt")
    >>> hdr["TELESCOP"]
    'NenuFAR'
    """
    from astropy.io import fits as _fits

    hdr = _fits.Header()
    with open(template_path, "r") as fh:
        for line in fh:
            line = line.replace("\\\n", "")
            if "END" in line:
                break
            if "=" not in line:
                continue
            key, rest = line.split("=", 1)
            try:
                value, comment = rest.split("/", 1)
            except ValueError:
                value, comment = rest, ""
            value = value.replace("'", "").replace(" ", "")
            try:
                value = float(value)
            except ValueError:
                pass
            if "BIT" in key or "NAXIS" in key:
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    pass
            hdr.set(key.strip(), value, comment.strip())
    return hdr
    

def freq_axis_from_fits(
    hdu: fits.HDUList,
    axis: int = 3,
) -> np.ndarray:
    """
    Reconstruct a linear frequency axis from FITS WCS keywords.
 
    Reads ``CRVAL``, ``CDELT``, ``CRPIX``, and ``NAXIS`` from the primary
    HDU header and returns the corresponding frequency array.
 
    This is equivalent to the manual four-line reconstruction used in the
    stacking notebooks::
 
        CRVAL = hdr['CRVAL3']
        CDELT = hdr['CDELT3']
        CRPIX = hdr['CRPIX3']
        NAXIS = hdr['NAXIS3']
        f = np.array([CRVAL + (i - CRPIX) * CDELT for i in range(NAXIS)])
 
    Parameters
    ----------
    hdu : astropy.io.fits.HDUList
        Opened FITS file (e.g. ``fits.open(path)``).
    axis : int, optional
        FITS axis number (1-based). Default is ``3`` (the spectral axis
        in NenuFAR alltime cubes).
 
    Returns
    -------
    np.ndarray
        1-D frequency array in the same units as ``CRVAL``
        (typically GHz for NenuFAR data).
 
    Raises
    ------
    KeyError
        If any of the required WCS keywords are missing from the header.
 
    Examples
    --------
    >>> from astropy.io import fits
    >>> from spectral_tools.io import freq_axis_from_fits
    >>> hdu = fits.open("alltime_TAUA_CLOUDS_Calph_OFF.fits")
    >>> f = freq_axis_from_fits(hdu)
    >>> print(f"Frequency range: {f[0]:.3f} – {f[-1]:.3f} GHz")
    """
    hdr   = hdu[0].header
    crval = hdr[f"CRVAL{axis}"]
    cdelt = hdr[f"CDELT{axis}"]   # Hz (or GHz) per channel
    crpix = hdr[f"CRPIX{axis}"]   # Reference pixel (1-based in FITS)
    naxis = hdr[f"NAXIS{axis}"]   # Total number of channels
 
    # Standard WCS linear transform: f[i] = CRVAL + (i - CRPIX) * CDELT
    # numpy uses 0-based indices, so i runs from 0 to NAXIS-1.
    return crval + (np.arange(naxis) - crpix) * cdelt
 

