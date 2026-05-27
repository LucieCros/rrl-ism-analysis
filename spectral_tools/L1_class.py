# -*- coding: utf-8 -*-
"""
L1_class.py
===========
Parser and container for NenuFAR/LOFAR Level-1 FITS observation blocs.

The :class:`L1` class reads a pipeline-produced FITS file, exposes all
header keywords and data arrays as attributes, and provides convenience
methods for Doppler correction, sub-band extraction, and time-averaged
integration.

.. note::
    This module imports :mod:`spectral_tools.tools` (for Doppler utilities)
    but is **not** imported by it — the dependency is one-directional.
    The recommended way to instantiate an :class:`L1` object from user code
    is through :func:`spectral_tools.io.load_bloc`.

Dependencies
------------
astropy >= 5.0, numpy, pandas
Internal: spectral_tools.tools
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy import units as u

import spectral_tools.tools as tools


# ---------------------------------------------------------------------------
# Default display colours per RRL series
# ---------------------------------------------------------------------------

#: Matplotlib colour assigned to each RRL series for quick plotting.
SERIES_COLORS: dict[str, str] = {
    "Halph":  "darkblue",
    "Hbeta":  "mediumblue",
    "Hgamm":  "b",
    "Hdelt":  "blue",
    "Hepsi":  "royalblue",
    "Healph": "orange",
    "Hebeta": "gold",
    "Calph":  "darkviolet",
    "Cbeta":  "violet",
    "Salph":  "yellowgreen",
}


class L1:
    """
    Container for a NenuFAR/LOFAR Level-1 spectral observation bloc.

    Instantiation opens and parses the FITS file. All header columns from
    the input and output binary tables (HDU 1 and 2) are exposed as instance
    attributes. Data arrays (``DATA``, ``NDATA``) are set from subsequent
    HDUs.

    Parameters
    ----------
    file : str
        Absolute path to the ``.fits`` observation bloc.
    rrlfile : str
        Path to the RRL catalogue CSV used by :meth:`import_rrls`.

    Attributes
    ----------
    filename : str
        Path passed at construction time.
    rrlfile : str
        Path to the RRL catalogue.
    hdu : astropy.io.fits.HDUList
        Raw HDU list (kept open for inspection; close with ``self.hdu.close()``).
    command_line : str
        Original pipeline command line stored in the primary HDU.
    freq : numpy.ndarray
        Frequency axis [MHz], built from ``FMIN``, ``FMAX``, ``DF``.
    df : float
        Channel width [MHz].
    names_cols : dict
        Colour map for each RRL series (see :data:`SERIES_COLORS`).
    date : astropy.time.Time
        Observation date (set by :meth:`get_date`).
    year : str
        ISO date string ``'YYYY-MM-DD'`` (set by :meth:`get_date`).
    time : str
        Time string ``'HH:MM:SS'`` (set by :meth:`get_date`).

    Notes
    -----
    Dynamic attributes (``FMIN``, ``FMAX``, ``NBEAMLETS``, ``NCHANNELS``,
    etc.) are set via :func:`setattr` from the FITS binary table columns —
    their exact names depend on the pipeline version that produced the file.
    """

    def __init__(self, file: str, rrlfile: str):
        hdu            = fits.open(file)
        self.hdu       = hdu
        self.filename  = file
        self.rrlfile   = rrlfile
        self.names_cols = SERIES_COLORS.copy()

        # ── HDU 1 : input parameters ────────────────────────────────────────
        for col in hdu[1].data.columns:
            setattr(self, col.name, hdu[1].data[col.name])

        # ── HDU 2 : output parameters ───────────────────────────────────────
        for col in hdu[2].data.columns:
            setattr(self, col.name, hdu[2].data[col.name])

        # ── HDU 3+ : data arrays ─────────────────────────────────────────────
        # Recognised ARRAY keys: 'DATA', 'NDATA', 'DATA0'
        # 'DATA0' is an alias for 'DATA' produced by older pipeline versions.
        for i in range(3, len(hdu)):
            key = hdu[i].header["ARRAY"]
            if "DATA" not in key:
                setattr(self, key, hdu[i].data)
                continue
            if key == "DATA":
                setattr(self, "DATA", hdu[i].data[0])
            elif key == "NDATA":
                setattr(self, "NDATA", hdu[i].data)
            elif key == "DATA0":
                # Legacy key from older pipeline versions
                setattr(self, "DATA", hdu[i].data)
            # Any other DATA-like key is silently skipped

        # ── Primary HDU : command line ───────────────────────────────────────
        self.command_line = bytearray(hdu[0].data).decode()

        # ── Frequency axis ───────────────────────────────────────────────────
        # DF is stored with a unit string in DFUNIT; convert to MHz.
        df_quantity    = eval(f"{self.DF[0]}*u.{self.DFUNIT[0]}")
        self.df        = df_quantity.to(u.MHz).value
        self.freq      = np.arange(self.FMIN[0], self.FMAX[0], self.df)

    # -----------------------------------------------------------------------
    # Date extraction
    # -----------------------------------------------------------------------

    def get_date(self) -> tuple:
        """
        Extract the observation date from the pipeline command line string.

        The date and time are encoded in the original filename embedded in
        the command line as ``YYYYMMDD_HHMMSS``.

        Returns
        -------
        date : astropy.time.Time
            Full observation timestamp.
        year : str
            ISO date ``'YYYY-MM-DD'``.
        time : str
            Time ``'HH:MM:SS'``.

        Notes
        -----
        Also sets ``self.date``, ``self.year``, and ``self.time``.
        """
        original_file = self.command_line.split(",")[1].replace("'", "")
        fname         = original_file.split("/")[-1].split("_")
        date_str      = fname[-3]   # 'YYYYMMDD'
        hour_str      = fname[-2]   # 'HHMMSS'

        iso = (f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
               f"T{hour_str[:2]}:{hour_str[2:4]}:{hour_str[4:]}")

        self.date = Time(iso)
        self.year = self.date.value.split("T")[0]
        self.time = self.date.value.split("T")[1].split(".")[0]
        return self.date, self.year, self.time

    # -----------------------------------------------------------------------
    # Integration
    # -----------------------------------------------------------------------

    def integration(self) -> np.ndarray:
        """
        Compute the time-averaged (integrated) spectrum, weighted by ``NDATA``.

        The weighted mean is:

        .. math::

            I_\\nu = \\frac{\\sum_t N_{t,\\nu}\\,D_{t,\\nu}}{\\sum_t N_{t,\\nu}}

        Zero-valued channels (fully flagged) are masked and set to ``NaN``.

        Returns
        -------
        numpy.ndarray
            Integrated spectrum, stored as ``self.I``. Same length as ``self.freq``.
        """
        NDATA     = self.NDATA
        DATA      = self.DATA
        sum_cols  = DATA.shape[1] if self.NS[0] > 1 else np.sum(NDATA, axis=1)
        I         = np.sum(DATA * NDATA, axis=1) / sum_cols

        masked    = np.ma.masked_equal(I, 0)
        self.I    = np.ma.masked_array(I, mask=masked.mask,
                                        fill_value=np.nan).filled()
        return self.I

    # -----------------------------------------------------------------------
    # Sub-band utilities
    # -----------------------------------------------------------------------

    def get_subband(self, array: np.ndarray, k: int) -> np.ndarray:
        """
        Extract sub-band ``k`` from a full-band array.

        Parameters
        ----------
        array : numpy.ndarray
            Full-band 1-D array (length = ``NBEAMLETS × NCHANNELS``).
        k : int
            Sub-band index (0-based).

        Returns
        -------
        numpy.ndarray
            Slice of length ``NCHANNELS`` corresponding to sub-band ``k``.
        """
        n = int(len(self.freq) / self.NBEAMLETS)
        return array[k * n:(k + 1) * n]

    def get_line_frequencies(self, k: int, velo: float,
                              myLine: str = "Calph") -> np.ndarray:
        """
        Return Doppler-shifted RRL frequencies within sub-band ``k``.

        Parameters
        ----------
        k : int
            Sub-band index (0-based).
        velo : float
            Source radial velocity [km/s].
        myLine : str, optional
            RRL series name. Default ``'Calph'``.

        Returns
        -------
        numpy.ndarray
            Array of line frequencies [MHz] within the sub-band.
        """
        rrls  = self.import_rrls(v=velo, correction=False)
        n     = int(len(self.freq) / self.NBEAMLETS[0])
        fmin  = self.freq[n * k]
        fmax  = self.freq[(k + 1) * n - 1]

        band  = rrls[myLine]
        lines = np.array(band.where(band > fmin).where(band < fmax).dropna())
        return lines

    # -----------------------------------------------------------------------
    # RRL catalogue
    # -----------------------------------------------------------------------

    def import_rrls(self, v: float = -47.0,
                    correction: bool = True) -> dict:
        """
        Load the RRL catalogue and apply a Doppler shift.

        Parameters
        ----------
        v : float, optional
            Source radial velocity [km/s]. Default -47 km/s.
        correction : bool, optional
            If ``True``, subtract the LSR correction computed from the
            observation date before applying ``v``. Default ``True``.

        Returns
        -------
        dict
            ``{series_name: pandas.Series}`` mapping each RRL series to its
            Doppler-shifted frequency array. Also stored as
            ``self.RRLS_doppler`` and ``self.RRLS``.

        Notes
        -----
        When ``correction=False``, ``v`` is treated as already in m/s
        (multiplied by 1000 internally) — this is a legacy behaviour kept
        for compatibility with :meth:`get_line_frequencies`.
        """
        RRL = pd.read_csv(self.rrlfile, sep=",")
        RRL.set_index("n", inplace=True)

        self.get_date()
        if correction:
            lsr_corr = tools.doppler_corrections(self.date)[-1].to(u.km / u.s).value
            V = v - lsr_corr   # velocity in km/s, will be converted below
            V_ms = V * 1000.0  # m/s for doppler_correction()
        else:
            V_ms = v * 1000.0  # legacy: v already in km/s, ×1000 → m/s

        A = {
            col: tools.doppler_correction(RRL[col].dropna(), V_ms)
            for col in RRL.columns
        }

        self.RRLS_doppler = A
        self.RRLS         = RRL
        return A
