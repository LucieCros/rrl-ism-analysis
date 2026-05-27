# -*- coding: utf-8 -*-
"""
maps.py
=======
Spatial and spectral map utilities for ISM multi-tracer analysis.

This module provides two main components:

**1. Dust extinction** (:class:`DustMap`)
    Query the Edenhofer et al. (2023) 3-D dust map over a field of view or
    along a single line of sight, and estimate ISM cloud path lengths from
    extinction profiles.

**2. PPV cube handling** (:class:`MapLoader`)
    Load CO and HI position–position–velocity (PPV) cubes, crop them to a
    field of view (square or circular aperture), compute moment maps, and
    extract mean spectra.

Architecture
------------
Both classes manage their data objects internally, avoiding module-level
global variables. The dust interpolator (``Edenhofer2023Query``) is expensive
to load, so it is stored as a **class-level attribute** of :class:`DustMap`
and shared across all instances.

FITS HDUs opened by :class:`MapLoader` are closed when the instance is
deleted, preventing memory leaks during long notebook sessions.

Dependencies
------------
numpy, scipy, tqdm, astropy
Optional: dustmaps (only required for :class:`DustMap`)

References
----------
- Edenhofer et al. (2023) — 3-D dust map.
- Dame et al. (2001) — CO survey used as default CO tracer.
"""

import os

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from tqdm import tqdm


# ===========================================================================
# Dust extinction
# ===========================================================================

class DustMap:
    """
    Interface to the Edenhofer et al. (2023) 3-D dust map.

    The dust interpolator is loaded once at the class level and shared across
    all instances. Call :meth:`fetch` before instantiating if the map has
    never been downloaded.

    Parameters
    ----------
    flavor : str, optional
        Dataset flavor passed to ``Edenhofer2023Query``.
        Default ``'less_data_but_2kpc'`` (covers 0–2 kpc, smaller download).

    Examples
    --------
    >>> dm = DustMap()
    >>> coord = SkyCoord("23h23m24s", "58d48m54s", frame="icrs")
    >>> extinc = dm.extinction_fov(coord, fov=(2*u.deg, 2*u.deg), delta=0.1*u.deg)
    """

    #: Shared Edenhofer2023Query interpolator — loaded once for all instances.
    _query = None

    @classmethod
    def fetch(cls, fetch_2kpc: bool = True) -> None:
        """
        Download the Edenhofer (2023) dust map files (run once).

        Parameters
        ----------
        fetch_2kpc : bool, optional
            If ``True``, download the extended 2 kpc version. Default ``True``.
        """
        import dustmaps
        import dustmaps.edenhofer2023
        dustmaps.edenhofer2023.fetch(fetch_2kpc=fetch_2kpc)

    @classmethod
    def _ensure_loaded(cls, flavor: str) -> None:
        """Load the interpolator if not already done (lazy initialisation)."""
        if cls._query is None:
            from dustmaps.edenhofer2023 import Edenhofer2023Query
            cls._query = Edenhofer2023Query(flavor=flavor)

    def __init__(self, flavor: str = "less_data_but_2kpc"):
        self._flavor = flavor
        self._ensure_loaded(flavor)

    # -----------------------------------------------------------------------
    # Extinction queries
    # -----------------------------------------------------------------------

    def extinction_fov(self, coord: SkyCoord,
                       fov: tuple,
                       delta: float,
                       r_min: float = 0.03,
                       r_max: float = 2.0,
                       dr: float = 0.01) -> np.ndarray:
        """
        Compute the cumulative dust extinction integrated over distance,
        on a regular spatial grid covering the field of view.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Centre of the field of view.
        fov : tuple of astropy.units.Quantity
            ``(delta_l, delta_b)`` — angular size of the field [deg].
        delta : float
            Spatial grid spacing [deg].
        r_min : float, optional
            Minimum distance [kpc]. Default 0.03.
        r_max : float, optional
            Maximum distance [kpc]. Default 2.0.
        dr : float, optional
            Distance step [kpc]. Default 0.01.

        Returns
        -------
        numpy.ndarray, shape (n_l, n_b)
            Cumulative extinction map (sum over all distance shells).
        """
        Dl, Db   = fov
        l_center = coord.galactic.l
        b_center = coord.galactic.b
        nl       = int(Dl // delta)
        nb       = int(Db // delta)

        longitudes = np.linspace(l_center - Dl / 2, l_center + Dl / 2, nl)
        latitudes  = np.linspace(b_center - Db / 2, b_center + Db / 2, nb)
        l_grid, b_grid = np.meshgrid(longitudes, latitudes)

        radii  = np.arange(r_min, r_max, dr) * u.kpc
        extinc = np.zeros((nl, nb))

        for r in tqdm(radii, desc="Dust extinction (FoV)"):
            los        = SkyCoord(l=l_grid, b=b_grid, frame="galactic", distance=r)
            shell      = self._query.query(los)
            shell      = np.where(np.isnan(shell), 0.0, shell)
            extinc    += shell

        return extinc

    def extinction_los(self, coord: SkyCoord,
                       fov: tuple,
                       delta: float,
                       r_min: float = 0.03,
                       r_max: float = 2.0,
                       dr: float = 0.01) -> np.ndarray:
        """
        Compute the spatially-averaged extinction as a function of distance
        (line-of-sight profile).

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Centre of the field of view.
        fov : tuple of astropy.units.Quantity
            ``(delta_l, delta_b)`` — angular size of the field [deg].
        delta : float
            Spatial grid spacing [deg].
        r_min, r_max, dr : float, optional
            Distance range and step [kpc]. Defaults: 0.03, 2.0, 0.01.

        Returns
        -------
        numpy.ndarray, shape (n_radii,)
            Spatially-summed extinction per distance shell.
        """
        Dl, Db   = fov
        l_center = coord.galactic.l
        b_center = coord.galactic.b
        nl       = int(Dl // delta)
        nb       = int(Db // delta)

        longitudes = np.linspace(l_center - Dl / 2, l_center + Dl / 2, nl)
        latitudes  = np.linspace(b_center - Db / 2, b_center + Db / 2, nb)
        l_grid, b_grid = np.meshgrid(longitudes, latitudes)

        radii  = np.arange(r_min, r_max, dr) * u.kpc
        extinc = np.zeros(len(radii))

        for i, r in enumerate(tqdm(radii, desc="Dust extinction (LOS)")):
            los        = SkyCoord(l=l_grid, b=b_grid, frame="galactic", distance=r)
            shell      = self._query.query(los)
            shell      = np.where(np.isnan(shell), 0.0, shell)
            extinc[i]  = np.sum(shell)

        return extinc

    # -----------------------------------------------------------------------
    # Cloud path length estimation
    # -----------------------------------------------------------------------

    @staticmethod
    def _gaussian(x, amplitude, center, sigma):
        """1-D Gaussian model for cloud peak fitting."""
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)

    def cloud_path_lengths(self, extinctions: np.ndarray,
                           r_min: float = 0.0,
                           r_max: float = 2000.0,
                           search_radius: int = 30,
                           flags: list = [],
                           large_peaks: list = [],
                           small_peaks: list = [],
                           fit_gaussian: bool = False,
                           gaussian_peaks: list = []) -> tuple:
        """
        Estimate the path length (depth) of ISM clouds from an extinction
        profile by measuring the FWHM of each extinction peak.

        Two methods are available per peak:

        - **FWHM from data** (default): finds the half-maximum crossing points
          directly on the profile. Robust for well-resolved peaks.
        - **Gaussian fit** (``fit_gaussian=True``): fits a Gaussian to estimate
          the FWHM. Better for blended or noisy peaks.

        Parameters
        ----------
        extinctions : numpy.ndarray
            1-D extinction profile (e.g. from :meth:`extinction_los`).
        r_min : float, optional
            Start of the distance axis [pc]. Default 0.
        r_max : float, optional
            End of the distance axis [pc]. Default 2000.
        search_radius : int, optional
            Half-window in samples used to find half-maximum crossings.
            Default 30. Use ``large_peaks`` / ``small_peaks`` to override
            per-peak.
        flags : list of int, optional
            Peak indices to exclude from the analysis. Default [].
        large_peaks : list of int, optional
            Peak indices for which ``search_radius`` is doubled. Default [].
        small_peaks : list of int, optional
            Peak indices for which ``search_radius`` is halved. Default [].
        fit_gaussian : bool, optional
            If ``True``, fit a Gaussian to all peaks not in ``gaussian_peaks``.
            Default ``False``.
        gaussian_peaks : list of int, optional
            Peak indices to force Gaussian fitting regardless of
            ``fit_gaussian``. Default [].

        Returns
        -------
        peak_indices : numpy.ndarray of int
            Indices of detected peaks in ``extinctions``.
        peak_distances : numpy.ndarray
            Distances of the peaks [pc].
        path_lengths : numpy.ndarray
            Estimated path length (FWHM) of each cloud [pc].

        Notes
        -----
        Plotting of the results (FWHM markers, annotations) is intentionally
        **not** included here — pass the returned arrays to your notebook
        or :mod:`spectral_tools.graphics` for display.
        """
        Y  = extinctions
        X  = np.linspace(r_min, r_max, len(Y))
        r0 = search_radius

        # Detect peaks with a minimum prominence of 1 % of the global max
        peak_indices = find_peaks(Y, prominence=0.01 * np.nanmax(Y))[0]

        # Remove flagged indices
        for f in flags:
            peak_indices = peak_indices[peak_indices != f]

        path_lengths = np.zeros(len(peak_indices))

        for k, i in enumerate(peak_indices):
            half_max = Y[i] / 2.0

            # Determine local search radius for this peak
            if i in small_peaks:
                rad = r0 // 2
            elif i in large_peaks:
                rad = 2 * r0
            else:
                rad = r0

            use_gaussian = fit_gaussian or (i in gaussian_peaks)

            if not use_gaussian:
                # --- Direct FWHM from data ---
                lo  = max(i - rad, 0)
                hi  = min(i + rad, len(X))
                above_half = np.where(Y[lo:hi] > half_max)[0]
                if len(above_half) < 2:
                    path_lengths[k] = 0.0
                    continue
                start, end        = above_half[0], above_half[-1]
                path_lengths[k]   = (end - start) * 2  # symmetric FWHM estimate

            else:
                # --- Gaussian fit ---
                rad_fit = 2 * r0
                bounds  = np.array([
                    [0.9 * Y[i],   1.1 * Y[i]],
                    [X[i] - 10.0,  X[i] + 10.0],
                    [0.0,          2.0 * rad_fit],
                ])
                try:
                    popt, _ = curve_fit(
                        self._gaussian, X, Y,
                        bounds=bounds.T,
                        nan_policy="omit",
                    )
                    fwhm            = 2.0 * np.sqrt(2.0 * np.log(2.0)) * popt[-1]
                    path_lengths[k] = fwhm
                except RuntimeError:
                    # Fit did not converge — fall back to direct FWHM
                    lo  = max(i - rad, 0)
                    hi  = min(i + rad, len(X))
                    above_half      = np.where(Y[lo:hi] > half_max)[0]
                    path_lengths[k] = (above_half[-1] - above_half[0]) * 2 \
                                      if len(above_half) >= 2 else 0.0

        return peak_indices, X[peak_indices], path_lengths[::-1]


# ===========================================================================
# PPV cube loader
# ===========================================================================

class MapLoader:
    """
    Load and crop CO and HI position–position–velocity (PPV) cubes.

    One instance corresponds to one astronomical source. FITS files are
    opened lazily (on first access) and closed when the instance is deleted.

    Parameters
    ----------
    source : str
        Source identifier used to select the correct FITS file
        (e.g. ``'Tau'``, ``'Cas'``).
    path_co : str, optional
        Directory containing CO FITS cubes. Default ``'data/COmaps/'``.
    path_hi : str, optional
        Directory containing HI FITS cubes. Default ``'data/HImaps/'``.

    Examples
    --------
    >>> loader = MapLoader("Tau")
    >>> coord  = SkyCoord("05h34m32s", "22d00m52s", frame="icrs")
    >>> fov    = (2 * u.deg, 2 * u.deg)
    >>> cube, vel, spec, lon, lat, mom0 = loader.crop_circle(coord, "CO", fov)
    >>> del loader   # closes FITS files and frees memory
    """

    #: Velocity dimension index for each tracer in the cube axis ordering.
    _VDIM: dict[str, int] = {"CO": 1, "HI": 3}

    #: Velocity unit scale factor (HI axes are in m/s, converted to km/s).
    _VSCALE: dict[str, float] = {"CO": 1.0, "HI": 1e-3}

    def __init__(self, source: str,
                 path_co: str = "data/COmaps/",
                 path_hi: str = "data/HImaps/"):
        self.source   = source
        self._path_co = path_co
        self._path_hi = path_hi
        self._hdus: dict[str, fits.PrimaryHDU] = {}

    def __del__(self):
        """Close all open FITS HDUs on garbage collection."""
        for tracer, hdu in self._hdus.items():
            try:
                hdu._file.close()
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_hdu(self, tracer: str) -> fits.PrimaryHDU:
        """
        Return the HDU for ``tracer``, loading it on first access.

        Parameters
        ----------
        tracer : str
            ``'CO'`` or ``'HI'``.

        Returns
        -------
        astropy.io.fits.PrimaryHDU
        """
        if tracer not in self._hdus:
            path = self._path_co if tracer == "CO" else self._path_hi
            files = np.array(os.listdir(path))
            match = [f for f in files if self.source in f]
            if not match:
                raise FileNotFoundError(
                    f"No {tracer} cube found for source '{self.source}' in '{path}'"
                )
            self._hdus[tracer] = fits.open(os.path.join(path, match[0]))[0]
        return self._hdus[tracer]

    @staticmethod
    def _get_axis(hdu: fits.PrimaryHDU, i: int) -> np.ndarray:
        """
        Build the world-coordinate axis ``i`` from FITS WCS keywords.

        Parameters
        ----------
        hdu : fits.PrimaryHDU
            FITS HDU containing the WCS header.
        i : int
            1-based axis index (FITS convention).

        Returns
        -------
        numpy.ndarray
            World-coordinate values along axis ``i``.
        """
        hdr   = hdu.header
        v_ref = float(hdr[f"CRVAL{i}"])
        k_ref = float(hdr[f"CRPIX{i}"])
        delta = float(hdr[f"CDELT{i}"])
        n_pix = float(hdr[f"NAXIS{i}"])
        return v_ref + (np.arange(1, n_pix + 1) - k_ref) * delta

    def _get_velocity(self, hdu: fits.PrimaryHDU,
                      tracer: str) -> np.ndarray:
        """
        Return the velocity axis [km/s] for ``tracer``.

        Parameters
        ----------
        hdu : fits.PrimaryHDU
        tracer : str

        Returns
        -------
        numpy.ndarray
            Velocity axis [km/s].
        """
        vdim  = self._VDIM[tracer]
        scale = self._VSCALE[tracer]
        axis_index = (vdim + 2) % 3 + 1
        return self._get_axis(hdu, axis_index) * scale

    # -----------------------------------------------------------------------
    # Moment maps
    # -----------------------------------------------------------------------

    @staticmethod
    def moment0(cube: np.ndarray, cutoff: float = 0.0,
                vdim: int = 1) -> np.ndarray:
        """
        Compute the zeroth moment (integrated intensity) map.

        Parameters
        ----------
        cube : numpy.ndarray
            PPV data cube.
        cutoff : float, optional
            Pixels below this value are masked. Default 0.
        vdim : int, optional
            Velocity axis index (0-based). Default 1.

        Returns
        -------
        numpy.ndarray
            Moment-0 map (NaN where no valid data).
        """
        c = np.copy(cube).astype(float)
        c[c < cutoff] = np.nan
        m = np.nanmean(c, axis=vdim - 1)
        return np.where(np.isfinite(m), m, np.nan)

    # -----------------------------------------------------------------------
    # Cube cropping
    # -----------------------------------------------------------------------

    def crop_square(self, coord: SkyCoord, tracer: str,
                    fov: tuple) -> tuple:
        """
        Crop a PPV cube to a square field of view centred on ``coord``.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Centre of the field of view.
        tracer : str
            ``'CO'`` or ``'HI'``.
        fov : tuple of astropy.units.Quantity
            ``(delta_l, delta_b)`` angular size of the field.

        Returns
        -------
        cube_crop : numpy.ndarray
            Cropped PPV sub-cube.
        vel : numpy.ndarray
            Velocity axis [km/s].
        spectrum : numpy.ndarray
            Spatially-averaged spectrum over the cropped region.
        lon_crop : numpy.ndarray
            Galactic longitude axis of the cropped region [deg].
        lat_crop : numpy.ndarray
            Galactic latitude axis of the cropped region [deg].
        mom0 : numpy.ndarray
            Moment-0 map over the full velocity range.
        """
        hdu   = self._get_hdu(tracer)
        vdim  = self._VDIM[tracer]
        cube  = np.copy(hdu.data).astype(float)

        vel = self._get_velocity(hdu, tracer)
        if tracer == "HI":
            mask = np.abs(vel) < 100.0
            cube = cube[mask]
            vel  = vel[mask]

        lon = self._get_axis(hdu, vdim % 3 + 1)
        lat = self._get_axis(hdu, (vdim + 1) % 3 + 1)

        l = coord.galactic.l.value
        b = coord.galactic.b.value
        ext = 0.5 * (fov[0].value + fov[1].value) / 2.0

        il0 = np.where(np.abs(lon - l) < ext)[0][0]
        il1 = np.where(np.abs(lon - l) < ext)[0][-1] + 1
        ib0 = np.where(np.abs(lat - b) < ext)[0][0]
        ib1 = np.where(np.abs(lat - b) < ext)[0][-1] + 1

        # Crop — axis ordering differs between CO and HI
        if tracer == "CO":
            crop = cube[ib0:ib1, il0:il1, :]
        else:  # HI : (vel, lat, lon)
            crop = cube[:, ib0:ib1, il0:il1]

        spatial_axes = [0, 1, 2]
        spatial_axes.remove(2 * vdim % 3)
        spectrum = np.nanmean(crop, axis=tuple(spatial_axes))

        mom0_full = self.moment0(hdu.data, cutoff=0, vdim=vdim)
        mom0      = mom0_full[ib0:ib1, il0:il1]

        return crop, vel, spectrum, lon[il0:il1], lat[ib0:ib1], mom0

    def crop_circle(self, coord: SkyCoord, tracer: str,
                    fov: tuple) -> tuple:
        """
        Crop a PPV cube to a **circular** aperture centred on ``coord``.

        Pixels outside the circle are set to ``NaN`` before averaging.
        Internally calls :meth:`crop_square` then applies the circular mask.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Centre of the aperture.
        tracer : str
            ``'CO'`` or ``'HI'``.
        fov : tuple of astropy.units.Quantity
            ``(delta_l, delta_b)`` used to define the circular radius as
            half the mean of the two angular extents.

        Returns
        -------
        Same as :meth:`crop_square`, with pixels outside the circle masked.
        """
        vdim = self._VDIM[tracer]
        crop, vel, _, lon_c, lat_c, mom0 = self.crop_square(coord, tracer, fov)

        ext = 0.5 * (fov[0].value + fov[1].value) / 2.0
        l   = coord.galactic.l.value
        b   = coord.galactic.b.value

        LON, LAT = np.meshgrid(lon_c, lat_c)
        outside  = (LAT - b)**2 + (LON - l)**2 > ext**2

        crop_masked = np.copy(crop).astype(float)
        if tracer == "CO":
            crop_masked[outside, :] = np.nan
        else:
            crop_masked[:, outside] = np.nan

        spatial_axes = [0, 1, 2]
        spatial_axes.remove(2 * vdim % 3)
        spectrum = np.nanmean(crop_masked, axis=tuple(spatial_axes))

        mom0_masked             = np.copy(mom0).astype(float)
        mom0_masked[outside]    = np.nan

        return crop_masked, vel, spectrum, lon_c, lat_c, mom0_masked

    def moment0_crop(self, coord: SkyCoord, tracer: str,
                     fov: tuple,
                     v_start: float, v_stop: float,
                     shape: str = "circle") -> tuple:
        """
        Compute the moment-0 map integrated over a velocity range.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Field centre.
        tracer : str
            ``'CO'`` or ``'HI'``.
        fov : tuple of astropy.units.Quantity
            Field of view.
        v_start : float
            Lower velocity bound [km/s].
        v_stop : float
            Upper velocity bound [km/s].
        shape : str, optional
            Aperture shape: ``'circle'`` or ``'square'``. Default ``'circle'``.

        Returns
        -------
        cube : numpy.ndarray
            Cropped PPV sub-cube.
        vel : numpy.ndarray
            Velocity axis [km/s].
        spectrum : numpy.ndarray
            Mean spectrum over the aperture.
        lon_crop, lat_crop : numpy.ndarray
            Spatial axes.
        mom0 : numpy.ndarray
            Moment-0 map integrated between ``v_start`` and ``v_stop``.
        """
        crop_fn = self.crop_circle if shape == "circle" else self.crop_square
        cube, vel, spectrum, lon_c, lat_c, _ = crop_fn(coord, tracer, fov)

        vdim       = self._VDIM[tracer]
        vel_mask   = (vel > v_start) & (vel < v_stop)

        cube_vel   = np.copy(cube).astype(float)
        if tracer == "CO":
            cube_vel[~vel_mask, :, :] = np.nan   # zero out outside range — CO: (lat,lon,vel)
        else:
            cube_vel[~vel_mask, :, :] = np.nan   # HI: (vel, lat, lon)

        spatial_axes = [0, 1, 2]
        spatial_axes.remove(2 * vdim % 3)
        mom0 = np.nanmean(cube_vel, axis=tuple(spatial_axes))

        return cube, vel, spectrum, lon_c, lat_c, mom0

    def mean_spectrum(self, coord: SkyCoord, tracer: str,
                      extension_fov: float) -> tuple:
        """
        Extract the mean spectrum over a circular aperture of radius
        ``extension_fov`` degrees.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Aperture centre.
        tracer : str
            ``'CO'`` or ``'HI'``.
        extension_fov : float
            Aperture radius [deg].

        Returns
        -------
        vel : numpy.ndarray
            Velocity axis [km/s].
        spectrum : numpy.ndarray
            Mean spectrum over the aperture.
        """
        hdu  = self._get_hdu(tracer)
        vdim = self._VDIM[tracer]
        cube = np.copy(hdu.data).astype(float)

        vel = self._get_velocity(hdu, tracer)
        if tracer == "HI":
            mask = np.abs(vel) < 100.0
            cube = cube[mask]
            vel  = vel[mask]

        lon = self._get_axis(hdu, vdim % 3 + 1)
        lat = self._get_axis(hdu, (vdim + 1) % 3 + 1)
        l   = coord.galactic.l.value
        b   = coord.galactic.b.value

        LON, LAT = np.meshgrid(lon, lat)
        outside  = (LAT - b)**2 + (LON - l)**2 > extension_fov**2

        if tracer == "HI":
            cube[:, outside] = np.nan
            spectrum = np.nanmean(cube, axis=(1, 2))
        else:
            cube[outside] = np.nan
            spectrum = np.nanmean(cube, axis=(0, 1))

        return vel, spectrum

    # -----------------------------------------------------------------------
    # Spatial axes utility
    # -----------------------------------------------------------------------

    def get_spatial_axes(self, coord: SkyCoord,
                         fov: tuple,
                         delta: float) -> tuple:
        """
        Return galactic longitude and latitude axes for a regular grid.

        Parameters
        ----------
        coord : astropy.coordinates.SkyCoord
            Grid centre.
        fov : tuple
            ``(delta_l, delta_b)`` angular extents.
        delta : float
            Grid spacing [deg].

        Returns
        -------
        longitudes : numpy.ndarray
            Galactic longitudes [deg].
        latitudes : numpy.ndarray
            Galactic latitudes [deg].
        """
        Dl, Db   = fov
        l_center = coord.galactic.l
        b_center = coord.galactic.b
        nl       = int(Dl // delta)
        nb       = int(Db // delta)
        return (
            np.linspace(l_center - Dl / 2, l_center + Dl / 2, nl).value,
            np.linspace(b_center - Db / 2, b_center + Db / 2, nb).value,
        )
