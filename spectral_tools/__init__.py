# -*- coding: utf-8 -*-
"""
spectral_tools
==============
Python toolkit for the analysis of carbon radio recombination lines (CRRLs)
observed with NenuFAR/LOFAR.

Package structure
-----------------
.. code-block:: none

    spectral_tools/
    ├── atoms.py            Atomic species data and RRL frequency computation
    ├── tools.py            Signal processing, Doppler corrections, utilities
    ├── io.py               FITS I/O, source catalogue reader, FITS header builder
    ├── L1_class.py         NenuFAR Level-1 FITS observation container
    ├── line_fitting.py     Voigt profile fitting and uncertainty propagation
    ├── modeling.py         Physical CRRL line modeling (broadening, area, chi²)
    ├── graphics.py         Matplotlib plotting utilities
    ├── maps.py             Dust extinction and PPV cube handling (CO, HI)
    └── grid_exploration.py Chi² grid exploration and parameter-space visualisation

Quick start
-----------
::

    # Load an observation
    from spectral_tools.io import load_bloc
    hdu = load_bloc("/path/to/observation.fits")

    # Compute a line frequency
    from spectral_tools.atoms import line_freq
    freq = line_freq(500)          # Cα n=500, returns Quantity in MHz

    # Doppler correction
    import spectral_tools.tools as tools
    v_lsr = tools.doppler_corrections("2021-06-09T12:00:00")[-1]

    # Fit a Voigt profile
    import spectral_tools.line_fitting as lf
    area = lf.voigt_area(peak=-0.05, fwhm_G=3.2, fwhm_L=1.1)

    # Load a source's CO cube and crop it
    from spectral_tools.maps import MapLoader
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    loader = MapLoader("Tau")
    cube, vel, spec, lon, lat, mom0 = loader.crop_circle(
        SkyCoord("05h34m32s", "22d00m52s"), "CO", (2*u.deg, 2*u.deg)
    )
"""

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "0.1.0"
__author__  = "Lucie Cros"
__license__ = "CC BY 4.0"

# ---------------------------------------------------------------------------
# Public API — explicit re-exports for IDE auto-completion and `from
# spectral_tools import *` usage. Import only the most commonly used
# symbols; detailed utilities should be accessed via their submodule.
# ---------------------------------------------------------------------------

# Atomic physics
from spectral_tools.atoms import line_freq, set_specie          # noqa: F401

# Core utilities
from spectral_tools.tools import (                               # noqa: F401
    doppler_correction,
    doppler_corrections,
    f_to_v,
    v_to_f,
    get_line,
    get_linefree,
    slice_line,
    rebinning,
    moving_avg,
    sliding_rms,
    recursive_clipping,
    adaptive_sigma_clip_loop,
    fill_edges,
    weighted_avg_and_std,
    continuum_polyn,
    continuum_powerlaw,
)

# I/O
from spectral_tools.io import (                                  # noqa: F401
    load_bloc,
    read_source_info,
    read_source_velocity,
    normalise_source_name,
    build_fits_header,
)

# Line fitting
from spectral_tools.line_fitting import (                        # noqa: F401
    voigt,
    multiple_voigt,
    voigt_fwhm,
    voigt_area,
    voigt_area_inv,
    lorentz_amplitude,
    fit_multi_voigt,
)

# Physical modeling
from spectral_tools.modeling import (                            # noqa: F401
    doppler_broadening,
    pressure_broadening,
    radiation_broadening,
    lorentzian_broadening,
    natural_broadening,
    total_broadening,
    total_broadening_kms,
    integrated_area,
    create_surface,
    create_surface_multi,
    cost_function,
    cost_function_multi,
)

# Grid exploration
from spectral_tools.grid_exploration import (                    # noqa: F401
    compute_chi2_split,
    find_best_parameters,
    plot_chi2_projections,
    plot_chi2_projections_log,
)

# ---------------------------------------------------------------------------
# Submodules available for explicit import
# ---------------------------------------------------------------------------
# spectral_tools.graphics       → plot_overview, plot_subband, set_axes, ...
# spectral_tools.maps           → DustMap, MapLoader
# spectral_tools.L1_class       → L1
