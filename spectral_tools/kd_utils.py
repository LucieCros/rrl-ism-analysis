# -*- coding: utf-8 -*-
"""
kd_utils.py
===========
Galactic kinematic distance utilities based on the Reid et al. (2019) rotation model.

Provides low-level functions for converting heliocentric (l, b, d) coordinates
to Galactocentric radius, Galactic Cartesian coordinates, and LSR velocity.
Used internally by :mod:`spectral_tools.reid19_rotcurve`.

Dependencies
------------
numpy
Standard library: os
"""

import os
import numpy as np

# IAU-defined solar motion parameters (km/s)
__Ustd = 10.27
__Vstd = 15.32
__Wstd = 7.74

# Reid+2019 Galactocentric radius and solar motion parameters
__R0 = 8.15  # kpc
__Usun = 10.6  # km/s
__Vsun = 10.7  # km/s
__Wsun = 7.6  # km/s

def calc_Rgal(glong, glat, dist, R0=__R0):
    """
    Return the Galactocentric radius of an object with a given
    Galacitic longitude, latitude, and distance.

    Parameters:
      glong, glat :: scalars or arrays of scalars
        Galactic longitude and latitude (deg).

      dist :: scalar or array of scalars
        line-of-sight distance (kpc).

      R0 :: scalar (optional)
        Galactocentric radius of the Sun.

    Returns: R
      Rgal :: scalar or array of scalars
        Galactocentric radius (kpc).
    """
    #
    # law of cosines
    #
    dist_cos_glat = dist * np.cos(np.deg2rad(glat))
    Rgal2 = R0 ** 2.0 + dist_cos_glat ** 2.0
    Rgal2 = Rgal2 - 2.0 * R0 * dist_cos_glat * np.cos(np.deg2rad(glong))
    Rgal = np.sqrt(Rgal2)
    return Rgal
    
    
def calc_az(glong, glat, dist, R0=__R0):
    """
    Return the Galactocentric azimuth of an object with a given
    Galacitic longitude, latitude, and distance. Galactocentric
    azimuth is defined as zero in the direction of the Sun and
    increasing in the direction of the Solar orbit.

    Parameters:
      glong, glat :: scalars or arrays of scalars
        Galactic longitude and latitude (deg).

      dist :: scalar or array of scalars
        line-of-sight distance (kpc).

      R0 :: scalar (optional)
        Galactocentric radius of the Sun.

    Returns: az
      az :: scalar or array of scalars
        Galactocentric azimuth (degs).

    """
    input_scalar = np.isscalar(glong) and np.isscalar(glat) and np.isscalar(dist)
    glong, glat, dist = np.atleast_1d(glong, glat, dist)
    # ensure longitude range [0,360) degrees
    glong = glong % 360.0
    #
    # Compute Rgal
    #
    Rgal = calc_Rgal(glong, glat, dist, R0=R0)
    #
    # law of cosines
    #
    dist_cos_glat = dist * np.cos(np.deg2rad(glat))
    cos_az = (R0 ** 2.0 + Rgal ** 2.0 - dist_cos_glat ** 2.0) / (2.0 * Rgal * R0)
    
    #
    # Catch fringe cases
    #
    cos_az[cos_az > 1.0] = 1.0
    cos_az[cos_az < -1.0] = -1.0
    az = np.rad2deg(np.arccos(cos_az))
    #
    # Correct azimuth in 3rd and 4th quadrants
    #
    az[glong > 180.0] = 360.0 - az[glong > 180.0]
    if input_scalar:
        return az[0]
    return az

