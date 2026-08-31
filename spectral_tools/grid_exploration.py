# -*- coding: utf-8 -*-
"""
grid_exploration.py
===================
Chi² grid exploration and visualisation for CRRL physical parameter fitting.

This module provides tools to compare a pre-computed parameter-space grid
(produced by :mod:`pipelines.generate_grid`) against observed RRL line
widths and integrated areas, and to identify the best-fit physical conditions.

This module provides:

- **Chi² computation** : :func:`compute_chi2_split` — evaluate the reduced
  chi² over a (Te, Ne, T0, L, vt) grid from per-n NetCDF model files
- **Best-fit extraction** : :func:`find_best_parameters` — identify
  parameter combinations within a tolerance of the minimum chi²
- **Visualisation** : :func:`plot_chi2_projections`,
  :func:`plot_chi2_projections_log` — 2-D marginalised chi² heatmaps for
  all parameter pairs

Changes from the original ``gridexplo.py``
------------------------------------------
- ``set_specie()``, ``line_freq()``, ``v_to_f()`` → removed; imported from
  :mod:`spectral_tools.atoms` and :mod:`spectral_tools.tools` instead.
- French docstrings → English.
- ``plot_chi2_projections_full_fast`` → :func:`plot_chi2_projections`
- ``plot_chi2_projections_full_log``  → :func:`plot_chi2_projections_log`

Dependencies
------------
numpy, pandas, xarray, matplotlib
Internal: spectral_tools.atoms, spectral_tools.tools
"""

import itertools
import os

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from spectral_tools.atoms import line_freq
from spectral_tools.tools import v_to_f
from spectral_tools import graphics
# ---------------------------------------------------------------------------
# Matplotlib style (publication quality)
# ---------------------------------------------------------------------------

#matplotlib.rcParams.update({
#    "mathtext.fontset":  "stix",
#    "font.family":       "serif",
#    "axes.linewidth":    1.5,
#    "lines.linewidth":   1.7,
#    "font.size":         20,
#    "xtick.labelsize":   13,
#    "ytick.labelsize":   13,
#    "xtick.direction":   "in",
#    "ytick.direction":   "in",
#    "xtick.major.size":  10,
#    "ytick.major.size":  10,
#    "xtick.minor.size":   7,
#    "ytick.minor.size":   7,
#    "xtick.major.width":  1,
#    "ytick.major.width":  1,
#    "xtick.minor.width":  1,
#    "ytick.minor.width":  1,
#})
graphics.set_style()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default names of the five physical parameters.
PARAM_NAMES: list[str] = ["Te", "Ne", "T0", "L", "vt"]

PARAM_LABELS: dict = {
        'Te': '$T_e$', 'Ne': '$n_e$',
        'T0': '$T_{\\rm rad}$', 'L':  '$L$',
        'vt': '$v_t$',
    }
PARAM_UNITS: dict = {'Te': 'K', 'Ne': 'cm$^{-3}$', 'T0': 'K', 'L': 'pc', 'vt': 'km/s'}


#: Number of free parameters in the chi² model (Te, Ne, T0, L, vt).
N_FREE_PARAMS: int = 5

#: Number of observables per transition (deltaf + area).
N_OBS_PER_TRANSITION: int = 2


# ---------------------------------------------------------------------------
# Observation loader
# ---------------------------------------------------------------------------

def load_observations(csv_file: str,
                      n_subset=None) -> tuple:
    """
    Load and prepare fitted line parameters from a CSV observation file.

    The CSV must contain the following columns:

    - ``wV``   : measured Voigt FWHM
    - ``dwV``  : uncertainty on ``wV``
    - ``Ifit`` : measured integrated area
    - ``dIfit``: uncertainty on ``Ifit``

    The index column should contain the principal quantum numbers n.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file of fitted line parameters.
    n_subset : array-like of int, optional
        Subset of quantum numbers to load. If ``None``, all rows are loaded.
    oldvals : bool, optional
        If ``True``, the CSV values are in km/s and will be converted to Hz
        using the central frequency of each transition. Default ``False``.
    tau : bool, optional
        Controls which column is used as the index (``index_col=1`` if
        ``True`` or ``oldvals``). Default ``True``.

    Returns
    -------
    obs_n : numpy.ndarray of int
        Principal quantum numbers.
    obs_deltaf : numpy.ndarray
        Measured Voigt FWHM [Hz].
    obs_ddeltaf : numpy.ndarray
        Uncertainty on ``obs_deltaf`` [Hz].
    obs_area : numpy.ndarray
        Measured integrated area.
    obs_darea : numpy.ndarray
        Uncertainty on ``obs_area``.
    """
    import astropy.units as u

    index_col = 0
    df = pd.read_csv(csv_file, index_col=index_col)

    if n_subset is not None:
        df = df[df.index.isin(n_subset)]

    obs_n       = df.index.values.astype(int)
    obs_deltaf  = df["wV"].values
    obs_ddeltaf = df["dwV"].values
    obs_area    = df["Ifit"].values
    obs_darea   = df["dIfit"].values

    return obs_n, obs_deltaf, obs_ddeltaf, obs_area, obs_darea


# ---------------------------------------------------------------------------
# Chi² computation
# ---------------------------------------------------------------------------

def compute_chi2_split(path_xrs: str, filepattern: str, csv_file: str,
                       n_subset=None, chi_type: type=np.float32,
                       weight_fwhm: np.float32 = 1) -> xr.DataArray:
    """
    Compute the reduced chi² between a parameter-space grid and observations.

    Iterates over per-n NetCDF model files produced by
    :mod:`pipelines.generate_grid`, accumulating the chi² contribution from
    each transition:

    .. math::

        \\chi^2_{\\rm red} = \\frac{\\sum_n
            \\left[\\left(\\frac{\\Delta f_{\\rm mod} - \\Delta f_{\\rm obs}}
            {\\sigma_{\\Delta f}}\\right)^2
            + \\left(\\frac{A_{\\rm mod} - A_{\\rm obs}}{\\sigma_A}\\right)^2
            \\right]}{2N - 5}

    where :math:`N` is the number of transitions with finite chi²
    contributions and 5 is the number of free parameters
    (Te, Ne, T0, L, vt).

    Parameters
    ----------
    path_xrs : str
        Directory containing the per-n NetCDF grid files.
    filepattern : str
        Filename pattern with a single ``{}`` placeholder for n,
        e.g. ``'grid-{}.nc'``.
    csv_file : str
        Path to the CSV file of fitted observations (see :func:`load_observations`).
    n_subset : array-like of int, optional
        Subset of quantum numbers to include. Default: all in the CSV.
    chi_type : type        
        Computation dtype (float32 recommended given the grid size). Default: float32
    Returns
    -------
    xr.DataArray
        Reduced chi² grid, dimensions (Te, Ne, T0, L, vt).
    """
    obs_n, obs_deltaf, obs_ddeltaf, obs_area, obs_darea = load_observations(
        csv_file, n_subset
    )
    
    # choix du dtype adapté au nombre de transitions, avec marge de sécurité
#    n_transitions = len(obs_n)
#    if n_transitions < 2**8 - 1:
#        count_dtype = np.uint8
#    elif n_transitions < 2**16 - 1:
#        count_dtype = np.uint16
#    else:
#        count_dtype = np.uint32
    count_dtype = np.int64   
    chi2_accum       = None
    number_of_points = None
    last_coords      = None

    for i, n_val in enumerate(obs_n):
        filepath = os.path.join(path_xrs, filepattern.format(n_val))

        with xr.open_dataset(filepath) as ds:
            # Shape: (1, Te, Ne, T0, L, vt) — squeeze the n axis
            model_deltaf = ds["deltaf"].values[0, ...]
            model_area   = ds["area"].values[0, ...]
            last_coords  = {dim: ds[dim].values for dim in PARAM_NAMES}

        # Chi² contribution from this transition
        chi2_n = (
              weight_fwhm*((model_deltaf - obs_deltaf[i])  / obs_ddeltaf[i]) ** 2
            + ((model_area - obs_area[i])    / obs_darea[i])   ** 2
        ) 
        # Exclude non-finite values (model outside valid range)
        chi2_n[~np.isfinite(chi2_n)] = np.nan
        finite_mask = np.isfinite(chi2_n).astype(count_dtype)

        if chi2_accum is None:
            chi2_accum       = chi2_n.astype(chi_type).copy()
            number_of_points = finite_mask
        else:
            chi2_accum       = np.nansum([chi2_accum, chi2_n], axis=0, dtype=chi_type)
            number_of_points += finite_mask

    # Reduced chi²: divide by (2N - N_FREE_PARAMS)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi2_red = chi2_accum / ((weight_fwhm +1) * number_of_points
                                 - N_FREE_PARAMS)
    chi2_da = xr.DataArray(chi2_red.astype(chi_type), coords=last_coords, dims=PARAM_NAMES)
    print(f"[compute_chi2_split] Chi² grid shape: {chi2_da.shape}")
    return chi2_da

def compute_chi2_split_old(path_xrs: str, filepattern: str, csv_file: str,
                       n_subset=None) -> xr.DataArray:
    """
    Compute the reduced chi² between a parameter-space grid and observations.

    Iterates over per-n NetCDF model files produced by
    :mod:`pipelines.generate_grid`, accumulating the chi² contribution from
    each transition:

    .. math::

        \\chi^2_{\\rm red} = \\frac{\\sum_n
            \\left[\\left(\\frac{\\Delta f_{\\rm mod} - \\Delta f_{\\rm obs}}
            {\\sigma_{\\Delta f}}\\right)^2
            + \\left(\\frac{A_{\\rm mod} - A_{\\rm obs}}{\\sigma_A}\\right)^2
            \\right]}{2N - 5}

    where :math:`N` is the number of transitions with finite chi²
    contributions and 5 is the number of free parameters
    (Te, Ne, T0, L, vt).

    Parameters
    ----------
    path_xrs : str
        Directory containing the per-n NetCDF grid files.
    filepattern : str
        Filename pattern with a single ``{}`` placeholder for n,
        e.g. ``'grid-{}.nc'``.
    csv_file : str
        Path to the CSV file of fitted observations (see :func:`load_observations`).
    n_subset : array-like of int, optional
        Subset of quantum numbers to include. Default: all in the CSV.
    oldvals : bool, optional
        If ``True``, convert CSV values from km/s to Hz. Default ``False``.
    tau : bool, optional
        Controls the CSV index column. Default ``True``.

    Returns
    -------
    xr.DataArray
        Reduced chi² grid, dimensions (Te, Ne, T0, L, vt).
    """
    obs_n, obs_deltaf, obs_ddeltaf, obs_area, obs_darea = load_observations(
        csv_file, n_subset
    )

    chi2_accum       = None
    number_of_points = None
    last_coords      = None

    for i, n_val in enumerate(obs_n):
        filepath = os.path.join(path_xrs, filepattern.format(n_val))

        with xr.open_dataset(filepath) as ds:
            # Shape: (1, Te, Ne, T0, L, vt) — squeeze the n axis
            model_deltaf = ds["deltaf"].values[0, ...]
            model_area   = ds["area"].values[0, ...]
            last_coords  = {dim: ds[dim].values for dim in PARAM_NAMES}

        # Chi² contribution from this transition
        chi2_n = (
            ((model_deltaf - obs_deltaf[i])  / obs_ddeltaf[i]) ** 2
            + ((model_area - obs_area[i])    / obs_darea[i])   ** 2
        )
        # Exclude non-finite values (model outside valid range)
        chi2_n[~np.isfinite(chi2_n)] = np.nan
        finite_mask = np.isfinite(chi2_n).astype(np.int64)

        if chi2_accum is None:
            chi2_accum       = chi2_n.copy()
            number_of_points = finite_mask
        else:
            chi2_accum       = np.nansum([chi2_accum, chi2_n], axis=0)
            number_of_points += finite_mask

    # Reduced chi²: divide by (2N - N_FREE_PARAMS)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi2_red = chi2_accum / (N_OBS_PER_TRANSITION * number_of_points
                                 - N_FREE_PARAMS)

    chi2_da = xr.DataArray(chi2_red, coords=last_coords, dims=PARAM_NAMES)
    print(f"[compute_chi2_split] Chi² grid shape: {chi2_da.shape}")
    return chi2_da



# ---------------------------------------------------------------------------
# Best-fit extraction
# ---------------------------------------------------------------------------

def find_best_parameters(chi2_grid: xr.DataArray,
                          percentile: float = 5.0) -> tuple:
    """
    Extract parameter combinations compatible with the chi² minimum.

    A point is considered compatible if its chi² satisfies:

    .. math::

        \\chi^2 \\leq \\chi^2_{\\rm min} \\times \\left(1 + \\frac{p}{100}\\right)

    where ``p`` is the ``percentile`` argument.

    Parameters
    ----------
    chi2_grid : xr.DataArray
        Reduced chi² grid from :func:`compute_chi2_split`.
    percentile : float, optional
        Tolerance above the minimum in percent. Default 5 %.

    Returns
    -------
    df_best : pandas.DataFrame
        All parameter combinations within the tolerance, sorted by chi²
        in ascending order.
    best_params : dict
        Parameters at the global minimum, plus the key ``'chi2'``.

    Examples
    --------
    >>> df, best = find_best_parameters(chi2_grid, percentile=10.0)
    >>> print(best)
    {'Te': 80.0, 'Ne': 0.03, 'T0': 1000.0, 'L': 5.0, 'vt': 2.0, 'chi2': 1.23}
    """
    chi2_min  = float(chi2_grid.min())
    threshold = chi2_min * (1.0 + percentile / 100.0)

    df_best = (
        chi2_grid
        .where(chi2_grid <= threshold)
        .to_dataframe(name="chi2")
        .reset_index()
        .dropna(subset=["chi2"])
        .sort_values("chi2")
        .reset_index(drop=True)
    )

    # Global minimum coordinates (order-independent)
    min_idx_flat  = int(np.nanargmin(chi2_grid.values))
    min_idx_multi = np.unravel_index(min_idx_flat, chi2_grid.shape)
    best_params   = {
        dim: float(chi2_grid[dim].values[i])
        for dim, i in zip(chi2_grid.dims, min_idx_multi)
    }
    best_params["chi2"] = chi2_min

    return df_best, best_params
def min_connected_region(chi2_grid: xr.DataArray, Lmin: float = None,
                          Lmax: float = None, rel_threshold: float = 0.15,
                          connectivity: int = None,
                          dtype: type = np.float32) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Identify all arc-connected regions ("bubbles") below the chi2 threshold
    in a regular xarray grid (Te, Ne, T0, L, vt).

    Parameters
    ----------
    chi2_grid : xr.DataArray
        N-D chi2 grid, one dimension per parameter.
    Lmin, Lmax : float, optional
        Bounds to restrict the grid along the L axis before analysis
        (keeps a regular grid, just smaller along this axis).
    rel_threshold : float
        Upper threshold = chi2_min * (1 + rel_threshold).
    connectivity : int, optional
        Connectivity (1 = faces only, up to ndim = including diagonals).
        Default: max connectivity.
    dtype : type
        Computation dtype (float32 recommended given the grid size).

    Returns
    -------
    df_bubbles : pd.DataFrame
        Points of ALL connected regions below the threshold, sorted by
        ascending chi2. Includes a 'bubble_id' column (1-indexed, as
        returned by scipy.ndimage.label) and a boolean 'is_min_bubble'
        column flagging the bubble that contains the global minimum.
    labels : np.ndarray
        Integer label array, same shape as the (possibly L-restricted)
        grid. 0 = outside any bubble, >0 = bubble id.
    info : dict
        'chi2_min', 'threshold', 'n_features', 'shape', 'min_bubble_id'.
    """
    da = chi2_grid

    # --- optional restriction on L, keeping a regular grid ---
    if Lmin is not None or Lmax is not None:
        L_vals = da['L'].values
        keep = np.ones_like(L_vals, dtype=bool)
        if Lmin is not None:
            keep &= L_vals > Lmin
        if Lmax is not None:
            keep &= L_vals < Lmax
        da = da.isel(L=keep)

    dims = da.dims
    grid = da.values.astype(dtype)

    # --- labelling (nanmin/nanargmin in case NaNs are present) ---
    chi2_min = np.nanmin(grid)
    threshold = chi2_min * (1 + rel_threshold)
    mask = grid <= threshold  # NaNs are automatically excluded (comparison -> False)

    if connectivity is None:
        connectivity = grid.ndim
    structure = ndimage.generate_binary_structure(grid.ndim, connectivity)
    labels, n_features = ndimage.label(mask, structure=structure)

    idx_min = np.unravel_index(np.nanargmin(grid), grid.shape)
    min_bubble_id = labels[idx_min]

    # --- extraction of ALL bubbles at once ---
    bubble_mask = labels > 0
    region_coords = np.argwhere(bubble_mask)
    cols = {dim: da.coords[dim].values[region_coords[:, i]]
            for i, dim in enumerate(dims)}
    cols['chi2'] = grid[bubble_mask]
    cols['bubble_id'] = labels[bubble_mask]

    df_bubbles = pd.DataFrame(cols)
    df_bubbles['is_min_bubble'] = df_bubbles['bubble_id'] == min_bubble_id
    df_bubbles = df_bubbles.sort_values('chi2').reset_index(drop=True)

    info = {'chi2_min': chi2_min, 'threshold': threshold,
            'n_features': n_features, 'shape': grid.shape,
            'min_bubble_id': min_bubble_id}

    return df_bubbles, labels, info
    
    
def print_region_summary(df_region, value_col='chi2',save=False):
    """
    Print min, max, and best-fit value of each parameter
    within the connected region of the minimum.

    Parameters
    ----------
    df_region : pd.DataFrame
        Points of the connected region (columns = params + value_col).
    value_col : str
        Name of the chi2 column (excluded from the displayed parameters).
    """
    params = [c for c in df_region.columns if c != value_col]
    row_opt = df_region.loc[df_region[value_col].idxmin()]

    print(f"{'Parameter':<10}{'min':>12}{'opt':>12}{'max':>12}")
    saving = []
    for p in params:
        vmin = df_region[p].min()
        vmax = df_region[p].max()
        vopt = row_opt[p]
        saving.append([vmin, vmax, vopt])
        print(f"{p:<10}{vmin:>12.4g}{vopt:>12.4g}{vmax:>12.4g}")
    if save :
        return saving
# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def corner_plot_chi2_old(da, mask_region, params=None, cmap='viridis',
                      bubble_color='red', figsize=None, log_scale=None,
                      threshold=1.30, thresholds=(1.0, 1.15, 1.30),
                      wspace=0.05, hspace=0.05, diag_gap=0.15):
    """
    Corner plot of chi2 minimized by projection, with the selected
    connected region (bubble) overlaid as a contour on each panel.

    chi2_min and the best-fit coordinates (opt_vals) are computed
    directly from `da` (position of the global minimum). Heatmap
    vmax = threshold * chi2_min; vmin = chi2_min.

    Parameters
    ----------
    da : xr.DataArray
        N-D chi2 grid (same dims used for labelling).
    mask_region : np.ndarray (bool)
        Boolean mask, same shape as da.values, True for the selected bubble.
    params : list of str, optional
        Subset/order of dimensions to display. Default: all dims of da.
    cmap : str
        Colormap for the background (minimized chi2).
    bubble_color : str
        Color of the bubble's contour outline.
    log_scale : list of str, optional
        Parameter names to display on a log scale (e.g. ['Ne']).
    threshold : float
        Multiplicative factor of chi2_min for the colormap upper bound
        (vmax = threshold * chi2_min).
    thresholds : tuple of float
        Multiplicative factors of chi2_min for the horizontal lines
        on the diagonal panels (line styles: solid, ':', '-.', in order).
    wspace, hspace : float
        Horizontal/vertical spacing between panels (passed to subplots_adjust).
    diag_gap : float
        Fraction of panel size trimmed from each side of diagonal panels,
        to visually detach them (0 = flush, 0.15 = 15% margin per side).
    """

    dims = list(da.dims) if params is None else params
    n = len(dims)
    grid = da.values
    coords = {d: da.coords[d].values for d in dims}
    log_scale = log_scale or []

    # --- chi2_min and opt_vals computed directly from the grid ---
    idx_min = np.unravel_index(np.nanargmin(grid), grid.shape)
    chi2_min = grid[idx_min]
    all_dims = list(da.dims)
    opt_vals = {d: da.coords[d].values[idx_min[all_dims.index(d)]] for d in all_dims}

    vmin = chi2_min
    vmax = threshold * chi2_min

    line_styles = ['solid', ':', '-.', '--', (0, (1, 1))]

    fig, axes = plt.subplots(n, n, figsize=figsize or (2.3 * n, 2.3 * n))

    region_ranges = {}
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if j > i:
                ax.axis('off')
                continue
            
            if i == j:
                reduce_axes = tuple(k for k in range(n) if k != i)
                profile = np.nanmin(grid, axis=reduce_axes)
                x = coords[dims[i]]
                #ax.semilogy()
                graphics.set_axes(ax, yscale='log')
                ax.plot(x, profile, color='steelblue', lw=1.5)

                for k, s in enumerate(thresholds):
                    ax.axhline(s * chi2_min, color='gray',
                               ls=line_styles[k % len(line_styles)], lw=0.8)

                ax.axvline(opt_vals[dims[i]], color='r', ls='--', lw=1.2)

                proj_mask = np.any(mask_region, axis=reduce_axes)
                if proj_mask.any():
                    x_bubble = x[proj_mask]
                    region_ranges[dims[i]] = (x_bubble.min(), x_bubble.max())
                    ax.axvspan(x_bubble.min(), x_bubble.max(),
                               color=bubble_color, alpha=0.25)

                if dims[i] in log_scale:
                    ax.set_xscale('log')

                if i > 0:
                    ax.tick_params(labelleft=False, which="both")
            else:
                graphics.set_axes(ax)
                reduce_axes = tuple(k for k in range(n) if k not in (i, j))
                proj = np.nanmin(grid, axis=reduce_axes)
                proj_mask = np.any(mask_region, axis=reduce_axes)

                if i < j:
                    Z = proj
                    M = proj_mask
                else:
                    Z = proj.T
                    M = proj_mask.T

                x, y = coords[dims[j]], coords[dims[i]]
                ax.pcolormesh(x, y, Z, cmap=cmap, shading='auto',
                              vmin=vmin, vmax=vmax)
                if M.any():
                    ax.contour(x, y, M.astype(int), levels=[0.5],
                               colors=bubble_color, linewidths=1.5)

                ax.scatter([opt_vals[dims[j]]], [opt_vals[dims[i]]],
                           color='r', s=100, marker='*', zorder=5,
                           edgecolor='w')

                if dims[j] in log_scale:
                    ax.set_xscale('log')
                if dims[i] in log_scale:
                    ax.set_yscale('log')

            if i == n - 1:
                ax.set_xlabel(PARAM_LABELS[dims[j]] + " (" + PARAM_UNITS[dims[j]] + ")")
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(PARAM_LABELS[dims[i]] + " (" + PARAM_UNITS[dims[i]] + ")")
            elif j == 0 and i == 0:
                ax.set_ylabel("$\\chi^2_{min}$")
            else:
                ax.set_yticklabels([])

    fig.tight_layout()
    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # --- detach diagonal panels slightly ---
    for i in range(n):
        ax = axes[i, i]
        pos = ax.get_position()
        dx = pos.width * diag_gap
        dy = pos.height * diag_gap
        ax.set_position([pos.x0 + dx, pos.y0 + dy,
                          pos.width - 2 * dx, pos.height - 2 * dy])

    # --- text summary (min/opt/max) in the empty upper-right panel ---
    ax_text = axes[0, n - 3]
    ax_text.set_visible(True)
    ax_text.axis('off')

    lines = [f"{'Param':<8}{'min':>10}{'opt':>10}{'max':>10}"]
    for p in dims:
        label = PARAM_LABELS.get(p, p)
        unit = PARAM_UNITS.get(p, '')
        vmin_p, vmax_p = region_ranges.get(p, (np.nan, np.nan))
        vopt_p = opt_vals[p]
        name = f"{label} [{unit}]" if unit else label
        lines.append(f"{name:<14}{vmin_p:>8.3g}{vopt_p:>10.3g}{vmax_p:>10.3g}")

    ax_text.text(0.5, 1., "\n".join(lines), transform=ax_text.transAxes,
                 fontsize=10, family='monospace', va='top', ha='center')

    # --- global legend ---
    legend_handles = [
        mlines.Line2D([], [], color='r', marker='*', linestyle='None',
                      markersize=12, markeredgecolor='w', label='Best-fit'),
        mpatches.Patch(edgecolor=bubble_color, facecolor='none',
                       label=f'Connected region ($\\chi^2 \\leq$ {threshold:.2f}×'
                             '$\\chi^2_{\\rm min}$)'),
        mlines.Line2D([], [], color='gray', ls='solid', lw=1, label='$\\chi^2_{\\rm min}$'),
    ]
    for k, s in enumerate(thresholds[1:], start=1):
        legend_handles.append(
            mlines.Line2D([], [], color='gray', ls=line_styles[k % len(line_styles)],
                          lw=1, label=f'{s:.2f}' + '$× \\chi^2_{\\rm min}$')
        )

    fig.legend(handles=legend_handles, loc='upper right',
               bbox_to_anchor=(0.75, 0.8), fontsize=9, frameon=False)
    return fig, axes


def corner_plot_chi2(da: xr.DataArray, labels: np.ndarray,
                      bubble_ids: list[int] = None,
                      bubble_colors: dict[int, str] = None,
                      min_bubble_id: int = None,
                      params: list[str] = None, cmap: str = 'viridis',
                      figsize: tuple[float, float] = None,
                      log_scale: list[str] = None, threshold: float = 1.30,
                      thresholds: tuple[float, ...] = (1.0, 1.15, 1.30),
                      wspace: float = 0.05, hspace: float = 0.05) -> tuple[plt.Figure, np.ndarray]:
    """
    Corner plot of chi2 minimized by projection, with one or several
    connected regions ("bubbles") overlaid as colored contours on each panel.

    chi2_min and the best-fit coordinates (opt_vals) are computed
    directly from `da` (position of the global minimum). Heatmap
    vmax = threshold * chi2_min.

    Parameters
    ----------
    da : xr.DataArray
        N-D chi2 grid (same dims used for labelling).
    labels : np.ndarray
        Integer label array, same shape as da.values (0 = outside any
        bubble, >0 = bubble id), as returned by min_connected_region.
    bubble_ids : list of int, optional
        Which bubble ids to display. Default: all bubbles present in `labels`.
    bubble_colors : dict of {int: str}, optional
        Explicit color per bubble id. Default: auto-assigned from a
        qualitative colormap (tab10), cycling if more bubbles than colors.
    min_bubble_id : int, optional
        Bubble id containing the global minimum (e.g. info['min_bubble_id']
        from min_connected_region). If given, this bubble's legend entry
        and contour are marked as the best-fit region.
    params : list of str, optional
        Subset/order of dimensions to display. Default: all dims of da.
    cmap : str
        Colormap for the background (minimized chi2).
    log_scale : list of str, optional
        Parameter names to display on a log scale (e.g. ['Ne']).
    threshold : float
        Multiplicative factor of chi2_min for the colormap upper bound.
    thresholds : tuple of float
        Multiplicative factors of chi2_min for the horizontal lines
        on the diagonal panels.
    wspace, hspace : float
        Horizontal/vertical spacing between panels.

    """
    PARAM_LABELS = {
        'Te': '$T_e$', 'Ne': '$n_e$',
        'T0': '$T_{\\rm rad}$', 'L':  '$L$',
        'vt': '$v_t$',
    }
    PARAM_UNITS = {'Te': 'K', 'Ne': 'cm$^{-3}$', 'T0': 'K', 'L': 'pc', 'vt': 'km/s'}

    dims = list(da.dims) if params is None else params
    n = len(dims)
    grid = da.values
    coords = {d: da.coords[d].values for d in dims}
    log_scale = log_scale or []

    # --- which bubbles to draw, and their colors ---
    if bubble_ids is None:
        bubble_ids = sorted(int(b) for b in np.unique(labels) if b > 0)

    if bubble_colors is None:
        palette = plt.get_cmap('tab10').colors
        bubble_colors = {b: palette[k % len(palette)] for k, b in enumerate(bubble_ids)}

    # --- chi2_min and opt_vals computed directly from the grid ---
    idx_min = np.unravel_index(np.nanargmin(grid), grid.shape)
    chi2_min = grid[idx_min]
    all_dims = list(da.dims)
    opt_vals = {d: da.coords[d].values[idx_min[all_dims.index(d)]] for d in all_dims}

    vmin = chi2_min
    vmax = threshold * chi2_min

    line_styles = ['solid', ':', '-.', '--', (0, (1, 1))]

    fig, axes = plt.subplots(n, n, figsize=figsize or (2.3 * n, 2.3 * n))

    region_ranges = {b: {} for b in bubble_ids}  # per-bubble min/max per param
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if j > i:
                ax.axis('off')
                continue
            graphics.set_axes(ax)
            if i == j:
                reduce_axes = tuple(k for k in range(n) if k != i)
                profile = np.nanmin(grid, axis=reduce_axes)
                x = coords[dims[i]]
                ax.semilogy()
                ax.plot(x, profile, color='steelblue', lw=1.5)

                for k, s in enumerate(thresholds):
                    ax.axhline(s * chi2_min, color='gray',
                               ls=line_styles[k % len(line_styles)], lw=0.8)

                ax.axvline(opt_vals[dims[i]], color='k', ls='--', lw=1.2)

                for b in bubble_ids:
                    proj_mask = np.any(labels == b, axis=reduce_axes)
                    if proj_mask.any():
                        x_bubble = x[proj_mask]
                        region_ranges[b][dims[i]] = (x_bubble.min(), x_bubble.max())
                        ax.axvspan(x_bubble.min(), x_bubble.max(),
                                   color=bubble_colors[b], alpha=0.25)

                if dims[i] in log_scale:
                    ax.set_xscale('log')

                if i > 0:
                    ax.tick_params(labelleft=False, which="both")
            else:
                reduce_axes = tuple(k for k in range(n) if k not in (i, j))
                proj = np.nanmin(grid, axis=reduce_axes)

                Z = proj if i < j else proj.T
                x, y = coords[dims[j]], coords[dims[i]]
                ax.pcolormesh(x, y, Z, cmap=cmap, shading='auto',
                              vmin=vmin, vmax=vmax)

                for b in bubble_ids:
                    proj_mask = np.any(labels == b, axis=reduce_axes)
                    M = proj_mask if i < j else proj_mask.T
                    if M.any():
                        ax.contour(x, y, M.astype(int), levels=[0.5],
                                   colors=[bubble_colors[b]], linewidths=1.5)

                ax.scatter([opt_vals[dims[j]]], [opt_vals[dims[i]]],
                           color='k', s=100, marker='*', zorder=5,
                           edgecolor='w')

                if dims[j] in log_scale:
                    ax.set_xscale('log')
                if dims[i] in log_scale:
                    ax.set_yscale('log')

            if i == n - 1:
                ax.set_xlabel(PARAM_LABELS[params[j]] + " (" + PARAM_UNITS[params[j]] + ")")
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(PARAM_LABELS[params[i]] + " (" + PARAM_UNITS[params[i]] + ")")
            elif j == 0 and i == 0:
                ax.set_ylabel("$\\chi^2_{min}$")
            else:
                ax.set_yticklabels([])

    fig.tight_layout()
    fig.subplots_adjust(wspace=wspace, hspace=hspace)


    # --- text summary (min/opt/max) for the best-fit bubble only ---
    ax_text = axes[0, n - 3]
    ax_text.set_visible(True)
    ax_text.axis('off')

    summary_bubble = min_bubble_id if min_bubble_id in region_ranges else bubble_ids[0]
    lines = [f"{'Param':<8}{'min':>10}{'opt':>10}{'max':>10}"]
    for p in dims:
        label = PARAM_LABELS.get(p, p)
        unit = PARAM_UNITS.get(p, '')
        vmin_p, vmax_p = region_ranges[summary_bubble].get(p, (np.nan, np.nan))
        vopt_p = opt_vals[p]
        name = f"{label} [{unit}]" if unit else label
        lines.append(f"{name:<14}{vmin_p:>8.3g}{vopt_p:>10.3g}{vmax_p:>10.3g}")

    ax_text.text(0.5, 1., "\n".join(lines), transform=ax_text.transAxes,
                 fontsize=10, family='monospace', va='top', ha='center')

    # --- global legend ---
    legend_handles = [
        mlines.Line2D([], [], color='k', marker='*', linestyle='None',
                      markersize=12, markeredgecolor='w', label='Best-fit'),
        mlines.Line2D([], [], color='gray', ls='solid', lw=1, label='$\\chi^2_{\\rm min}$'),
    ]
    for k, s in enumerate(thresholds[1:], start=1):
        legend_handles.append(
            mlines.Line2D([], [], color='gray', ls=line_styles[k % len(line_styles)],
                          lw=1, label=f'{s:.2f}' + '$× \\chi^2_{\\rm min}$')
        )
    for b in bubble_ids:
        tag = ' (best-fit)' if b == min_bubble_id else ''
        legend_handles.append(
            mpatches.Patch(edgecolor=bubble_colors[b], facecolor='none',
                           label=f'Bubble {b}{tag} ($\\chi^2 \\leq$ {threshold:.2f}×'
                                 '$\\chi^2_{\\rm min}$)')
        )

    fig.legend(handles=legend_handles, loc='upper right',
               bbox_to_anchor=(0.75, 0.8), fontsize=9, frameon=False)
    return fig, axes
    
def _make_heatmap_axes(fig) -> dict:
    """
    Create a mosaic of 10 sub-panels (one per parameter pair C(5,2) = 10),
    plus a shared colorbar axis and a text annotation panel.

    Returns
    -------
    dict
        Axes dictionary keyed by mosaic labels.
    """
    return fig.subplot_mosaic(
        """
        cABCD
        cFGHI
        cKLtt
        """,
        width_ratios=(0.5, 5, 5, 5, 5),
    )


def make_masked_heatmap(df_subset, x_cols, y_idx, cond):
    """
    Pivot a χ² DataFrame subset into a (Te, L) heatmap, re-index onto
    the full grid, and apply a boolean exclusion mask (masked → NaN).

    Parameters
    ----------
    df_subset : pd.DataFrame
    x_cols    : array-like   — full L axis (columns)
    y_idx     : array-like   — full Te axis (index)
    cond      : np.ndarray   — True where values should be excluded

    Returns
    -------
    np.ndarray
    """
    hm = df_subset.pivot_table(
        index='Te', columns='L', values='chi2', aggfunc='min'
    ).reindex(columns=x_cols, index=y_idx)
    vals = hm.values.copy()
    vals[cond] = np.nan
    return vals



# ---------------------------------------------------------------------------
# Chi² projection plots
# ---------------------------------------------------------------------------

def plot_chi2_projections(chi2_grid: xr.DataArray,
                           best_params: dict,
                           df_best: pd.DataFrame,
                           filepath: str,
                           param_names: list = PARAM_NAMES) -> None:
    """
    Save a multi-panel figure of 2-D marginalised chi² projections.

    For each pair of parameters (p1, p2), the chi² is marginalised by taking
    the **minimum** over all other dimensions. Contours at +10 %, +20 %, and
    +30 % above the global minimum are drawn.

    Parameters
    ----------
    chi2_grid : xr.DataArray
        Reduced chi² grid from :func:`compute_chi2_split`.
    best_params : dict
        Global minimum parameters (from :func:`find_best_parameters`).
    df_best : pandas.DataFrame
        Best-fit parameter table (first rows shown as annotation).
    filepath : str
        Output path prefix (without extension). ``.png`` is appended.
    param_names : list of str, optional
        Parameter dimension names. Default :data:`PARAM_NAMES`.
    """
    chi2_min = best_params["chi2"]
    levels   = [chi2_min * (1.0 + k * 0.10) for k in (1, 2, 3)]
    vmax     = 1.5 * chi2_min

    df  = chi2_grid.to_dataframe(name="chi2").reset_index()
    fig = plt.figure(figsize=(6 * 4, 5 * 3))
    axs = _make_heatmap_axes(fig)

    plot_axes = [ax for key, ax in axs.items() if key not in ("c", "t")]
    im_ref    = None

    for ax, (p1, p2) in zip(plot_axes, itertools.combinations(param_names, 2)):
        heatmap = df.pivot_table(
            index=p1, columns=p2, values="chi2", aggfunc="min"
        )
        x_vals = heatmap.columns.values
        y_vals = heatmap.index.values
        z      = heatmap.values
        extent = [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]

        im_ref = ax.imshow(
            z, origin="lower", aspect="auto", extent=extent,
            cmap="viridis", vmin=chi2_min, vmax=vmax,
        )
        ax.contour(z, origin="lower", levels=levels, extent=extent, cmap="Reds")
        ax.set_xlabel(p2)
        ax.set_ylabel(p1)
        ax.set_title(f"{p1} vs {p2}")

    if im_ref is not None:
        plt.colorbar(im_ref, cax=axs["c"], label=r"$\chi^2$")

    fig.suptitle(r"$\chi^2$ projections over all parameter pairs")

    # Annotation panel
    ax_t = axs["t"]
    ax_t.set_axis_off()
    summary = (
        "  ".join(
            f"{k} = {v:.3g}"
            for k, v in best_params.items()
            if k != "chi2"
        )
        + f"  χ² = {chi2_min:.2f}"
    )
    for y_pos, content in zip(
        (0.90, 0.72, 0.52, 0.25),
        ("Best-fit parameters", summary,
         f"Within 30 % of minimum ({len(df_best)} solutions)",
         df_best.head().to_string(index=False)),
    ):
        ax_t.text(
            0.5, y_pos, str(content),
            ha="center", va="center",
            transform=ax_t.transAxes, fontsize=11,
        )

    plt.tight_layout()
    out_path = filepath + "_chi2.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot_chi2_projections] Saved: {out_path}")


def plot_chi2_projections_log(chi2_grid: xr.DataArray,
                               param_names: list = PARAM_NAMES,
                               out_dir: str = ".") -> None:
    """
    Save individual 2-D chi² projection figures using a **log colour scale**.

    Useful when the chi² spans several orders of magnitude, e.g. when
    exploring a wide coarse grid where many parameter combinations are far
    from the minimum.

    One PNG file is written per parameter pair, named
    ``chi2_log_{p1}_{p2}.png``.

    Parameters
    ----------
    chi2_grid : xr.DataArray
        Reduced chi² grid.
    param_names : list of str, optional
        Parameter dimension names. Default :data:`PARAM_NAMES`.
    out_dir : str, optional
        Output directory. Default ``'.'``.
    """
    chi2_min = float(chi2_grid.min())
    df = chi2_grid.to_dataframe(name="chi2").reset_index()
    df["chi2"] = df["chi2"].clip(lower=1e-12)

    norm = mcolors.LogNorm(vmin=chi2_min, vmax=1e3 * chi2_min)

    for p1, p2 in itertools.combinations(param_names, 2):
        heatmap = df.pivot_table(
            index=p1, columns=p2, values="chi2", aggfunc="min"
        )
        x_vals = heatmap.columns.values
        y_vals = heatmap.index.values
        z      = heatmap.values
        extent = [x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            z, origin="lower", aspect="auto", extent=extent,
            cmap="gray", norm=norm,
        )
        plt.colorbar(im, ax=ax, label=r"$\chi^2$ (log scale)")
        ax.set_xlabel(p2)
        ax.set_ylabel(p1)
        ax.set_title(rf"$\chi^2$ projection: {p1} vs {p2} (log)")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"chi2_log_{p1}_{p2}.png")
        fig.savefig(out_path, dpi=150)
        plt.close()
        print(f"[plot_chi2_projections_log] Saved: {out_path}")
