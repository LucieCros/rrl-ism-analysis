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
import numpy as np
import pandas as pd
import xarray as xr

from spectral_tools.atoms import line_freq
from spectral_tools.tools import v_to_f

# ---------------------------------------------------------------------------
# Matplotlib style (publication quality)
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    "mathtext.fontset":  "stix",
    "font.family":       "serif",
    "axes.linewidth":    1.5,
    "lines.linewidth":   1.7,
    "font.size":         20,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  10,
    "ytick.major.size":  10,
    "xtick.minor.size":   7,
    "ytick.minor.size":   7,
    "xtick.major.width":  1,
    "ytick.major.width":  1,
    "xtick.minor.width":  1,
    "ytick.minor.width":  1,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default names of the five physical parameters.
PARAM_NAMES: list[str] = ["Te", "Ne", "T0", "L", "vt"]

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


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

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
