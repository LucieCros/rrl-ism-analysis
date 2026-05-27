# -*- coding: utf-8 -*-
"""
graphics.py
===========
Matplotlib plotting utilities for RRL spectral data visualisation.

This module provides:

- :func:`set_axes`     : apply publication-quality tick styling to any Axes
- :func:`plot_lines`   : draw vertical markers at RRL positions
- :func:`color_fader`  : linearly interpolate between two colours
- :func:`plot_overview`: 4-panel overview of a raw observation bloc
- :func:`plot_subband` : 9-panel diagnostic plot of one sub-band reduction

All functions follow a **non-interactive** pattern: figures are saved to
disk and closed immediately. No ``plt.show()`` is called, making this
module safe to use in pipeline scripts and batch jobs.

Dependencies
------------
matplotlib >= 3.5, numpy
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

#: Default colour cycle used across multi-panel diagnostic plots.
#: Ordered from warm (step 0) to cool (step 6) for visual contrast.
COLORS: list[str] = [
    "tomato",        # 0 — raw / input
    "darkorchid",    # 1 — reference / after 0th-order mitigation
    "darkblue",      # 2 — after 1st-order flattening
    "cornflowerblue",# 3 — after 1st-order RFI mitigation
    "lightblue",     # 4 — after 2nd-order flattening
    "darkgreen",     # 5 — after 3rd-order mitigation
    "olive",         # 6 — final cleaned spectrum
]


# ---------------------------------------------------------------------------
# Axes styling
# ---------------------------------------------------------------------------

def set_axes(ax: plt.Axes,
             x_minor_ticks: int = 10,
             y_minor_ticks: int = 5,
             major_len: float = 5.0,
             minor_len: float = 2.0) -> None:
    """
    Apply publication-quality tick styling to a Matplotlib Axes.

    Sets inward-pointing ticks with configurable minor subdivisions on both
    axes. Mirrors the style used throughout the NenuFAR spectral pipeline
    figures.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes object to style.
    x_minor_ticks : int, optional
        Number of minor tick subdivisions on the x-axis. Default 10.
    y_minor_ticks : int, optional
        Number of minor tick subdivisions on the y-axis. Default 5.
    major_len : float, optional
        Length of major ticks in points. Default 5.
    minor_len : float, optional
        Length of minor ticks in points. Default 2.
    """
    ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor_ticks))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor_ticks))
    ax.tick_params(axis="x", which="major", length=major_len)
    ax.tick_params(axis="x", which="minor", length=minor_len)
    ax.tick_params(axis="y", which="major", length=major_len)
    ax.tick_params(axis="y", which="minor", length=minor_len)
    ax.tick_params(axis="both", which="both", direction="in")


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def color_fader(c1: str, c2: str, mix: float = 0.0) -> str:
    """
    Linearly interpolate between two Matplotlib colours.

    Parameters
    ----------
    c1 : str
        Start colour (``mix = 0``), any Matplotlib colour string.
    c2 : str
        End colour (``mix = 1``), any Matplotlib colour string.
    mix : float, optional
        Interpolation factor in [0, 1]. Default 0 (returns ``c1``).

    Returns
    -------
    str
        Hex colour string of the interpolated colour.

    Examples
    --------
    >>> color_fader("black", "white", 0.5)
    '#7f7f7f'
    """
    rgb1 = np.array(mpl.colors.to_rgb(c1))
    rgb2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1.0 - mix) * rgb1 + mix * rgb2)


# ---------------------------------------------------------------------------
# RRL line markers
# ---------------------------------------------------------------------------

def plot_lines(lines, ax: plt.Axes, arr_ref: np.ndarray,
               n_transitions: list = [],
               n_sigma: float = 3.0,
               color: str = "black",
               linewidth: float = 0.5) -> None:
    """
    Draw vertical markers at known RRL positions on a spectrum axes.

    The vertical extent of each marker is set to ±``n_sigma`` standard
    deviations around the mean of ``arr_ref``, so markers scale
    automatically with the data range.

    Parameters
    ----------
    lines : array-like
        Frequencies of the lines to mark [MHz].
    ax : matplotlib.axes.Axes
        Target axes.
    arr_ref : numpy.ndarray
        Reference spectrum used to set the marker height.
    n_transitions : list of int, optional
        Quantum numbers to annotate above each marker.
        If provided, one label is drawn per ``len(lines) // len(n_transitions)``
        lines. Default [] (no labels).
    n_sigma : float, optional
        Half-height of markers in units of ``std(arr_ref)``. Default 3.
    color : str, optional
        Marker colour. Default ``'black'``.
    linewidth : float, optional
        Marker line width. Default 0.5.
    """
    mean  = np.nanmean(arr_ref)
    sigma = np.nanstd(arr_ref)
    ymin  = mean - n_sigma * sigma
    ymax  = mean + n_sigma * sigma

    for i, freq in enumerate(lines):
        ax.vlines(freq, ymin, ymax, color=color, linewidth=linewidth)
        if n_transitions:
            nvelo = len(lines) // len(n_transitions)
            ax.text(freq + 9e-4, ymax, str(n_transitions[i // nvelo]),
                    fontsize=7)


# ---------------------------------------------------------------------------
# Overview plot (raw observation)
# ---------------------------------------------------------------------------

def plot_overview(DATA: np.ndarray, NDATA: np.ndarray,
                  freq: np.ndarray, integ: np.ndarray,
                  flat: np.ndarray, savename: str,
                  cmap: str = "Spectral",
                  polluted_sub: list = [],
                  n_chan: int = 1024) -> None:
    """
    Save a 4-panel overview figure for a raw observation bloc.

    Panels (top to bottom):

    1. ``DATA`` waterfall (time × frequency, colour-coded intensity)
    2. ``NDATA`` waterfall (normalisation / weights, grayscale)
    3. Time-integrated spectrum (arbitrary units)
    4. Flattened spectrum (optical depth), with polluted sub-bands in black

    Parameters
    ----------
    DATA : numpy.ndarray, shape (n_time, n_freq)
        Raw calibrated data array.
    NDATA : numpy.ndarray, shape (n_time, n_freq)
        Normalisation (weight) array.
    freq : numpy.ndarray
        Frequency axis [MHz].
    integ : numpy.ndarray
        Time-integrated spectrum.
    flat : numpy.ndarray
        Flattened (baseline-subtracted) spectrum.
    savename : str
        Full path of the output PNG file.
    cmap : str, optional
        Colormap for the DATA waterfall. Default ``'Spectral'``.
    polluted_sub : list of int, optional
        Sub-band indices to overplot in black on panel 4. Default [].
    n_chan : int, optional
        Number of channels per sub-band (used for polluted sub-band slicing).
        Default 1024.
    """
    df   = freq[1] - freq[0]
    fmin = freq[0]

    fig, axs2D = plt.subplots(
        4, 2,
        figsize=(10, 15),
        gridspec_kw={"width_ratios": (40, 1), "height_ratios": (5, 5, 3, 3)},
    )
    axs = axs2D[:, 0]

    # Panel 1 — DATA waterfall
    im0 = axs[0].imshow(DATA.T, aspect="auto", cmap=cmap)
    fig.colorbar(im0, cax=axs2D[0, 1])
    set_axes(axs[0])
    ydisplay = np.array(axs[0].get_yticks() // 2, dtype=int)
    axs[0].set_yticklabels(ydisplay)
    axs[0].set_xticklabels([])
    axs[0].set_ylabel("Time [min]")
    axs[0].set_xlabel("Frequency [MHz]")

    # Panel 2 — NDATA waterfall (weights)
    im1 = axs[1].imshow(NDATA.T, aspect="auto", cmap="gray")
    fig.colorbar(im1, cax=axs2D[1, 1])
    set_axes(axs[1])
    axs[1].set_yticklabels(ydisplay)
    axs[1].set_ylabel("Time [min]")
    axs[1].set_xlabel("Frequency [MHz]")
    xdisplay = np.round(axs[1].get_xticks() * df + fmin, 1)
    axs[1].set_xticklabels(xdisplay)

    # Panel 3 — integrated spectrum
    axs[2].step(freq, integ, color="chocolate")
    set_axes(axs[2])
    axs[2].set_xlabel("Frequency [MHz]")
    axs[2].set_ylabel("Arbitrary unit")

    # Panel 4 — flattened spectrum
    axs[3].step(freq, flat, color="tomato")
    set_axes(axs[3])
    axs[3].set_xlabel("Frequency [MHz]")
    axs[3].set_ylabel("Optical depth")
    for k in polluted_sub:
        axs[3].step(freq[k * n_chan:(k + 1) * n_chan],
                    flat[k * n_chan:(k + 1) * n_chan],
                    color="black")

    # Hide unused colorbar axes
    axs2D[2, 1].axis("off")
    axs2D[3, 1].axis("off")

    fig.savefig(savename)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sub-band diagnostic plot
# ---------------------------------------------------------------------------

def plot_subband(FREF: np.ndarray,
                 raw: np.ndarray,
                 after_spike: np.ndarray,
                 neighbors_mean: np.ndarray,
                 neighbors_smooth: np.ndarray,
                 flat1: np.ndarray,
                 pre_rfi: np.ndarray,
                 post_rfi: np.ndarray,
                 baseline: np.ndarray,
                 pre_flat2: np.ndarray,
                 flat2: np.ndarray,
                 pre_intra: np.ndarray,
                 intra_filter: np.ndarray,
                 post_intra: np.ndarray,
                 pre_final: np.ndarray,
                 final: np.ndarray,
                 SUBS2: np.ndarray,
                 n_transitions: list,
                 indexes_lo: int,
                 indexes_hi: int,
                 lines: list,
                 savename: str) -> None:
    """
    Save a 9-panel diagnostic figure for one sub-band reduction step.

    Each panel shows the spectrum after a successive cleaning stage,
    allowing visual inspection of the full reduction sequence.

    Panel layout
    ------------
    0. Raw sub-band + after spike removal
    1. Neighbouring sub-bands (waterfall-style offset display)
    2. Mean of neighbours + smoothed mean
    3. After-spike spectrum + 1st-order flat
    4. 1st-order flat + after 1st RFI mitigation (edges in gray)
    5. After-spike spectrum + fitted baseline
    6. 2nd-order flat (before / after)
    7. Pre / post intra-line filter + intra-line filter curve
    8. Pre / post final sigma clipping

    Parameters
    ----------
    FREF : numpy.ndarray
        Frequency axis for this sub-band [MHz].
    raw : numpy.ndarray
        Panel 0 — raw sub-band spectrum.
    after_spike : numpy.ndarray
        Panel 0 — spectrum after 0th-order spike removal.
    neighbors_mean : numpy.ndarray
        Panel 2 — mean of neighbouring sub-bands.
    neighbors_smooth : numpy.ndarray
        Panel 2 — Savitzky-Golay smoothed mean.
    flat1 : numpy.ndarray
        Panel 3 — spectrum after 1st-order flattening.
    pre_rfi : numpy.ndarray
        Panel 4 — spectrum before 1st-order RFI mitigation.
    post_rfi : numpy.ndarray
        Panel 4 — spectrum after 1st-order RFI mitigation.
    baseline : numpy.ndarray
        Panel 5 — fitted baseline (total: 1st + 2nd order).
    pre_flat2 : numpy.ndarray
        Panel 6 — spectrum before 2nd-order flattening.
    flat2 : numpy.ndarray
        Panel 6 — spectrum after 2nd-order flattening.
    pre_intra : numpy.ndarray
        Panel 7 — spectrum before intra-line filter.
    intra_filter : numpy.ndarray
        Panel 7 — intra-line Savitzky-Golay filter curve.
    post_intra : numpy.ndarray
        Panel 7 — spectrum after intra-line filter.
    pre_final : numpy.ndarray
        Panel 8 — spectrum before final sigma clipping.
    final : numpy.ndarray
        Panel 8 — final cleaned spectrum.
    SUBS2 : numpy.ndarray, shape (11, n_chan)
        Panel 1 — stack of 11 neighbouring sub-bands (central = index 5).
    n_transitions : list of int
        Quantum numbers for RRL label annotation.
    indexes_lo : int
        Left edge channel index of the clean sub-band interior.
    indexes_hi : int
        Right edge channel index of the clean sub-band interior.
    lines : list of float
        RRL frequencies within this sub-band [MHz].
    savename : str
        Full path of the output PNG file.
    """
    c_dark  = "dimgray"
    c_light = "lightgray"

    fig, axs = plt.subplots(
        9, 1,
        figsize=(10, 9 * 3 + 0.5),
        gridspec_kw={"height_ratios": (2.5, 4, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5)},
    )

    # Panel 0 — raw + after spike removal
    axs[0].step(FREF, raw,         color=COLORS[0])
    axs[0].step(FREF, after_spike, color=COLORS[1])
    plot_lines(lines, axs[0], after_spike, n_transitions, n_sigma=1.5)
    set_axes(axs[0])

    # Panel 1 — neighbouring sub-bands (offset waterfall)
    axs[1].set_yticklabels([])
    for i in range(11):
        col = COLORS[1] if i == 5 else color_fader(c_dark, c_light, i / 11)
        axs[1].step(FREF, (i - 5) * 0.01 + SUBS2[i], color=col)
    set_axes(axs[1])

    # Panel 2 — mean of neighbours + smoothed mean
    axs[2].step(FREF, neighbors_mean,   color=color_fader(c_dark, c_light, 5 / 11))
    axs[2].plot(FREF, neighbors_smooth, "--", color="black")
    set_axes(axs[2])

    # Panel 3 — after-spike + 1st-order flat
    axs[3].step(FREF, after_spike, color=COLORS[1])
    axs[3].step(FREF, flat1,       color=COLORS[2])
    plot_lines(lines, axs[3], after_spike, n_transitions)
    set_axes(axs[3])

    # Panel 4 — before/after 1st-order RFI mitigation (edges in gray)
    axs[4].step(FREF, pre_rfi,  color=COLORS[2])
    axs[4].step(FREF, post_rfi, color=COLORS[3])
    axs[4].step(FREF[:indexes_lo],  post_rfi[:indexes_lo],  color="gray")
    axs[4].step(FREF[indexes_hi:],  post_rfi[indexes_hi:],  color="gray")
    plot_lines(lines, axs[4], pre_rfi, n_transitions, n_sigma=7)
    set_axes(axs[4])

    # Panel 5 — after-spike + fitted baseline
    axs[5].step(FREF, after_spike, color=COLORS[1])
    axs[5].plot(FREF, baseline,    "--", color="black")
    plot_lines(lines, axs[5], baseline, n_transitions)
    set_axes(axs[5])

    # Panel 6 — 2nd-order flat (before / after)
    axs[6].step(FREF, pre_flat2, color=COLORS[3])
    axs[6].step(FREF, flat2,     color=COLORS[4])
    plot_lines(lines, axs[6], pre_flat2, n_transitions, n_sigma=7)
    set_axes(axs[6])

    # Panel 7 — intra-line filter
    axs[7].step(FREF, pre_intra,    color=COLORS[4])
    axs[7].step(FREF, post_intra,   color=COLORS[5])
    axs[7].plot(FREF, intra_filter, "--", color="black")
    plot_lines(lines, axs[7], pre_intra, n_transitions, n_sigma=7)
    set_axes(axs[7])

    # Panel 8 — final sigma clipping
    axs[8].step(FREF, pre_final, color=COLORS[5])
    axs[8].step(FREF, final,     color=COLORS[6])
    axs[8].hlines(0, FREF[0], FREF[-1], color="k")
    plot_lines(lines, axs[8], pre_final, n_transitions, n_sigma=7)
    set_axes(axs[8])
    axs[8].set_xlabel("Frequency [MHz]")

    for i in range(9):
        if i != 1:
            axs[i].set_ylabel("Optical depth")

    fig.savefig(savename)
    plt.close(fig)
