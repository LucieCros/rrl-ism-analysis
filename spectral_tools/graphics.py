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
from astropy.coordinates import SkyCoord

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
# Style settings
# ---------------------------------------------------------------------------

def set_style(font_size: float = 15,
              label_size: float = 13,
              axes_linewidth: float = 2.5,
              lines_linewidth: float = 1.7,
              major_tick_size: float = 5.0,
              major_tick_width: float = 1.0,
              minor_tick_size: float = 4.5,
              minor_tick_width: float = 1.0,
              font_family: str = "serif",
              mathtext_fontset: str = "stix") -> None:
    """
    Apply the NenuFAR publication style to Matplotlib's global ``rcParams``.

    Sets font, axis line widths and tick appearance consistently across all
    subsequent figures in a session.  Call once at the top of a notebook or
    script, before creating any figure.

    All parameters have sensible defaults matching the style used throughout
    the NenuFAR spectral pipeline notebooks.  Override individual values as
    needed — for example, use ``font_size=17`` for larger slide figures.

    Parameters
    ----------
    font_size : float, optional
        Base font size for titles, labels and annotations [pt].  Default 15.
    label_size : float, optional
        Font size for tick labels [pt].  Default 13.
    axes_linewidth : float, optional
        Line width of the axes frame (spines) [pt].  Default 2.5.
    lines_linewidth : float, optional
        Default line width for plotted data lines [pt].  Default 1.7.
    major_tick_size : float, optional
        Length of major ticks [pt].  Default 5.0.
    major_tick_width : float, optional
        Width of major ticks [pt].  Default 1.0.
    minor_tick_size : float, optional
        Length of minor ticks [pt].  Default 4.5.
    minor_tick_width : float, optional
        Width of minor ticks [pt].  Default 1.0.
    font_family : str, optional
        Matplotlib font family string (e.g. ``'serif'``, ``'sans-serif'``).
        Default ``'serif'``.
    mathtext_fontset : str, optional
        Matplotlib mathtext font set (e.g. ``'stix'``, ``'cm'``).
        Default ``'stix'``.

    Examples
    --------
    Default style (used in most notebooks):

    >>> import spectral_tools.graphics as graphics
    >>> graphics.set_style()

    Larger fonts for presentation slides:

    >>> graphics.set_style(font_size=17, label_size=15)

    Sans-serif variant:

    >>> graphics.set_style(font_family="sans-serif", mathtext_fontset="cm")
    """
    mpl.rcParams["mathtext.fontset"] = mathtext_fontset
    mpl.rcParams["font.family"]      = font_family
    mpl.rcParams["font.size"]        = font_size

    mpl.rcParams["axes.linewidth"]   = axes_linewidth
    mpl.rcParams["lines.linewidth"]  = lines_linewidth

    for axis in ("xtick", "ytick"):
        mpl.rcParams[f"{axis}.direction"]   = "in"
        mpl.rcParams[f"{axis}.labelsize"]   = label_size
        mpl.rcParams[f"{axis}.major.size"]  = major_tick_size
        mpl.rcParams[f"{axis}.major.width"] = major_tick_width
        mpl.rcParams[f"{axis}.minor.size"]  = minor_tick_size
        mpl.rcParams[f"{axis}.minor.width"] = minor_tick_width

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


# ---------------------------------------------------------------------------
# Support functions for multi tracers plotting
# ---------------------------------------------------------------------------

def plot_spectrum_with_range(ax: plt.Axes,
                             vel: np.ndarray,
                             spectrum: np.ndarray,
                             vmin: float,
                             vmax: float,
                             color: str = "blue",
                             label: str | None = None,
                             normalize: bool = True,
                             xlim: tuple = (-20, 30),
                             **set_axes_kwargs) -> None:
    """
    Plot a normalised PPV spectrum and highlight a selected velocity range.
 
    The full spectrum is drawn in gray; the selected velocity interval
    ``[vmin, vmax]`` is overdrawn in ``color`` with dotted boundary markers.
    Intended for HI and CO spectra extracted from PPV cubes.
 
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    vel : numpy.ndarray
        Velocity axis [km/s].
    spectrum : numpy.ndarray
        Brightness-temperature spectrum [K] (or any consistent unit).
    vmin, vmax : float
        Lower and upper velocity bounds to highlight [km/s].
    color : str, optional
        Colour used for the highlighted range and boundary markers.
        Default ``'blue'``.
    label : str or None, optional
        Legend label for the highlighted segment.  If ``None``, a default
        ``'{vmin}–{vmax} km/s'`` label is used.
    normalize : bool, optional
        If ``True`` (default), divide the spectrum by its maximum before
        plotting so the y-axis is in relative units [0–1].
    xlim : tuple of float, optional
        x-axis limits [km/s].  Default ``(-20, 30)``.
    **set_axes_kwargs
        Extra keyword arguments forwarded to :func:`set_axes`
        (e.g. ``pad=5``, ``xgraduation=5``).
 
    Examples
    --------
    >>> plot_spectrum_with_range(ax, velHI, specHI, vmin=-5, vmax=12,
    ...                          color='steelblue', label='cloud A')
    """
    norm = np.nanmax(spectrum) if normalize else 1.0
    spec_n = spectrum / norm
    cond = (vel >= vmin) & (vel <= vmax)
    lbl = label if label is not None else f"{vmin}–{vmax} km/s"
 
    ax.plot(vel, spec_n, c="gray", zorder=1)
    ax.plot(vel[cond], spec_n[cond], c=color, label=lbl, zorder=2)
    ax.axvline(vmin, 0.05, 0.95, color=color, linestyle="dotted", linewidth=1)
    ax.axvline(vmax, 0.05, 0.95, color=color, linestyle="dotted", linewidth=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Relative intensity")
    set_axes(ax, **set_axes_kwargs)
 
 
def plot_dust_profile(ax: plt.Axes,
                      radii: np.ndarray,
                      extinc: np.ndarray,
                      cloud_distances_pc: np.ndarray | None = None,
                      cloud_path_lengths_pc: np.ndarray | None = None,
                      normalize: bool = True,
                      color: str = "gray",
                      cloud_color: str = "orange",
                      **set_axes_kwargs) -> None:
    """
    Plot a 1-D dust extinction profile as a function of heliocentric distance.
 
    Optionally overlay shaded bands that mark the extent of detected ISM
    clouds (from :meth:`~spectral_tools.maps.DustMap.cloud_path_lengths`).
 
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    radii : numpy.ndarray
        Distance axis [kpc], shape ``(n_r,)``.
    extinc : numpy.ndarray
        Extinction profile [arbitrary], shape ``(n_r,)``.
    cloud_distances_pc : numpy.ndarray or None, optional
        Central distances of detected clouds [pc].  If provided together
        with ``cloud_path_lengths_pc``, a shaded band and a dashed vertical
        line are drawn for each cloud.
    cloud_path_lengths_pc : numpy.ndarray or None, optional
        FWHM path lengths of detected clouds [pc].  Must have the same
        length as ``cloud_distances_pc``.
    normalize : bool, optional
        Divide the extinction by its maximum before plotting.  Default True.
    color : str, optional
        Line colour for the extinction profile.  Default ``'gray'``.
    cloud_color : str, optional
        Fill colour for the cloud FWHM bands.  Default ``'orange'``.
    **set_axes_kwargs
        Extra keyword arguments forwarded to :func:`set_axes`.
 
    Examples
    --------
    >>> plot_dust_profile(ax, radiitot, extinc,
    ...                   cloud_distances_pc=peak_dist,
    ...                   cloud_path_lengths_pc=path_lengths)
    """
    norm = np.nanmax(extinc) if normalize else 1.0
    ax.plot(radii, extinc / norm, c=color)
 
    if cloud_distances_pc is not None and cloud_path_lengths_pc is not None:
        for d_pc, pl_pc in zip(cloud_distances_pc, cloud_path_lengths_pc):
            d_kpc      = d_pc  / 1000.0
            half_kpc   = pl_pc / 2000.0
            ax.axvspan(d_kpc - half_kpc, d_kpc + half_kpc,
                       alpha=0.25, color=cloud_color, zorder=0)
            ax.axvline(d_kpc, color=cloud_color, linestyle="--",
                       linewidth=1, zorder=1)
 
    ax.set_xlabel("Distance from the Sun (kpc)")
    ax.set_ylabel("Relative absorption")
    ax.set_title("Spatial distribution of dust")
    ax.set_ylim(-0.1, 1.1)
    set_axes(ax, **set_axes_kwargs)
 
 
def overlay_positions(ax: plt.Axes,
                      coord_source: SkyCoord,
                      coords_off: list[SkyCoord],
                      colors: list[str],
                      source_label: str = "Source",
                      off_labels: list[str] | None = None,
                      source_marker: str = "*",
                      off_marker: str = ".",
                      source_ms: float = 12,
                      off_ms: float = 10) -> None:
    """
    Overlay a primary source and a list of offset positions on a sky map.
 
    Positions are plotted in Galactic coordinates (longitude, latitude).
    The primary source is drawn as a star; offsets as filled circles.
 
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes, assumed to have Galactic longitude on x and latitude
        on y (consistent with :func:`imshow` extent in Galactic frame).
    coord_source : astropy.coordinates.SkyCoord
        Primary source position.
    coords_off : list of astropy.coordinates.SkyCoord
        Offset positions, one per entry.
    colors : list of str
        Colours for each offset position.  Must have the same length as
        ``coords_off``.
    source_label : str, optional
        Legend label for the primary source.  Default ``'Source'``.
    off_labels : list of str or None, optional
        Legend labels for the offset positions.  If ``None``, labels are
        generated automatically as ``'off_1'``, ``'off_2'``, …
    source_marker : str, optional
        Matplotlib marker for the primary source.  Default ``'*'``.
    off_marker : str, optional
        Matplotlib marker for offset positions.  Default ``'.'``.
    source_ms : float, optional
        Marker size for the primary source.  Default 12.
    off_ms : float, optional
        Marker size for offset positions.  Default 10.
 
    Examples
    --------
    >>> overlay_positions(axmap, CoordSource, CoordsOff, colors,
    ...                   source_label="Tau A")
    """
    if off_labels is None:
        off_labels = [f"off_{i + 1}" for i in range(len(coords_off))]
 
    ax.plot(coord_source.galactic.l.value,
            coord_source.galactic.b.value,
            marker=source_marker, c="black", ms=source_ms,
            label=source_label, zorder=5)
 
    for coord, col, lbl in zip(coords_off, colors, off_labels):
        ax.plot(coord.galactic.l.value,
                coord.galactic.b.value,
                marker=off_marker, c=col, ms=off_ms,
                label=lbl, zorder=4)
 
 
def annotate_spectral_peaks(ax: plt.Axes,
                             vel: np.ndarray,
                             spectrum: np.ndarray,
                             color: str = "black",
                             vel_min: float = -40.0,
                             vel_max: float = 40.0,
                             text_offset_pixels: tuple = (-150, -50),
                             arrow_kwargs: dict | None = None) -> np.ndarray:
    """
    Auto-detect and annotate spectral peaks in a velocity range.
 
    Uses :func:`scipy.signal.find_peaks` to locate peaks in the window
    ``[vel_min, vel_max]`` and draws an annotated arrow for each one.
    Intended for HI or CO spectra displayed in offset-position panels.
 
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (the spectrum must already be plotted on it).
    vel : numpy.ndarray
        Velocity axis [km/s].
    spectrum : numpy.ndarray
        Spectrum values [K or relative].
    color : str, optional
        Arrow and text colour.  Default ``'black'``.
    vel_min, vel_max : float, optional
        Velocity window used for peak detection [km/s].
        Default ``-40`` and ``40``.
    text_offset_pixels : tuple of (int, int), optional
        ``(dx, dy)`` pixel offset of the annotation text relative to the
        peak.  Default ``(-150, -50)``.
    arrow_kwargs : dict or None, optional
        Extra keyword arguments merged into the ``arrowprops`` dict passed
        to :func:`~matplotlib.axes.Axes.annotate`.  Defaults:
        ``{'width': 1, 'headwidth': 5, 'headlength': 5}``.
 
    Returns
    -------
    peak_velocities : numpy.ndarray
        Velocities of the detected peaks [km/s].
 
    Examples
    --------
    >>> annotate_spectral_peaks(ax, velHI, specHI, color='steelblue',
    ...                          vel_min=-20, vel_max=20)
    """
    from scipy.signal import find_peaks
 
    default_arrow = {"width": 1, "headwidth": 5, "headlength": 5}
    if arrow_kwargs is not None:
        default_arrow.update(arrow_kwargs)
    default_arrow["color"] = color
 
    cond = (vel >= vel_min) & (vel <= vel_max)
    peak_idx = find_peaks(spectrum[cond])[0]
 
    for km in peak_idx:
        xm = vel[cond][km]
        ym = spectrum[cond][km]
        ax.annotate(
            f"{xm:.0f} km/s",
            xy=(xm, ym),
            xytext=text_offset_pixels,
            textcoords="offset pixels",
            color=color,
            arrowprops=default_arrow,
        )
 
    return vel[cond][peak_idx]
 
