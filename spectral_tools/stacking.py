"""
spectral_tools/stacking.py
===========================
Weighted stacking utilities for NenuFAR Radio Recombination Lines (RRLs).

This module provides:

extract_weighted_lines   — slice transitions + compute 1/rms² weights
compute_line_snr         — signal-to-noise ratio of a single line profile
build_quantum_intervals  — define (n_min, n_max) stacking intervals
weighted_stack           — combine lines into stacked spectra

Dependencies
----------
numpy, tqdm

"""

import warnings

import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
def compute_line_snr(
    line: np.ndarray,
    width: int,
) -> tuple[float, float]:
    """
    Estimate the signal-to-noise ratio of an extracted line profile.

    The estimators follow the convention used throughout the stacking
    notebooks:

    * **Signal** — ``nanmean`` of the *central third* of the window,
      i.e. channels ``[width//3 : 2*width//3]``.
    * **Noise**  — ``nanstd`` of the *wings*, defined as the outer
      portion of the window with the central half masked
      (channels ``[width//6 : 5*width//6]`` set to ``NaN``).
    * **Sign convention** — the ratio is returned as ``-signal / noise``
      so that *absorption* lines (negative intensity) yield a *positive*
      S/N value.

    Parameters
    ----------
    line : np.ndarray
        Extracted line profile of length ``width``.
        May contain ``NaN`` for blanked channels.
    width : int
        Total window width in channels.

    Returns
    -------
    snr : float
        Signal-to-noise ratio (positive for absorption lines).
        Returns ``np.nan`` if noise is zero or if the entire window
        is blank.
    noise : float
        Estimated RMS noise measured on the wings.
        Returns ``np.nan`` if the wings are entirely blank.

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_tools.stacking import compute_line_snr
    >>> rng = np.random.default_rng(0)
    >>> line = rng.normal(0, 0.01, 1000)
    >>> line[400:600] -= 0.05          # inject an absorption dip
    >>> snr, noise = compute_line_snr(line, width=1000)
    >>> print(f"S/N = {snr:.1f},  noise = {noise:.4f}")
    """
    # ── Signal: mean of the central third ─────────────────────────────────
    signal = np.nanmean(line[width // 3 : 2 * width // 3])

    # ── Noise: std of the wings (central part masked) ─────────────────────
    wings = np.copy(line)
    wings[width // 6 : 5 * width // 6] = np.nan
    noise = np.nanstd(wings)

    if noise == 0 or not np.isfinite(noise):
        return np.nan, np.nan

    # Negative sign: absorption lines have negative intensity
    return float(-signal / noise), float(noise)


# ─────────────────────────────────────────────────────────────────────────────
def extract_weighted_lines(
    f: np.ndarray,
    I: np.ndarray,
    RMS: np.ndarray,
    Lines,
    width: int,
    lowest_n: int,
    highest_n: int,
    og: float = 1e4,
    cap_snr: bool = True,
    max_snr: float = 6.0,
    nan_fraction_threshold: float = 0.4,
    n_min_band: int = 426,
    n_max_band: int = 850,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract individual RRL transitions and compute their inverse-variance
    weights.

    For each quantum number ``n`` in
    ``[max(lowest_n, n_min_band), min(highest_n, n_max_band))``,
    the function:

    1. Looks up the expected line frequency from *Lines*.
    2. Skips lines outside the observed frequency band ``[f[0], f[-1]]``.
    3. Slices a window of ``width`` channels centred on the line.
    4. Blanks channels with ``NaN`` or ``|amplitude| > 100 / og``.
    5. **Rejects** lines where more than ``nan_fraction_threshold`` of
       channels are blank (sets profile to ``NaN``, weight to ``NaN``).
    6. Computes the weight as ``1 / rms²``; if ``cap_snr=True`` the rms
       is floored so that no line's effective S/N exceeds ``max_snr``,
       preventing a single bright line from monopolising the stack.

    Parameters
    ----------
    f : np.ndarray, shape (N,)
        Frequency axis of the full spectrum [GHz].
    I : np.ndarray, shape (N,)
        Intensity (or optical-depth) spectrum.
    RMS : np.ndarray, shape (N,)
        Per-channel RMS noise array.
    Lines : pandas.Series
        Line-frequency table indexed by quantum number ``n`` [GHz].
        Typically obtained from ``tools.get_line(...)``.
    width : int
        Extraction window width in channels (centred on the line).
        Must be even.
    lowest_n, highest_n : int
        Quantum number range available in the *Lines* table.
    og : float, optional
        Optical-depth gain factor.  Raw amplitudes are compared to
        ``100 / og`` to flag outlier channels.  Default ``1e4``.
    cap_snr : bool, optional
        If ``True``, cap the effective S/N of each line to ``max_snr``
        before computing the weight.  Default ``True``.
    max_snr : float, optional
        Maximum allowed effective S/N per line (only used when
        ``cap_snr=True``).  Default ``6.0``.
    nan_fraction_threshold : float, optional
        Lines with a fraction of blank channels above this threshold are
        rejected entirely.  Default ``0.4`` (40 %).
    n_min_band : int, optional
        Hardware lower limit on the quantum number for this frequency
        band (NenuFAR lane 0: 426).  Default ``426``.
    n_max_band : int, optional
        Hardware upper limit.  Default ``850``.
    verbose : bool, optional
        If ``True``, print a summary at the end.  Default ``True``.

    Returns
    -------
    isolated_lines : np.ndarray, shape (M, width)
        Extracted line profiles (dtype object; ``NaN`` where blanked).
        ``M = min(highest_n, n_max_band) - max(lowest_n, n_min_band)``.
    weights : np.ndarray, shape (M,)
        Inverse-variance weights (``NaN`` for rejected lines).
    freq_window : np.ndarray, shape (width,)
        Reference frequency axis centred on zero [Hz], shared by all
        profiles.  ``freq_window[i] = (i - width/2) * df`` where ``df``
        is the channel width.
    snr_all : np.ndarray, shape (M,)
        S/N value of each extracted line (computed via
        :func:`compute_line_snr`).  ``NaN`` for rejected / out-of-band
        lines.

    Raises
    ------
    ValueError
        If ``width`` is not a positive even integer.

    Examples
    --------
    >>> from astropy.io import fits
    >>> from spectral_tools.io import freq_axis_from_fits
    >>> import spectral_tools.tools as tools
    >>> from spectral_tools.stacking import extract_weighted_lines
    >>>
    >>> hdu   = fits.open("alltime_TAUA_CLOUDS_Calph_OFF.fits")
    >>> f     = freq_axis_from_fits(hdu)
    >>> I     = hdu[0].data[:, 0, 0]
    >>> RMS   = hdu[2].data[:, 0, 0]
    >>> Lines = tools.get_line(hdu[0].header["FMIN"], 47.56, 0,
    ...                        line="Calph", path="rrlines.csv")
    >>> lines, weights, fref, snrs = extract_weighted_lines(
    ...     f, I, RMS, Lines, width=1000,
    ...     lowest_n=int(Lines.index.min()),
    ...     highest_n=int(Lines.index.max()),
    ... )
    """
    if width <= 0 or width % 2 != 0:
        raise ValueError(f"width must be a positive even integer, got {width}.")

    # Channel width [Hz] — used to build the reference frequency window
    df = float(np.abs(f[1] - f[0]))

    # Reference frequency window centred on zero [Hz]
    freq_window = np.linspace(-df * (width // 2), df * (width // 2), width)

    n_start = max(lowest_n, n_min_band)
    n_stop  = min(highest_n, n_max_band)

    isolated_lines = []
    weights_list   = []
    snr_list       = []

    n_extracted = 0
    n_rejected  = 0
    n_outofband = 0

    for n in tqdm(range(n_start, n_stop), desc="Extracting lines", disable=not verbose):
        f0 = Lines.loc[n]

        # ── Out-of-band: line outside the observed frequency range ─────────
        if f0 < f[0] or f0 > f[-1]:
            isolated_lines.append(np.full(width, np.nan))
            weights_list.append(np.nan)
            snr_list.append(np.nan)
            n_outofband += 1
            continue

        # ── Slice a window centred on the line ─────────────────────────────
        # tools.slice_line returns an array of length `width`
        from spectral_tools import tools as _tools
        slicedline = _tools.slice_line(f0, I, f, cut_width=width // 2)

        # ── Blank outlier channels ──────────────────────────────────────────
        slicedline[np.isnan(slicedline)]           = 0
        slicedline[np.abs(slicedline) > 100 / og]  = 0

        # ── Reject lines with too many blank channels ───────────────────────
        blank_fraction = np.sum(slicedline == 0) / width
        if blank_fraction > nan_fraction_threshold:
            isolated_lines.append(np.full(width, np.nan))
            weights_list.append(np.nan)
            snr_list.append(np.nan)
            n_rejected += 1
            continue

        # Replace remaining zeros with NaN so they are ignored in nanmean
        slicedline[slicedline == 0] = np.nan
        isolated_lines.append(slicedline)
        n_extracted += 1

        # ── S/N (stored but also used later for rejection) ──────────────────
        snr, _ = compute_line_snr(slicedline, width)
        snr_list.append(snr)

        # ── Weight = 1 / rms² with optional S/N cap ────────────────────────
        rms_slice = np.nanmean(
            _tools.slice_line(f0, RMS, f, cut_width=width // 2)
        )
        if cap_snr and np.isfinite(snr):
            # Floor the rms so that effective S/N ≤ max_snr
            signal_min = np.nanmin(slicedline)
            rms_floor  = signal_min / max_snr
            rms_slice  = max(rms_slice, rms_floor)

        weights_list.append(1.0 / rms_slice**2)

    if verbose:
        print(
            f"Extraction complete: "
            f"{n_extracted} accepted, "
            f"{n_rejected} rejected (>{100*nan_fraction_threshold:.0f}% blank), "
            f"{n_outofband} out-of-band."
        )

    isolated_lines = np.array(isolated_lines, dtype=object)
    weights        = np.array(weights_list,   dtype=float)
    snr_all        = np.array(snr_list,       dtype=float)

    return isolated_lines, weights, freq_window, snr_all


# ─────────────────────────────────────────────────────────────────────────────
def build_quantum_intervals(
    lowest_n: int,
    highest_n: int,
    qi: list[int],
    n0: int,
    n1: int,
) -> np.ndarray:
    """
    Build the boundaries of stacking intervals with three distinct step sizes.

    The quantum number range ``[lowest_n, highest_n]`` is divided into
    three zones with different bin widths:

    * ``n  < n0`` — coarse bins of width ``qi[0]`` (strong, low-n lines)
    * ``n0 ≤ n ≤ n1`` — fine bins of width ``qi[1]`` (best-S/N mid-n range)
    * ``n  > n1`` — coarse bins of width ``qi[2]`` (weak, high-n lines)

    Parameters
    ----------
    lowest_n, highest_n : int
        Quantum number range to cover (inclusive).
    qi : list of int, length 3
        Step sizes ``[step_low, step_mid, step_high]``.
    n0, n1 : int
        Transition quantum numbers between the three zones.

    Returns
    -------
    np.ndarray of int
        Sorted array of interval boundaries of length ``nstacks + 1``.
        The k-th stacking interval covers
        ``[boundaries[k], boundaries[k+1])``.

    Raises
    ------
    ValueError
        If ``len(qi) != 3`` or if ``lowest_n >= highest_n``.

    Examples
    --------
    >>> from spectral_tools.stacking import build_quantum_intervals
    >>> bounds = build_quantum_intervals(
    ...     lowest_n=517, highest_n=837,
    ...     qi=[40, 20, 50], n0=500, n1=730
    ... )
    >>> print(bounds)
    [517 557 ... 837]
    >>> print(f"{len(bounds)-1} stacks")
    """
    if len(qi) != 3:
        raise ValueError(f"qi must have exactly 3 elements, got {len(qi)}.")
    if lowest_n >= highest_n:
        raise ValueError(
            f"lowest_n ({lowest_n}) must be strictly less than highest_n ({highest_n})."
        )

    q0 = [int(x) for x in range(lowest_n,   n0 + qi[0], qi[0])]
    q1 = [int(x) for x in range(q0.pop(-1), n1 + qi[1], qi[1])]
    q2 = [int(x) for x in range(q1.pop(-1), highest_n + 1, qi[2])]
    return np.array(q0 + q1 + q2, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
def weighted_stack(
    isolated_lines: np.ndarray,
    weights: np.ndarray,
    snr_all: np.ndarray,
    quantum_intervals: np.ndarray,
    lowest_n: int,
    width: int,
    snr_min: float = 0.0,
    snr_max: float = 0.8,
    verbose: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """
    Combine extracted RRL transitions into weighted-average stacks.

    Within each quantum-number interval ``[quantum_intervals[k],
    quantum_intervals[k+1])``, lines are combined using their
    ``1/σ²`` weights.  A line is **excluded** from its stack by setting
    its weight to ``NaN`` if its S/N falls outside
    ``[snr_min, snr_max]``:

    * ``snr < snr_min`` → likely emission-contaminated
    * ``snr > snr_max`` → too faint / undetected

    The normalised weighted average is then::

        stack[k] = Σ_i (w_i / Σ_j w_j) * profile_i

    where the sums ignore ``NaN`` entries.

    Parameters
    ----------
    isolated_lines : np.ndarray, shape (M, width)
        Extracted line profiles as returned by
        :func:`extract_weighted_lines`.
    weights : np.ndarray, shape (M,)
        Inverse-variance weights (``NaN`` for rejected lines).
    snr_all : np.ndarray, shape (M,)
        S/N of each line as returned by :func:`extract_weighted_lines`.
    quantum_intervals : np.ndarray of int, shape (K+1,)
        Interval boundaries as returned by
        :func:`build_quantum_intervals`.
    lowest_n : int
        Quantum number corresponding to index 0 in *isolated_lines*
        (i.e. ``isolated_lines[0]`` belongs to quantum number
        ``lowest_n``).
    width : int
        Number of channels in each line profile.
    snr_min : float, optional
        Lower S/N threshold.  Lines below this are excluded.
        Default ``0.0``.
    snr_max : float, optional
        Upper S/N threshold.  Lines above this are excluded.
        Default ``0.8``.
    verbose : bool, optional
        If ``True``, print a per-interval summary table.  Default ``True``.

    Returns
    -------
    stacks : np.ndarray, shape (K, width)
        Stacked spectra, one per quantum-number interval.
        Intervals with no valid lines contain a zero profile.
    n_used : list of int, length K
        Number of lines that contributed to each stack
        (after S/N rejection).

    Examples
    --------
    >>> from spectral_tools.stacking import (
    ...     extract_weighted_lines, build_quantum_intervals, weighted_stack
    ... )
    >>> # ... (after running extract_weighted_lines) ...
    >>> qi = build_quantum_intervals(lowest_n, highest_n,
    ...                              qi=[40, 20, 50], n0=500, n1=730)
    >>> stacks, n_used = weighted_stack(
    ...     isolated_lines, weights, snr_all,
    ...     quantum_intervals=qi, lowest_n=lowest_n, width=1000,
    ...     snr_min=0.0, snr_max=0.8,
    ... )
    """
    nstacks = len(quantum_intervals) - 1
    stacks  = []
    n_used  = []

    for k in tqdm(range(nstacks), desc="Stacking", disable=not verbose):
        N0_k = quantum_intervals[k]
        N_k  = quantum_intervals[k + 1]

        # ── Map global quantum numbers to local array indices ──────────────
        idx_start = N0_k - lowest_n
        idx_end   = N_k  - lowest_n

        w     = weights[idx_start:idx_end].copy().astype(float)
        snrs  = snr_all[idx_start:idx_end]
        block = np.array(list(isolated_lines[idx_start:idx_end]),
                         dtype=float)   # shape (interval_width, width)

        # ── Reject lines whose S/N falls outside [snr_min, snr_max] ────────
        for local_idx, snr in enumerate(snrs):
            if not np.isfinite(snr) or snr < snr_min or snr > snr_max:
                w[local_idx] = np.nan

        n_valid = int(np.sum(np.isfinite(w)))
        n_used.append(n_valid)

        # ── Normalised weighted average ─────────────────────────────────────
        norm = float(np.nansum(w))
        if norm == 0 or n_valid == 0:
            stacks.append(np.zeros(width))
            continue

        # Broadcasting: multiply each profile (row) by its normalised weight
        weighted = block * (w / norm)[:, np.newaxis]
        stacks.append(np.nansum(weighted, axis=0))

    stacks = np.array(stacks)

    if verbose:
        print("\nStacking complete.")
        print(f"  {'Interval':>20}  {'n_used':>6}")
        for k in range(nstacks):
            label = f"n = {quantum_intervals[k]}…{quantum_intervals[k+1]}"
            print(f"  {label:>20}  {n_used[k]:>6}")

    return stacks, n_used
