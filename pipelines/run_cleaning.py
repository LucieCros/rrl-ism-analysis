#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cleaning.py
===============
Batch pipeline launcher for spectral cleaning of NenuFAR observation blocs.

For each raw ``.fits`` file found in the source data directory, this script
builds and dispatches a ``Cleaning_special.py`` subprocess. Up to 8 workers
run in parallel; the script waits for all of them to complete before exiting.

Usage
-----
::

    python3 run_cleaning.py -source CASA \\
        -l Calph -mask 1.8 -cw 200

    python3 run_cleaning.py -source CYGA -gal -nomask -l Calph -cw 200

    python3 run_cleaning.py -source TAUA -off -l Calph -mask 2.0 -cw 150

Command-line arguments
----------------------
-source     Source identifier (required), e.g. ``CASA``, ``TAUA``.
-l          RRL series to process (default: ``Calph``).
-mask       Mask half-width in units of the expected line width. Pass 0 to
            disable masking.
-cw         Cut width in channels for sub-band extraction.
-nocorr     Disable LSR Doppler correction.
-CAL        Process calibrator observations.
-gal        Galaxy mode: use galactic velocity and reduced-gal output path.
-rw         Rewrite already-reduced files (default: skip existing).
-spec       Special mode (relaxed sigma-clipping for bright sources).
-off        Process OFF-beam observations.
-nomask     Disable all line masking.
-velo       Override source velocity [km/s].

Dependencies
------------
Standard library: sys, os, subprocess, time, argparse, warnings
Third-party: tqdm
Internal: spectral_tools.io (read_source_info)
"""

import argparse
import os
import subprocess
import time
import warnings

from tqdm import tqdm

from spectral_tools.io import read_source_info

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Maximum number of parallel cleaning subprocesses.
MAX_WORKERS: int = 8

#: Polling interval [s] when the worker queue is full.
POLL_INTERVAL: int = 180

#: Subprocess timeout [s] per observation (100 minutes).
SUBPROCESS_TIMEOUT: int = 100 * 60

#: Root directory for raw data.
DATA_ROOT: str = "/juliette2/datartemix2/lt10"

#: Root directory for reduced outputs.
OUTPUT_ROOT: str = "/home/lcros"

#: Source info table (tab-separated: name, ra, dec, velo, coeff).
SOURCE_INFO: str = "../files/source_info.txt"

#: Cleaning script to dispatch.
CLEANING_SCRIPT: str = "clean_observation.py"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch launcher for NenuFAR spectral cleaning pipeline."
    )
    parser.add_argument("-source", "--source",  type=str, required=True,
                        help="Source identifier, e.g. CASA, TAUA.")
    parser.add_argument("-nocorr", "--nocorr",  action="store_true",
                        help="Disable LSR Doppler correction.")
    parser.add_argument("-CAL",    "--CAL",     action="store_true",
                        help="Process calibrator observations.")
    parser.add_argument("-gal",    "--galaxy",  action="store_true",
                        help="Galaxy mode (galactic velocity, gal output path).")
    parser.add_argument("-rw",     "--rewrite", action="store_true",
                        help="Rewrite already-reduced files.")
    parser.add_argument("-l",      "--line",    type=str, default="Calph",
                        help="RRL series to process (default: Calph).")
    parser.add_argument("-spec",   "--special", action="store_true",
                        help="Special mode for bright sources.")
    parser.add_argument("-off",    "--off",     action="store_true",
                        help="Process OFF-beam observations.")
    parser.add_argument("-nomask", "--nomask",  action="store_true",
                        help="Disable line masking.")
    parser.add_argument("-velo",   "--velo",
                        help="Override source velocity [km/s].")
    parser.add_argument("-mask",   "--mask",    type=float,
                        help="Mask half-width in units of expected line width.")
    parser.add_argument("-cw",     "--cutwidth", type=int,
                        help="Cut width in channels.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Worker queue helpers
# ---------------------------------------------------------------------------

def _wait_for_slot(processes: list) -> None:
    """
    Block until at least one worker slot is free (queue length < MAX_WORKERS).

    Parameters
    ----------
    processes : list of subprocess.Popen
        Currently running subprocesses.
    """
    return_types = [type(processes[-i].poll()) for i in range(1, MAX_WORKERS)]
    while int not in return_types:
        time.sleep(POLL_INTERVAL)
        return_types = [type(processes[-i].poll()) for i in range(1, MAX_WORKERS)]
    print("Worker slot freed — return codes:", return_types)


def _wait_all(processes: list) -> None:
    """
    Wait for all subprocesses to complete, then terminate them.

    Parameters
    ----------
    processes : list of subprocess.Popen
        All dispatched subprocesses.
    """
    for proc in tqdm(processes, desc="Waiting for workers"):
        proc.wait(timeout=SUBPROCESS_TIMEOUT)
        proc.terminate()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args       = parse_args()
    sourcename = args.source
    my_line    = args.line
    start      = time.time()

    # ── Collect raw FITS files ──────────────────────────────────────────────
    data_path = os.path.join(DATA_ROOT, f"DATA-{sourcename}")
    all_fits  = [
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if f.endswith(".fits") and "LOG" not in f
    ]
    print(f"Found {len(all_fits)} FITS files for source '{sourcename}'")
    print("=" * 60)

    processes: list[subprocess.Popen] = []

    for fits_path in all_fits:

        # ── Parse filename to extract output path, name, extension ─────────
        output_path = os.path.join(OUTPUT_ROOT, fits_path.split("/")[-2])

        # Source and day identification differ for special programmes
        if sourcename == "NT04":
            day    = fits_path.split("COSMIC_DAWN")[1].split("spectra")[0].split("_")[1]
            source = fits_path.split("/")[-1].split("COSMIC_DAWN")[0]
        else:
            day    = fits_path.split("TRACKING")[1].split("spectra")[0].split("_")[1]
            source = fits_path.split("/")[-1].split("TRACKING")[0]

        # Strip J2000 suffix for cloud observations
        if "CLOUDS" in source:
            source = source.split("_J2000")[0]

        name = fits_path.split("/")[-1].split(".spectra")[0][:-1]
        ext  = ".spectra" + fits_path.split("spectra")[1]

        # ── Filter by beam polarity (ON/OFF) ────────────────────────────────
        if args.off:
            if "_2" not in fits_path:
                continue
        else:
            if "_0.spectra" not in fits_path:
                continue

        # ── Check output directory ──────────────────────────────────────────
        out_subdir = (
            os.path.join(output_path, f"reduced-gal/{my_line}/transitions/")
            if args.galaxy
            else os.path.join(output_path, f"{my_line}/transitions/")
        )
        # condition = (output_name not in os.listdir(out_subdir)) or args.rewrite
        # Currently always reprocess; uncomment above to skip existing files.
        condition = True

        if not condition:
            continue

        # ── Look up source parameters ───────────────────────────────────────
        info = read_source_info(source, sourcename, is_off=args.off)
        if info is None:
            print(f"[SKIP] No info found for source '{source}' — skipping.")
            continue

        ra    = info["ra"]
        dec   = info["dec"]
        coeff = info["coeff"]

        # Velocity: CLI override > source info table > galaxy default
        if args.velo is not None:
            velo = args.velo
        elif args.galaxy:
            velo = 16811
        else:
            velo = info["velo"]

        # Mask width: galaxy and nomask flags override the CLI value
        mask = args.mask
        if args.galaxy or args.nomask:
            mask = 0

        # ── Build and dispatch subprocess ───────────────────────────────────
        cmd = (
            f"python3 {CLEANING_SCRIPT}"
            f" -path={output_path}/"
            f" -name={name}"
            f" -ext={ext}"
            f" -v={velo}"
            f" -ra={ra}"
            f" -dec={dec}"
            f" -coeff={coeff}"
            f" -line={my_line}"
            f" -mask={mask}"
            f" -cw={args.cutwidth}"
        )
        if args.special:
            cmd += " -spec"
        if args.off:
            cmd += " -off"

        print("=" * 60)
        print(f"Dispatching: {name}")
        print(f"  Command : {cmd}")

        proc = subprocess.Popen(cmd.split())
        processes.append(proc)

        # Throttle: wait for a free slot when queue is full
        if len(processes) > MAX_WORKERS:
            _wait_for_slot(processes)

    # ── Final wait ──────────────────────────────────────────────────────────
    print("\nAll jobs dispatched — waiting for completion...")
    for proc in processes:
        print(f"  PID {proc.pid} status: {proc.poll()}")
    _wait_all(processes)

    elapsed = (time.time() - start) / 60.0
    print(f"\nTotal processing time: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
