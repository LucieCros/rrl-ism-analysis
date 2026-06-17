# rrl-ism-analysis

A Python toolkit for the analysis of **carbon radio recombination lines (CRRLs)**
observed with NenuFAR.

The package covers the full data analysis chain: from raw FITS blocs to
physical parameter fitting, including spectral cleaning, stacking, Voigt
profile fitting, parameter-space grid generation, and multi-tracer ISM
mapping (dust, CO, HI).

---

## Repository structure

```
spectral_tools/           Core library (importable package)
├── atoms.py              Atomic species data, RRL frequency computation
├── tools.py              Signal processing, Doppler corrections, utilities
├── io.py                 FITS I/O, source catalogue, FITS header builder
├── L1_class.py           NenuFAR Level-1 FITS observation container
├── line_fitting.py       Voigt profile fitting, uncertainty propagation
├── modeling.py           Physical CRRL broadening and area modeling
├── graphics.py           Matplotlib plotting utilities
├── maps.py               Dust extinction (Edenhofer 2023), CO/HI cube handling, beam computation
├── grid_exploration.py   Chi² grid exploration and parameter visualisation
├── kd_utils.py           Galactic kinematic distance utilities (Reid+2019)
└── reid19_rotcurve.py    LSR velocity model (Reid et al. 2019 rotation curve)

pipelines/                Executable scripts
├── run_cleaning.py       Batch launcher for spectral cleaning
├── clean_observation.py  Single-observation cleaning pipeline
├── run_time_average.py   Weighted temporal averaging of cleaned observations
└── generate_grid.py      Generate parameter-space grids (coarse or fine)

notebooks/                Jupyter exploration notebooks
├── 1_Cloud-id.ipynb                     ISM cloud identification (dust, CO, HI)
├── 3_Stacking.ipynb                     RRL profile stacking and visualisation
├── 4_Line-fitting.ipynb                 Voigt fitting on detected transitions
└── 5_Grid-exploration.ipynb             Chi² grid exploration and parameter fitting

files/                    Reference data (tracked on git)
├── rrlines.csv           RRL frequency catalogue (Halph, Calph, Hbeta, ...)
├── source_info.txt       Source astrometric and velocity parameters
├── fitsheader.txt        FITS header template for pipeline output
├── grid_config.yaml      Parameter grid configuration (ranges, quantum numbers)
├── B1B2.pickle           Pre-computed bn·βn interpolators
└── alphagamma.pickle     Pre-computed collisional broadening interpolators
```

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/LucieCros/rrl-ism-analysis.git
cd rrl-ism-analysis
pip install -e .
```

Dependencies are listed in `requirements.txt`.

---

## Quick start

```python
# Load an observation bloc
from spectral_tools.io import load_bloc
hdu = load_bloc("/path/to/observation.fits")
print(hdu.freq)         # frequency axis [MHz]

# Compute a CRRL frequency
from spectral_tools.atoms import line_freq
freq = line_freq(500)   # Cα n=500 → Quantity in MHz

# Apply LSR Doppler correction
import spectral_tools.tools as tools
v_lsr = tools.doppler_corrections("2021-06-09T12:00:00")[-1]

# Fit a Voigt profile
import spectral_tools.line_fitting as lf
area = lf.voigt_area(peak=-0.05, fwhm_G=3.2, fwhm_L=1.1)

# Load CO cube and crop to a circular aperture
from spectral_tools.maps import MapLoader
from astropy.coordinates import SkyCoord
import astropy.units as u

loader = MapLoader("Tau")
cube, vel, spec, lon, lat, mom0 = loader.crop_circle(
    SkyCoord("05h34m32s", "22d00m52s"), "CO", (2*u.deg, 2*u.deg)
)
del loader   # closes FITS files and frees memory
```

---

## Pipeline usage

### 1. Spectral cleaning

Run the cleaning pipeline on all observations of a source in parallel
(up to 8 workers):

```bash
python3 pipelines/run_cleaning.py -source CASA -l Calph -mask 1.8 -cw 200
```

Each observation is processed by `clean_observation.py`, which can also
be run standalone for debugging:

```bash
python3 pipelines/clean_observation.py \
    -path /data/CASA/Calph/ -name obs_20210609 \
    -ext .spectra.0.fits -v -47 -ra 23h23m24s -dec 58d48m54 \
    -coeff [1,1] -line Calph -mask 1.8 -cw 200
```

### 2. Temporal averaging

Compute the weighted time-average of all cleaned observations:

```bash
python3 pipelines/run_time_average.py -source CASA -l Calph
```

### 3. Parameter-space grid generation

Edit `files/grid_config.yaml` to set the parameter ranges and quantum
numbers for your source, then generate the grids:

```bash
# Coarse exploration grid
python3 pipelines/generate_grid.py --mode coarse

# Fine refinement grid (after identifying the best-fit region)
python3 pipelines/generate_grid.py --mode fine
```

The `grid_config.yaml` file controls all grid parameters without touching
the code:

```yaml
coarse:
  quantum_numbers: [446, 486, 516, ...]
  parameters:
    Te: [10, 500, 5]    # [start, stop, step] in K
    Ne: [0.01, 0.1, 0.005]
    ...
```

### 4. Chi² fitting

Use the `5_Grid-exploration.ipynb` notebook interactively, or call the
module directly:

```python
import spectral_tools.grid_exploration as grid

chi2 = grid.compute_chi2_split(
    path_xrs="grids/",
    filepattern="grid-{}.nc",
    csv_file="results/fitted_lines_CASA.csv",
)
df_best, best = grid.find_best_parameters(chi2, percentile=5.0)
grid.plot_chi2_projections(chi2, best, df_best, filepath="results/CASA")
```

---

## Notebooks

The Jupyter notebooks in `notebooks/` provide an interactive interface for
the exploration, stacking, fitting, and parameter analysis steps. They
consume the outputs of the cleaning pipelines and form the downstream part
of the analysis chain.

### Overview

| Notebook | Purpose |
|---|---|
| `1_Cloud-id.ipynb` | ISM cloud identification (dust, CO, HI) |
| `3_Stacking.ipynb` | Weighted stacking of CI α radio recombination lines |
| `4_Line-fitting.ipynb` | Voigt profile fitting on detected transitions |
| `5_Grid-exploration.ipynb` | Chi² grid exploration and physical parameter fitting |

---

### `1_Cloud-id.ipynb` — ISM cloud identification

**Purpose:** Identify interstellar medium clouds by cross-matching multiple
tracers (3D dust extinction, CO line emission, HI emission) in order to
define the velocity components to be modelled with the CRRLs.

**Module dependencies:**
- `spectral_tools.maps` — dust map queries (`DustMap`), PPV cube loading (`MapLoader`), beam computation
- `spectral_tools.tools` — coordinate conversions (`galactic_to_cartesian`, `cartesian_to_galactic`)
- `spectral_tools.graphics` — axis styling helpers
- `spectral_tools.reid19_rotcurve` — LSR velocity model for iso-velocity surfaces
- `nenupy` — NenuFAR beam simulation and angular resolution
- `astropy`, `numpy`, `scipy`, `pandas`, `dask`, `xarray`, `matplotlib`, `plotly`, `kaleido`, `scikit-image`

**Data dependencies:**
- CO and HI PPV cubes (FITS, external, not tracked)
- Edenhofer et al. (2023) 3D dust extinction map (fetched via `maps.DustMap`)
- NenuFAR observation log (CSV)
- `files/source_info.txt` — source coordinates and systemic velocity

---

### `3_Stacking.ipynb` — Weighted RRL stacking

**Purpose:** Stack individual CI α transitions over a wide range of quantum
number *n* to increase the signal-to-noise ratio. The pipeline:
1. loads the time-averaged spectrum (FITS `alltime` file);
2. extracts each transition in a frequency window centred on *f₀(n)*;
3. evaluates the S/N of each line and rejects contaminated ones;
4. computes the inverse-variance weighted average (weight = 1/RMS²) in
   intervals of *n*;
5. saves the stacks and produces a mosaic visualisation.

**Module dependencies:**
- `spectral_tools.tools` — `get_line`, `slice_line`, `line_freq`, `rebinning`
- `spectral_tools.graphics` — `set_axes`
- `numpy`, `matplotlib`, `astropy` (`fits`, `units`),
  `scipy` (`interp1d`, `curve_fit`, `linregress`), `tqdm`

**Data dependencies:**
- `alltime_files/alltime_{SOURCE}_{LINE}[_OFF].fits` — time-averaged
  spectrum produced by `pipelines/run_time_average.py`
- `files/rrlines.csv` — RRL frequency catalogue
- `files/source_info.txt` — systemic velocity of the source

**Outputs:**
- `stacks/STACKS-{SOURCE}[_OFF].txt` — stacked profiles (quantum number
  bounds + spectra)
- `{SOURCE}[_OFF]-detections.pdf` — mosaic of all stacked profiles

---

### `4_Line-fitting.ipynb` — Voigt profile fitting

**Purpose:** Fit a Voigt profile to each detected transition to extract
the integrated area, the Gaussian FWHM (thermal/turbulent broadening), and
the Lorentzian FWHM (radiation broadening), together with propagated
uncertainties.

**Module dependencies:**
- `spectral_tools.line_fitting` — `voigt_area` and fitting routines
- `spectral_tools.tools`, `spectral_tools.graphics`
- `numpy`, `matplotlib`, `scipy`

**Data dependencies:**
- `stacks/STACKS-{SOURCE}[_OFF].txt` — produced by `3_Stacking.ipynb`
- `files/rrlines.csv`

**Outputs:**
- `results/fitted_lines_{SOURCE}.csv` — fitted parameters (areas, FWHMs,
  uncertainties) for each detected transition

---

### `5_Grid-exploration.ipynb` — Chi² grid exploration

**Purpose:** Compare the measured spectral parameters (area, line widths)
against a grid of physical CRRL models to constrain the physical conditions
of the medium (electron temperature *Tₑ*, electron density *Nₑ*, …).
Generates chi-squared projection maps and identifies the best-fit region.

**Module dependencies:**
- `spectral_tools.grid_exploration` — `compute_chi2_split`,
  `find_best_parameters`, `plot_chi2_projections`
- `spectral_tools.atoms`, `spectral_tools.tools`
- `numpy`, `matplotlib`, `xarray` (NetCDF grid reading)

**Data dependencies:**
- `grids/grid-{n}.nc` — model grids generated by
  `pipelines/generate_grid.py`
- `results/fitted_lines_{SOURCE}.csv` — produced by `4_Line-fitting.ipynb`
- `files/B1B2.pickle` — pre-computed *bₙ·βₙ* interpolators
- `files/alphagamma.pickle` — pre-computed collisional broadening
  interpolators
- `files/grid_config.yaml` — grid configuration (parameter ranges, quantum
  numbers)

**Outputs:**
- `results/{SOURCE}_chi2_*.pdf` — chi-squared projection figures
- Best-fit physical parameter table (*Tₑ*, *Nₑ*, …)

---

## Module dependency graph

```
atoms.py          (no internal imports)
    ↑
tools.py          (imports atoms)
    ↑             ↑
L1_class.py       line_fitting.py   (imports tools)
    ↑             ↑
io.py             modeling.py       (imports atoms, tools, line_fitting)
    ↑
kd_utils.py       (no internal imports)
    ↑
reid19_rotcurve.py (imports kd_utils)

graphics.py       (no internal imports)
maps.py           (no internal imports — uses nenupy, dustmaps externally)
grid_exploration  (imports atoms, tools)
```

---

## Reference data files

| File | Description | Tracked on git |
|---|---|---|
| `rrlines.csv` | RRL frequency catalogue | ✓ |
| `source_info.txt` | Source astrometry and velocities | ✓ |
| `fitsheader.txt` | FITS header template | ✓ |
| `grid_config.yaml` | Grid parameter configuration | ✓ |
| `B1B2.pickle` | bn·βn interpolators | ✓ |
| `alphagamma.pickle` | Collisional broadening interpolators | ✓ |
| `data/raw/` | Raw observation FITS blocs | ✗ (too large) |
| `data/processed/` | Cleaned spectra | ✗ (generated) |
| `outputs/` | Figures, results, cached grids | ✗ (generated) |

---

## References

- Gordon & Sorochenko (1992) — *Radio Recombination Lines*
- Salgado et al. (2017a, 2017b) — radiation broadening of CRRLs
- Edenhofer et al. (2023) — 3-D dust extinction map
- Thompson (1987) — Voigt FWHM approximation
- McKean et al. (2016) — log-polynomial continuum model

---

## License

CC BY 4.0 — see `LICENSE` for details. If you use this software, please cite it using `CITATION.cff`.
