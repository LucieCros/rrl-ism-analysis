import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.edenhofer2023 import Edenhofer2023Query
import dustmaps
from tqdm import tqdm

# =========================
# PARAMÈTRES
# =========================

#off_1 = [181.90999999, -5.83000002]
#off_2 = [183.75863572, -3.75428324]
off_3 = [183.69200623, -4.93698766]
#off_4 = [184.97599442, -6.13300623]

#offsets = np.array([off_1, off_2, off_3, off_4])
offsets = np.array([off_3])
radii = np.arange(0.04, 1.99, 0.001)

r_taua = 2000  # pc
extent_in_pc = int(np.ceil(6 / 180 *np.pi * r_taua))

# =========================
# DUST MAP
# =========================

dustmaps.edenhofer2023.fetch(fetch_2kpc=True)
eden = Edenhofer2023Query(flavor="less_data_but_2kpc")

# =========================
# AXES
# =========================

X = np.array(radii * 1000, dtype=float)  # pc
Y = np.arange(-extent_in_pc // 2, (extent_in_pc // 2) + 1)
Z = np.arange(-extent_in_pc // 2, (extent_in_pc // 2) + 1)

# =========================
# GRILLE 3D
# =========================

Xg, Yg, Zg = np.meshgrid(X, Y, Z, indexing="ij")

r = np.sqrt(Xg**2 + Yg**2 + Zg**2)

# éviter division par zéro
r[r == 0] = np.nan

phi = np.degrees(np.arccos(Zg / r))
theta = np.degrees(np.arctan2(Yg, Xg))

# =========================
# MASQUE FOV
# =========================

mask = (np.abs(theta) <= 6  / 2) & (np.abs(90 - phi) <= 3.5 / 2)

# flatten (gros gain perf)
theta_flat = theta[mask]
phi_flat = phi[mask]
r_flat = r[mask]

# =========================
# TABLEAU RÉSULTAT
# =========================

data = np.full(Xg.shape + (1,), np.nan, dtype=np.float32)

# =========================
# CALCUL PRINCIPAL
# =========================

for i, (l0, b0) in enumerate(offsets):
    print(f"Processing offset {i+1}/4")

    l = l0 + theta_flat
    b = b0 + (90 - phi_flat)

    coord = SkyCoord(
        l=l * u.deg,
        b=b * u.deg,
        frame="galactic",
        distance=r_flat * u.pc
    )

    values = eden(coord).astype(np.float32)

    tmp = np.full(r.shape, np.nan, dtype=np.float32)
    tmp[mask] = values

    data[..., i] = tmp

# =========================
# XARRAY
# =========================

#da = xr.DataArray(
#    data,
#    dims=("x", "y", "z", "offset"),
#    coords={
#        "x": X,
#        "y": Y,
#        "z": Z,
#        "offset": ["off1", "off2", "off3", "off4"]
#    },
#    name="dust_absorption"
#)

da = xr.DataArray(
    data,
    dims=("x", "y", "z", "offset"),
    coords={
        "x": X,
        "y": Y,
        "z": Z,
        "offset": ["off3"]
    },
    name="dust_absorption"
)

# =========================
# SAUVEGARDE (NetCDF optimisé)
# =========================

output_file = "dust_map.nc"

encoding = {
    "dust_absorption": {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32"
    }
}

da.to_netcdf(output_file, encoding=encoding)

print(f"\nSaved to {output_file}")
