from setuptools import setup, find_packages

setup(
    name="rrl-ism-analysis",
    version="0.1.0",
    author="Lucie Cros",
    description="CRRL spectral analysis toolkit for NenuFAR data",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "astropy>=5.0",
        "matplotlib",
        "xarray",
        "netCDF4",
        "tqdm",
        "pyyaml",
    ],
    extras_require={
        "maps": ["dustmaps", "scikit-image"],
        "grid": ["dask"],
        "nenupy": ["nenupy"],
    },
)
