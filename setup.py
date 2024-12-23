from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        "SpotDIPy.cutils",
        sources=["src/SpotDIPy/cutils.pyx"],
        include_dirs=[np.get_include()]
    )
]

setup(
    name="SpotDIPy",
    version="0.0.5",
    author="Engin Bahar",
    author_email="enbahar@ankara.edu.tr",
    description="An Easy Way for Stellar Doppler Imaging of Cool Single Stars",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/EnginBahar/SpotDIPy",
    project_urls={
        "Homepage": "https://github.com/EnginBahar/SpotDIPy",
        "Issues": "https://github.com/EnginBahar/SpotDIPy/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "Cython",
        "matplotlib",
        "basemap",
        "astropy",
        "PyAstronomy",
        "exotic_ld",
        "PyDynamic",
        "tqdm",
        "PyQt5",
        "PyQt5-Qt!=5.15.14",
        "phoebe",
        "kneebow",
        "mayavi",
        "traits",
        "traitsui",
        "vtk==9.3.0",
        "configobj",
        "jax",
        "jaxopt",
        'healpy; platform_system=="Linux"',
    ],
    setup_requires=['wheel', "Cython"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions),
    include_package_data=True,
)
