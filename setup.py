from distutils import sysconfig
from setuptools import setup, Extension, find_packages
import os
import sys
import setuptools
from copy import deepcopy

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="fastsrm",
    install_requires=[
        "scipy>=0.18.0"
        "numpy>=1.12"
        "scikit-learn>=0.23"
        "joblib>=1.1.0"
        "matplotlib>=2.0.0"
        "pytest>=6.2.5"
    ],
    version="0.0.4",
    license="MIT",
    author="Hugo RICHARD",
    download_url="https://github.com/hugorichard/FastSRM/archive/v_004.tar.gz",
    author_email="hugo.richard@inria.fr",
    url="https://github.com/hugorichard/FastSRM",
    long_description=long_description,
    long_description_content_type='text/markdown',
    description="Fast Shared response model",
    keywords="Component analysis, fMRI",
    packages=find_packages(),
    python_requires=">=3",
)
