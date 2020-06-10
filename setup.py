from distutils import sysconfig
from setuptools import setup, Extension, find_packages
import os
import sys
import setuptools
from copy import deepcopy

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='fastsrm',
    install_requires=[
        'scikit-learn',
        'joblib',
        'python-picard'
    ],
    version = '0.0.2',
    license='MIT',
    author='Hugo RICHARD',
    author_email='hugo.richard@inria.fr',
    url='https://github.com/hugorichard/FastSRM',
    long_description=long_description,
    description='Fast Shared response model',
    keywords='Component analysis, fMRI',
    packages=find_packages(),
    python_requires='>=3',
)
