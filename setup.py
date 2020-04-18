#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'gpseer'
DESCRIPTION = '''Python API to infer missing data in sparsely sampled \
genotype-phenotype maps.'''
URL = 'https://github.com/harmslab/gpseer'
EMAIL = 'zachsailer@gmail.com'
AUTHOR = 'Zach Sailer and Mike Harms'

# What packages are required for this module to be executed?
REQUIRED = ['requests', 'numpy', 'pandas', 'gpmap', 'epistasis', 'tqdm','matplotlib']
TESTS_REQUIRE = ['pytest', 'pytest-cov', 'pytest-console-scripts', 'pytest-datafiles']

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    entry_points={
        'console_scripts': ['{} = {}.main:entrypoint'.format(NAME, NAME)],
    },
    install_requires=REQUIRED,
    extras_require = {
        'test': TESTS_REQUIRE,
    },
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ]
)
