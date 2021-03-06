Install
=======

Basic install
-------------

Install GPSeer from PyPI using pip.

.. code-block:: console

    > pip install gpseer

Dependencies
------------

1. `numpy <https://docs.scipy.org/doc/numpy/reference/>`_: Python's numerical array computation library.
2. `pandas <https://pandas.pydata.org/>`_: Python's DataFrame library.
3. `tqdm <https://github.com/tqdm/tqdm>`_: Progress bars in Python.
4. `gpmap <https://gpmap.readthedocs.io/en/latest/>`_: Python API for storing and manipulating genotype-phenotype maps
5. `epistasis <https://epistasis.readthedocs.io/>`_: Python API for extracting and analyzing epistasis in genotype-phenotype maps
6. `matplotlib <https://matplotlib.org/>`_: Python API for plotting.
7. `requests <https://requests.readthedocs.io/en/master/>`_: Python API for downloading data via HTTP.

Installing from source
----------------------

Clone the ``gpseer`` repository from Github.

.. code-block:: console

    > git clone https://github.com/harmslab/gpseer
    > cd gpseer
    > pip install -e .
