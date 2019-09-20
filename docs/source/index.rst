.. gpseer documentation master file, created by
   sphinx-quickstart on Sat Sep 14 06:17:16 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======
gpseer
======

Simple software for inferring missing data in sparsely measured genotype-phenotype maps
=======================================================================================

Experimentally characterizing genotype-phenotype maps can be very challenging because the size of a map expands exponentially as the number of mutations increases.  For example, a map with four mutational sites, each existing in one of two states, includes 16 genotypes (:math:`2^{4}`).  By contrast, a map with 15 mutational sites consists of 32,768 genotypes (:math:`2^{15}`).  Exhaustive characterization of the phenotypes in a map is often infeasible, particularly for phenotypes that are difficult to characterize by high-throughput methods.  To address this shortfall, we have developed a straightforward approach to infer the missing phenotypes from an incomplete genotype-phenotype map, with well-characterized uncertainty in our predictions.  Such knowledge allows robust and statistically-informed analyses of features of the map, such as knowledge of possible evolutionary trajectories.

Install
=======

Clone this repository and install with pip:

.. code-block:: bash

    git clone https://github.com/harmslab/gpseer.git
    cd gpseer
    pip install gpseer


Dependencies
============

1. `gpmap <https://gpmap.readthedocs.io/en/latest/>`_: Python API for storing and manipulating genotype-phenotype maps
2. `epistasis <https://epistasis.readthedocs.io/>`_: Python API for extracting and analyzing epistasis in genotype-phenotype maps
3. `traitlets <https://traitlets.readthedocs.io/en/stable/>`_ : static typing and configurable objects in Python

Quick start
===========

The simplest use-case is to call ``gpseer`` on an input ``.csv`` file containing genotype-phenotype data.

To try it out, see the ``examples/`` directory.

**For example**

.. code-block:: bash

    cd gpseer
    cd examples/

    # Fit the observations in phenotypes.csv.  The instrument detection
    # threshold is set to 1.  This will first classify each genotype as
    # detectable or undetectable, interpolate across the map using a second-
    # order spline, and then describe the effect of each mutation as 
    # additive.   
    gpseer -i phenotypes.csv --threshold 1  --spline_order 2    

*Output*

::

    [GPSeer] + Reading data...
    [GPSeer] └── Done reading data.

    [GPSeer] + Fitting data...
    [GPSeer] └── Done fitting data.

    [GPSeer] + Predicting missing data...
    [GPSeer] └── Done predicting...

    [GPSeer] GPSeer finished!

which returns a set of phenotype predictions stored in `predictions.csv`.

Command-line options
====================

.. code-block:: bash

    -i <Unicode> (GPSeer.infile)
        Default: 'test'
        Input file.
    -o <Unicode> (GPSeer.outfile)
        Default: 'predictions.csv'
        Output file
    --model_definition=<Instance> (GPSeer.model_definition)
        Default: None
        An epistasis model definition written in Python.
    --wildtype=<Unicode> (GPSeer.wildtype)
        Default: ''
        The wildtype sequence
    --threshold=<Float> (GPSeer.threshold)
        Default: 0.0
        Experimental detection threshold, used by classifer.
    --spline_order=<Int> (GPSeer.spline_order)
        Default: 0.0
        Order of spline..
    --spline_smoothness=<Int> (GPSeer.spline_smoothness)
        Default: 10
        Smoothness of spline.
    --epistasis_order=<Int> (GPSeer.epistasis_order)
        Default: 1
        Order of epistasis in the model.
    --nreplicates=<Int> (GPSeer.nreplicates)
        Default: None
        Number of replicates for calculating uncertainty.
    --model_file=<Unicode> (GPSeer.model_file)
        Default: ''
        File containing epistasis model definition.

To see all configuration items, call ``gpseer --help``:


Advanced epistasis models
=========================

More advanced models are possible by writing a short models file:

.. code-block:: python

    # model.py
    from epistasis.models import (
        EpistasisPipeline,
        EpistasisLogisticRegression,
        EpistasisSpline,
        EpistasisLinearRegression
    )

    c.GPSeer.model_definition = EpistasisPipeline([
        EpistasisLogisticRegression(threshold=5),
        EpistasisSpline(k=3),
        EpistasisLinearRegression(order=3)
    ])

then call the ``gpseer`` command.

.. code-block:: bash

    gpseer -i phenotypes.csv --model_file=model.py

For documentation on how to write these models, see the `epistasis <<https://epistasis.readthedocs.io/>`_
package.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


