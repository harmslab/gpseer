.. gpseer documentation master file, created by
   sphinx-quickstart on Mon Oct 31 09:50:26 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GPSeer
======

GPSeer provides a Python object that predicts phenotypes in sparsely sampled genotype-phenotypes maps.

Maximum Likelihood Approach
---------------------------


.. code-block:: python

  from gpseer import GPSeer

  # Initialize the seer
  seer = GPSeer(gpm, model)

  # Setup the models. This constructs a genotype-phenotype map from all reference states.
  seer.setup()
  
  # Run a Maximum likelihood group of models to predict phenotypes
  seer.run_ml_pipeline()

Bayesian Approach
-----------------

.. code-block:: python

  from gpseer import GPSeer

  # Initialize the seer
  seer = GPSeer(gpm, model)

  # Setup the models.
  seer.setup()
  
  # 
  seer.run_bayesian_pipeline()


Parallelized Computation
------------------------

The computations in GPSeer are extremely computationally expensive. It requires constructing
unique epistasis models for all missing phenotypes. Fortunately, 
these computations are also easy to parallelize. GPSeer 

GPSeer is built on top of Dask_. It easily distributes the computing across all
resources provided.

Contents:

.. toctree::
   :maxdepth: 2

   _pages/


.. Indices and tables
.. ==================
.. 
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
