Computing engines
=================

Phenotype prediction computations using epistasis models are expensive. To help
alleviate this problem, GPSeer offers a distributed computing engine powered by
Dask. Dask offers an API that makes parallelization, setup and teardown of nodes, 
and collection of data to a central scheduler simple. GPSeer connects to Dask 
through the ``Engine`` object. 

Distributed Engine
------------------

To access the distributed engine, simply start a Dask ``Client`` and pass it to GPSeer.
Dask allows you to control the input

.. code-block:: python

  # Normal imports
  from gpmap import GenotypePhenotypeMap
  from epistasis.models import EpistasisLinearRegression
  from gpseer import GPSeer

  # Import Dask Client.
  from dask.distributed import Client
  
  # Initialize a client.
  client = Client()
  
  # Initialize data and model.
  gpm = GenotypePhenotypeMap.read_csv('data.csv')
  model = EpistasisLinearRegression(order=2, model_type='local')
  
  # Pass the client to the GPSeer object
  seer = GPSeer(gpm, model, client=client)


Serial Engine
-------------

The default in GPSeer is to use a serial computation engine. If you don't pass
a Client to GPSeer, all computations will be done on a single node in series

.. code-block:: python

  # Normal imports
  from gpmap import GenotypePhenotypeMap
  from epistasis.models import EpistasisLinearRegression
  from gpseer import GPSeer
  
  # Initialize data and model.
  gpm = GenotypePhenotypeMap.read_csv('data.csv')
  model = EpistasisLinearRegression(order=2, model_type='local')
  
  # Pass the client to the GPSeer object
  seer = GPSeer(gpm, model)
