# GPSeer: a genotype-phenotype predictor

An opinionated sampler for sampling high-order epistasis models and predicting phenotypes in a sparsely sampled genotype-phenotype map. 
The distributed version of the API is powered by [Dask](https://github.com/dask/dask).

Still under HEAVY development. Please don't use yet. The API will change very rapidly.

## Basic Example

1. Load a genotype-phenotype map and construct a model for predicting missing phenotypes.

```python
from gpmap import GenotypePhenotypeMap
from epistasis.models import EpistasisMixedRegression

# Load data
gpm = GenotypePhenotypeMap.read_json("data.json")

# Initialize a model to use
model = EpistasisMixedRegression(
    order=3,                # Set order of model
    threshold=5,            # Threshold for classification as viable/nonviable
    lmbda=1, A=1, B=1,      # nonlinear scale parameters
    model_type="local")     # Type of high-order epistais model to use.

```
2. **Construct** a GPSeer model (distributed for faster computation) and **train** the model.

```python
from dask.distributed import Client
from gpseer import GPSeer

# Start a Dask Client
client = Client()

# Sample directory name
db_dir = "samples"

# Initialize a GPSeer object and give it data.
seer = GPSeer(client=client).setup(gpm, model, db_dir=db_dir)

# Run Pipeline.
# This samples predictions from many epistasis models.
# Then, fits histograms to the large array of predictions and
# saves these as Snapshot objects.
seer.run(n_samples=10000)

# Collect the results
seer.collect()
```

Use the trained model to approximate the posterior distribution of an unknown 
genotype (predict the phenotype). 

```python
# Approximate the posterior distribution.
seer.approximate_posterior()
```

## Install

Clone this repository and install with pip:

```
pip install -e .
```

## Dependencies

1. gpmap : Python API for analyzing genotype phenotype maps.
2. epistasis : Python API for extracting high-order epistasis in genotype-phenotype maps
4. Dask : Distributed array computing made easy in Python.
5. dask.distributed : distributing scheduler made easy in Python.
