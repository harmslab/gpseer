# GPSeer
*Infer missing data in sparsely measured genotype-phenotype maps.*

An opinionated library for sampling high-order epistasis models to predict
phenotypes in a sparsely sampled genotype-phenotype map. This is an extremely
computationally expensive task, so GPSeer does its best break up the problem
into independent chunks distributed to as many workers as your machine allows.
This is powered by the fantastic [Dask](https://github.com/dask/dask) library.

Still under HEAVY development. Please don't use yet. The API is still changing very rapidly.

## Basic examples

### API example

1. Load a genotype-phenotype map and initialize an epistasis model for predicting missing phenotypes.

```python
from gpmap import GenotypePhenotypeMap
from epistasis.models import EpistasisMixedRegression

# Load data
gpm = GenotypePhenotypeMap.read_json("data.json")

# Initialize a model to use
model = EpistasisLinearRegression(order=1, model_type='local')
```
2. **Construct** a GPSeer model (distributed for faster computation) and **train** the model.

```python
from dask.distributed import Client
from gpseer import GPSeer

# Start a Dask Client
client = Client()

# Posterior window
bins = np.arange(0, 10, 0.1)

# Initialize a GPSeer object and give it data.
seer = GPSeer(gpm, model, bins, client=client, db_dir=db_dir)

# Run Pipeline.
# This samples predictions from many epistasis models and stores as histograms.
seer.sample_pipeline(n_samples=1000)

# See results
seer.results
```

Use the trained model to approximate the posterior distribution of an unknown
genotype (predict the phenotype).

### CLI example

GPSeer install a few scripts that are runnable from anywhere on the commandline.
To sample a genotype-phenotype map, simply run `gpseer-predict`.

```
gpseer predict --input data.json --output results.csv \
      --model EpistasisLinearRegression --nsamples 100 \
      --range 0 200 --order 1 --db_dir gpseer-db \
      --preclassify False --model_type global
```

If you've already sampled a dataset and would like to sample it further, try
`gpseer-continue`. Just point this script to the directory that contains
gpseer data.

```
gpseer continue --db_dir gpseer-db --output results.csv --nsamples 100
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
