# GPSeer: a genotype-phenotype predictor

A Python API that predicts unknown phenotypes genotype-phenotype maps from known phenotypes, powered by [Dask](https://github.com/dask/dask).

Still under development.

`GPSeer`, fits a high-order epistasis model (using Nonlinear regression and classification) to an incomplete genotype-phenotype map. Due to the growing size of experimental genotype-phenotype maps and the necessity to bootstrap these maps, we've built GPSeer on top of `h5py`. This enables us to manage and search large datasets (on the order of Terabytes) efficiently.

GPSeer uses the Python APIs, GPMap and Epistasis, to draw out as much information as possible from a measured genotype-phenotype map, and then makes predictions about unknown genotype-phenotypes.

## Basic Example

```python
from gpmap import GenotypePhenotypeMap
from epistasis.models import EpistasisMixedRegression
from gpseer import GPSeer
from dask.distributed import Client

# Start a Dask Client
client = Client()

# Sample directory name
db_dir = "samples"

# Load data
gpm = GenotypePhenotypeMap.read_json("data.json")

# Initialize a model to use
model = EpistasisMixedRegression(
    order=3,                # Set order of model
    threshold=5,            # Threshold for classification as viable/nonviable
    lmbda=1, A=1, B=1,      # nonlinear scale parameters
    model_type="local")     # Type of high-order epistais model to use.

# Initialize a GPSeer object and give it data.
seer = GPSeer(client=client).setup(gpm, model, db_dir=db_dir)

# Run Pipeline.
# This samples predictions from many epistasis models.
# Then, fits histograms to the large array of predictions and
# saves these as Snapshot objects.
seer.run(n_samples=10000)

# Plot the histogram to visualize the prediction probability distribution.
seer.snaphots["000"].plot()
```

## Install

Clone this repository and install with pip:

```
pip install -e .
```

## Dependencies

1. gpmap : Python API for analyzing genotype phenotype maps.
2. epistasis : Python API for extracting high-order epistasis in genotype-phenotype maps
3. h5py : Python API for working with HDF5 files (writing large arrays to disk).
4. Dask : Distributed array computing made easy in Python.
5. dask.distributed : distributing scheduler made easy in Python.
