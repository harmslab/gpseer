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

# Sample directory name
db_dir = "samples"

# Load data
gpm = GenotypePhenotypeMap.from_json("data.json")

# Initialize a model to use
model = EpistasisMixedRegression(order=3, threshold=5, model_type="local")

# Initialize a GPSeer object
seer = GPSeer(gpm, model, db_dir=db_dir)

# Run Pipeline
seer.add_ml_fits()        # Fit maximum likelihood models.
seer.add_samples(10000)   # Sample the likelihood function.
seer.add_predictions()    # Predict from samples.
seer.add_posteriors()     # Construct a posterior distribution.
```

## Install

```
pip install -e .
```

## Dependencies

1. gpmap : Python API for analyzing genotype phenotype maps.
2. epistasis :  
3. h5py :
