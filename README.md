# GPSeer: a genotype-phenotype predictor

A Python API that predicts unknown phenotypes genotype-phenotype maps from known phenotypes.

Still under development.

The base class, `Predictor`, fits a high-order epistasis model (using Linear Regression) to an incomplete genotype-phenotype map. Due to the growing size of experimental genotype-phenotype maps and the necessity to bootstrap these maps, we've built GPSeer on top of `h5py`. This enables us to manage and search large datasets (on the order of Terabytes) efficiently.

GPSeer uses the Python APIs, GPMap and Epistasis, to draw out as much information as possible from a measured genotype-phenotype map, and then makes predictions about unknown genotype-phenotypes.

## Basic Example

```python
from gpmap import GenotypePhenotypeMap
from epistasis.models import EpistasisMixedRegression
from epistasis.sampling import BayesianSampler
from gpseer import Predictor

# Read in data
gpm = GenotypePhenotypeMap.from_json("example.json")

# Initialize the predictor
predictor = Predictor(gpm,
    Model= EpistasisMixedRegression,    # Type of epistasis model
    Sampler=BayesianSampler,            # Sampling method
    order=2                             # Order of epistasis model (optional argument)
)

# Prepare the predictor models
predictor.setup()

# Fit the ML for all models
predictor.fit(lmbda=1, A=1, B=0)

# Generate 10000 samples
predictor.sample(n_samples=10000)

# Sample from a given references state
predictor.sample_posterior("0000")
```

## Install

```
pip install -e .
```

## Dependencies

1. gpmap : Python API for analyzing genotype phenotype maps.
2. epistasis :  
3. h5py :
