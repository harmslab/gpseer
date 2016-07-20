# GPSeer: a genotype-phenotype predictor

A Python API that predicts unknown phenotypes genotype-phenotype maps from known phenotypes.

Still under development.

The base class, `Predictor`, uses linear regression to fit a high-order epistasis model to an incomplete genotype-phenotype map. Due to the growing size of experimental genotype-phenotype maps and the necessity to bootstrap these maps, we've built GPSeer on top of `h5py`. This enables us to manage and search large datasets (on the order of Terabytes) efficiently.

GPSeer uses the Python APIs, SeqSpace and Epistasis, to draw out as much information as possible from a measured genotype-phenotype map, and then makes predictions about unknown genotype-phenotypes.

# Basic Usage

```python
from seqspace import GenotypePhenotypeMap
from epistasis.models import LinearEpistasisRegression
from gpseer import Predictor

# Read in data
gpm = GenotypePhenotypeMap.from_json("example.json")

# Initialize the predictor
predictor = Predictor(gpm,
    LinearEpistasisRegression,  # Type of epistasis model
    order=4                     # Order of epistasis model (optional argument)
)

# Sample from a given references state
reference= "0000"
predictor.sample_to_convergence(reference=reference)
print(predictor.genotypes.g0000.samples())
```

# Install

```
pip install -e .
```

# Dependencies

1. seqspace : Python API for analyzing genotype phenotype maps.
2. epistasis :  
3. h5py:
