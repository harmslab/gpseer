# GPSeer: a genotype-phenotype predictor

A Python API that predicts unknown phenotypes genotype-phenotype maps from known phenotypes.

Still under development.

The base class, `Predictor`, uses linear regression to fit a high-order epistasis model to an incompletely genotype-phenotype map. Due to the growing size of experimental genotype-phenotype maps and the necessity to bootstrap these maps, we've built GPSeer on top of `h5py`. This enables us to manage and search large datasets (on the order of Terabytes) efficiently. 

# Basic Usage

```python
import gpseer
```

# Install

```
pip install -e .
```

# Dependencies

1. seqspace:
2. epistasis:
3. h5py:
