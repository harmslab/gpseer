# GPSeer
*An opinionated approach to inferring missing data in sparsely measured genotype-phenotype maps.*

## Basic Usage

The simplest use-case is to call GPSeer on a CSV File.
```
gpseer -i phenotypes.csv
```
This returns a set of phenotype predictions using a third-order spline.

## Advanced Usage

More advanced models are possible by writing a short models file:
```python
# model.py
from epistasis.models import (
      EpistasisPipeline,
      EpistasisLogisticRegression,
      EpistasisSpline,
      EpistasisLinearRegression
)

c.GPSeer.epistasis_model = EpistasisPipeline([
      EpistasisLogisticRegression(threshold=5),
      EpistasisSpline(k=3),
      EpistasisLinearRegression(order=3)
])
```
then call the `gpseer` command.
```
gpseer -i phenotypes.csv --model_file=model.py
```


## Install

Clone this repository and install with pip:

```
pip install gpseer
```

## Dependencies

1. gpmap : Python API for analyzing genotype phenotype maps.
2. epistasis : Python API for extracting high-order epistasis in genotype-phenotype maps
3. traitlets: static typing and configurable objects in Python