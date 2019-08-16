# GPSeer
*An opinionated approach to inferring missing data in sparsely measured genotype-phenotype maps.*

## Basic Usage

The simplest use-case is to call `gpseer` on an input `.csv` file containing genotype-phenotype data.

For example:
```bash
gpseer -i phenotypes.csv    \
      --wildtype "00000"    \
      --threshold 1         \
      --spline_order 2      \
      --epistasis_order 2
```
Output:
```bash
[GPSeer] Running GPSeer on phenotypes.csv. Look for a predictions.csv file with your results.

[GPSeer] + Reading data...
[GPSeer] └── Done reading data.

[GPSeer] + Fitting data...
[GPSeer] └── Done fitting data.

[GPSeer] + Predicting missing data...
[GPSeer] └── Done predicting...

[GPSeer] GPSeer finished!
```
which returns a set of phenotype predictions using a 2nd-order spline .

## Command-line options
To see all configuration items, call `gpseer --help`:
```
A tool for predicting phenotypes in a sparsely sampled genotype-phenotype maps.

Options
-------

-i <Unicode> (GPSeer.infile)
    Default: 'test'
    Input file.
-o <Unicode> (GPSeer.outfile)
    Default: 'predictions.csv'
    Output file
--model_definition=<Instance> (GPSeer.model_definition)
    Default: None
    An epistasis model definition written in Python.
--wildtype=<Unicode> (GPSeer.wildtype)
    Default: ''
    The wildtype sequence
--threshold=<Float> (GPSeer.threshold)
    Default: 0.0
    Experimental detection threshold, used by classifer.
--spline_order=<Int> (GPSeer.spline_order)
    Default: 0.0
    Order of spline..
--spline_smoothness=<Int> (GPSeer.spline_smoothness)
    Default: 10
    Smoothness of spline.
--epistasis_order=<Int> (GPSeer.epistasis_order)
    Default: 1
    Order of epistasis in the model.
--nreplicates=<Int> (GPSeer.nreplicates)
    Default: None
    Number of replicates for calculating uncertainty.
--model_file=<Unicode> (GPSeer.model_file)
    Default: ''
    File containing epistasis model definition.
```

## Advanced epistasis models

More advanced models are possible by writing a short models file:
```python
# model.py
from epistasis.models import (
      EpistasisPipeline,
      EpistasisLogisticRegression,
      EpistasisSpline,
      EpistasisLinearRegression
)

c.GPSeer.model_definition = EpistasisPipeline([
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