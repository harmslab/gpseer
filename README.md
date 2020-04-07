# GPSeer
*Simple software for inferring missing data in sparsely measured genotype-phenotype maps*

![GPSeer tests](https://github.com/harmslab/gpseer/workflows/GPSeer%20tests/badge.svg)

## Basic Usage

The simplest use-case is to call `gpseer` on an input `.csv` file containing genotype-phenotype data.

To try it out, see the `examples/` directory.

### For example

```bash
cd gpseer
cd examples/

# Fit the observations in phenotypes.csv.  The instrument detection
# threshold is set to 1.  This will first classify each genotype as
# detectable or undetectable, interpolate across the map using a second-
# order spline, and then describe the effect of each mutation as
# additive.
gpseer estimate-ml phenotypes.csv output.csv --threshold 1  --spline_order 2
```

*Output*

```bash
[GPSeer] + Reading data...
[GPSeer] └── Done reading data.

[GPSeer] + Fitting data...
[GPSeer] └── Done fitting data.

[GPSeer] + Predicting missing data...
[GPSeer] └── Done predicting...

[GPSeer] GPSeer finished!
```

which returns a set of phenotype predictions stored in `predictions.csv`.

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

## Install

Clone this repository and install with pip:

```
git clone https://github.com/harmslab/gpseer.git
cd gpseer
pip install gpseer
```

## Dependencies

1. gpmap : Python API for analyzing genotype phenotype maps.
2. epistasis : Python API for extracting high-order epistasis in genotype-phenotype maps
