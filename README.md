# GPSeer
*Simple software for inferring missing data in sparsely measured genotype-phenotype maps*

[![GPSeer tests](https://github.com/harmslab/gpseer/workflows/GPSeer%20tests/badge.svg)](https://github.com/harmslab/gpseer/actions?query=workflow%3A%22GPSeer+tests%22)

## Basic Usage

Install gpseer using pip:
```bash
pip install gpseer
```

To use as a command line, call `gpseer` on an input `.csv` file containing genotype-phenotype data.

The [API Demo.ipynb](https://nbviewer.jupyter.org/github/harmslab/gpseer/blob/master/examples/API%20Demo.ipynb)
demonstrates how to use GPSeer in a Jupyter notebook.

+ [Documentation](https://gpseer.readthedocs.io) 
+ [Tutorial](https://gpseer.readthedocs.io/en/latest/tutorial.html)



### Downloading the example

To get started, use GPSeer's `fetch-example` command to download an example from its Github repo.

Download the gpseer example and explore the example input data:
```
# fetch data from Github page.
> gpseer fetch-example

[GPSeer] Downloading files to /examples...
[GPSeer] └──>: 100%|██████████████████| 3/3 [00:00<00:00,  9.16it/s]
[GPSeer] └──> Done!

# Change into the example directory and checkout the files that were downloaded
> cd examples/
> ls

API Demo.ipynb
example-full.csv
example-test.csv
example-train.csv
Generate Dataset.ipynb
genotypes.txt
pfcrt-raw-data.csv
```

### Predicting missing data using ML model.

Estimate the maximum likelihood additive model on the training set and predict all missing genotypes. The predictions will be written to a file named `"example-train_predictions.csv"`.

```
> gpseer estimate-ml example-train.csv

[GPSeer] Reading data from example-train.csv...
[GPSeer] └──> Done reading data.
[GPSeer] Constructing a model...
[GPSeer] └──> Done constructing model.
[GPSeer] Fitting data...
[GPSeer] └──> Done fitting data.
[GPSeer] Predicting missing data...
[GPSeer] └──> Done predicting.
[GPSeer] Calculating fit statistics...
[GPSeer]

Fit statistics:
---------------

              parameter     value
0         num_genotypes       128
1  num_unique_mutations         8
2   explained_variation  0.985186
3        num_parameters         9
4   num_obs_to_converge   2.82714
5             threshold      None
6          spline_order      None
7     spline_smoothness      None
8       epistasis_order         1


[GPSeer]

Convergence:
------------

  mutation  num_obs  num_obs_above  fold_target  converged
0      F0K       64             64    22.637735       True
1      S1Y       69             69    24.406308       True
2      Q2T       63             63    22.284020       True
3      R3V       70             70    24.760023       True
4      N4D       62             62    21.930306       True
5      A5C       69             69    24.406308       True
6      C6D       65             65    22.991450       True
7      C7A       64             64    22.637735       True


[GPSeer] └──> Done.
[GPSeer] Writing phenotypes to example-train_predictions.csv...
[GPSeer] └──> Done writing predictions!
[GPSeer] Writing plots...
[GPSeer] Writing example-train_correlation-plot.pdf...
[GPSeer] Writing example-train_phenotype-histograms.pdf...
[GPSeer] └──> Done plotting!
[GPSeer] GPSeer finished!
```

### Compute the predictive power of the model by cross-validation

Estimate how well your model is predicting data using the "cross-validate"
subcommand. Try the example below where we generate 100 subsets from the data
and compute your prediction scores.

```
> gpseer cross-fit example-test.csv

[GPSeer] Reading data from example-train.csv...
[GPSeer] └──> Done reading data.
[GPSeer] Fitting all data data...
[GPSeer] └──> Done fitting data.
[GPSeer] Sampling the data...
[GPSeer] └──>: 100%|████████████████████| 100/100 [00:03<00:00, 25.90it/s]
[GPSeer] └──> Done sampling data.
[GPSeer] Plotting example-train_cross-validation-plot.pdf...
[GPSeer] └──> Done writing data.
[GPSeer] Writing scores to example-train_cross-validation-scores.csv...
[GPSeer] └──> Done writing data
```
