
Quick Start
===========

This is a very quick introduction to gpseer that demonstrates the basic modes of
the software. A complete tutorial--including how to interpret the outputs of the
calculation--is given on the `tutorial <tutorial.html>`_ page.

The `API Demo.ipynb <https://nbviewer.jupyter.org/github/harmslab/gpseer/blob/master/examples/API%20Demo.ipynb>`_
demonstrates how to use GPSeer in a Jupyter notebook.

Downloading examples
--------------------

Download the gpseer examples:

.. code-block:: console

    > gpseer fetch-example

    [GPSeer] Downloading files to /examples...
    [GPSeer] └──>: 100%|██████████████████| 3/3 [00:00<00:00,  9.16it/s]
    [GPSeer] └──> Done!


.. code-block:: console

    > cd examples/
    > ls

    API Demo.ipynb
    example-full.csv
    example-test.csv
    example-train.csv
    Generate Dataset.ipynb
    genotypes.txt
    pfcrt-raw-data.csv


Predicting unmeasured phenotypes of from a collection of measured phenotypes
----------------------------------------------------------------------------

Estimate the maximum likelihood additive model on the training set and predict
all missing genotypes.

.. code-block:: console

    > gpseer estimate-ml example-train.csv

    [GPSeer] Reading data from example-test.csv...
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
    2   explained_variation  0.984436
    3   num_obs_to_converge   2.89004
    4             threshold      None
    5          spline_order      None
    6     spline_smoothness      None
    7       epistasis_order         1


    [GPSeer]

    Convergence:
    ------------

      mutation  num_obs  fold_target  converged
    0      K0F       64    22.145002       True
    1      S1Y       59    20.414924       True
    2      T2Q       63    21.798986       True
    3      R3V       58    20.068908       True
    4      D4N       62    21.452971       True
    5      C5A       69    23.875080       True
    6      D6C       65    22.491018       True
    7      A7C       64    22.145002       True


    [GPSeer] └──> Done.
    [GPSeer] Writing phenotypes to example-test_predictions.csv...
    [GPSeer] └──> Done writing predictions!
    [GPSeer] Writing plots...
    [GPSeer] └──> Done plotting!
    [GPSeer] GPSeer finished!

Your predictions will be in :code:`example-train_predictions.csv`.  Several
other .csv files and graphs will be generated.  For a full description of
this output, see the `Input/Output <io.html>`_ page.


Compute the predictive power of the model by cross-validation
-------------------------------------------------------------

Estimate how well your model is predicting data using the "cross-validate"
subcommand. Try the example below where we generate 10 subsets from the data
and compute our prediction scores.

.. code-block:: console

    > gpseer goodness-of-fit example-train.csv

    [GPSeer] Reading data from example-train.csv...
    [GPSeer] └──> Done reading data.
    [GPSeer] Fitting all data data...
    [GPSeer] └──> Done fitting data.
    [GPSeer] Sampling the data...
    [GPSeer] └──>: 100%█████████████   ████| 100/100 [00:03<00:00, 26.76it/s]
    [GPSeer] └──> Done sampling data.
    [GPSeer] Plotting example-train_cross-validation-plot.pdf...
    [GPSeer] └──> Done writing data.
    [GPSeer] Writing scores to example-train_cross-validation-scores.csv...
    [GPSeer] └──> Done writing data.
    [GPSeer] GPSeer finished!

A cross-validation plot called :code:`example_train_cross-validation-plot.pdf`
will be created, along with a csv file.  For a full description of
this output, see the `Input/Output`_ page.
