Input/Output
============

Input
-----

csv file with training data
...........................

GPSeer requires a csv file with the following columns:

* :code:`genotypes`: A string representation of each genotype. Genotype strings
  must all have the same length.  GPSeer interprets each position in the string
  as a site.  It builds the possible set of states at each site based on what
  letters are observed at position.  There can be any number of states at each
  site. If a site does not vary (meaning, for example, it is `G` in every
  genotype), this will be dropped from the model.
* :code:`phenotypes`: A decimal (float) number describing the quantitative
  phenotype of that genotype.

See the example input file `here <https://github.com/harmslab/gpseer/raw/master/examples/example-full.csv>`_.

It can also optionally take columns:

* :code:`stdeviations`: standard deviations on the experimental estimates of the
  input phenotypes.
* :code:`n_replicates`: The number of replicates for this measurement.  This is
  used in conjunction with the :code:`stdeviations` to estimate the standard
  error on the mean of input phenotype estimates.

If standard deviations are included, regression is weighted by phenotype standard
deviation.  If standard deviations and number of replicates are included,
regression is weighted by the standard error on the mean.

Genotypes file (optional)
.........................

The genotypes file is used by the ``estimate-ml`` subcommand to predict the
phenotypes of a user-defined set of genotypes. This file is a list of genotypes
in a text file. See an example `here <https://github.com/harmslab/gpseer/raw/master/examples/genotypes.txt>`_.
If this file is not given, GPSeer will predict the phenotypes of all possible
genotypes given the input data.  Depending on the nature of the map, this could
be a very large number of genotypes.


Output
------

``estimate-ml``
...............

* **{output_root}_predictions.csv**: A csv file containing predicted phenotypes.
  This has predicted phenotypes for both the requested genotypes and the input
  (training) genotypes.

  The first three columns (*genotypes*, *phenotypes*, and *uncertainty*) are
  designed to be passed on to downstream applications, assigning each genotype
  the best guess for their phenotypes (measured, predicted, or below the
  detection threshold).  A few rows of output are shown below.

  +-----------+------------+-------------+----------+--------------+--------------+------------+----------------+-----------------+----------+-------------+
  | genotypes | phenotypes | uncertainty | measured | measured_err | n_replicates | prediction | prediction_err | phenotype_class | binary   | n_mutations |
  +===========+============+=============+==========+==============+==============+============+================+=================+==========+=============+
  | KSTRDCDA  | 12.24      |             | 12.24    |              | 1            | 11.83      | 0.206          | above           | 00000000 | 0           |
  +-----------+------------+-------------+----------+--------------+--------------+------------+----------------+-----------------+----------+-------------+
  | KSTRDCDC  | 8.31       |             | 8.31     |              | 1            | 8.93       | 0.206          | above           | 00000001 | 1           |
  +-----------+------------+-------------+----------+--------------+--------------+------------+----------------+-----------------+----------+-------------+
  | KSTVNCCC  | 3.0        | 0.0         |          |              | 1            | 2.0        | 0.206          | below           | 00011011 | 4           |
  +-----------+------------+-------------+----------+--------------+--------------+------------+----------------+-----------------+----------+-------------+
  | KSTVDCDC  | 5.99       | 0.206       |          |              | 1            | 5.99       | 0.206          | above           | 00010001 | 2           |
  +-----------+------------+-------------+----------+--------------+--------------+------------+----------------+-----------------+----------+-------------+

  + *genotypes*: genotype
  + *phenotypes*: Best guess for phenotype of this genotype.  If this phenotype
    measured, this will be the *measured* value ("KSTRDCDA").  Otherwise, this
    will be the predicted value.  If no threshold is used or the genotype is
    predicted to be above the threshold, this column will have the quantitative
    output of the epistasis model + spline ("KSTVDCDC" above).  If a threshold
    was used and the genotype is predicted to be below it, this will be
    the threshold value ("KSTVNCCC" above).
  + *uncertainty*: Phenotype uncertainty.  If measured, this is the measurement
    uncertainty; if no threshold or above the threshold, this is the epistasis
    in the system; if below the threshold, no uncertainty is assigned.
  + *measured*: Measured phenotype for this genotype.  If this is a prediction,
    this will be empty.
  + *measured_err*: Measured error for this phenotype.  If this is a prediction,
    or no standard deviations were fed into the prediction, this will be empty.
  + *n_replicates*: how many replicates were used to measure phenotype.  If this
    is a prediction or this was not fed into the prediction, this will be 1.
  + *prediction*: Quantitative prediction for this genotype.  A quantitative
    prediction will be made for all genotypes, even those with measured values
    or predicted to be below the threshold.
  + *prediction_err*: Quantitative uncertainty in the prediction.  This will be
    the same for all genotypes: it is :math:`(1 - R^{2})\times mean(phenotype)`
    for the quantitative model.
  + *phenotype_class*: If a threshold was used, this will indicate whether the
    genotype was classified to be above or below the detection threshold.
  + *binary*: Binary representation of the genotype.
  + *n_mutations*: Number of sequence differences between this genotype and the
    wildtype genotype.


* **{output_root}_fit-information.csv**: A csv file containing various output
  from the fit.

  + *num_genotypes*: Number of genotypes in the map.
  + *num_mutations*: Number of unique mutations in the map.
  + *explained_variation*: :math:`R^2` for the quantitiative model
  + *num_obs_converge*: Number of observations of each genotype required to
    allow convergence of an additive model given the amount of epistasis
    in the map.
  + *num_parameters*: Number of parameters in the model.
  + *threshold*: threshold value used in the fit.  If no threshold was used,
    this will be empty.
  + *spline_order*: spline order used in the fit.  If no spline was used, this
    will be empty.
  + *spline_smoothness*: spline smoothness parameter.  If no spline was used,
    this will be empty.
  + *epistasis_order*: the order of the epistatic model used.
  + XXXX NUMBER OF FIT PARAMETERS, FINAL LIKELIHOOD


* **{output_root}_convergence.csv**: A csv file containing information about the
  number of times each mutation was seen across the map and what this predicts
  about model convergence given the epistasis in the model.

  + *mutation*: a unique mutation seen in the map
  + *num_obs*: number of unique genotypes that contain this mutation in the
    experimental observations.
  + *num_obs_above*: number of unique genotypes above the threshold that contain
    this mutation in the experimental observations.
  + *fold_target*: ratio of the number of times this mutation was seen above
    the threshold to the number of observations predicted for model convergence
    (*num_obs_converge*) from the "fit-information.csv" file.  A ratio above one
    means we expect convergence; a ratio less than one means you might be able
    to squeeze more predictive power out of the model by observing that mutation
    across more genetic backgrounds.
  + *converged*: Whether *fold_target* is above one.


* **{output_root}_correlation-plot.pdf**: Correlation plot for
  genotypes with both measured and predicted phenotypes. Each point is a
  genotype.  Both prediction and measurement uncertainty (if available) are
  plotted with error bars.  If a threshold was applied, phenotypes below the
  threshold will be plotted as gray points.  The lower panel indicates the fit
  residuals.  A perfect fit will be randomly distributed about zero.

  .. image:: correlation-plot.png
    :align: center

* **{output_root}_spline-fit.pdf**: If a spline was used, this plot will show
  the transformation mapping the model to the measurements.  Each point is a
  genotype.  The red line indicates the spline fit.  Graphically, the model
  transforms the data such that the red line becomes linear. If a threshold was
  applied, phenotypes below the threshold will be plotted as gray points.

  .. image:: spline-fit.png
    :align: center

* **{output_root}_phenotype-histograms.pdf**: Each panel shows histograms for
  phenotype values.  The top panel shows the histogram for the measured values.
  The middle panel shows the histogram for the model *predictions* of the
  training (measured) values.  The bottom panel shows the distribution of the
  values predicted for the unmeasured values.  A radical mismatch between the
  training set and test set predictions may indicate a mismatch between the
  genotypes used to train the model and the genotypes that are being predicted.

  .. image:: phenotype-histograms.png
    :align: center

``cross-validate``
..................

* **{output_root}_cross-validation-scores.csv**: A csv file containing the
  :math:`R^{2}_{train}` and :math:`R^{2}_{test}` for each resampling of the
  training data.

* **{output_root}_ross-validation-plot.pdf**: Two-dimensional histogram plotting
  :math:`R^{2}_{train}` against :math:`R^{2}_{test}`. Bright colors indicate
  populated regions of the histogram.  The dashed lines indicate the mode of the
  distribution in each dimension.  When :math:`R^{2}_{train} \gg R^{2}_{test}`,
  it indicates the model is being overfit.

  .. image:: cross-validation-plot.png
    :align: center
