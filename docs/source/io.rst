Input/Output
============

Input (CSV) file
----------------

GPSeer requires a csv file with the following columns:

* :code:`genotypes`: A string representation of each genotype. Genotype strings
  must all have the same length.  GPSeer interprets each position in the string
  as a site.  It builds the possible set of states at each site based on what
  letters are observed at position.
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

Genotypes file
--------------

The genotypes file is used by the ``estimate-ml`` subcommand to predict a
user-defined list of genotypes. This file is a simple list of genotypes in a
text file. See an example `here <https://github.com/harmslab/gpseer/blob/master/examples/genotypes.txt>`_


``estimate-ml`` output
----------------------

The output for ``estimate-ml`` is a CSV file (``predictions.csv``) with the following columns:

* genotypes
* phenotypes
* uncertainty
* measured
* measured_err
* n_replicates
* prediction
* prediction_err
* phenotype_class
* binary
* n_mutations

``goodness-of-fit`` output
--------------------------

The output for ``goodness-of-fit`` is a CSV file (``scores.csv``) with the scores for a given number of samples.
