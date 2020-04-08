Input/Output
============

Input (CSV) file
----------------

The input file to GPSeer always requires CSV with the following column required:

* genotypes
* phenotypes
* stdeviations
* n_replicates

See the example input file `here <https://github.com/harmslab/gpseer/blob/master/examples/example-full.csv>`_.

Genotypes file
--------------

The genotypes file is used by the ``estimate-ml`` subcommand to predict a user-defined list of genotypes. This file is a simple list of genotypes in a text file. See an example `here <https://github.com/harmslab/gpseer/blob/master/examples/genotypes.txt>`_


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
