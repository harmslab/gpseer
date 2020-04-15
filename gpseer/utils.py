import pandas as pd

from gpmap import GenotypePhenotypeMap
from epistasis.models import (
    EpistasisPipeline,
    EpistasisLogisticRegression,
    EpistasisSpline,
    EpistasisLasso
)

import os

def read_file_to_gpmap(
    input_file_name,
    wildtype=None,
):
    """Read the input file for GPSeer.

    This should be a CSV file with the following columns:
    genotypes, phenotypes, n_replicates, stdeviations
    """
    df = pd.read_csv(input_file_name)
    required_columns = ["genotypes","phenotypes"]
    optional_columns = ["stdeviations","n_replicates"]
    for c in required_columns:
        try:
            df[c]
        except AttributeError:
            err = "input file ({}) must contain a column labeled '{}'".format(input_file_name, c)
            return AttributeError(err)

    # If wildtype is not given, use the first genotype in the input file.
    if not wildtype:
        wildtype = df.loc[0, 'genotypes']

    # Fill in missing columns for the GenotypePhenotypeMap
    for col in optional_columns:
        if col not in df.columns:
            df[col] = None

    gpm = GenotypePhenotypeMap.read_dataframe(df, wildtype)
    return gpm


def _raise_line_err(msg,line):
    err = "\n\n{}\n\n".format(msg)
    err += "Line:\n\n{}\n\n".format(line.strip())
    raise ValueError(err)


def read_genotype_file(wildtype, genotype_file_name):
    """Read a file with a list of genotypes to predict.
    """
    genotype_size = len(wildtype)

    out_genotypes = []
    with open(genotype_file_name) as f:
        for line in f.readlines():
            genotype = line.strip()

            # Skip blank lines and # comments
            if genotype == "" or genotype.startswith("#"):
                continue

            # Look for line with more than one genotype
            if len(genotype.split()) > 1:
                _raise_line_err("Mangled line. More than one genotype?",line)

            # Look for line with incorrect number of sites
            if len(genotype) != genotype_size:
                _raise_line_err("Mangled line. Genotype length does not match {}".format(wildtype),line)

            out_genotypes.append(genotype)

    return out_genotypes


def gpmap_from_gpmap(
    original_gpmap,
    new_genotypes,
    new_phenotypes,
    new_n_replicates=1,
    new_stdeviations=None,
):
    """Generate a GenotypePhenotypeMap from another GenotypePhenotypeMap
    with new genotypes, phenotypes, n_replicates, and stdevations.
    """
    gpm = original_gpmap
    return GenotypePhenotypeMap(
        wildtype=gpm.wildtype,
        mutations=gpm.mutations,
        genotypes=new_genotypes,
        phenotypes=new_phenotypes,
        stdeviations=new_stdeviations,
        n_replicates=new_n_replicates
    )

def construct_model(
    threshold=None,
    spline_order=None,
    spline_smoothness=10,
    epistasis_order=1,
    alpha=1,
):
    """Build an epistasis pipeline based on model
    parameters given.

    If a threshold is given, add a logistic classifier to
    the pipeline first; otherwise, no classifier is applied.

    If a spline_order or smoothness is given, add a nonlinear
    spline model with the given 'smoothness' and order.

    alpha is a control parameter for the lasso regression

    Returns
    -------
    model : EpistasisPipeline
        an epistasis pipeline with a the pieces mentioned above
        based on the arguments given.
    """
    model = EpistasisPipeline()

    if threshold is not None:
        model.append(EpistasisLogisticRegression(threshold=threshold))

    if spline_order and spline_smoothness:
        model.append(EpistasisSpline(k=spline_order, s=spline_smoothness))

    model.append(EpistasisLasso(order=epistasis_order,alpha=alpha))

    return model

def prep_for_output(input_file,output_root=None,overwrite=False,expected_outputs=[]):
    """
    Prep output files and output root.

    input_file: input file for calculation.
    output_root: output_root to stick on all output files.  If None, build from
                 input_file.
    overwrite: whether or not to overwrite existing files.
    expected_outputs: expected output file suffixes to check for.

    returns: output_root.
    """

    # Construct an output_root if not specified
    if output_root is None:
        split = input_file.split(".")
        if len(split) == 1:
            output_root = split[0]
        else:
            output_root = ".".join(split[:-1])


    # Make sure we're not going to wipe out an existing file
    for e in expected_outputs:
        output_file = "{}{}".format(output_root,e)
        if os.path.isfile(output_file):
            if not overwrite:
                err = "output_file '{}' already exists.\n".format(output_file)
                raise FileExistsError(err)
            else:
                os.remove(output_file)

    return output_root
