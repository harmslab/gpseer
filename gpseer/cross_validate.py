import pandas as pd
import numpy as np
from tqdm import tqdm

from epistasis.stats import split_gpm

from .utils import (
    read_file_to_gpmap,
    construct_model,
    prep_for_output
)
from . import plot

SUBCOMMAND = "cross-validate"

DESCRIPTION = """
cross-validate: Estimate the predictive power of a given model by generating
multiple samples of the training + test subsets of the data and then calculating
a training and test pearson coefficient for each sample.
"""

HELP = """
Estimate the predictive power of a given model by generating
multiple samples of the training + test subsets of the data and then calculating
a training and test pearson coefficient for each sample.
"""

ARGUMENTS = {}

OPTIONAL_ARGUMENTS = {
    "--n_samples": dict(
        type=int,
        help="""
        A CSV file GPSeer will create with final predictions.
        """,
        default=100
    ),
    "--output_root": dict(
        type=str,
        help="""
        Root for all output files (e.g. {root}_predictions.csv,
        {root}_spline-fit.pdf, etc.).  If none, this will be made from the
        input file name.
        """,
        default=None
    ),
    "--train_fraction": dict(
        type=float,
        help="""
        Fraction of data to include in training set.
        """,
        default=0.8
    )
}

def cross_validate_to_dataframe(model,gpm,n_samples=100,train_fraction=0.8):
    """
    Cross validate predictive power of model and return as a data frame of
    training and test scores.

    model: epistasis model to test
    gpm: genotype-phenotype map to train and validate on
    n_samples: number of n_samples
    train_fraction: what fraction of the data to train on

    returns: dataframe with test and train scores
    """

    test_scores = []
    train_scores = []
    for _ in tqdm(range(n_samples), desc="[GPSeer] └──>"):

        # Split the data.
        train_gpm, test_gpm = split_gpm(gpm, fraction=train_fraction)

        # Fit model to data.
        model.add_gpm(train_gpm)
        model.fit()

        X = train_gpm.genotypes
        y = train_gpm.phenotypes
        train_score = model.score(X=X, y=y)
        train_scores.append(train_score)

        X = test_gpm.genotypes
        y = test_gpm.phenotypes
        test_score = model.score(X=X, y=y)
        test_scores.append(test_score)

    df = pd.DataFrame({'test_scores': test_scores,
                       'train_scores': train_scores})

    return df


def main(
    logger,
    input_file,
    n_samples,
    output_root=None,
    train_fraction=0.8,
    wildtype=None,
    threshold=None,
    spline_order=None,
    spline_smoothness=10,
    epistasis_order=1,
    alpha=1,
    overwrite=False
):

    # Expected files this will create
    expected_outputs = ["_cross-validation-scores.csv",
                        "_cross-validation-plot.pdf"]
    output_root = prep_for_output(input_file,output_root,overwrite,expected_outputs)

    # Read data
    logger.info(f"Reading data from {input_file}...")
    gpm = read_file_to_gpmap(input_file, wildtype=wildtype)
    logger.info("└──> Done reading data.")

    # Fit the model by itself
    logger.info("Fitting all data data...")
    full_model = construct_model(
        threshold=threshold,
        spline_order=spline_order,
        spline_smoothness=spline_smoothness,
        epistasis_order=epistasis_order,
        alpha=alpha
    )
    full_model.add_gpm(gpm)
    full_model.fit()
    logger.info("└──> Done fitting data.")

    logger.info("Sampling the data...")
    sub_model = construct_model(
        threshold=threshold,
        spline_order=spline_order,
        spline_smoothness=spline_smoothness,
        epistasis_order=epistasis_order,
        alpha=alpha
    )
    df = cross_validate_to_dataframe(sub_model,gpm,n_samples,train_fraction)
    logger.info("└──> Done sampling data.")

    # -------------------------------------------------------------------------
    # Plot results
    # -------------------------------------------------------------------------

    output_pdf = "{}_cross-validation-plot.pdf".format(output_root)
    logger.info(f"Plotting {output_pdf}...")
    fig, ax = plot.plot_test_train(df,bin_scalar=5)
    fig.savefig(output_pdf)
    logger.info("└──> Done writing data.")

    # -------------------------------------------------------------------------
    # Write output file
    # -------------------------------------------------------------------------

    output_file = "{}_cross-validation-scores.csv".format(output_root)
    logger.info(f"Writing scores to {output_file}...")
    df.to_csv(output_file)
    logger.info("└──> Done writing data.")

    logger.info("GPSeer finished!")
