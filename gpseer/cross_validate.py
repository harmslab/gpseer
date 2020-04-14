import pandas as pd
import numpy as np
from tqdm import tqdm

from epistasis.stats import split_gpm

from .utils import (
    read_file_to_gpmap,
    construct_model
)
from . import plot

import os

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
    "--output_file": dict(
        type=str,
        help="""
        A CSV with the scores returned by each sample from the goodness-of-fit calculation.
        """,
        default="scores.csv"
    ),
    "--train_fraction": dict(
        type=float,
        help="""
        Fraction of data to include in training set.
        """,
        default=0.8
    )
}


def main(
    logger,
    input_file,
    n_samples,
    output_file='scores.csv',
    train_fraction=0.8,
    wildtype=None,
    threshold=None,
    spline_order=None,
    spline_smoothness=10,
    epistasis_order=1,
    overwrite=False
):

    if os.path.isfile(output_file):
        if not overwrite:
            err = "output_file '{}' already exists.\n".format(output_file)
            raise FileExistsError(err)
        else:
            os.remove(output_file)

    logger.info(f"Reading data from {input_file}...")
    gpm = read_file_to_gpmap(input_file, wildtype=wildtype)
    logger.info("└──> Done reading data.")

    # Fit the model by itself
    logger.info("Fitting all data data...")
    full_model = construct_model(
        threshold=threshold,
        spline_order=spline_order,
        spline_smoothness=spline_smoothness,
        epistasis_order=epistasis_order
    )
    full_model.add_gpm(gpm)
    full_model.fit()
    logger.info("└──> Done fitting data.")

    logger.info("Sampling the data...")
    test_scores = []
    train_scores = []
    for _ in tqdm(range(n_samples), desc="[GPSeer] └──>"):

        # Split the data.
        train_gpm, test_gpm = split_gpm(gpm, fraction=train_fraction)
        # Fit model to data.
        model = construct_model(
            threshold=threshold,
            spline_order=spline_order,
            spline_smoothness=spline_smoothness,
            epistasis_order=epistasis_order
        )
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

    logger.info("└──> Done sampling data.")

    # -------------------------------------------------------------------------
    # Plot results
    # -------------------------------------------------------------------------

    # Figure out the root for output graph
    out_root = output_file.split(".")
    if out_root[-1] in ["csv","txt","text","xls","xlsx"]:
        out_root = ".".join(out_root[:-1])
    else:
        out_root = output_file

    output_pdf = "{}_correlation-plot.pdf".format(out_root)
    logger.info("Plotting {}...".format(output_pdf))
    fig, ax = plot.plot_test_train(df,bin_scalar=5)
    fig.savefig(output_pdf)
    logger.info("└──> Done writing data.")

    # -------------------------------------------------------------------------
    # Write output file
    # -------------------------------------------------------------------------

    logger.info(f"Writing scores to {output_file}...")
    df.to_csv(output_file)
    logger.info("└──> Done writing data.")

    logger.info("GPSeer finished!")
