import pandas as pd
import numpy as np
from tqdm import tqdm

from epistasis.stats import split_gpm

from .utils import (
    read_file_to_gpmap,
    construct_model
)

SUBCOMMAND = "goodness-of-fit"

DESCRIPTION = """
goodness-of-fit: Estimate the 'goodness of fit' for a given model
by generating multiple samples of training + test subsets of the data and
calculate the pearson coefficient for each sample.
"""

HELP = """
Estimate the 'goodness of fit' for a given model
by generating multiple samples of training + test subsets of the data and
calculate the pearson coefficient for each sample.
"""

ARGUMENTS = {
    "n_samples": dict(
        type=int,
        help="""
        A CSV file GPSeer will create with final predictions.
        """,
        default=100
    )
}

OPTIONAL_ARGUMENTS = {
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
):
    logger.info(f"Reading data from {input_file}...")
    gpm = read_file_to_gpmap(input_file, wildtype=wildtype)
    logger.info("└──> Done reading data.")

    logger.info("Sampling the data...")
    scores = []
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

        X = test_gpm.genotypes
        y = test_gpm.phenotypes
        score = model.score(X=X, y=y)
        scores.append(score)
    logger.info("└──> Done sampling data.")

    logger.info(f"Writing scores to {output_file}...")
    df = pd.DataFrame({'scores': scores})
    df.to_csv(output_file)
    logger.info("└──> Done writing data.")

    logger.info("GPSeer finished!")
