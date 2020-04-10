import pandas as pd
import numpy as np

from epistasis.models import (
    EpistasisPipeline,
    EpistasisLogisticRegression,
    EpistasisSpline,
    EpistasisLinearRegression
)

from gpmap import GenotypePhenotypeMap
from gpmap.utils import genotypes_to_mutations
from .utils import (
    gpmap_from_gpmap,
    read_file_to_gpmap,
    read_genotype_file,
    construct_model
)

from .plot import plots_to_pdf

import os

# Cutoff for zero
NUMERICAL_CUTOFF = 1e-10

SUBCOMMAND = "estimate-ml"

DESCRIPTION = """
estimate-ml: GPSeer's maximum likelihood calculator—
predicts the maximum-likelihood estimates for missing
phenotypes in a sparsely sampled genotype-phenotype map.
"""

HELP = """
Predict the maximum-likelihood estimates for missing
phenotypes in a sparsely sampled genotype-phenotype map.
"""

ARGUMENTS = {}
OPTIONAL_ARGUMENTS = {
    "--output_file": dict(
        type=str,
        help="""
        A CSV file GPSeer will create with final predictions.
        """,
        default="predictions.csv"
    ),
    "--genotype_file": dict(
        type=str,
        help="""
        A text file with a list of genotypes to predict given the input_file
        and epistasis model.
        """,
        default=None
    ),
}



def predict_to_dataframe(
    ml_model,
    genotypes_to_predict=None
):
    """
    Predict a list of genotypes using an ML model.

    The predictions are returned in a dataframe with the following columns:
    "genotypes", "phenotypes", "uncertainty", "measured", "measured_err",
    "n_replicates", "prediction", "prediction_err", "phenotype_class",
    "binary", "n_mutations"

    Parameters
    ----------
    ml_model : Epistasis model or EpistasisPipeline
        Fitted model.

    genotypes_to_predict : list
        List of genotypes to predict.

    Returns
    -------
    df : DataFrame
        Formatted and sorted dataframe with predictions from the given model.
    """

    # If no genotypes are specified, predict them all
    if not genotypes_to_predict:
        genotypes_to_predict = ml_model.gpm.get_missing_genotypes()

    # Predict on the training data as well as the missing data
    measured_genotypes = ml_model.gpm.genotypes[:]
    genotypes_to_predict.extend(measured_genotypes)

    # Predict!
    predicted_phenotypes = ml_model.predict(X=genotypes_to_predict)

    # If there is a detection threshold, we want to make sure
    # that genotypes below the detection threshold are not
    # counted as genotypes contributing to the uncertainty
    # in the epistasis model for the predicted genotypes
    # above the threshold.
    if isinstance(ml_model[0], EpistasisLogisticRegression):
        above = ml_model[0].classes == 1
        above_genotypes = ml_model.gpm.genotypes[above]
        above_phenotypes = ml_model.gpm.phenotypes[above]
        predicted_err = (
            (1 - ml_model.score(X=above_genotypes, y=above_phenotypes))
            * np.mean(above_phenotypes)
        )
    else:
        predicted_err = ((1 - ml_model.score()) * np.mean(ml_model.gpm.phenotypes))


    # Drop any nonsense uncertainty.
    if predicted_err < 0 and np.abs(predicted_err) < NUMERICAL_CUTOFF:
        predicted_err = 0
    predicted_err = np.ones(len(predicted_phenotypes)) * predicted_err

    # Construct a dataframe from predictions
    output_gpm = gpmap_from_gpmap(
        ml_model.gpm,
        genotypes_to_predict,
        predicted_phenotypes,
    )

    out_data = output_gpm.data
    out_data["prediction"] = predicted_phenotypes
    out_data["prediction_err"] = predicted_err
    out_data["uncertainty"] = predicted_err

    # Maps genotype to phenotype and stdeviation within dataset
    phenotype_mapper = ml_model.gpm.map("genotypes", "phenotypes")
    err_mapper = ml_model.gpm.map("genotypes", "stdeviations")

    # Get any measured genotypes found in the original dataset.  Stick them
    # into the "measured" and "phenotypes" columns
    out_data["measured"] = [phenotype_mapper[g] if g in phenotype_mapper else None
                            for g in genotypes_to_predict]

    # Get any measured error in original dataset
    out_data["measured_err"] = [err_mapper[g] if g in err_mapper else None
                                for g in genotypes_to_predict]

    # Add a column for classifier predictions if a classifier was used.  Give
    # the sane name "above" or "below"
    if isinstance(ml_model[0], EpistasisLogisticRegression):
        classes = ml_model[0].predict(X=genotypes_to_predict)
        out_data["phenotype_class"] = ["above" if c == 1 else "below" for c in classes]

    # Do some clean up on the phenotypes and uncertainty so they make sense
    # whatever precise model was used.
    phenotypes = []
    uncertainty = []
    for i, g in enumerate(genotypes_to_predict):

        # Set phenotypes for the measured values to be the measured values, not
        # the predictions.
        if g in measured_genotypes:
            phenotypes.append(phenotype_mapper[g])
            uncertainty.append(err_mapper[g])
        else:
            phenotypes.append(out_data["phenotypes"].iloc[i])

            # Deal with uncertainty if threshold was used
            try:
                if out_data["phenotype_class"].iloc[i] == "below":
                    uncertainty.append(0)
                else:
                    uncertainty.append(out_data["uncertainty"].iloc[i])
            except KeyError:
                uncertainty.append(out_data["uncertainty"].iloc[i])

    # Record phenotypes and uncertainties
    out_data["phenotypes"] = phenotypes
    out_data["uncertainty"] = uncertainty

    # Make sane column order
    column_order = ["genotypes","phenotypes","uncertainty",
                    "measured","measured_err","n_replicates",
                    "prediction","prediction_err","phenotype_class",
                    "binary","n_mutations"]
    try:
        out_data["phenotype_class"]
    except KeyError:
        column_order.remove("phenotype_class")

    df = (
        out_data[column_order]
        .sort_values("binary")
        .reset_index(drop=True)
    )
    return df


def main(
    logger,
    input_file,
    output_file="predictions.csv",
    wildtype=None,
    threshold=None,
    spline_order=None,
    spline_smoothness=10,
    epistasis_order=1,
    nreplicates=None,
    genotype_file=None,
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

    logger.info("Constructing a model...")
    model = construct_model(
        threshold=threshold,
        spline_order=spline_order,
        spline_smoothness=spline_smoothness,
        epistasis_order=epistasis_order
    )
    model.add_gpm(gpm)
    logger.info("└──> Done constructing model.")

    logger.info("Fitting data...")
    model.fit()
    logger.info("└──> Done fitting data.")

    genotypes_to_predict = None
    if genotype_file:
        genotypes_to_predict = read_genotype_file(wildtype, genotype_file)

    logger.info("Predicting missing data...")
    out_df = predict_to_dataframe(
        model,
        genotypes_to_predict=genotypes_to_predict,
    )
    logger.info("└──> Done predicting.")

    # Figure out the root for any output graphs
    out_root = output_file.split(".")
    if out_root[-1] in ["csv","txt","text","xls","xlsx"]:
        out_root = ".".join(out_root[:-1])
    else:
        out_root = output_file

    # Plot pdfs of diagnostic graphs
    plots_to_pdf(model,out_df,out_root)

    logger.info(f"Writing phenotypes to {output_file}...")
    out_df.to_csv(output_file)
    logger.info("└──> Done writing predictions!")

    logger.info("GPSeer finished!")
