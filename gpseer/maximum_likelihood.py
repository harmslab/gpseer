import pandas as pd
import numpy as np
from epistasis.models import (
    EpistasisPipeline,
    EpistasisLogisticRegression,
    EpistasisSpline,
    EpistasisLinearRegression
)

from gpmap import GenotypePhenotypeMap
from .io import (
    gpmap_from_gpmap,
    read_input_file,
    read_genotype_file,
    write_output_file
)

# Cutoff for zero
NUMERICAL_CUTOFF = 1e-10


def construct_model(
    threshold=None,
    spline_order=None,
    spline_smoothness=None,
    epistasis_order=1,
    ):
    """Build an epistasis pipeline based on model
    parameters given.

    If a threshold is given, add a logistic classifier to
    the pipeline first; otherwise, no classifier is applied.

    If a spline_order or smoothness is given, add a nonlinear
    spline model with the given 'smoothness' and order.

    Returns
    -------
    model : EpistasisPipeline
        an epistasis pipeline with a the pieces mentioned above
        based on the arguments given.
    """
    model = EpistasisPipeline([])

    if threshold:
        model.append(EpistasisLogisticRegression(threshold=threshold))

    if spline_order and spline_smoothness:
        model.append(EpistasisSpline(k=spline_order, s=spline_smoothness))

    model.append(EpistasisLinearRegression(order=epistasis_order))

    return model


def fit_ml_model(
    model,
    df,
    wildtype,
    ):
    """Estimate the maximum likelihood model for a given
    genotype-phenotype map.
    """
    gpm = GenotypePhenotypeMap.read_dataframe(df, wildtype)
    model.add_gpm(gpm)
    model.fit()
    return model


def get_ml_predictions_df(
        ml_model,
        genotypes_to_predict=None
    ):
    """Build a dataframe of predictions from the ML model
    for a given list of genotypes. If not genotypes are given, the model
    will predict for all missing genotypes.
    """
    if not genotypes_to_predict:
        genotypes_to_predict = ml_model.gpm.get_missing_genotypes()

    predicted_phenotypes = ml_model.predict(X=genotypes_to_predict)
    predicted_err = (1 - ml_model.score()) * np.mean(ml_model.gpm.phenotypes)
    # Drop any nonsense uncertainty.
    if predicted_err < 0 and np.abs(predicted_unpredicted_errcertainty) < NUMERICAL_CUTOFF:
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

    # Get any measured genotypes found in the original dataset
    mapper = ml_model.gpm.map("genotypes", "phenotypes")
    out_data["measured"] = [mapper[g] if g in mapper else None for g in genotypes_to_predict]

    # Get any measured error in origina dataset
    mapper = ml_model.gpm.map("genotypes", "stdeviations")
    out_data["measured_err"] = [mapper[g] if g in mapper else None for g in genotypes_to_predict]

    # Make sane column order
    column_order = ["genotypes","phenotypes","uncertainty",
                    "measured","measured_err","n_replicates",
                    "prediction","prediction_err","phenotype_class",
                    "binary","n_mutations"]

    # Add a column for classifier predictions if a classifier was used.
    if isinstance(ml_model[0], EpistasisLogisticRegression):
        out_data["phenotype_class"] = ml_model[0].predict(X=genotypes_to_predict)
    else:
        column_order.remove("phenotype_class")

    df = (
        out_data[column_order]
        .sort_values("binary")
        .reset_index(drop=True)
    )
    return df


def run_estimate_ml(
        logger,
        input_file,
        output_file,
        wildtype=None,
        threshold=None,
        spline_order=None,
        spline_smoothness=10,
        epistasis_order=1,
        nreplicates=None,
        genotype_file=None,
    ):
    logger.info("Reading input data...")
    input_df = read_input_file(input_file)
    logger.info("Finished reading input data.")

    logger.info("Constructing a model...")
    model = construct_model(
        threshold=threshold,
        spline_order=spline_order,
        spline_smoothness=spline_smoothness,
        epistasis_order=epistasis_order
    )

    logger.info("Fitting model to data...")
    model = fit_ml_model(
        model,
        input_df,
        wildtype
    )

    genotypes_to_predict = None
    if genotype_file:
        genotypes_to_predict = read_genotype_file(wildtype, genotype_file)

    logger.info("Predicting phenotypes...")
    out_df = get_ml_predictions_df(
        model,
        genotypes_to_predict=genotypes_to_predict,
    )

    logger.info("Writing phenotypes to file...")
    write_output_file(output_file, out_df)
    logger.info("Done!")
