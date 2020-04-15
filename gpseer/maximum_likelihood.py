import pandas as pd
import numpy as np

import epistasis

from epistasis.models import (
    EpistasisPipeline,
    EpistasisLogisticRegression,
    EpistasisSpline,
    EpistasisLasso
)

from gpmap import GenotypePhenotypeMap
from gpmap.utils import genotypes_to_mutations
from .utils import (
    gpmap_from_gpmap,
    read_file_to_gpmap,
    read_genotype_file,
    construct_model,
    prep_for_output
)

from . import plot

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
    "--output_root": dict(
        type=str,
        help="""
        Root for all output files (e.g. {root}_predictions.csv,
        {root}_spline-fit.pdf, etc.).  If none, this will be made from the
        input file name
        """,
        default=None
    ),
    "--genotype_file": dict(
        type=str,
        help="""
        A text file with a list of genotypes to predict given the input_file
        and epistasis model.
        """,
        default=None
    ),
    "--overwrite": dict(
        action="store_true",
        help="""
        Overwrite existing output.
        """,
        default=False
    )
}



def predict_to_dataframe(ml_model,genotypes_to_predict=None):
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

def create_stats_output(ml_model):
    """
    Return some stats about the predicted fits.

    Parameters
    ----------
    ml_model : Epistasis model or EpistasisPipeline
        Fitted model.

    Returns
    -------
    stats_df : data frame containing statistics about the fits
    convergence_df : data frame containing information about whether the model
                     for each mutation has converged.
    """

    # Rip out the model parameters
    threshold = None
    spline_order = None
    spline_smoothness = None
    epistasis_order = 1
    for m in ml_model:
        if isinstance(m,EpistasisLogisticRegression):
            threshold = m.threshold
        elif isinstance(m,EpistasisSpline):
            spline_order = m.k
            spline_smoothness = m.s
        elif isinstance(m,EpistasisLasso):
            epistasis_order = m.order
            alpha = m.alpha
        else:
            err = "epistasis model {} not recognized\n".format(m)
            raise RuntimeError(err)

    # Construct a list of mutation names corresponding to each
    # position in the binary genotypes vector.
    mutation_names = []
    for index in ml_model.gpm.encoding_table.index:
        row = ml_model.gpm.encoding_table.loc[index,:]
        if row.wildtype_letter == row.mutation_letter:
            continue

        mutation = "{}{}{}".format(row.wildtype_letter,
                                   row.site_label,
                                   row.mutation_letter)
        mutation_names.append(mutation)

    # Grab binary genotypes above the threshold (if applied)
    if isinstance(ml_model[0], EpistasisLogisticRegression):
        above = ml_model[0].classes == 1
        above_binary = ml_model.gpm.binary[above]
    else:
        above_binary = ml_model.gpm.binary

    binary = ml_model.gpm.binary

    # Record all genotypes seen as as numpy array of binary integers
    genotypes_as_int = np.array([[int(m) for m in bin_genotype]
                                 for bin_genotype in binary],
                                dtype=np.int)

    genotypes_above_as_int = np.array([[int(m) for m in bin_genotype]
                                       for bin_genotype in above_binary],
                                       dtype=np.int)

    # Record the number of times each genotype was seen
    num_obs = np.sum(genotypes_as_int,axis=0)
    num_obs_above = np.sum(genotypes_above_as_int,axis=0)

    # Calculate the epistasis remaining in the map to estimate how many times
    # we need to see each mutation to converge.
    if isinstance(ml_model[0], EpistasisLogisticRegression):
        above = ml_model[0].classes == 1
        above_genotypes = ml_model.gpm.genotypes[above]
        above_phenotypes = ml_model.gpm.phenotypes[above]
        epistasis = 1 - ml_model.score(X=above_genotypes, y=above_phenotypes)
    else:
        epistasis = 1 - ml_model.score()

    # Calculate how many times we need to see each genotype to resolve
    # the map given this amount of epistasis.  This is from Figure 5
    # "epistasis as uncertainty" in the manuscript describing gpseer.
    num_obs_for_convergence = 83.896*epistasis + 1.5843

    # Calculate whether each mutation is expected to be converged
    converged = [n > num_obs_for_convergence for n in num_obs_above]

    # Calculate how high above (or below) convergence this
    # mutation is
    fold_target = [n/num_obs_for_convergence for n in num_obs_above]

    # Construct a data frame with summary statistics
    to_add = {"num_genotypes":len(genotypes_as_int),
              "num_unique_mutations":len(num_obs),
              "explained_variation":ml_model.score(),
              "num_parameters":ml_model.num_of_params,
              "num_obs_to_converge":num_obs_for_convergence,
              "threshold":threshold,
              "spline_order":spline_order,
              "spline_smoothness":spline_smoothness,
              "epistasis_order":epistasis_order,
              "lasso_alpha":alpha}

    stats_df = pd.DataFrame(columns=["parameter","value"])
    for i, a in enumerate(to_add.keys()):
        stats_df.loc[i] = [a,to_add[a]]

    # Construct a data frame describing whether each mutation
    # is converged or not
    convergence_df = pd.DataFrame({"mutation":mutation_names,
                                   "num_obs":num_obs,
                                   "num_obs_above":num_obs_above,
                                   "fold_target":fold_target,
                                   "converged":converged})

    return stats_df, convergence_df

def plots_to_pdf(model,prediction_df,out_root):
    """
    Plot a collection of summary graphs for a prediction, writing them to pdf.
    model: EpistasisPipline object containing completed fit
    prediction_df: prediction_to_dataframe output, containing finalized dataframe
                  with predictions
    out_root: root name for all output pdfs

    returns a list of the plots generated
    """

    plots_written = []

    # Create a spline.  If no spline in pipeline, will be None.
    fig, ax = plot.plot_spline(model,prediction_df)
    if fig is not None:
        fig.savefig("{}_spline-fit.pdf".format(out_root))
        plots_written.append("{}_spline-fit.pdf".format(out_root))

    # Plot correlation between predicted and observed values for training set
    fig, ax = plot.plot_correlation(model,prediction_df)
    fig.savefig("{}_correlation-plot.pdf".format(out_root))
    plots_written.append("{}_correlation-plot.pdf".format(out_root))

    # Plot histograms of values for measured values, training set predictions,
    # and test set predictions
    fig, ax = plot.plot_histograms(model,prediction_df)
    fig.savefig("{}_phenotype-histograms.pdf".format(out_root))
    plots_written.append("{}_phenotype-histograms.pdf".format(out_root))

    return plots_written


def main(
    logger,
    input_file,
    output_root=None,
    wildtype=None,
    threshold=None,
    spline_order=None,
    spline_smoothness=10,
    epistasis_order=1,
    alpha=1,
    nreplicates=None,
    genotype_file=None,
    overwrite=False
):

    expected_outputs = ["_predictions.csv",
                        "_fit-information.csv",
                        "_convergence.csv",
                        "_spline-fit.pdf",
                        "_correlation-plot.pdf",
                        "_phenotype-histograms.pdf"]
    output_root = prep_for_output(input_file,output_root,overwrite,expected_outputs)

    # Read the input file
    logger.info(f"Reading data from {input_file}...")
    gpm = read_file_to_gpmap(input_file, wildtype=wildtype)
    logger.info("└──> Done reading data.")

    # Construct a model based on the input parameters
    logger.info("Constructing a model...")
    model = construct_model(
        threshold=threshold,
        spline_order=spline_order,
        spline_smoothness=spline_smoothness,
        epistasis_order=epistasis_order,
        alpha=alpha
    )
    model.add_gpm(gpm)
    logger.info("└──> Done constructing model.")

    # Fit the model to the data
    logger.info("Fitting data...")

    try:
        model.fit()
    except epistasis.models.utils.FittingError as err:

        # Brittle check to see if this is the univariate spline error.  If it
        # is, return a useful help message.
        if err.args[0].startswith("scipy.interpolate.UnivariateSpline fitting returned more parameters"):
            informative_err = "\n\nspline fit failed.  Try increasing --spline_smoothness\n"
            raise RuntimeError(informative_err)

        # If not, re-raise the error.  If our brittle check for the specific
        # error fails, we'll still get an informative error back from epistasis
        # (and sklearn/scipy)
        raise err

    logger.info("└──> Done fitting data.")

    # Figure out which genotypes to predict.
    genotypes_to_predict = None
    if genotype_file:
        genotypes_to_predict = read_genotype_file(wildtype, genotype_file)

    # Do the actual prediction
    logger.info("Predicting missing data...")
    out_df = predict_to_dataframe(
        model,
        genotypes_to_predict=genotypes_to_predict,
    )
    logger.info("└──> Done predicting.")

    # Calculate some stats
    logger.info("Calculating fit statistics...")
    stats_df, convergence_df = create_stats_output(model)
    logger.info(f"\n\nFit statistics:\n---------------\n\n{stats_df}\n\n")
    logger.info(f"\n\nConvergence:\n------------\n\n{convergence_df}\n\n")
    stats_df.to_csv("{}_fit-information.csv".format(output_root))
    convergence_df.to_csv("{}_convergence.csv".format(output_root))
    logger.info("└──> Done.")

    # Write phenotypes
    output_file = "{}_predictions.csv".format(output_root)
    logger.info(f"Writing phenotypes to {output_file}...")
    out_df.to_csv(output_file)
    logger.info("└──> Done writing predictions!")

    # Plot pdfs of diagnostic graphs
    logger.info(f"Writing plots...")
    plots_written = plots_to_pdf(model,out_df,output_root)
    for w in plots_written:
        logger.info(f"Writing {w}...")
    logger.info("└──> Done plotting!")

    logger.info("GPSeer finished!")
