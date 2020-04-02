import sys, os, warnings
import logging
from copy import deepcopy
from traitlets.config import Application, Config, catch_config_error
from traitlets import (
    Unicode,
    Instance,
    Int,
    Float,
    Enum,
    Bool
)

import pandas as pd
import numpy as np

from gpmap import GenotypePhenotypeMap

from .base import BaseApp

# Cutoff for zero
NUMERICAL_CUTOFF = 1e-10


def _raise_line_err(msg,line):
    err = "\n\n{}\n\n".format(msg)
    err += "Line:\n\n{}\n\n".format(line.strip())
    raise ValueError(err)

def _load_genotype_file(genotype_file,wildtype):
    """
    Function for loading file containing genotypes to predict.
    """

    genotype_size = len(wildtype)

    out_genotypes = []
    with open(genotype_file) as f:
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

def _construct_output(to_df,original_df,genotypes_to_predict,wildtype):
    out = original_df.copy()
    out = out.rename(columns={"phenotypes":"measured",
                              "stdeviations":"measured_err"})
    out["phenotypes"] = out["measured"]
    out["uncertainty"] = out["measured_err"]

    out = pd.merge(pd.DataFrame(to_df),out,how="left",sort=True,on="genotypes")
    #out = out.drop("Unnamed: 0",axis=1)

    mask = np.isnan(out["phenotypes"])
    out.loc[mask,"phenotypes"] = out["prediction"][mask]
    out.loc[mask,"uncertainty"] = out["prediction_err"][mask]

    # Clean up binary and n_mutations columns
    tmp_gpm = GenotypePhenotypeMap(genotypes=genotypes_to_predict,wildtype=wildtype)
    all_columns = np.copy(tmp_gpm.data.columns)
    all_columns = [k for k in all_columns if k not in ["genotypes","binary","n_mutations"]]
    tmp_gpm = tmp_gpm.data.drop(columns=all_columns)

    out = out.drop(columns=["binary","n_mutations"])
    out = pd.merge(out,tmp_gpm)

    # Make sane column order
    column_order = ["genotypes","phenotypes","uncertainty",
                    "measured","measured_err","n_replicates",
                    "prediction","prediction_err","phenotype_class",
                    "binary","n_mutations"]

    # See if we're writing out a phenotype class.  If not, remove it
    try:
        to_df["phenotype_class"]
    except KeyError:
        column_order.remove("phenotype_class")

    # Order and sort by binary
    out = out[column_order]
    out = out.sort_values("binary")
    out = out.reset_index(drop=True)

    return out


class GPSeer(BaseApp):

    description = "A tool for predicting phenotypes in a sparsely sampled genotype-phenotype maps."

    # --------------------- Traits specific to Seer App ------------------

    nreplicates = Int(
        None,
        allow_none=True,
        help="Number of replicates for sampling classifier/spline uncertainty.",
        config=True
    )

    genotype_file = Unicode(
        None,
        allow_none=True,
        help="File with with set of genotypes to predict",
        config=True
    )

    def start(self):
        self.zinitialize()

        self.log.info("Running GPSeer on {}. Look for a {} file with your results.".format(self.infile, self.outfile))

        # -------------------------------------------------------------------
        # Load data and construct gpmap to do fit
        gpm = self.read_data()

        # Load in genotypes to predict
        if self.genotype_file:
            genotypes_to_predict = _load_genotype_file(self.genotype_file,wildtype)
            genotypes_to_predict.extend(list(df.genotypes))
            genotypes_to_predict = list(set(genotypes_to_predict))

        # If none are specified, predict all genotypes
        else:
            genotypes_to_predict = gpm.get_all_possible_genotypes()

        # Decide whether or not to do logistic regression
        if self.use_logistic:
            # Make sure that the dataset contains data on both sides of specified
            # threshold before trying to train
            if np.sum(gpm.data.phenotypes < self.threshold) == 0:
                self.log.warning("No data found below threshold.  No classifier will be applied.")
                # Pop the classifier from the model pipeline if no phenotypes are below threshold.
                self.model.pop(0)

            elif np.sum(df.phenotypes >= self.threshold)  == 0:

                self.log.warning("No data found above threshold.  All predictions will be below detection limit.")

                # For the case in which all training genotypes have phenotypes below
                # the detection threshold, spit out a dataframe with everyone
                # below the detection limit.
                to_df = {"genotypes":genotypes_to_predict}

                template = np.ones(len(to_df["genotypes"]))

                # Create predictions
                to_df["prediction"] = template*self.threshold

                # Determine the uncertainty on each prediction
                to_df["prediction_err"] = template*np.nan

                # Record classifier, if requested
                if self.use_logistic:
                    to_df["phenotype_class"] = 0

                self.log.info("+ Writing output...")
                out = _construct_output(to_df,df,genotypes_to_predict,wildtype)
                out.to_csv(self.outfile)
                self.log.info("└── Done writing.")
                self.log.info("GPSeer finished!")
                return

        # -------------------------------------------------------------------
        # Do fit

        self.log.info("+ Fitting data...")
        self.model.add_gpm(gpm)
        self.model.fit()
        self.log.info("└── Done fitting data.")

        # -------------------------------------------------------------------
        # Predict on requested genotypes and construct output

        self.log.info("+ Predicting missing data...")

        to_df = {"genotypes":genotypes_to_predict}

        # Create predictions
        to_df["prediction"] = self.model.predict(X=genotypes_to_predict)

        # Determine the uncertainty on each prediction
        err = (1-self.model.score())*np.mean(gpm.phenotypes)
        if err < 0 and np.abs(err) < NUMERICAL_CUTOFF:
            err = 0
        to_df["prediction_err"] = np.ones(len(to_df["prediction"]))*err

        # Record classifier, if requested
        if self.use_logistic:
            to_df["phenotype_class"] = self.model[0].predict(X=genotypes_to_predict)

        self.log.info("└── Done predicting.")

        self.log.info("+ Writing output...")
        out = _construct_output(to_df, gpm.data, genotypes_to_predict, self.wildtype)
        out.to_csv(self.outfile)
        self.log.info("└── Done writing.")

        self.log.info("GPSeer finished!")


# How to run the
main = GPSeer.launch_instance
