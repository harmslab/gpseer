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

from gpmap import GenotypePhenotypeMap

from epistasis.models import (
    EpistasisPipeline,
    EpistasisLogisticRegression,
    EpistasisSpline,
    EpistasisLinearRegression
)

from epistasis.models.base import BaseModel

import pandas as pd
import numpy as np

_aliases = {
    'i': 'GPSeer.infile',
    'o': 'GPSeer.outfile',
    'model_definition': 'GPSeer.model_definition',
    'wildtype': 'GPSeer.wildtype',
    'threshold': 'GPSeer.threshold',
    'spline_order': 'GPSeer.spline_order',
    'spline_smoothness': 'GPSeer.spline_smoothness',
    'epistasis_order': 'GPSeer.epistasis_order',
    'nreplicates': 'GPSeer.nreplicates',
    'model_file': 'GPSeer.model_file',
    'genotype_file':'GPSeer.genotype_file'
}

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
    out = out.drop("Unnamed: 0",axis=1)

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




class GPSeer(Application):

    description = "A tool for predicting phenotypes in a sparsely sampled genotype-phenotype maps."
    aliases = _aliases

    # The log level for the application
    log_level = Enum((0,10,20,30,40,50,'DEBUG','INFO','WARN','ERROR','CRITICAL'),
                    default_value=logging.INFO,
                    help="Set the log level by value or name.").tag(config=True)

    # ----------------  I/O settings ----------------------

    infile = Unicode(
        default_value=u'',
        help="Input file.",
        config=True
    ).tag(config=True)

    outfile = Unicode(
        default_value=u'predictions.csv',
        help="Output file",
        config=True
    )

    # ----------------  GPSeer settings ----------------------

    wildtype = Unicode(
        default_value=None,
        allow_none=True,
        help='The wildtype sequence (if not specified, take first genotype in file as wildtype)',
        config=True
    )

    threshold = Float(
        allow_none=True,
        help='Experimental detection threshold, used by classifer.',
        config=True
    )

    spline_order = Int(
        allow_none=True,
        help="Order of spline.",
        config=True
    )

    spline_smoothness = Int(
        10,
        help="Smoothness of spline.",
        config=True,
    )

    epistasis_order = Int(
        1,
        help="Order of epistasis in the model.",
        config=True
    )

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


    # ----------------  Model Definitions ----------------------

    model_definition = Instance(
        BaseModel,
        allow_none=True,
        help="An epistasis model definition written in Python.",
        config=True
    )

    model_file = Unicode(
        help="File containing epistasis model definition.",
        config=True
    )

    def initialize(self, argv=None):
        self.parse_command_line(argv)
        # Load config file if it exists.
        if self.model_file:
            self.load_config_file(self.model_file)

    @catch_config_error
    def start(self):

        if self.infile == "":
            err = "an input file must be specified using: gpseer -i INPUT_FILE"
            self.log.critical(err)
            return

        if os.path.isfile(self.outfile):
            err = "Output file ({}) already exists. Delete or ".format(self.outfile)
            err += "specify a new output file with: "
            err += "gpseer -o OUTPUT_FILE"
            self.log.critical(err)
            return

        self.log.info("Running GPSeer on {}. Look for a {} file with your results.".format(self.infile, self.outfile))

        # -------------------------------------------------------------------
        # Load data and construct gpmap to do fit

        # Read the input file
        self.log.info("+ Reading data...")
        df = pd.read_csv(self.infile)

        # Quick sanity check on inputs
        required_columns = ["genotypes","phenotypes"]
        for c in required_columns:
            try:
                df[c]
            except AttributeError:
                err = "input file ({}) must contain a column labeled '{}'".format(self.infile,c)
                self.log.critical(err)
                return

        # Decide whether or not to fit a spline to the data
        use_spline = False
        if self.spline_order:
            use_spline = True

        # If no wildtype is explicitly specified, load from the first genotype in the file
        if self.wildtype:
            wildtype = self.wildtype

            # sanity check to make sure wildtype is compatible with genotypes
            all_genotypes = np.copy(df.genotypes)
            if len(wildtype) != len(all_genotypes[0]):
                err = "wildtype sequence length does not match sequence lengths in file."
                self.log.critical(err)
                return

            # Find unique residues at all sites and make sure that the
            # specified wildtype sequence matches at least one of them
            arr = np.array([list(g) for g in all_genotypes])
            for i, col in enumerate(arr.T):
                site_states = list(np.unique(col))
                if wildtype[i] not in site_states:
                    err = "wildtype state ({}) at position {} not found in sequences in file. States at this position in the file are: {}".format(wildtype[i],i,str(site_states))
                    self.log.critical(err)
                    return

        else:
            wildtype = df.iloc[0].genotypes

        # Create a genotype phenotype map from the input file
        gpm = GenotypePhenotypeMap.read_dataframe(df,wildtype)

        # Load in genotypes to predict
        if self.genotype_file:
            genotypes_to_predict = _load_genotype_file(self.genotype_file,wildtype)
            genotypes_to_predict.extend(list(df.genotypes))
            genotypes_to_predict = list(set(genotypes_to_predict))

        # If none are specified, predict all genotypes
        else:
            genotypes_to_predict = gpm.get_all_possible_genotypes()

        # Decide whether or not to do logistic regression
        use_logistic = False
        all_dead = False
        if self.threshold:
            # Make sure that the dataset contains data on both sides of specified
            # threshold before trying to train
            if np.sum(df.phenotypes < self.threshold) == 0:
                self.log.warning("No data found below threshold.  No classifier will be applied.")

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
                if use_logistic:
                    to_df["phenotype_class"] = 0

                self.log.info("+ Writing output...")
                out = _construct_output(to_df,df,genotypes_to_predict,wildtype)
                out.to_csv(self.outfile)
                self.log.info("└── Done writing.")
                self.log.info("GPSeer finished!")

                return

            else:
                use_logistic = True

        self.log.info("└── Done reading data.")


        # -------------------------------------------------------------------
        # Do fit

        self.log.info("+ Fitting data...")

        # Construct model
        model = EpistasisPipeline([])
        if use_logistic:
            model.append(EpistasisLogisticRegression(threshold=self.threshold))
        if use_spline:
            model.append(EpistasisSpline(k=self.spline_order, s=self.spline_smoothness))
        model.append(EpistasisLinearRegression(order=self.epistasis_order))

        # Add the GPMap to the epistasis model.
        model.add_gpm(gpm)

        # Fit the epistasis model.
        model.fit()

        self.log.info("└── Done fitting data.")

        # -------------------------------------------------------------------
        # Predict on requested genotypes and construct output

        self.log.info("+ Predicting missing data...")

        to_df = {"genotypes":genotypes_to_predict}

        # Create predictions
        to_df["prediction"] = model.predict(X=genotypes_to_predict)

        # Determine the uncertainty on each prediction
        err = (1-model.score())*np.mean(gpm.phenotypes)
        if err < 0 and np.abs(err) < NUMERICAL_CUTOFF:
            err = 0
        to_df["prediction_err"] = np.ones(len(to_df["prediction"]))*err

        # Record classifier, if requested
        if use_logistic:
            to_df["phenotype_class"] = model[0].predict(X=genotypes_to_predict)

        self.log.info("└── Done predicting.")

        self.log.info("+ Writing output...")
        out = _construct_output(to_df,df,genotypes_to_predict,wildtype)
        out.to_csv(self.outfile)
        self.log.info("└── Done writing.")


        self.log.info("GPSeer finished!")


# How to run the
main = GPSeer.launch_instance
