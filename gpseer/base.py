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


class BaseApp(Application):

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

    def initialize_settings(self):
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

        # Decide whether or not to fit a spline to the data
        self.use_spline = False
        if self.spline_order:
            self.use_spline = True

        self.use_logistic = False
        if self.threshold:
            self.use_logistic = True

    def initialize_model(self):
        self.model = EpistasisPipeline([])
        if self.use_logistic:
            self.model.append(EpistasisLogisticRegression(threshold=self.threshold))
        if self.use_spline:
            self.model.append(EpistasisSpline(k=self.spline_order, s=self.spline_smoothness))
        self.model.append(EpistasisLinearRegression(order=self.epistasis_order))

    def zinitialize(self):
        self.initialize_settings()
        self.initialize_model()

    def read_data(self):
        # Read the input file
        self.log.info("+ Reading data...")
        df = pd.read_csv(
            self.infile,
            dtype={
                'genotypes': str,
                'phenotypes': float
            }
        )

        # Quick sanity check on inputs
        required_columns = ["genotypes","phenotypes"]
        for c in required_columns:
            try:
                df[c]
            except AttributeError:
                err = "input file ({}) must contain a column labeled '{}'".format(self.infile,c)
                self.log.critical(err)
                return


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

        self.log.info("└── Done reading data.")
        return gpm





