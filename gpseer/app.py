import sys
import logging
from copy import deepcopy
from traitlets.config import Application, Config, catch_config_error
from traitlets import (
    Unicode,
    Instance,
    Int,
    Float,
    Enum
)

from gpmap import GenotypePhenotypeMap

from epistasis.models import (
    EpistasisPipeline, 
    EpistasisLogisticRegression,
    EpistasisSpline, 
    EpistasisLinearRegression
)

from epistasis.models.base import BaseModel


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
    'model_file': 'GPSeer.model_file'
}


class GPSeer(Application):

    description = "A tool for predicting phenotypes in a sparsely sampled genotype-phenotype maps."
    aliases = _aliases
    
    # The log level for the application
    log_level = Enum((0,10,20,30,40,50,'DEBUG','INFO','WARN','ERROR','CRITICAL'),
                    default_value=logging.INFO,
                    help="Set the log level by value or name.").tag(config=True)

    # ----------------  I/O settings ----------------------

    infile = Unicode(
        default_value=u'test',
        help="Input file.",
        config=True
    ).tag(config=True)

    outfile = Unicode(
        u'predictions.csv',
        help="Output file",
        config=True
    )

    # ----------------  GPSeer settings ----------------------

    wildtype = Unicode(
        help='The wildtype sequence',
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
        help="Number of replicates for calculating uncertainty.",
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
        self.log.info("Running GPSeer on {}. Look for a {} file with your results.\n".format(self.infile, self.outfile))
        
        # Read a genotype phenotype map.
        self.log.info("+ Reading data...")
        gpm = GenotypePhenotypeMap.read_csv(self.infile, self.wildtype)
        genotypes_to_predict = gpm.get_all_possible_genotypes
        self.log.info("└── Done reading data.\n")

        # Build the epistasis model based on configuration.
        if self.model_definition:
            model = self.model_definition
        else:
            model = EpistasisPipeline([])
            if self.threshold:
                model.append(EpistasisLogisticRegression(threshold=self.threshold))
            if self.spline_order:
                model.append(EpistasisSpline(k=self.spline_order, s=self.spline_smoothness))
            model.append(EpistasisLinearRegression(order=self.epistasis_order))

        # Add GPMap to the epistasis model.
        model.add_gpm(gpm)

        # Fit the epistasis model.
        self.log.info("+ Fitting data...")
        model.fit()
        self.log.info("└── Done fitting data.\n")

        # Predict the 
        self.log.info("+ Predicting missing data...")
        model.predict_to_csv(self.outfile)
        self.log.info("└── Done predicting...\n")

        self.log.info("GPSeer finished!")

# How to run the 
main = GPSeer.launch_instance