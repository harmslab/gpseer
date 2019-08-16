import sys
from traitlets.config import Application, Config
from traitlets import (
    Unicode,
    TraitType
)

from gpmap import GenotypePhenotypeMap

from epistasis.models import (
    EpistasisPipeline, 
    EpistasisSpline, 
    EpistasisLinearRegression
)

from epistasis.models.base import BaseModel


_aliases = {
    'i': 'GPSeer.infile',
    'o': 'GPSeer.outfile',
    'm': 'GPSeer.epistasis_model',
    'w': 'GPSeer.wildtype',
    'model_file': 'GPSeer.model_file'
}

class EpistasisModel(TraitType):
    """Trait for epistasis models.
    """
    info_text = "an epistasis model"

    def validate(self, obj, value):
        # Verify
        if issubclass(value.__class__, BaseModel):
            return value
        else:
            self.error(obj, value)

class GPSeer(Application):

    description = "Predict phenotypes in a sparsely sampled genotype-phenotype maps."

    aliases = _aliases

    infile = Unicode(
        help="Input file.",
        config=True
    )

    outfile = Unicode(
        u'predictions.csv',
        help="Output file",
        config=True
    )

    wildtype = Unicode(
        help='The wildtype sequence',
        config=True
    )
    
    epistasis_model = EpistasisModel(
        EpistasisSpline(),
        help="""Epistasis model""",
        config=True
    )


    model_file = Unicode(
        help="File containing epistasis model definition.",
        config=True
    )

    def _load_config(self, cfg, section_names=None, traits=None):
        """A work around to avoid traitlets' requirement that 
        traits cannot be instances.
        """
        my_cfg = self._find_my_config(cfg)
        epistasis_model = my_cfg.pop("epistasis_model", [])

        # Turn handlers list into a pickeable function
        def get_epistasis_model():
            return epistasis_model

        my_cfg["epistasis_model"] = epistasis_model

        # Build a new eventlog config object.
        model_cfg = Config({"EpistasisModel": my_cfg})
        super(GPSeer, self)._load_config(model_cfg, section_names=None, traits=None)

    def initialize(self, argv=None):
        self.parse_command_line(argv)
        # Load config file if it exists.
        if self.model_file:
            self.load_config_file(self.model_file)

    def start(self):
        # Read a genotype phenotype map.
        self.log.info("Reading data...")
        gpm = GenotypePhenotypeMap.read_csv(self.infile, self.wildtype)
        self.log.info("Done reading data.")

        # Add GPMap to the epistasis model.
        self.epistasis_model.add_gpm(gpm)

        # Fit the epistasis model.
        self.log.info("Fitting data...")
        self.epistasis_model.fit()
        self.log.info("Done fitting data.")

        # Predict the 
        self.log.info("Predicting missing data...")
        df = self.epistasis_model.predict_to_csv(self.outfile)
        self.log.info("Done predicting...")


# How to run the 
main = GPSeer.launch_instance