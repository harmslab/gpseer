import os
import numpy as np
import pickle

from .utils import EngineError, SubclassError

class Engine(object):
    """Base class for running sampling on genotype-phenotype maps.
    """
    def __init__(self, gpm, model, db_path="database/"):
        self.gpm = gpm
        self.model = model
        self.db_path = db_path

        # Create database folder
        if not os.path.exists(self.db_path):
            # Create the directory for saving sampler data.
            os.makedirs(self.db_path)
            
            path = os.path.join(self.db_path, 'gpm.pickle')
            with open(path, 'wb') as f:
                pickle.dump(self.gpm, f)
            
            path = os.path.join(self.db_path, 'model.pickle')
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)

    def setup(self):
        """Initialize models for each reference state in the genotype-phenotype map."""
        raise SubclassError("Must be defined in a subclass.")

    def run_ml_fits(self):
        """Call the `fit` methods on all models. GPSeer assumes these are the maximum
        likelihood solution."""
        raise SubclassError("Must be defined in a subclass.")
    
    def run_ml_predictions(self):
        """Call the `predict` methods on all models. GPSeer assumes these are the maximum
        likelihood solution."""
        raise SubclassError("Must be defined in a subclass.")    
    
    def sample_models(self, n_samples=10):
        """Sample the posterior distributions for each model using an MCMC sampling
        method (see the `emcee` library)."""
        raise SubclassError("Must be defined in a subclass.")
    
    def sample_predictions(self):
        """Use the samples to predict all possible phenotypes in the genotype-phenotype map."""
        raise SubclassError("Must be defined in a subclass.")
    
    def run(self, n_samples=10):
        """Run the full pipeline, from setup to predictig phenotypes."""
        raise SubclassError("Must be defined in a subclass.")

    def collect(self, n_samples=10):
        """Collect the results from all models."""
        raise SubclassError("Must be defined in a subclass.")
