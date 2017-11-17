import os
import numpy as np
import pickle

from .utils import EngineError, SubclassError

class Engine(object):
    """Base class for running sampling on genotype-phenotype maps.
    """
    def __init__(self, gpm, model, db_path="database/"):
        if model.model_type != 'local':
            raise Exception('model_type in model must be set to `local`.')
        
        self.gpm = gpm
        self.model = model
        self.db_path = db_path
        self.references = self.gpm.complete_genotypes
        self.map_of_mcmc_states = {ref : None for ref in self.references}

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

    def run_fits(self):
        """Call the `fit` methods on all models. GPSeer assumes these are the maximum
        likelihood solution."""
        raise SubclassError("Must be defined in a subclass.")
    
    def run_predictions(self):
        """Call the `predict` methods on all models. GPSeer assumes these are the maximum
        likelihood solution."""
        raise SubclassError("Must be defined in a subclass.")    
    
    def run_pipeline(self):
        """Call run_fits and run_predictions together. Useful for distributed computing. 
        Prevents multiple calls to nodes."""
        raise SubclassError("Must be defined in a subclass.")    

    def sample_fits(self, n_samples=10, previous_state=None):
        """Sample the likelihood function of the model and return an array of all
        sets of model coefficients sampled.
        
        Parameters
        ----------
        n_samples : int
            number of steps to take in sampler.
        previous_state : dict
            dictionary with data from latest step in MCMC walk. Must have 'rstate', 'lnprob',
            and 'pos' as keys.
            
        Returns
        -------
        map_of_model_samples : dict
            dictionary mapping each model's reference state to their model samples.
            shape of array: (n_samples, number of coefs) 
        """
        raise SubclassError("Must be defined in a subclass.")
    
    def sample_predictions(self, samples, bins, genotypes='missing'):
        """Use the samples to predict all possible phenotypes in the 
        genotype-phenotype map.
        
        Parameters
        ----------
        samples : numpy.ndarray
            array of model coefficients returned by an MCMC walk. 
        bins : numpy.ndarray
            array of histogram bins.
        genotypes : str
            the group of genotypes to predict. Must be either 'missing', 'obs',
            or 'complete'.
        """
        raise SubclassError("Must be defined in a subclass.")

    def sample_pipeline(self):
        """Call sample_fits and sample_predictions together. Useful for distributed computing. 
        Prevents multiple calls to nodes."""
        raise SubclassError("Must be defined in a subclass.")   
