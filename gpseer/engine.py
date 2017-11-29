import os
import pickle
from functools import wraps
import numpy as np

from .utils import EngineError, SubclassError

def save_engine(method):
    """Save the current state of the GPSeer engine."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        out = method(*args, **kwargs)
        
        # Load reference genotypes
        path = os.path.join(self.db_dir, 'reference_genotypes')
        with open(path, 'rb') as f:
            self.reference_genotypes = pickle.load(f)
        
        # Load predicted genotypes
        path = os.path.join(self.db_dir, 'predicted_genotypes')
        with open(path, 'rb') as f:
            self.predicted_genotypes = pickle.load(f)        
        
        # Load reference genotypes
        path = os.path.join(self.db_dir, 'reference_genotypes')
        with open(path, 'rb') as f:
            self.reference_genotypes = pickle.load(f)
        return out
    return wrapper


class Engine(object):
    """Engine for sampling epistasis model on sparsely sampled genotype-phenotype
    maps.
    
    Parameters
    ----------
    gpm : gpmap.GenotypePhenotypeMap
        object containing data for a sparse genotype-phenotype map.
    model :  
        epistasis model to use when predicting the genotype-phenotype map.
    bins :
        array of bins used in bayesian posterior histogram.
    db_dir : str
        directory to save data from this sampling engine.
    
    Attributes
    ----------
    reference_genotypes : numpy.ndarray
        Array containing all possible genotypes in the GenotypePhenotypeMap
    predicted_genotypes : numpy.ndarray
        array of genotypes that were predicted by sampling engine.
    map_of_mcmc_states : dict
        Information about the engine's last step in the MCMC walk. This is
        important for continuing MCMC walks. If not steps have been taken, 
        the values are set to None.
    map_of_models : dict
        A mapping of epistasis models taken from different reference states. Key
        is the reference genotype and the value of an epistasis model object.
    map_of_predictions : dict
        A mapping of the maximum likelihood phenotypes predicted by epistasis
        models for each reference state. Key is the references genotype and the
        value is a DataFrame of predictions. 
    map_of_sampled_predictions : dict
        A mapping of the posterior probability distributions for all predicted 
        phenotypes. Key is the reference genotype and the value is a DataFrame 
        (histogram bins as the index and predicted genotypes as columns.) 
    """
    def __init__(self, gpm, model, bins, sample_weight=None, genotypes='missing', db_dir="database/"):
        if model.model_type != 'local':
            raise Exception('model_type in model must be set to `local`.')
        
        self.gpm = gpm
        self.bins = bins
        self.model = model    
        self.db_dir = db_dir
        self.genotypes = genotypes
        self.sample_weight = sample_weight

        # Store the predicted genotypes
        if genotypes in ['missing', 'complete', 'obs']:
            if genotypes == 'missing':
                self.predicted_genotypes = self.gpm.missing_genotypes
            elif genotypes == 'obs':
                self.predicted_genotypes = self.gpm.genotypes
            else:
                self.predicted_genotypes = self.gpm.complete_genotypes
        else:
            raise ValueError("genotypes must be 'missing', 'obs', or 'complete'.")        

        # Create database folder
        if not os.path.exists(self.db_dir):
            # Create the directory for saving sampler data.
            os.makedirs(self.db_dir)
            
            path = os.path.join(self.db_dir, 'gpm.pickle')
            with open(path, 'wb') as f:
                pickle.dump(self.gpm, f)
            
            path = os.path.join(self.db_dir, 'model.pickle')
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
    
    def sample_predictions(self, samples, genotypes='missing'):
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
