import os, glob
import numpy as np
import pandas as pd
from functools import wraps
from collections import Counter
from gpmap.utils import hamming_distance

from . import workers
from .engine import Engine

# Import Dask stuff for distributed computing!
from dask import delayed, compute, array
import dask.array as da
import dask.dataframe as ddf
#from dask.distributed import Client

class DistributedEngine(Engine):
    """GPSeer engine that distributes the work across all resources using Dask.
    """
    @wraps(Engine)
    def __init__(self, client=None, *args, **kwargs):
        super(DistributedEngine, self).__init__(*args, **kwargs)
        self.client = client

    @wraps(Engine.setup)    
    def setup(self, references=None):
        # Get references for all models.
        if references is None:
            # Get references
            references = self.gpm.complete_genotypes
            
        # Store the references
        self.references = references

        # Distribute the work using Dask.
        items = [delayed(workers.setup)(ref, self.gpm, self.model) for ref in references]
        results = compute(*items, get=self.client.get)
        
        # Organize the results
        self.map_of_models = {}
        for i, ref in enumerate(references):
            # Get model from results distributed
            new_model = results[i]
            
            # Store model
            self.map_of_models[ref] = new_model
            
    @wraps(Engine.run_fits)        
    def run_fits(self):
        # Unzip map_of_models dictionary
        references = list(self.map_of_models.keys())
        models = list(self.map_of_models.values())
        
        # Distribute the work using Dask.
        processes = [delayed(workers.run_fits)(model) for model in models]
        results = compute(*processes, get=self.client.get)
        
        # Zip map_of_models back
        self.map_of_models = dict(zip(references, results))

    @wraps(Engine.run_predictions)
    def run_predictions(self, genotypes='missing'):
        # Proper order check
        if hasattr(self, 'map_of_models') is False:
            raise Exception('Try running `run_fits` before running this method.')
        
        # Unzip map_of_models dictionary
        references = list(self.map_of_models.keys())
        models = list(self.map_of_models.values())
        
        # Distribute the work using Dask.
        processes = [delayed(workers.run_predictions)(model, genotypes=genotypes) for model in models]
        results = compute(*processes, get=self.client.get)

        # Zip predictions
        self.map_of_predictions = dict(zip(references, results))

    @wraps(Engine.run_ml_pipeline)
    def run_ml_pipeline(self, references=None, genotypes='missing'):
        # Get references for all models.
        if references is None:
            # Get references
            references = self.gpm.complete_genotypes
            
        # Run pipeline for each reference state.
        processes = [delayed(workers.run_ml_pipeline)(ref, self.gpm, self.model) for ref in references]
        results = compute(*processes, get=self.client.get)
        
        # Collect results
        self.map_of_models = {ref : results[i][0] for i, ref in enumerate(references)}
        self.map_of_predictions = {ref : results[i][1] for i, ref in enumerate(references)}

    @property
    def results(self):
        """Get dataframe of results."""
        # Proper order check
        if hasattr(self, 'map_of_predictions') is False:
            raise Exception('Try running `run_fits` before running this method.')        

        output = {}
        
        # Unzip map_of_models dictionary
        references = list(self.map_of_predictions.keys())
        predictions = list(self.map_of_predictions.values())        
        
        # Get columns for results df (genotypes)
        col = list(predictions[0])        
        ml = [self.map_of_predictions[c][c] for c in col]
        output = dict(zip(col, ml))
        
        return pd.DataFrame(output, index=['max_likelihood'])


    # def run_ml_pipeline(self):
    #     self.run_fits()
    #     self.run_predictions()
    #     
    # 
    # @wraps(Engine.sample_fits)        
    # def sample_fits(self, n_samples=10):
    #     # Distribute the work using Dask.
    #     items = [delayed(workers.sample)(ref, items['model'], n_samples=n_samples) for ref, items in self.model_map.items()]
    #     results = compute(*items, get=self.client.get)   
    #     
    #     # Organize the results.
    #     for i, ref in enumerate(self.model_map.keys()):
    #         sampler = results[i]
    #         self.model_map[ref]['sampler'] = sampler     
    # 
    # @wraps(Engine.sample_predictions)    
    # def sample_predictions(self):
    #     # Distribute the work using Dask.
    #     items = [delayed(workers.predict)(ref, items['sampler'], db_path=self.db_path) for ref, items in self.model_map.items()]
    #     results = compute(*items, get=self.client.get)
    #     
    #     # Organize the results.
    #     for i, ref in enumerate(self.model_map.keys()):
    #         sampler = results[i]
    #     
    # @wraps(Engine.sample_bayes_pipeline)    
    # def sample_bayes_pipeline(self, n_samples=10):
    #     # Get references
    #     references = self.references
    #     
    #     # Distribute the work using Dask.
    #     items = [delayed(workers.run)(ref, self.gpm, self.model, n_samples=n_samples, db_path=self.db_path) for ref in references]
    #     results = compute(*items, get=self.client.get)
    # 
    # def _get_model_priors(self, genotype, flat_prior=False):
    #     """Get a set of priors for a given genotype."""
    #     # Build priors.
    #     if flat_prior:
    #         # List references states.
    #         references = self.references
    #         priors = np.ones(len(references)) * 1.0 / len(references)
    #     else:        
    #         # List references states.
    #         references = self.references
    # 
    #         # Generate prior distribution
    #         weights = np.array([10**(-hamming_distance(ref, genotype)) for ref in references])
    #         weights = weights/weights.sum()    
    # 
    #     # Build dictionary
    #     priors = dict(zip(references, weights))
    #     return priors
