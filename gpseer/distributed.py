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
    """GPSeer engine that distributes the work across all resources using Dask."""
    @wraps(Engine)
    def __init__(self, client=None, *args, **kwargs):
        super(DistributedEngine, self).__init__(*args, **kwargs)
        self.client = client

    @wraps(Engine.setup)    
    def setup(self):
        # Distribute the work using Dask.
        processes = []
        for ref in self.references:
            # Build process for this model.            
            process = delayed(workers.setup)(ref, self.gpm, self.model)
            
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)    
        
        # Organize the results
        self.map_of_models = {}
        for i, ref in enumerate(self.references):
            # Get model from results distributed
            new_model = results[i]
            
            # Store model
            self.map_of_models[ref] = new_model
            
    @wraps(Engine.run_fits)        
    def run_fits(self):
        # Distribute the work using Dask.
        processes = []
        for ref in self.references:
            # Build process for this model.            
            process = delayed(workers.run_fits)(self.map_of_models[ref])
            
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)        
        
        # Zip map_of_models back
        self.map_of_models = dict(zip(self.references, results))

    @wraps(Engine.run_predictions)
    def run_predictions(self, genotypes='missing'):        
        # Distribute the work using Dask.
        processes = []
        for ref in self.references:
            # Build process for this model.            
            process = delayed(workers.run_predictions)(self.map_of_models[ref], 
                genotypes=genotypes)
            
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)

        # Zip predictions
        self.map_of_predictions = dict(zip(self.references, results))

    @wraps(Engine.run_pipeline)
    def run_pipeline(self, genotypes='missing'):            
        # Distribute the work using Dask.
        processes = []
        for ref in self.references:
            # Build process for this model.            
            process = delayed(workers.run_pipeline)(ref, self.gpm, self.model)
            
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)

        # Collect results
        self.map_of_models = {ref : results[i][0] for i, ref in enumerate(self.references)}
        self.map_of_predictions = {ref : results[i][1] for i, ref in enumerate(self.references)}

    @wraps(Engine.sample_fits)        
    def sample_fits(self, n_samples):
        # Proper order check
        if hasattr(self, 'map_of_models') is False:
            raise Exception('Try running `run_fits` before running this method.')        
        
        # Distribute the work using Dask.
        processes = []
        for ref in self.references:
            # Build process for this model.            
            process = delayed(workers.sample_fits)(self.map_of_models[ref], 
                n_samples=n_samples, 
                previous_state=self.map_of_mcmc_states[ref])
            
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)
        
        map_of_model_samples = {ref : results[i][0] for i, ref in enumerate(self.references)}
        self.map_of_mcmc_states = {ref : results[i][1] for i, ref in enumerate(self.references)}
        return map_of_model_samples

    @wraps(Engine.sample_predictions)    
    def sample_predictions(self, map_of_model_samples, bins, genotypes='missing'):
        # Distribute the work using Dask.
        processes = []
        for ref in self.references:
            process = delayed(workers.sample_predictions)(
                self.map_of_models[ref], 
                map_of_model_samples[ref], 
                bins, genotypes=genotypes) 
            
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)
        
        # Organize the results.
        self.map_of_sampled_predictions = {ref : results[i] for i, ref in enumerate(self.references)}
        
    @wraps(Engine.sample_pipeline)    
    def sample_pipeline(self, n_samples, bins, genotypes='missing'):
        # Distribute the work using Dask.
        processes = []
        for ref in self.references:
            process = delayed(workers.sample_pipeline)(ref, self.gpm, self.model,
                n_samples, bins, 
                genotypes=genotypes,
                previous_state=self.map_of_mcmc_states[ref])
                
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)

        # Parse results from workers.
        self.map_of_models = {}
        self.map_of_mcmc_states = {}
        self.map_of_predictions = {}
        self.map_of_sampled_predictions = {}
        for i, ref in enumerate(self.references):
            self.map_of_models[ref] = results[i][0]
            self.map_of_mcmc_states[ref] = results[i][1]
            self.map_of_predictions[ref] = results[i][2]
            self.map_of_sampled_predictions[ref] = results[i][3]
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


    @property
    def results(self):
        """Get dataframe of results."""

        if len(self.map_of_predictions) != len(self.map_of_sampled_predictions):
            raise Exception    
        
        # Get columns for results df (genotypes)
        # col = list(predictions[0])        
        # ml = [self.map_of_predictions[c][c] for c in col]
        # output = dict(zip(col, ml))
        
        # Get example predictions DataFrame
        df = list(self.map_of_sampled_predictions.values())[0]
        columns = df.columns
        index = list(df.index)
        
        # Build output array
        data = {}
        for col in columns:
            posterior = []
            for ref in self.references:
                posterior.append(np.array(self.map_of_sampled_predictions[ref][col]))
            posterior = list(np.sum(posterior, axis=0))
            
            ml_val = [self.map_of_predictions[col][col]['max_likelihood']]
            print(ml_val, posterior)
            # Concatenate data
            col_data = ml_val + posterior
            data[col] = col_data
            
        
        return pd.DataFrame(data, index=['max_likelihood'] + index)
