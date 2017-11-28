import os, glob
import numpy as np
import pandas as pd
from functools import wraps
from collections import Counter
from gpmap.utils import hamming_distance

from .. import workers
from ..engine import Engine

# Import Dask stuff for distributed computing!
from dask import delayed, compute

class DistributedEngine(Engine):
    """GPSeer engine that distributes the work across all resources using Dask."""
    
    @wraps(Engine)
    def __init__(self, client=None, *args, **kwargs):
        # Set up Engine
        super(DistributedEngine, self).__init__(*args, **kwargs)
        
        # Reference client for distributed computing
        self.client = client
        self.max_workers = sum(self.client.ncores().values())
        
        # Construct map for MCMC states.
        self.reference_genotypes = self.gpm.complete_genotypes
        self.map_of_mcmc_states = {ref : None for ref in self.reference_genotypes}
        
    @wraps(Engine.setup)    
    def setup(self):
        
        # Distribute the work using Dask.
        processes = []
        for ref in self.reference_genotypes:
            # Build process for this model.            
            process = delayed(workers.setup)(ref, self.gpm, self.model)
            
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)    
        
        # Organize the results
        self.map_of_models = {}
        for i, ref in enumerate(self.reference_genotypes):
            # Get model from results distributed
            new_model = results[i]
            
            # Store model
            self.map_of_models[ref] = new_model
            
    @wraps(Engine.run_fits)        
    def run_fits(self):
        
        # Distribute the work using Dask.
        processes = []
        for ref in self.reference_genotypes:
            # Build process for this model.            
            process = delayed(workers.run_fits)(self.map_of_models[ref], sample_weight=self.sample_weight)
            
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)        
        
        # Zip map_of_models back
        self.map_of_models = dict(zip(self.reference_genotypes, results))

    @wraps(Engine.run_predictions)
    def run_predictions(self):
        # Distribute the work using Dask.
        processes = []
        for ref in self.reference_genotypes:
            # Build process for this model.            
            process = delayed(workers.run_predictions)(self.map_of_models[ref], 
                genotypes=self.genotypes)
            
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)

        # Zip predictions
        self.map_of_predictions = dict(zip(self.reference_genotypes, results))

    @wraps(Engine.run_pipeline)
    def run_pipeline(self):            
        # Distribute the work using Dask.
        processes = []
        for ref in self.reference_genotypes:
            # Build process for this model.            
            process = delayed(workers.run_pipeline)(ref, self.gpm, self.model, genotypes=self.genotypes)
            
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)

        # Collect results
        self.map_of_models = {ref : results[i][0] for i, ref in enumerate(self.reference_genotypes)}
        self.map_of_predictions = {ref : results[i][1] for i, ref in enumerate(self.reference_genotypes)}

    @wraps(Engine.sample_fits)        
    def sample_fits(self, n_samples):
        
        # Proper order check
        if hasattr(self, 'map_of_models') is False:
            raise Exception('Try running `run_fits` before running this method.')        
        
        # Distribute the work using Dask.
        processes = []
        for ref in self.reference_genotypes:
            # Build process for this model.            
            process = delayed(workers.sample_fits)(self.map_of_models[ref], 
                n_samples=n_samples, 
                previous_state=self.map_of_mcmc_states[ref])
            
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)
        
        map_of_model_samples = {ref : results[i][0] for i, ref in enumerate(self.reference_genotypes)}
        self.map_of_mcmc_states = {ref : results[i][1] for i, ref in enumerate(self.reference_genotypes)}
        return map_of_model_samples

    @wraps(Engine.sample_predictions)    
    def sample_predictions(self, map_of_model_samples):        
        # Distribute the work using Dask.
        processes = []
        for ref in self.reference_genotypes:
            process = delayed(workers.sample_predictions)(
                self.map_of_models[ref], 
                map_of_model_samples[ref], 
                self.bins, genotypes=self.genotypes) 
            
            # Add process to list of processes
            processes.append(process)
        
        # Compute processes on distributed network.
        results = compute(*processes, get=self.client.get)
        
        # Organize the results.
        self.map_of_sampled_predictions = {ref : results[i] for i, ref in enumerate(self.reference_genotypes)}
        
    @wraps(Engine.sample_pipeline)    
    def sample_pipeline(self, n_samples):
        
        # Distribute the work using Dask.
        processes = []
        for ref in self.reference_genotypes:
            process = delayed(workers.sample_pipeline)(ref, self.gpm, self.model,
                n_samples, self.bins, 
                genotypes=self.genotypes,
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
        for i, ref in enumerate(self.reference_genotypes):
            self.map_of_models[ref] = results[i][0]
            self.map_of_mcmc_states[ref] = results[i][1]
            self.map_of_predictions[ref] = results[i][2]
            self.map_of_sampled_predictions[ref] = results[i][3]


    @property
    def ml_results(self):
        """Get the maximum likelihood results"""
        # Get example predictions DataFrame
        data = {}        
        for genotype in self.predicted_genotypes:
            # Get max_likelihood
            val = self.map_of_predictions[genotype][genotype]['max_likelihood']
            data[genotype] = [val]
                
        df = pd.DataFrame(data, index=['max_likelihood'])
        return df       

    @property
    def results(self):
        """Get dataframe of prediction results."""
        df = self.ml_results
        
        # Add histograms
        data = {g : [] for g in self.predicted_genotypes}
        if hasattr(self, 'map_of_sampled_predictions'):
            # Get histograms
            mapping = self.map_of_sampled_predictions
            for genotype in self.predicted_genotypes:
                arr = np.zeros(len(self.bins)-1)
                
                # Build priors.
                priors = np.array([10**(-hamming_distance(ref, genotype)) for ref in self.reference_genotypes])
                priors = priors/priors.sum()
                
                # Construct histograms
                for i, ref in enumerate(self.reference_genotypes):               
                    arr += np.array(mapping[ref][genotype].values) * priors[i]
                data[genotype] += list(arr)
            
            # Append posterior distributions to dataframe
            df2 = pd.DataFrame(data, index=list(self.bins[1:]))
            df = df.append(df2)
            
        return df
