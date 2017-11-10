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
        if references is None    
            # Get references
            references = self.gpm.complete_genotypes
            
        # Store the references
        self.references = references

        # Distribute the work using Dask.
        items = [delayed(workers.setup)(ref, self.gpm, self.model) for ref in references]
        results = compute(*items, get=self.client.get)
        
        # Organize the results
        self.model_map = {}
        for i, ref in enumerate(references):
            # Get model from results distributed
            new_model = results[i]
            
            # Store model
            items = dict(model=new_model)
            self.model_map[ref] = items
            
    @wraps(Engine.fit)        
    def run_fits(self):
        # Distribute the work using Dask.
        items = [delayed(workers.fit)(ref, items['model']) for ref, items in self.model_map.items()]
        results = compute(*items, get=self.client.get)
        
        # Organize the results.
        for i, ref in enumerate(self.model_map.keys()):
            model = results[i]
            self.model_map[ref]['model'] = model

    @wraps(Engine.run)
    def run_predictions(self):
        pass

    def run_ml_pipeline(self):
        pass

    @wraps(Engine.sample)        
    def sample_fits(self, n_samples=10):
        # Distribute the work using Dask.
        items = [delayed(workers.sample)(ref, items['model'], n_samples=n_samples) for ref, items in self.model_map.items()]
        results = compute(*items, get=self.client.get)   
        
        # Organize the results.
        for i, ref in enumerate(self.model_map.keys()):
            sampler = results[i]
            self.model_map[ref]['sampler'] = sampler     

    @wraps(Engine.predict)    
    def sample_predictions(self):
        # Distribute the work using Dask.
        items = [delayed(workers.predict)(ref, items['sampler'], db_path=self.db_path) for ref, items in self.model_map.items()]
        results = compute(*items, get=self.client.get)
        
        # Organize the results.
        for i, ref in enumerate(self.model_map.keys()):
            sampler = results[i]
        
    @wraps(Engine.run)    
    def run_bayesian_pipeline(self, n_samples=10):
        # Get references
        references = self.references
        
        # Distribute the work using Dask.
        items = [delayed(workers.run)(ref, self.gpm, self.model, n_samples=n_samples, db_path=self.db_path) for ref in references]
        results = compute(*items, get=self.client.get)

    @wraps(Engine.collect)    
    def collect(self):
        # Get references
        references = self.references
        
        self.data = {}
        for ref in references:
            path = os.path.join(self.db_path, "{}.csv".format(ref))
            df = ddf.read_csv(path)
            self.data[ref] = df

    def get_model_priors(self, genotype, flat_prior=False):
        """Get a set of priors for a given genotype."""
        # Build priors.
        if flat_prior:
            # List references states.
            references = self.references
            priors = np.ones(len(references)) * 1.0 / len(references)
        else:        
            # List references states.
            references = self.references

            # Generate prior distribution
            weights = np.array([10**(-hamming_distance(ref, genotype)) for ref in references])
            weights = weights/weights.sum()    
    
        # Build dictionary
        priors = dict(zip(references, weights))
        return priors
    
    def compute_individual_histograms(self, genotype, bins, range):
        """Calculate the individual histograms comprised of all samples from all
        models for a given genotype.
        
        Note: histogram ignores nans
        
        Returns
        -------
        histograms : dictionary
            model reference state paired with their histogram data as numpy arrays.
        """
        if hasattr(self, "data") is False:
            raise Exception("Collect data first.")
        
        histograms = {}
        for ref in self.data:
            # Get data as dask.array
            data = self.data[ref][genotype].values
            useful_data = data[~da.isnan(data)]
            hist, bins = da.histogram(useful_data, bins=bins, range=range)
            histograms[ref] = hist.compute()
        
        return histograms
    
    def approximate_posterior(self, genotype, bins, range, flat_prior=False):
        """Approximates posterior distribution from samples.
        
        Note: histogram ignores nans
        """
        # Calculate histograms for each model
        histograms = self.compute_individual_histograms(genotype, bins=bins, range=range)
        
        # Apply a non-flat prior.
        if flat_prior is False:
            # Calculate priors for this dataset
            priors = self.get_model_priors(genotype, flat_prior=flat_prior)
            
            for ref, hist in histograms.items():
                # change heights of histogram by prior
                histograms[ref] = hist * priors[ref]
    
        # Sum all histograms to marginalize all models to a single histogram
        return np.sum(list(histograms.values()), axis=0)
    
    def sample_posterior(self, genotype, flat_prior=False, n_samples=10000):
        """"""
        # Calculate priors
        priors = self.get_model_priors(genotype)
        
        # Get elements from priors dictionary
        references = np.array(priors.keys())
        weights = np.array(priors.values())
        
        ########### Clever/efficient way to build prior sampling into mix.
            
        # Generate samples 
        samples = np.random.choice(references, size=n_samples, replace=True, p=weights)
        counts = Counter(samples)
        
        ########### End clever choice.
        
        dfs = []
        for ref, count in counts.items():
            # Get data
            data = self.data[ref][genotype]
            frac = count / len(data)

            # Only accept fractions that make sense.
            if 0 < frac <= 1:
                # Randomly sample data.
                df = data.sample(frac=frac, replace=True)
                dfs.append(df.compute())
        
        # Return DataFrame.
        return pd.concat(dfs)
