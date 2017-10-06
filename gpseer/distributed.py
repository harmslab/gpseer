import os, glob
import numpy as np
import pandas as pd
from functools import wraps
from collections import Counter
from gpmap.utils import hamming_distance

from . import workers
from .engine import Engine

# Import Dask stuff for distributed computing!
from dask import delayed, compute, dataframe
from dask.distributed import Client

class DistributedEngine(Engine):
    """GPSeer engine that distributes the work across all resources using Dask.
    """
    @wraps(Engine)
    def __init__(self, *args, **kwargs):
        super(DistributedEngine, self).__init__(*args, **kwargs)
        self.client = Client()
    
    def setup(self):    
        # Get references
        references = self.gpm.complete_genotypes

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
    
    def fit(self):
        """"""
        # Distribute the work using Dask.
        items = [delayed(workers.fit)(ref, items['model']) for ref, items in self.model_map.items()]
        results = compute(*items, get=self.client.get)
        
        # Organize the results.
        for i, ref in enumerate(self.model_map.keys()):
            model = results[i]
    
    def sample(self):
        """"""
        # Distribute the work using Dask.
        items = [delayed(workers.sample)(ref, items['model']) for ref, items in self.model_map.items()]
        results = compute(*items, get=self.client.get)   
        
        # Organize the results.
        for i, ref in enumerate(self.model_map.keys()):
            sampler = results[i]
            self.model_map[ref]['sampler'] = sampler     

    def predict(self):
        """"""
        # Distribute the work using Dask.
        items = [delayed(workers.predict)(ref, items['sampler'], db_path=self.db_path) for ref, items in self.model_map.items()]
        results = compute(*items, get=self.client.get)
        
        # Organize the results.
        for i, ref in enumerate(self.model_map.keys()):
            sampler = results[i]
        
    def run(self, n_samples=10):
        """"""        
        # Get references
        references = self.gpm.complete_genotypes
        
        # Distribute the work using Dask.
        items = [delayed(workers.run)(ref, self.gpm, self.model, n_samples=n_samples, db_path=self.db_path) for ref in references]
        results = compute(*items, get=self.client.get)

    def collect(self):
        # Get references
        references = self.gpm.complete_genotypes
        
        self.data = {}
        for ref in references:
            path = os.path.join(self.db_path, "{}.csv".format(ref))
            df = dataframe.read_csv(path)
            self.data[ref] = df
    
    def sample_posterior(self, genotype, n_samples=10000):
        """"""
        # List references states.
        references = self.gpm.complete_genotypes

        # Generate prior distribution
        priors = np.array([10**(-hamming_distance(ref, genotype)) for ref in references])
        priors = priors/priors.sum()
        
        # Generate samples 
        samples = np.random.choice(references, size=n_samples, replace=True, p=priors)
        counts = Counter(samples)

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
    
    
