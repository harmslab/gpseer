import os, glob
import numpy as np
import pandas as pd
from functools import wraps
from collections import Counter
from gpmap.utils import hamming_distance

from . import workers
from .engine import Engine

class SerialEngine(Engine):
    """"""
    @wraps(Engine)
    def setup(self):            
        # Get references
        references = self.gpm.complete_genotypes
        
        # Get models.
        self.model_map = {}
        for i, ref in enumerate(references):
            # Get model from worker function
            new_model = workers.setup(ref, self.gpm, self.model)

            # Store model
            items = dict(model=new_model)
            self.model_map[ref] = items
    
    def fit(self):
        """"""
        for ref, items in self.model_map.items():
            model = items['model']
            model = workers.fit(ref, model)

    def sample(self):
        """"""            
        for ref, items in self.model_map.items():
            # Sample model.
            model = items['model']
            sampler = workers.sample(ref, model)
            items['sampler']= sampler

    def predict(self):
        """"""
        for ref, items in self.model_map.items():
            # compute predictions from models
            sampler = items['sampler']
            sampler = workers.predict(ref, sampler, db_path=self.db_path)
    
    def run(self, n_samples=100):
        """"""
        # Get references
        references = self.gpm.complete_genotypes
        
        # Run models.
        for i, ref in enumerate(references):
            workers.run(ref, self.gpm, self.model, 
                n_samples=n_samples, 
                #starting_index=starting_index, 
                db_path=self.db_path)
    
    def collect(self):
        """"""
        # List references states.
        references = self.gpm.complete_genotypes
        self.data = {}
        for i, ref in enumerate(references):
            path = os.path.join(self.db_path, "{}.csv".format(ref))
            df = pd.read_csv(path, index_col=0)
            self.data[ref] = df
    
    def sample_posterior(self, genotype, n_samples=10000):
        """"""
        ########### Clever/efficient way to build prior sampling into mix.
        # Build priors.
        if flat_prior:
            # List references states.
            references = self.gpm.complete_genotypes
            priors = np.ones(len(references)) * 1.0 / len(references)
        else:        
            # List references states.
            references = self.gpm.complete_genotypes

            # Generate prior distribution
            priors = np.array([10**(-hamming_distance(ref, genotype)) for ref in references])
            priors = priors/priors.sum()
            
        # Generate samples 
        samples = np.random.choice(references, size=n_samples, replace=True, p=priors)
        counts = Counter(samples)
        
        ########### End clever choice.

        dfs = []
        for ref, count in counts.items():
            # Get data
            data = self.data[ref][genotype]
            frac = count / len(data)
            
            # Only accept fractions that make sense.
            if frac <= 1:
                # Randomly sample data.
                df = data.sample(frac=frac, replace=True)
                dfs.append(df)
        
        # Return DataFrame.
        return pd.concat(dfs)
