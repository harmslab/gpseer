import os
import numpy as np

from gpmap.utils import hamming_distance

from .utils import EngineError
from .serial import SerialEngine
from .distributed import DistributedEngine

class GPSeer(object):
    """"""
    def __init__(self, gpm, model, engine="serial", db_path="database/"):
        # Store input
        self.gpm = gpm
        self.model = model
        self.db_path = db_path
        
        # Tell whether to serialize or not.
        if engine == "serial": 
            self.engine = SerialEngine()
        elif engine == "distributed":
            self.engine = DistributedEngine()
        else:
            raise EngineError('client argument is invalid. Must be "serial" or "distributed".')

    def setup(self):
        """"""
        self.engine.setup(self.gpm, self.model, self.db_path)
        return self

    def fit(self):
        """"""
        self.engine.fit()
        return self
            
    def sample(self):
        """"""
        self.engine.sample()
        return self
            
    def predict(self):
        """"""
        self.engine.predict()
        return self

    def run(self):
        """"""
        self.engine.run(self.gpm, 
            self.model, 
            n_samples=5, 
            db_path=self.db_path)
        return self

    def collect(self):
        """"""
        self.df = self.engine.collect(self.db_path)
        return self.df
    
    def get_posterior(self, genotype, n_samples=10000):
        # List references states.
        references = self.gpm.complete_genotypes

        # Calculate how many samples exist
        dim1 = len(self.df)
        dim2 = len(references)
        nsamples = int(dim1 / dim2)

        # Prepare weights for a genotype
        weights = []
        for ref in references:
            prior = 10**(-hamming_distance(ref, genotype))
            weights.append(np.ones(nsamples)*prior)
            
        # Flatten
        weights = np.ravel(weights)
        # Normalize
        weights = weights / weights.sum()
        
        # Sample
        df = self.df[genotype]
        return df.sample(n_samples, replace=True, weights=weights)
        
