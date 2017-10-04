import os, glob

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
        elif client == "distributed":
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
        path = os.path.join(self.db_path, "*.csv")
        filenames = glob.glob(path)
        self.df = self.engine.collect(filenames)
        return self.df
    
    def get_posterior(self, genotype, n_samples=10000):
        # List references states.
        references = self.gpm.complete_genotypes

        # Calculate how many samples exist
        dims = self.df.shape
        nsamples = dims[0] / dims[1]
        
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
        
