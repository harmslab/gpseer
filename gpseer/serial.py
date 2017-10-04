import os
import numpy as np
import pandas as pd

from . import workers

class SerialEngine(object):
    """"""
    def setup(self, gpm, model):            
        # Get references
        references = gpm.complete_genotypes
        
        # Get models.
        self.model_map = {}
        for i, ref in enumerate(references):
            # Get model from worker function
            new_model = workers.setup(ref, gpm, model)

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

    def predict(self, db_path):
        """"""
        # Create database folder
        if not os.path.exists(db_path):
            os.makedirs(db_path)        

        for ref, items in self.model_map.items():
            # compute predictions from models
            sampler = items['sampler']
            sampler = workers.predict(ref, sampler, db_path=db_path)
    
    def run(self, gpm, model, n_samples=10, db_path="database/"):
        """"""        
        # Create database folder
        if not os.path.exists(db_path):
            os.makedirs(db_path)
    
        # Get references
        references = gpm.complete_genotypes
        
        # Run models.
        for ref in references:
            workers.run(ref, gpm, model, db_path=db_path)
    
    def collect(self, filenames):
        """"""
        # Read datasets
        dfs = []
        for fname in filenames:
            df = pd.read_csv(path, index_col=0)
            dfs.append(df)
        return pd.concat(dfs)
