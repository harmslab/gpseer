import os
import copy
import warnings
import numpy as np
import pandas as pd

from . import workers

# Import Dask stuff for distributed computing!
from dask import delayed, compute
from dask.distributed import Client

class DistributedEngine(object):
    """"""
    def __init__(self):
        self.client = Client()
    
    def setup(self, gpm, model, db_path):        
        # Create database folder
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
    
        # Get references
        references = gpm.complete_genotypes

        # Distribute the work using Dask.
        items = [workers.setup(ref, gpm, model) for ref in references]
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
        items = [workers.fit(ref, items['model']) for ref, items in self.model_map.items()]
        results = compute(*items, get=self.client.get)
        
        # Organize the results.
        for i, ref in enumerate(self.model_map.keys()):
            model = results[i]
    
    def sample(self):
        """"""
        # Distribute the work using Dask.
        items = [workers.sample(ref, items['model']) for ref, items in self.model_map.items()]
        results = compute(*items, get=self.client.get)   
        
        # Organize the results.
        for i, ref in enumerate(self.model_map.keys()):
            sampler = results[i]
            self.model_map[ref]['sampler'] = sampler     

    def predict(self, db_path):
        """"""
        # Distribute the work using Dask.
        items = [workers.predict(ref, items['sampler'], db_path=db_path) for ref, items in self.model_map.items()]
        results = compute(*items, get=self.client.get)
        
        # Organize the results.
        for i, ref in enumerate(self.model_map.keys()):
            sampler = results[i]
        
    def run(self, gpm, model, n_samples=10, db_path="database/"):
        """"""
        # Create database folder
        if not os.path.exists(db_path):
            os.makedirs(db_path)
    
        # Run models.
        for ref in references:
            workers.run(ref, gpm, model, db_path=db_path)

    def collect(self, model_map):
        pass
