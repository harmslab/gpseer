import copy
import numpy as np
import pandas as pd

from dask.distributed import Client
from gpmap import GenotypePhenotypeMap

from .utils import EngineError
from .serial import SerialEngine
from .distributed import DistributedEngine

class Thing(object):
    """"""
    def __init__(self, engine="serial"):
        # Tell whether to serialize or not.
        if engine == "serial": 
            self.engine = SerialEngine()
        elif client == "distributed":
            self.engine = DistributedEngine()
        else:
            raise ClientError('client argument is invalid. Must be "serial" or "distributed".')
        
    def setup(self, gpm, model, db_path="database/"):
        """"""
        # Store input
        self.gpm = gpm
        self.model = model
        self.db_path = db_path
        self.model_map = self.engine.setup(self.gpm, self.model)

    def fit(self):
        """"""
        self.engine.fit(self.model_map)
            
    def sample(self):
        """"""
        self.engine.sample(self.model_map)
            
    def predict(self):
        """"""
        self.engine.predict(self.model_map)
        
    def collect(self):
        """"""
        
    def run(self):
        """"""
        self.fit()
        self.sample()
        self.predict()
        self.collect()
