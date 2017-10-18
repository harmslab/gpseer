import os
import numpy as np
import pickle

class EngineError(Exception):
    """Rename exception for problems with distributed client."""

class Engine(object):
    
    def __init__(self, gpm, model, db_path="database/"):
        self.gpm = gpm
        self.model = model
        self.db_path = db_path

        # Create database folder
        if not os.path.exists(self.db_path):
            # Create the directory for saving sampler data.
            os.makedirs(self.db_path)
            
            path = os.path.join(self.db_path, 'gpm.pickle')
            with open(path, 'wb') as f:
                pickle.dump(self.gpm, f)
            
            path = os.path.join(self.db_path, 'model.pickle')
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)

    def set_starting_index(self, n_samples):
        # Encoding the indices
        self.prefix_index = len(str(len(self.gpm.complete_genotypes)))
        self.suffix_index = len(str(n_samples))
        self.starting_index = 10**(prefix_digits+suffix_digits)
