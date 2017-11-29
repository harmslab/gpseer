import os, pickle
from . import GPSeer

def load(db_dir, client=None):
    """Load a sampler already on disk."""
    path = os.path.join(db_dir, 'gpm.pickle')
    with open(path, 'rb') as f:
        gpm = pickle.load(f)
    
    path = os.path.join(db_dir, 'model.pickle')
    with open(path, 'rb') as f:
        model = pickle.load(f)    
        
    return GPSeer(gpm, model, client=client, db_dir=db_dir)


    
