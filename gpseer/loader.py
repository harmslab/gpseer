import os
import pickle
from . import GPSeer


def load(db_dir, client=None):
    """Load a GPSeer object already on disk."""
    # Build path on os.
    path = os.path.join(db_dir, 'gpseer-object.pickle')
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Initialize the GPSeer objects
    seer = GPSeer(data['gpm'],
                  data['model'],
                  data['bins'],
                  genotypes=data['genotypes'],
                  sample_weight=data['sample_weight'],
                  client=client,
                  perspective=data['perspective'],
                  db_dir=db_dir)

    # Update data!
    seer.predicted_genotypes = data['predicted_genotypes']
    seer.keys = data['keys']
    seer.map_of_mcmc_states = data['map_of_mcmc_states']
    seer.map_of_models = data['map_of_models']
    seer.map_of_predictions = data['map_of_predictions']
    seer.map_of_sampled_predictions = data['map_of_sampled_predictions']
    return seer
