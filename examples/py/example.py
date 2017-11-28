import numpy as np
from gpmap.simulate import MountFujiSimulation
from epistasis.models import EpistasisLinearRegression

from gpseer import GPSeer

def main(client):
    # Create an instance of the model. Using `from_length` makes this easy.
    gpm = MountFujiSimulation.from_length(3, field_strength=-1)

    # add roughness, sampling from a range of values.
    gpm.set_roughness(range=(-1,1))
    
    # Fit with epistasis model
    model = EpistasisLinearRegression(order=1, model_type='local')
    
    # Initialize a seer.
    bins = np.arange(-1,10, .1)
    seer = GPSeer(gpm, model, client=client, bins=bins, genotypes='complete', single_reference=True, db_path='database')
    
    # Sample posterior.
    seer.run_pipeline()
    return seer.results
    
# if __name__ == "__main__":
#     #from dask.distributed import Client
# 
#     #client = Client()
#     #from epistasis
#     main(client)
