import os
import glob
import copy
import h5py
import dask.array
from .likelihood import ModelLikelihood

def fit(reference, gpm=None, model=None, **kwargs):
    """
    """
    model = copy.deepcopy(model)
    # Extremely inefficient...
    gpm_ = copy.deepcopy(gpm)
    # Set the reference state for the binary representation of the map.
    gpm_.binary.wildtype = reference
    model.add_gpm(gpm_)
    model.fit(**kwargs)
    return model

def sample(model, n_samples=1000, db_dir=None):
    """"""
    reference = model.gpm.binary.wildtype
    path = os.path.join(db_dir, "models","{}".format(reference))
    likelihood = ModelLikelihood(model, db_dir=path)
    likelihood.add_samples(n_samples)

def predict(model, db_dir=None):
    """"""
    reference = model.gpm.binary.wildtype
    path = os.path.join(db_dir, "models","{}".format(reference))
    likelihood = ModelLikelihood(model, db_dir=path)
    likelihood.add_predictions()

def posterior(keypair, db_dir=None):
    """
    """
    index = keypair[0]
    genotype = keypair[1]
    path = os.path.join(db_dir, "models", "*", "sample-db.hdf5")
    filenames = glob.glob(path)
    slices = [h5py.File(fn, "r")["predictions"][:,index] for fn in filenames]
    pieces = [dask.array.from_array(ds, chunks=(1000,)) for ds in slices]
    array = dask.array.concatenate(pieces)
    new_path = os.path.join(db_dir, "likelihoods", genotype)
    # Create a folder for the database.
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    dask.array.to_hdf5(os.path.join(new_path, "posterior.hdf5"), '/posterior', array)
