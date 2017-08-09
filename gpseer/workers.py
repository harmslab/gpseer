"""
The worker module includes functions for single workers to do tasks scheduled
by a Dask.distributed Client.
"""
import os
import glob
import copy
import h5py
import numpy as np
import dask.array
from .model import ModelSampler
from .prediction import Prediction

def fit(reference, gpm=None, model=None, **kwargs):
    """Copy the genotype-phenotype map and epistasis model, set the reference
    state in the copy, and fit the model to the map.

    Parameters
    ----------
    reference : str
        reference genotype.
    gpm : GenotypePhenotypeMap
        genotype-phenotype-map dataset.
    model :
        epistasis model to fit the data

    Keyword Arguments
    -----------------
    Key word arguments are passed to the model.fit() method.
    """
    model_copy = copy.deepcopy(model)
    # Extremely inefficient...
    gpm_copy = copy.deepcopy(gpm)
    # Set the reference state for the binary representation of the map.
    gpm_copy.add_binary(reference)
    model_copy.add_gpm(gpm_copy)
    model_copy.fit(**kwargs)
    return model_copy

def sample(model, n_samples=1000, db_dir=None, **kwargs):
    """Sample the parameters of the model and store in a HDF5 file.

    Parameters
    ----------
    model :
        epistasis model.
    n_samples : int

    """
    reference = model.gpm.binary.wildtype
    path = os.path.join(db_dir, "models","{}".format(reference))
    sampler = ModelSampler(model, db_dir=path)
    sampler.add_samples(n_samples, **kwargs)

def predict(model, db_dir=None):
    """"""
    reference = model.gpm.binary.wildtype
    path = os.path.join(db_dir, "models","{}".format(reference))
    sampler = ModelSampler(model, db_dir=path)
    sampler.add_predictions()

def sort(model, db_dir=None):
    """"""
    genotypes = model.gpm.complete_genotypes
    reference = model.gpm.binary.wildtype
    index = genotypes[genotypes == reference].index[0]

    # Link to all h5py files (out of core).
    path = os.path.join(db_dir, "models")
    data = [h5py.File(os.path.join(path, genotype, "sample-db.hdf5"))["predictions"] for genotype in genotypes]

    # Concatenate all arrays
    chunk_shape = data[0].shape
    combined = dask.array.concatenate([dask.array.from_array(ds, chunks=chunk_shape) for ds in data], axis=0)

    # Get array
    arr = np.array(combined[:, index])

    # Write to file
    new_path = os.path.join(db_dir, "likelihoods", reference)

    # Create file
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    path = os.path.join(new_path, "likelihood.hdf5")

    # Write to HDF5 file
    f = h5py.File(path)
    ds = f.create_dataset("likelihood", data=arr, dtype=float)
    f.close()

def analyze(genotype, db_dir=None):
    """"""
    # Handle paths
    likelihood_path = os.path.join(db_dir, "likelihoods", genotype, "likelihood.hdf5")
    snapshot_path = os.path.join(db_dir, "snapshots", "{}.pickle".format(genotype))

    # Write to file
    new_path = os.path.join(db_dir, "snapshots")

    # Create file
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    # Build a predictions object
    prediction = Prediction(likelihood_path)

    # Histogram the data
    prediction.histogram()

    # Calculate the percentiles
    prediction.percentile((2.5, 97.5))

    # Snapshot and save
    snapshot = prediction.snapshot()
    snapshot.pickle(snapshot_path)
    return snapshot
