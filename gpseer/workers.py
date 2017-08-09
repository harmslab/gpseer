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
    """"""
    model_copy = copy.deepcopy(model)
    # Extremely inefficient...
    gpm_copy = copy.deepcopy(gpm)
    # Set the reference state for the binary representation of the map.
    gpm_copy.add_binary(reference)
    model_copy.add_gpm(gpm_copy)
    model_copy.fit(**kwargs)
    return model_copy

def sample(model, n_samples=1000, db_dir=None, **kwargs):
    """"""
    reference = model.gpm.binary.wildtype
    path = os.path.join(db_dir, "models","{}".format(reference))
    sampler = ModelSampler(model, db_dir=path)
    sampler.add_samples(n_samples, **kwargs)
    return model

def predict(model, db_dir=None):
    """"""
    reference = model.gpm.binary.wildtype
    path = os.path.join(db_dir, "models","{}".format(reference))
    sampler = ModelSampler(model, db_dir=path)
    sampler.add_predictions()
    return model

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

    return path

def analyze(likelihood_path, chunk=1000):
    """"""
    # Handle paths
    path, filename = os.path.split(likelihood_path)
    snapshot_path = os.path.join(path, "snapshot.pickle")

    # Build a predictions object
    prediction = Prediction(likelihood_path, chunks=chunk)

    # Histogram the data
    prediction.histogram()

    # Calculate the percentiles
    prediction.percentile((2.5, 97.5))

    # Snapshot and save
    snapshot = prediction.snapshot()
    snapshot.pickle(snapshot_path)

    return snapshot
