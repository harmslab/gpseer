# External dependencies
import os as _os
import h5py as _h5py
import numpy as _np
import json as _json
import copy as _copy
import multiprocessing as _mp

# Epistasis imports
from epistasis.sampling import BayesianSampler
from gpmap.utils import hamming_distance

# Local imports
from . import utils

class Predictor(object):
    """Genotype-phenotype map predictor.

    Construct

    Two step MCMC Bayesian algorithm.

    1. Construct a set of epistasis models from a single dataset and use
        MCMC algorithm to approximate the full posterior probabilities for all
        coefficients in the model.

    Parameters
    ----------
    """
    def __init__(self, gpm, model_class,
        sampler_class=BayesianSampler,
        db_dir=None,
        n_jobs=1,
        **kwargs):

        # Set parameters
        self.gpm = gpm
        self.model_class = model_class
        self.sampler_class = sampler_class
        self.model_kwargs = kwargs
        self.n_jobs = n_jobs

        # Set up the sampling database
        if db_dir is None:
            self._db_dir = utils.add_datetime_to_filename("predictor")
        else:
            self._db_dir = db_dir

        # Create a folder for the database.
        if not _os.path.exists(self._db_dir):
            _os.makedirs(self._db_dir)

        # Pull full genotype list to pass a references for models.
        self.references = list(self.gpm.complete_genotypes)

        # Store the location of models
        self.paths = {reference : _os.path.join(self._db_dir, reference)
            for reference in self.references}

        # Store models
        self.models = {reference : self.get_model_sampler(reference)
            for reference in self.references}


    def get_model_sampler(self, reference):
        """Given a reference state, return a likelihood calculator."""
        # Extremely inefficient...
        gpm = _copy.deepcopy(self.gpm)
        # Set the reference state for the binary representation of the map.
        gpm.binary.wildtype = reference
        # Initialize a model
        model = self.model_class.from_gpm(gpm,
            model_type="local",
            **self.model_kwargs)
        # Initialize a sampler. Sampler writes models and samples to disk.
        sampler = self.sampler_class(model, db_dir=self.paths[reference], n_jobs=self.n_jobs)
        # Return sampler
        return sampler

    def fit(self, **kwargs):
        """Estimate the maximum likelihood models for all reference states."""
        for ref in self.references:
            self.models[ref].model.fit(**kwargs)

    def sample(self, n):
        """Create N samples of the likelihood function."""
        for sampler in self.models.values():
            sampler.add_samples(n)

    def predict(self, genotype, nmax=1000):
        """Use hamming distance to re-weight models"""
        # Get reference
        priors = {ref: _np.exp(-hamming_distance(genotype, ref)) for ref in self.references}
        denom = sum(priors.values())
        posterior = []
        for ref, prior in priors.items():
            # Calculate the number of samples to draw
            # from this reference state
            weight = prior / denom
            nsamples = int(weight * nmax)
            # Draw samples from likelihood function
            if nsamples > 0:
                samples = self.models[ref].predict_from_random_samples(nsamples)
                # Get mapping for this model
                mapping = {g: i for i, g in enumerate(self.models[ref].model.gpm.complete_genotypes)}
                index = mapping[genotype]
                # Sort samples
                posterior += list(samples[:, index])

        return _np.array(posterior)
