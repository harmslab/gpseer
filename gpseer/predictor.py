# External dependencies
import os as _os
import h5py as _h5py
import numpy as _np
import json as _json
import copy as _copy

# Epistasis imports
from epistasis.sampling import BayesianSampler
from gpmap.utils import hamming_distance

# Local imports
from . import utils

class Predictor(object):
    """Genotype-phenotype map predictor.

    Two step MCMC Bayesian algorithm.

    1. Construct a set of epistasis models from a single dataset and use
        MCMC algorithm to approximate the full posterior probabilities for all
        coefficients in the model.


    Parameters
    ----------
    """
    def __init__(self, gpm, model_class, db_dir=None):

        self.gpm = gpm
        self.model_class = model_class

        # -----------------------------------
        # Set up the sampling database
        # -----------------------------------

        if db_dir is None:
            self._db_dir = utils.add_datetime_to_filename("sampler")
        else:
            self._db_dir = db_dir

        # Create a folder for the database.
        if not _os.path.exists(self._db_dir):
            _os.makedirs(self._db_dir)

        # Get all genotypes to create a model from.
        self.references = list(self.gpm.complete_genotypes)
        self.locations = {reference : _os.path.join(self._db_dir, reference) for reference in self.references}
        self.likelihood_samplers = {reference : self.add_likelihood_sampler(reference)
                                    for reference in self.references}

    def add_likelihood_sampler(self, reference):
        """Given a reference state, return a likelihood calculator."""
        # Extremely inefficient...
        gpm = _copy.deepcopy(self.gpm)
        # Set the reference state for the binary representation of the map.
        gpm.binary.reference = reference
        # Initialize a model and fit
        model = self.model_class.from_gpm(gpm, order=gpm.binary.length, model_type="local").fit()
        sampler = BayesianSampler(model, db_dir=self.locations[reference])
        return sampler

    def sample_likelihoods(self, n):
        """Create N samples of the likelihood function."""
        for sampler in self.likelihood_samplers.values():
            sampler.add_samples(n)

    def predict(self, genotype, nmax=10000):
        """"""
        # Get reference
        genotype_index = list(self.gpm.genotypes).index(genotype)
        priors = {ref: _np.exp(hamming_distance(genotype, ref)) for ref in self.references}
        denom = sum(priors.values())
        posterior = []
        for ref, prior in priors.items():
            # Calculate the number of samples to draw
            # from this reference state
            weight = prior / denom
            nsamples = int(weight * nmax)

            # Draw samples from likelihood function
            if nsamples > 0:
                samples = self.likelihood_samplers[ref].predict_from_weighted_samples(nsamples)
                posterior += list(samples[:,genotype_index])
        return posterior
