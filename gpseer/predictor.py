# External dependencies
import pickle as _pickle
import os as _os
import h5py as _h5py
import numpy as _np
import json as _json
import copy as _copy
import shutil as _shutil
import multiprocessing as _mp

# Epistasis imports
from epistasis.sampling import BayesianSampler
from gpmap.utils import hamming_distance

# Local imports
from . import utils

class Predictor(object):
    """Sample an Epistasis Model and build predictions for sparsely
    sampled genotype-phenotype maps.

    How it works
    ------------
    1. Creates a database directory to store model samples.
    2. Within the database directory, create a Sampler for each epistasis model
        to sample.
    3. Initialize a Sampler Object, creating a directory for that model, pickling
        the model into that directory, and starting an HDF5 file to store samples.
    4. One initialized, call `fit` to create a ML solution for all models. This
        will be a starting point for the Sampler.
    5. Call `sample` to sample all models. This may take a while.
    6. Call `sample_posterior` to built a posterior distribution for a selected
        genotype. This will return a Posterior object for the given genotype.

    Example
    -------
    The resulting predictor database will look something like:
    ```
    predictor/
        Model.pickle
        Sampler.pickle
        model_kwargs.json
        models/
            genotype-1/
                model.pickle
                samples.h5
            genotype-2/
                model.pickle
                samples.h5
            .
            .
            .
        posteriors/
            genotype-1.h5
            genotype-2.h5
            .
            .
            .

    ```

    Sampling protocol
    -----------------
    The predictor iteratively builds posterior distributions for each genotype by:

    1. sampling the likelihood functions of many epistasis models
    2. and weighting the final distributions by a prior.

    The prior states that models whose reference state is closer to the genotype
    of interest will best approximate the genotype.

    Parameters
    ----------

    """
    def __init__(self, gpm, Model, Sampler=BayesianSampler, db_dir=None, **kwargs):
        # Set parameters
        self.gpm = gpm
        self.Model = Model
        self.Sampler = Sampler
        self.model_kwargs = kwargs

        # Set up the sampling database
        if db_dir is None:
            self._db_dir = utils.add_datetime_to_filename("predictor")
        else:
            self._db_dir = db_dir

        # Create a folder for the database.
        if not _os.path.exists(self._db_dir):
            _os.makedirs(self._db_dir)

    def setup(self):
        """Setup the predictor class."""
        # Write out Model class
        with open(_os.path.join(self._db_dir, "gpm.pickle"), "wb") as f:
            _pickle.dump(self.gpm, f)

        with open(_os.path.join(self._db_dir, "Model.pickle"), "wb") as f:
            _pickle.dump(self.Model, f)

        # Write out Sampler class
        with open(_os.path.join(self._db_dir, "Sampler.pickle"), "wb") as f:
            _pickle.dump(self.Sampler, f)

        with open(_os.path.join(self._db_dir, "model_kwargs.json"), "w") as f:
            _json.dump(self.model_kwargs, f)

        # Create a folder for posterior
        post_path = _os.path.join(self._db_dir, "models")
        _os.makedirs(post_path)

        # Create a folder for posterior
        post_path = _os.path.join(self._db_dir, "posterior")
        _os.makedirs(post_path)

        # Pull full genotype list to pass a references for models.
        self.references = list(self.gpm.complete_genotypes)

        # Store the location of models
        self.paths = {reference : _os.path.join(self._db_dir, "models", reference)
            for reference in self.references}

        # Store models
        self.models = {reference : self.get_model_sampler(reference)
            for reference in self.references}

        # Set the predictor class as ready
        self.ready = True

    def update(self, gpm=None, Model=None, Sampler=None, purge=False):
        """Update items in the predictor."""
        if gpm is not None:
            self.gpm = gpm
        if Model is not None:
            self.Model = Model
        if Sampler is not None:
            self.Sampler = Sampler
        if purge:
            # Delete models directory
            path = _os.path.join(self.db_dir, "models")
            shutil.rmtree(path)
            # Create an empty directory
            _os.makedirs(self._db_dir)

    @classmethod
    def load(cls, db_dir):
        """Load a Predictor database already on disk."""
        # Get the genotype-phenotype map
        with open(_os.path.join(db_dir, "gpm.pickle"), "rb") as f:
            gpm = _pickle.load(f)

        # Get the Epistasis model class
        with open(_os.path.join(db_dir, "Model.pickle"), "rb") as f:
            Model = _pickle.load(f)

        # Get the Sampler object.
        with open(_os.path.join(db_dir, "Sampler.pickle"), "rb") as f:
            Sampler = _pickle.load(f)

        with open(_os.path.join(db_dir, "model_kwargs.json"), "r") as f:
            model_kwargs = _json.load(f)

        # Initialize the predictor
        self = cls(gpm, Model=Model, Sampler=Sampler, db_dir=db_dir)

        # Construct the model references.
        self.references = list(self.gpm.complete_genotypes)

        # Store the path to each model
        self.paths = {reference : _os.path.join(self._db_dir, "models", reference)
            for reference in self.references}

        # Load the h5 file for each model previously sampled.
        self.models = {reference : self.Sampler.from_db(self.paths[reference])
            for reference in self.references}

        # Setup ready.
        self.ready = True

        # Return the class.
        return self

    def get_model_sampler(self, reference):
        """Given a reference state, return a likelihood calculator."""
        # Extremely inefficient...
        gpm = _copy.deepcopy(self.gpm)
        # Set the reference state for the binary representation of the map.
        gpm.binary.wildtype = reference
        # Initialize a model
        model = self.Model.from_gpm(gpm,
            model_type="local",
            **self.model_kwargs)
        # Initialize a sampler. Sampler writes models and samples to disk.
        sampler = self.Sampler(model, db_dir=self.paths[reference])
        # Return sampler
        return sampler

    def fit(self, **kwargs):
        """Estimate the maximum likelihood models for all reference states."""
        for ref in self.references:
            self.models[ref].fit(**kwargs)

    def get_prior(self, genotype):
        """"""
        priors = {ref: _np.exp(-hamming_distance(genotype, ref)) for ref in self.references}
        return priors

    def sample_models(self, n):
        """Create N samples of the likelihood function."""
        for sampler in self.models.values():
            # Add samples to models
            samples = sampler.add_samples(n)

            # Predict the samples
            sampler.predict(samples)

            

    def sample_posteriors(self):
        """"""
        for sampler in self.models.values():
            output = sampler.predict(self.coef.values)













        """Use hamming distance to re-weight models"""
        # Get reference
        # priors = {ref: _np.exp(-hamming_distance(genotype, ref)) for ref in self.references}
        #denom = sum(priors.values())
        posterior = []
        nn = len(self.gpm.complete_genotypes)
        for ref in self.references:
            # Calculate the number of samples to draw
            # from this reference state
            # weight = prior / denom
            nsamples = int(1/nn * nmax)
            # Draw samples from likelihood function
            if nsamples > 0:
                samples = self.models[ref].predict_from_random_samples(nsamples)
                # Get mapping for this model
                mapping = {g: i for i, g in enumerate(self.models[ref].model.gpm.complete_genotypes)}
                index = mapping[genotype]
                # Sort samples
                posterior += list(samples[:, index])

        return _np.array(posterior)
