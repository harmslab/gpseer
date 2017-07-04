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
from .likelihood import LikelihoodDB
from .posterior import PosteriorDB
from gpmap.utils import hamming_distance

# Local imports
from . import utils

class Predictor(object):
    """API for sampling a set of epistasis models and inferring phenotypes from
    the resulting posterior distribution. All likelihood samples are saved in a directory
    with separate HDF5 database files for each model. Posterior distributions
    are also stored in separate HDF5 database files. See database architecture below.

    How it works
    ------------
    1. Creates a database directory to store model samples.
    2. Within the database directory, create a LikelihoodDB for each epistasis model
        to sample.
    3. Initialize a Likelihood Object, creating a directory for that model, pickling
        the model into that directory, and starting an HDF5 file to store samples.
    4. One initialized, call `fit` to create a ML solution for all models. This
        will be a starting point for the Likelihood.
    5. Call `sample` to sample all models. This may take a while.
    6. Call `sample_posterior` to built a posterior distribution for a selected
        genotype. This will return a Posterior object for the given genotype.

    .. math::
        P(H|E) = \\frac{ P(E|H) \cdot P(H) }{ P(E) }

    This reads: "the probability of epistasis model :math:`H` given the data
    :math:`E` is equal to the probability of the data given the model times the
    probability of the model."

    Example
    -------
    The resulting predictor database will look something like:
    ```
    predictor/
        Model.pickle
        model_kwargs.json
        models/
            genotype-1/
                model.pickle
                samples-db.h5
            genotype-2/
                model.pickle
                samples-db.h5
            .
            .
            .
        posteriors/
            genotype-1/
                posterior-db.h5
            genotype-2.h5
                posterior-db.h5
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
    gpm : GenotypePhenotypeMap object
        the dataset to predict.
    Model : model object
        epistasis model that will be used to fit the genotype-phenotype map.

    """
    def __init__(self, gpm, Model, db_dir=None, **kwargs):
        # Set parameters
        self.gpm = gpm
        self.Model = Model
        self.Likelihood = LikelihoodDB

        # Set up the sampling database
        if db_dir is None:
            self._db_dir = utils.add_datetime_to_filename("predictor")
        else:
            self._db_dir = db_dir

        # Create a folder for the database.
        if not _os.path.exists(self._db_dir):
            _os.makedirs(self._db_dir)

    def setup(self, **kwargs):
        """Setup the predictor class."""
        self.model_kwargs = kwargs

        # Write out Model class
        with open(_os.path.join(self._db_dir, "gpm.pickle"), "wb") as f:
            _pickle.dump(self.gpm, f)

        with open(_os.path.join(self._db_dir, "Model.pickle"), "wb") as f:
            _pickle.dump(self.Model, f)

        with open(_os.path.join(self._db_dir, "model_kwargs.json"), "w") as f:
            _json.dump(self.model_kwargs, f)

        # Pull full genotype list to pass a references for models.
        self.references = list(self.gpm.complete_genotypes)

        # -----------------------------------------------------------
        # Prepare Likelihood databases
        # -----------------------------------------------------------

        # Create a folder for posterior
        model_path = _os.path.join(self._db_dir, "models")
        _os.makedirs(model_path)

        # Store the location of models
        self.model_paths = {reference : _os.path.join(model_path, reference)
            for reference in self.references}

        # Store models
        self.models = {reference : self.get_model_likelihood(reference)
            for reference in self.references}

        # -----------------------------------------------------------
        # Prepare Posterior databases
        # -----------------------------------------------------------

        # Create a folder for posterior
        posterior_path = _os.path.join(self._db_dir, "posteriors")
        _os.makedirs(posterior_path)

        # Store the location of models
        self.posterior_paths = {reference : _os.path.join(posterior_path, reference)
            for reference in self.references}

        # Store posterior DB
        self.posteriors = {ref : PosteriorDB(path) for ref, path in self.posterior_paths.items()}

        # Set the predictor class as ready
        self.ready = True
        return self

    @classmethod
    def load(cls, db_dir):
        """Load a Predictor database already on disk."""
        # Get the genotype-phenotype map
        with open(_os.path.join(db_dir, "gpm.pickle"), "rb") as f:
            gpm = _pickle.load(f)

        # Get the Epistasis model class
        with open(_os.path.join(db_dir, "Model.pickle"), "rb") as f:
            Model = _pickle.load(f)

        with open(_os.path.join(db_dir, "model_kwargs.json"), "r") as f:
            model_kwargs = _json.load(f)

        # Initialize the predictor
        self = cls(gpm, Model=Model, db_dir=db_dir)
        self.model_kwargs = model_kwargs

        # Construct the model references.
        self.references = list(self.gpm.complete_genotypes)

        # -----------------------------------------------------------
        # Prepare Likelihood databases
        # -----------------------------------------------------------

        # Store the path to each model
        self.model_paths = {reference : _os.path.join(self._db_dir, "models", reference)
            for reference in self.references}

        # Load the h5 file for each model previously sampled.
        self.models = {reference : self.Likelihood.from_db(self.model_paths[reference])
            for reference in self.references}

        # -----------------------------------------------------------
        # Prepare Posterior databases
        # -----------------------------------------------------------

        # Store the location of models
        self.posterior_paths = {reference : _os.path.join(self._db_dir, "posteriors", reference)
            for reference in self.references}

        # Store posterior DB
        self.posteriors = {ref : PosteriorDB.from_db(path) for ref, path in self.posterior_paths.items()}

        # Setup ready.
        self.ready = True

        # Return the class.
        return self

    def get_model_likelihood(self, reference):
        """Given a reference state, return a likelihood calculator."""
        # Extremely inefficient...
        gpm = _copy.deepcopy(self.gpm)
        # Set the reference state for the binary representation of the map.
        gpm.binary.wildtype = reference
        # Initialize a model
        model = self.Model.from_gpm(gpm,
            model_type="local",
            **self.model_kwargs)
        # Initialize a likelihood db. Likelihood writes models and samples to disk.
        likelihood = self.Likelihood(model, db_dir=self.model_paths[reference])
        # Return likelihood
        return likelihood

    def set_prior(self, genotype):
        """"""
        priors = {ref: _np.exp(-hamming_distance(genotype, ref)) for ref in self.references}
        return priors

    def add_ml_fits(self, **kwargs):
        """Estimate the maximum likelihood models for all reference states."""
        for ref in self.references:
            self.models[ref].fit(**kwargs)

    def add_samples(self, n):
        """Sample the likelihood function of all models"""
        for likelihood in self.models.values():
            # Add samples to models
            likelihood.add_samples(n)

    def add_predictions(self):
        """Predict phenotypes from all samples in all the models."""
        for likelihood in self.models.values():
            likelihood.add_predictions()

    def add_posteriors(self):
        """"""
        for ref in self.references:
            path = _os.path.join(self._db_dir, "posteriors", ref)
            posterior = PosteriorDB(db_dir=path)
            # Sort predictins from likelihoods
            for likelihood in self.models.values():
                mapping = likelihood.prediction_map
                predictions = likelihood.predictions[:, mapping[ref]]
                posterior.add_model_posteriors(ref, predictions)
