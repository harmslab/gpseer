import os
import glob
import h5py
import copy
import pickle
import dask.array
import dask.distributed
from . import workers

from .prediction import Prediction

class GPSeer(object):
    """API for sampling a set of epistasis models and inferring phenotypes from
    the resulting posterior distribution, powered by Dask. All likelihood samples
    are saved in a directory with separate HDF5 database files for each model.
    Posterior distributions are also stored in separate HDF5 database files.
    See database architecture below.

    .. math::
        P(H|E) = \\frac{ P(E|H) \cdot P(H) }{ P(E) }

    This reads: "the probability of epistasis model :math:`H` given the data
    :math:`E` is equal to the probability of the data given the model times the
    probability of the model."


    Parameters
    ----------
    gpm: GenotypePhenotypeMap
        dataset as a GenotypePhenotypeMap object
    model:
        epistasis model to fit the data.
    client: dask.distributed.Client
        A client to parallelize the gpseer
    db_dir: str


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
        likelihoods/
            genotype-1/
                likelihoods-db.h5
            genotype-2.h5
                likelihoods-db.h5
            .
            .
            .
    """
    def __init__(self, gpm, model, client=None, db_dir=None, **kwargs):
        # Set parameters
        self.gpm = gpm
        self.model = model

        # Set up the sampling database
        if db_dir is None:
            self._db_dir = utils.add_datetime_to_filename("GPSeer")
        else:
            self._db_dir = db_dir

        # Create a folder for the database.
        if not os.path.exists(self._db_dir):
            os.makedirs(self._db_dir)

        # Scheduler
        if client is None:
            raise Exception("Start a dask distributed client first.")
        self.client = client

        with open(os.path.join(self._db_dir, "model.pickle"), "wb") as f:
            pickle.dump(self.model, f)

        with open(os.path.join(self._db_dir, "gpm.pickle"), "wb") as f:
            pickle.dump(self.gpm, f)

    @property
    def genotypes(self):
        """Full set of genotypes in genotype-phenotype map."""
        return self.gpm.complete_genotypes

    @classmethod
    def load(cls, client, db_dir):
        """Load GPSeer for a database of samples.
        """
        # Load models and gpm for GPSeer class
        model_path = os.path.join(db_dir, "model.pickle")
        gpm_path = os.path.join(db_dir, "gpm.pickle")

        # Load files
        with open(model_path, "rb") as f: model = pickle.load(model_path);
        with open(gpm_path, "rb") as f: gpm = pickle.load(gpm_path);

        # Initialize class
        self = cls(gpm, model, client, db_dir=None)
        return self

    def _add_ml_fits(self):
        genotypes = self.gpm.complete_genotypes
        fits = self.client.map(workers.fit, genotypes, gpm=self.gpm, model=self.model)
        fits = self.client.gather(fits)
        self.fits = dict(zip(genotypes, fits))

    def _add_samples(self, n_samples=10000, **kwargs):
        genotypes = list(self.fits.keys())
        fits = list(self.fits.values())
        fits = self.client.map(workers.sample, fits, n_samples=n_samples, db_dir=self._db_dir, **kwargs)
        fits = self.client.gather(fits)
        self.fits = dict(zip(genotypes, fits))

    def _add_predictions(self):
        genotypes = list(self.fits.keys())
        fits = list(self.fits.values())
        fits = self.client.map(workers.predict, fits, db_dir=self._db_dir)
        fits = self.client.gather(fits)
        self.fits = dict(zip(genotypes, fits))

    def _add_sorted_likelihoods(self):
        genotypes = list(self.fits.keys())
        fits = list(self.fits.values())
        fits = self.client.map(workers.sort, fits, db_dir=self._db_dir)
        paths_to_likelihoods = self.client.gather(fits)
        self.paths_to_likelihoods = dict(zip(genotypes, paths_to_likelihoods))

    def _add_analysis(self):
        """"""
        genotypes = list(self.paths_to_likelihoods.keys())
        likelihood_paths = list(self.paths_to_likelihoods.values())
        chunk = len(genotypes)
        predictions = self.client.map(workers.analyze, likelihood_paths, chunk=chunk)
        snapshots = self.client.gather(predictions)
        self.snapshots = dict(zip(genotypes, snapshots))

    def run(self, n_samples=10000, **kwargs):
        """Run GPSeer out-of-box"""
        # Run the Pipeline to predict phenotypes
        self._add_ml_fits()
        self._add_samples(n_samples=n_samples, **kwargs)
        self._add_predictions()
        self._add_sorted_likelihoods()
        self._add_analysis()
