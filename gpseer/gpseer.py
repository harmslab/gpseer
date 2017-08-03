import os
import glob
import h5py
import copy
import pickle
import dask
import dask.distributed
from . import workers

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

    def add_ml_fits(self):
        genotypes = self.gpm.complete_genotypes
        fits = self.client.map(workers.fit, genotypes, gpm=self.gpm, model=self.model)
        fits = self.client.gather(fits)
        self.fits = dict(zip(genotypes, fits))

    def add_samples(self, n_samples=10000, **kwargs):
        genotypes = self.gpm.complete_genotypes
        models = list(self.fits.values())
        likelihoods = self.client.map(workers.sample, models, n_samples=n_samples, db_dir=self._db_dir, **kwargs)
        likelihoods = self.client.gather(likelihoods)
        self.likelihoods = dict(zip(genotypes, likelihoods))

    def add_predictions(self):
        genotypes = self.gpm.complete_genotypes
        fits = list(self.fits.values())
        results = self.client.map(workers.predict, fits, db_dir=self._db_dir)
        self.client.gather(results)

    def add_likelihoods(self):
        genotypes = self.gpm.complete_genotypes
        indices = list(range(len(genotypes)))
        items = tuple(zip(indices,genotypes))
        slices = self.client.map(workers.likelihood, items, db_dir=self._db_dir)
        self.client.gather(slices)

    def run(self, n_samples=10000):
        """Run GPSeer out-of-box"""
        self.add_ml_fits()
        self.add_samples(n_samples=n_samples)
        self.add_predictions()
        self.add_likelihoods()
