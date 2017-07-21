import os
import glob
import h5py
import copy
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
    """
    def __init__(self, gpm, model, db_dir=None, **kwargs):
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
        self.client = dask.distributed.Client()

    @classmethod
    def load(cls, db_dir):
        self = cls()
        return self

    def add_ml_fits(self, **kwargs):
        genotypes = self.gpm.complete_genotypes
        fits = self.client.map(workers.fit, genotypes, gpm=self.gpm, model=self.model, kwargs=kwargs)
        fits = self.client.gather(fits)
        self.fits = dict(zip(genotypes, fits))

    def add_samples(self, n_samples):
        genotypes = self.gpm.complete_genotypes
        models = list(self.fits.values())
        likelihoods = self.client.map(workers.sample, models, n_samples=n_samples, db_dir=self._db_dir)
        likelihoods = self.client.gather(likelihoods)
        self.likelihoods = dict(zip(genotypes, likelihoods))

    def add_predictions(self):
        genotypes = self.gpm.complete_genotypes
        fits = list(self.fits.values())
        results = self.client.map(workers.predict, fits, db_dir=self._db_dir)
        self.client.gather(results)

    def add_posteriors(self):
        genotypes = self.gpm.complete_genotypes
        indices = list(range(len(genotypes)))
        items = tuple(zip(indices,genotypes))
        slices = self.client.map(workers.posterior, items, db_dir=self._db_dir)
        self.client.gather(slices)
