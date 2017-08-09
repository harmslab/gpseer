import os
import glob
import h5py
import copy
import pickle
import dask.array
import dask.distributed
from . import workers

from . import prediction

class GPSeer(object):
    """General parallel Sampler for inferring phenotypes in a sparse
    genotype-phenotype map, powered by Dask. All likelihood samples
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
    client: dask.distributed.Client
        A client to parallelize the gpseer

    Example
    -------
    The resulting predictor database will look something like:

    .. code-block:: bash

        predictor/
            gpm.pickle
            model.pickle
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
                    likelihood.h5
                genotype-2/
                    likelihood.h5
                .
                .
                .

    """
    def __init__(self, client=None):
        # Scheduler
        if client is None:
            raise Exception("Start a dask distributed client first.")
        self.client = client

    def setup(self, gpm=None, model=None, db_dir=None):
        """Setup the GPSeer object with data and models for sampling.

        Parameters
        ----------
        gpm: GenotypePhenotypeMap
            dataset as a GenotypePhenotypeMap object
        model:
            epistasis model to fit the data.
        db_dir: str
            path to sample database.
        """
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

        # Pickle models and datasets.
        with open(os.path.join(self._db_dir, "model.pickle"), "wb") as f:
            pickle.dump(self.model, f)

        with open(os.path.join(self._db_dir, "gpm.pickle"), "wb") as f:
            pickle.dump(self.gpm, f)

        return self

    def load(self, db_dir):
        """Load a GPSeer from a sample database that already exists."""
        self._db_dir = db_dir

        # Read Pickled models and datasets.
        with open(os.path.join(self._db_dir, "model.pickle"), "rb") as f:
            self.model = pickle.load(f)

        with open(os.path.join(self._db_dir, "gpm.pickle"), "rb") as f:
            self.gpm = pickle.load(f)

        # Read ML fits
        try:
            self.fits = {}
            for genotype in self.genotypes:
                path = os.path.join(self._db_dir, "models", genotype, "model.pickle")
                with open(path, "rb") as f:
                    self.fits[genotype] = pickle.load(f)

        except IOError:
            print("No ML fits found. Might be a good idea to call `add_fits`.")

        # Read snapshots
        try:
            self.snapshots = {}
            for genotype in self.genotypes:
                path = os.path.join(self._db_dir, "snapshots", "{}.pickle".format(genotype))
                self.snapshots[genotype] = prediction.Snapshot.load(path)

        except IOError:
            print("No snapshots found.")

        return self

    @property
    def genotypes(self):
        """Full set of genotypes in genotype-phenotype map."""
        return self.gpm.complete_genotypes

    def add_fits(self):
        """Method that distributes a fitter method to many workers, powered by Dask

        See the ``gpseer.workers.fit`` docstring for documentation on the workers.
        """
        genotypes = self.gpm.complete_genotypes
        fits = self.client.map(workers.fit, genotypes, gpm=self.gpm, model=self.model)
        fits = self.client.gather(fits)
        self.fits = dict(zip(genotypes, fits))

    def add_samples(self, n_samples=10000, **kwargs):
        """Method that distributes a sampler method to many workers, powered by Dask

        See the ``gpseer.workers.sample`` docstring for documentation on the workers.
        """
        genotypes = list(self.fits.keys())
        fits = list(self.fits.values())
        fits = self.client.map(workers.sample, fits, n_samples=n_samples, db_dir=self._db_dir, **kwargs)
        out = self.client.gather(fits)

    def add_predictions(self):
        """Method that distributes a predictions method to many workers, powered by Dask

        See the ``gpseer.workers.predict`` docstring for documentation on the workers.
        """
        genotypes = list(self.fits.keys())
        fits = list(self.fits.values())
        fits = self.client.map(workers.predict, fits, db_dir=self._db_dir)
        out = self.client.gather(fits)

    def add_sorted_likelihoods(self):
        """Method that distributes a sort method to many workers, powered by Dask

        See the ``gpseer.workers.sort`` docstring for documentation on the workers.
        """
        genotypes = list(self.fits.keys())
        fits = list(self.fits.values())
        fits = self.client.map(workers.sort, fits, db_dir=self._db_dir)
        out = self.client.gather(fits)

    def add_analysis(self):
        """Method that distributes a analysis method to many workers, powered by Dask

        See the ``gpseer.workers.analysis`` docstring for documentation on the workers.
        """
        predictions = self.client.map(workers.analyze, self.genotypes, db_dir=self._db_dir)
        snapshots = self.client.gather(predictions)
        self.snapshots = dict(zip(self.genotypes, snapshots))

    def run(self, n_samples=10000, **kwargs):
        """Run GPSeer out-of-box"""
        # Run the Pipeline to predict phenotypes
        self.add_fits()
        self.add_samples(n_samples=n_samples, **kwargs)
        self.add_predictions()
        self.add_sorted_likelihoods()
        self.add_analysis()
