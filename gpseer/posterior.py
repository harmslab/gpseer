import os
import h5py
import numpy as np

from epistasis.sampling.base import file_handler

class PosteriorDB(object):
    """API for storing posterior samples from different models in HDF5 files (via h5py)

    Parameters
    ----------
    db_dir : str
        directory name for storing the Posterior database.
    """
    def __init__(self, db_dir=None):
        # -----------------------------------
        # Set up the sampling database
        # -----------------------------------

        if db_dir is None:
            self._db_dir = add_datetime_to_filename("posterior")
        else:
            self._db_dir = db_dir

        # Create a folder for the database.
        if not os.path.exists(self._db_dir):
            os.makedirs(self._db_dir)

        self._db_path = os.path.join(self._db_dir, "posterior-db.hdf5")

        # Create the hdf5 file for saving samples.
        self.File = h5py.File(self._db_path, "a")

        # Add database
        if "posterior" not in self.File:
            self.File.create_dataset("posterior", (0,0), maxshape=(None,None), compression="gzip")

        self.models = {}

    @classmethod
    def from_db(cls, db_dir, overwrite=True):
        """Initialize a PosteriorDB object from a database directory

        Parameters
        ----------
        db_dir : str
            directory name for database already on disk
        overwrite : bool

        Returns
        -------
        PosteriorDB :
            PosteriorDB object.
        """
        if overwrite:
            self = cls(db_dir=db_dir)
        else:
            # New database directory
            new_db_dir = add_datetime_to_filename("posterior")

            # Create a folder for the database.
            if not os.path.exists(new_db_dir):
                os.makedirs(new_db_dir)

            # Old database path
            old_db_path = os.path.join(db_dir, "posterior-db.hdf5")

            # Copy the old database to new database
            shutil.copyfile(db_dir, new_db_dir)

            self = cls(db_dir=new_db_dir)
        return self

    @file_handler
    def add_model_posteriors(self, key, samples, overwrite=False):
        """Append posterior samples from a given model to the Posterior database

        Parameters
        ----------
        key : str
            name/label for the given model.
        samples :
        """
        # Get dataset
        ds = self.File["posterior"]
        # Get dimensions
        dim = ds.shape
        npred = samples.shape[0]
        if key in self.models:
            # Get the column assigned to this model
            col = self.models[key]
            # Reshape the array if needed
            if dim[1] < npred:
                ds.resize((npred, dim[1]))
        else:
            # Reshape the array if needed
            if dim[1] < npred:
                ds.resize((npred, dim[1]+1))
            col = dim[1]
            self.models[key] = col
        # Set the predictions
        ds[:npred, col] = samples

    @property
    @file_handler
    def posterior(self):
        """Posterior samples from all models as 2D array."""
        return self.File("posterior")

    @file_handler
    def flatten(self):
        """Get the full set of posterior samples in the database, flattened
        as a 1D array.
        """
        return self.File["posterior"].value.flatten()

    def percentile(self, *percentiles):
        """Return a list of percentiles from the posterior distribution."""
        return np.percentile(self.flatten(), percentiles)

    def histogram(self, **kwargs):
        """Return a histogram of the posterior samples"""
        return np.histogram(self.flatten(), **kwargs)
