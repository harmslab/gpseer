import os
import shutil
import h5py
from datetime import datetime
import pickle
import numpy as np
from epistasis.sampling.base import Sampler, file_handler
from epistasis.sampling import BayesianSampler, BootstrapSampler

class LikelihoodDB(BayesianSampler):
    """API for sampling the likelihood of an epistasis model and inferring
    phenotypes from its likelihood.

    Parameters
    ----------
    model :
        Epistasis model to run a bootstrap calculation.
    db_dir : str (default=None)
        Name a the database directory for storing samples.

    Attributes
    ----------
    coefs : array
        samples for the coefs in the epistasis model.
    scores : array
        Log probabilities for each sample.
    best_coefs : array
        most probable model.
    """
    @file_handler
    def _try_to_create_file(self):
        """Try to create contents for database HDF5 file.
        """
        # Add database
        if "coefs" not in self.File:
            self.File.create_dataset("coefs", (0,0), maxshape=(None,None), compression="gzip")
        if "scores" not in self.File:
            self.File.create_dataset("scores", (0,), maxshape=(None,), compression="gzip")
        if "predictions" not in self.File:
            self.File.create_dataset("predictions", (0,0), maxshape=(None,None), compression="gzip")

    @property
    def prediction_map(self):
        """Genotype mapped to their index in the prediction database."""
        keys = self.model.gpm.complete_genotypes
        values = range(len(keys))
        return dict(zip(keys, values))

    @property
    @file_handler
    def predictions(self):
        """Samples of epistatic coefficients. Rows are samples, Columns are coefs."""
        return self.File["predictions"].value

    @file_handler
    def add_predictions(self, overwrite=False):
        """Take model samples and make predictions from each sample. If overwrite
        is False (default), the method will continue from sample that don't have
        corresponding predictions yet.
        """
        if overwrite is False:
            # Get samples that don't have corresponding predictions.
            s_dims = self.File["coefs"].shape
            p_dims = self.File["predictions"].shape
            diff = s_dims[0] - p_dims[0]
            samples = self.coefs[-diff:, :]

            # Make predictions
            predictions = self.predict(samples)

            # Write to disk
            self.write_dataset("predictions", predictions)

        else:
            # Create samples.
            samples = self.coefs.value

            # Make predictions from samples
            predictions = self.predict(samples)

            # Write to disk
            ds = self.File["predictions"]
            ds.resize(predictions.shape) # Resize predictions
            self.File["predictions"][:,:] = predictions

    def predict(self, samples):
        """Use a set of models to predict pseudodata.

        Parameters
        ----------
        samples : 2d array
            Sets of parameters from different models to predict.

        Returns
        -------
        predictions : 2d array
            Sets of data predicted from the sampled models.
        """
        predictions = np.empty((samples.shape[0], len(self.model.gpm.complete_genotypes)), dtype=float)
        for i in range(len(samples)):
            predictions[i,:] = self.model.hypothesis(thetas=samples[i,:])
        return predictions

    def predict_from_random_samples(self, n):
        """Randomly draw from sampled models and predict phenotypes.

        Parameters
        ----------
        n : int
            Number of models to randomly draw to create a set of predictions.

        Returns
        -------
        predictions : 2d array
            Sets of data predicted from the sampled models.
        """
        sample_size, coef_size = self.coefs.shape
        indices = np.random.choice(np.arange(sample_size), n, replace=True)
        return self.predict(samples=self.coefs.value[indices,:])

    def predict_from_top_samples(self, n):
        """Draw from top sampled models and predict phenotypes.

        Parameters
        ----------
        n : int
            Number of top models to draw to create a set of predictions.

        Returns
        -------
        predictions : 2d array
            Sets of data predicted from the sampled models.
        """
        sample_size, coef_size = self.coefs.shape
        model_indices = np.argsort(self.scores)[::-1]
        samples = np.empty((n, coef_size))
        for i, index in enumerate(model_indices[:n]):
            samples[i,:] = self.coefs[index, :]
        return self.predict(samples=samples)
