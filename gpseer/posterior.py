import os
import shutil
import h5py
from datetime import datetime
import pickle
import numpy as np
from epistasis.sampling import BayesianSampler

class BasePredictor(BasePredictor):
    """"""
    def __init__(self, model, db_dir=None):
        # Initialize sampler
        super(self).__init__(model, db_dir=db_dir)

        # Add database
        if "predictions" not in self.File:
            self.File.create_dataset("predictions", (0,0), maxshape=(None,None), compression="gzip")

    @property
    def predictions(self):
        """Samples of epistatic coefficients. Rows are samples, Columns are coefs."""
        return self.File["predictions"]

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

class BayesianPredictor(BasePredictor, BayesianSampler):
    """"""
    def add_samples(self, n_mcsteps, nwalkers=None, equil_steps=100):
        """"""
        # Sample Bayesian Predictor
        samples = super(BayesianPredictor, self).add_samples(n_mcsteps, nwalkers, equil_steps)
        # Make predictions
        predictions = self.predict(samples)
        # Write samples.
        self.write_dataset("predictions", predictions)
