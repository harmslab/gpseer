import matplotlib.pyplot as plt
import numpy as np
import seqspace

from .h5map import H5Map
from .utils import resample_to_convergence

class Predictor(object):
    """Genotype-phenotype map predictor.

    Parameters
    ----------
    space : seqspace.GenotypePhenotypeMap object
        The genotype-phenotype map to fit and predict.
    """
    def __init__(self, space, model, fname="predictions.hdf5", **kwargs):
        if type(space) != seqspace.gpm.GenotypePhenotypeMap:
            raise Exception("""space must be a GenotypePhenotypeMap object""")
        self.space = space
        self.model = model
        self.kwargs = kwargs
        self.predictions = H5Map(fname, self)

    def sample_ref(self, nsamples, reference):
        """Sample the genotype-phenotype map, and fit with epistasis model."""
        phenotypes = np.empty((nsamples, len(self.space.complete_genotypes)))
        for i in range(nsamples):
            sample = self.space.sample()
            modeli = self.model(
                reference,
                sample.genotypes,
                sample.phenotypes,
                log_transform=self.space.log_transform,
                n_replicates=self.space.n_replicates,
                logbase = self.space.logbase,
                mutations = self.space.mutations,
                **self.kwargs
            )
            # Add this model to the HDF5 datatable
            # fit the model
            modeli.fit()
            # Add predictions to dataset
            phenotypes[i, :] = modeli.statistics.predict()

        # Try creating a new dataset, otherwise if it exists, don't create
        try:
            self.predictions.add(reference, self.space.complete_genotypes)
        except ValueError:
            pass
        self.predictions.get(reference).add(phenotypes)

    def sample_to_convergence(self, reference, sample_size=50, rtol=1e-2):
        """Sample the phenotypes until predictions converge to a tolerance of rtol.

        Parameters
        ----------
        reference : string
            reference genotype for epistasis model
        sample_size : int
            number of samples to create before checking the mean/std
        rtol : float
            tolerance for convergence.
        """
        tests = (False, False)
        count = 0

        while False in tests:
            ############### Create a sample ###############
            self.sample_ref(sample_size, reference)
            ############ Check for convergence ##############
            try:
                # Calculate the statistics on the mean and sample.
                mean = np.mean(self.predictions.get(reference).samples, axis=0)
                std = np.std(self.predictions.get(reference).samples, ddof=1, axis=0)

                # Check values for
                check_mean = np.isclose(mean, old_mean, rtol=rtol)
                check_std = np.isclose(std, old_std, rtol=rtol)
                test1 = np.all(check_mean)
                test2 = np.all(check_std)
                tests = (test1, test2)

            # If this is the first iteration, don't check for convergence.
            except NameError:
                mean = np.mean(self.predictions.get(reference).samples, axis=0)
                std = np.std(self.predictions.get(reference).samples, ddof=1, axis=0)

            old_mean = mean
            old_std = std

            count += sample_size

    def full_prediction(self, genotype):
        """Get all predictions from all references states for a given genotype.

        Parameters
        ----------
        genotype : string
            genotype to get statistics.
        """
        for reference in self.predictions.references:
            genotypes = self.predictions.get(reference).genotypes
            mapping = dict(zip(genotypes, np.arange(len(genotypes)))
            index = mapping[genotypes[reference]]
            data = np.predictions.get(reference).samples[:,index]
            # Concatenate to full set, unless this is the first reference.
            try:
                samples = np.concatenate((samples, data))
            except NameError:
                samples = data
        return samples

    def full_statistics(self, genotype):
        pass
