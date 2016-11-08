import numpy as _np
from epistasis.models import LinearEpistasisRegression as _LinearEpistasisRegression

class Model(object):
    """A model object to sample from a given reference state and error distribution.

    Parameters
    ----------
    Group : ModelGroup object
        Group object
    Iteration : Iteration object
        Iteration with initial parameters from which this model is constructed.
    reference : string
        reference genotype for epistasis model
    """
    def __init__(self, Iteration, reference, **options):
        self.reference = reference
        self.label = self.reference
        self.Iteration = Iteration
        self.options = options
        self.base_model = _LinearEpistasisRegression(
            self.reference,
            self.Iteration.Priors.genotypes,
            self.Iteration.Priors.phenotypes,
            stdeviations=self.Iteration.Priors.stdeviations,
            **self.options
        )
        self.base_model.sort(self.Iteration.GenotypePhenotypeMap.genotypes)
        self.base_model.sort_missing(self.Iteration.GenotypePhenotypeMap.missing_genotypes)
        self.Group = self.Iteration.model_group

    @classmethod
    def read(cls, Iteration, Dataset, **options):
        """Read the model data from a Model Dataset."""
        reference = Dataset.name.split("/")[-1]
        self = cls(Iteration, reference, **options)
        self.Dataset = Dataset
        return self

    def sample(self, nsamples):
        """Sample the genotype-phenotype map, and fit with epistasis model."""
        phenotypes = _np.empty((nsamples, len(self.base_model.complete_genotypes)))
        for i in range(nsamples):
            # Draw a pseudo sample from the genotype-phenotype map
            sample = self.base_model.sample()
            modeli = _LinearEpistasisRegression.from_gpm(sample.get_gpm(), order=self.base_model.order)
            # fit the model
            modeli.fit()
            # Add predictions to dataset
            phenotypes[i, :] = modeli.statistics.predict()
        # Resize the dataset
        try:
            dims = self.Dataset.shape
            newdims = (dims[0] + nsamples, dims[1])
            self.Dataset.resize(newdims)
            # Add dataset
            self.Dataset[dims[0]:dims[0]+nsamples,:] = phenotypes
        except AttributeError:
            # Initiate a dataset
            self.Dataset = self.Group.create_dataset(self.label,
                data=phenotypes,
                maxshape=(None, len(self.base_model.complete_genotypes))
            )

    def sample_to_convergence(self, sample_size=50, rtol=1e-2):
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
            self.sample(sample_size, reference)
            ############ Check for convergence ##############
            try:
                # Calculate the statistics on the mean and sample.
                mean = _np.mean(self.predictions.get(reference).samples, axis=0)
                std = _np.std(self.predictions.get(reference).samples, ddof=1, axis=0)

                # Check values for
                check_mean = _np.isclose(mean, old_mean, rtol=rtol)
                check_std = _np.isclose(std, old_std, rtol=rtol)
                test1 = _np.all(check_mean)
                test2 = _np.all(check_std)
                tests = (test1, test2)
            # If this is the first iteration, don't check for convergence.
            except NameError:
                mean = _np.mean(self.predictions.get(reference).samples, axis=0)
                std = _np.std(self.predictions.get(reference).samples, ddof=1, axis=0)
            old_mean = mean
            old_std = std
            count += sample_size
