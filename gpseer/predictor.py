import os as _os
import h5py as _h5py
import matplotlib.pyplot as _plt
import numpy as _np
import json as _json

import seqspace as _seqspace
from .iteration import Iteration as _Iteration
from .results import Results as _Results

class Predictor(object):
    """Genotype-phenotype map predictor.

    The Predictor class creates an HDF5 file that has a predefined hierarchy for
    predicting phenotypes in a genotype-phenotype map.

     uses a linear,
    high-order epistasis model to fit a set of observed genotype-phenotypes, and
    predicts unknown genotypes-phenotypes.

    Parameters
    ----------
    """
    def __init__(self, known_genotypes, known_phenotypes, known_stdeviations, mutations,
            fname="predictor.hdf5",
            overwrite=False,
            **options
        ):
        if overwrite:
            _os.remove(fname)
        # Construct
        self.__construct__(**options)
        # Create an HDF5 for predictor class
        self.File = _h5py.File(fname, "a")
        self.genotypes = known_genotypes
        self.phenotypes = known_phenotypes
        self.stdeviations = known_stdeviations
        self.mutations = mutations

    def __construct__(self, **options):
        """Set all default options from predictor and prepare for learning.
        """
        self.iterations = {}
        self.latest_iteration = None
        # Default options
        self.options = dict(
            nbins=100,
            range=(0,100),
            order=None,
            mutations=None,
            log_transform=False,
            n_replicates=1,
            logbase=_np.log10
        )
        self.options.update(**options)

    @classmethod
    def read(cls, fname, **options):
        """Read a Predictor from a HDF5 file.
        """
        self = cls.__new__(cls)
        self.__construct__(**options)
        self.File = _h5py.File(fname, "r")
        self._genotypes = self.File["genotypes"]
        self._phenotypes = self.File["phenotypes"]
        self._stdeviations = self.File["stdeviations"]
        self._mutations = self.File["mutations"]
        items = list(self.File.keys())
        items.remove("genotypes")
        items.remove("phenotypes")
        items.remove("stdeviations")
        items.remove("mutations")
        # Get all iterations
        for item in items:
            Group = self.File[item]
            Iteration = _Iteration.read(self, Group)
            self.iterations[item] = Iteration
            if Group.attrs["latest"] == 1:
                self.latest_iteration = Iteration
        return self

    @classmethod
    def from_json(cls, jsonfile, fname="predictor.h5", **options):
        """Read a genotype-phenotype map directly from a json file.
        """
        space = _seqspace.GenotypePhenotypeMap.from_json(jsonfile)
        self = cls.from_gpm(space, fname=fname, **options)
        return self

    @classmethod
    def from_gpm(cls, space, fname="predictor.h5", overwrite=False, **options):
        """"""
        if overwrite:
            _os.remove(fname)
        self = cls.__new__(cls)
        # Load all options from the genotype-phenotype map.
        opt1 = dict(
            log_transform=space.log_transform,
            n_replicates=space.n_replicates,
            logbase=space.logbase,
            mutations=space.mutations,
        )
        # Update with manual options
        opt1.update(**options)
        # Construct initial attributes in predictor
        self.__construct__(**opt1)
        # Create HDF5 file
        self.File = _h5py.File(fname, "a")
        #self.GenotypePhenotypeMap = space
        # Write out main datasets
        self.genotypes = space.genotypes
        self.phenotypes = space.phenotypes
        self.stdeviations = space.stdeviations
        self.mutations = space.mutations
        return self

    def _suggest_iteration_label(self):
        """Return a suggested label for an iteration of the model.
        """
        return "iteration-" + str(len(self.iterations))

    def _suboptions(self, keys):
        """Get subset of options from options dictionary"""
        opts = {}
        for key in keys:
            try: opts[key] = self.options.get(key)
            except KeyError: pass
        return opts

    @property
    def genotypes(self):
        """Convert the genotype to an array of strings"""
        return self._genotypes.value.astype(str)

    @property
    def phenotypes(self):
        """Get phenotypes as numpy array in memory"""
        return self._phenotypes.value

    @property
    def stdeviations(self):
        """Get phenotypes as numpy array in memory"""
        return self._stdeviations.value

    @property
    def mutations(self):
        """Get mutations as dictionary in memory."""
        mutations = {}
        for key, dataset in self._mutations.items():
            mutations[int(key)] = list(dataset.value.astype(str))
        return mutations

    @genotypes.setter
    def genotypes(self, genotypes):
        """"""
        self._genotypes = self.File.create_dataset("genotypes",
            data=genotypes.astype("S" + str(len(genotypes[0]))))

    @phenotypes.setter
    def phenotypes(self, phenotypes):
        """Writes phenotypes to datasets"""
        self._phenotypes = self.File.create_dataset("phenotypes", data=phenotypes)

    @stdeviations.setter
    def stdeviations(self, stdeviations):
        """Writes phenotypes to datasets"""
        self._stdeviations = self.File.create_dataset("stdeviations", data=stdeviations)

    @mutations.setter
    def mutations(self, mutations):
        """Writes a mutation dictionary to hdf5 file."""
        self._mutations = self.File.create_group("mutations")
        for key, value in mutations.items():
            label = str(key)
            dataset = _np.array(value, dtype="S1")
            self._mutations.create_dataset(label, data=dataset)

    @property
    def options_model(self):
        """ Return options for models"""
        sub_options = ["order", "log_transform", "mutations", "logbase"]
        return self._suboptions(sub_options)

    @property
    def options_gpm(self):
        """ Return options for genotype-phenotype map"""
        sub_options = ["log_transform", "mutations", "logbase"]
        return self._suboptions(sub_options)

    @property
    def options_genotypes(self):
        """ Return options for genotypes"""
        sub_options = ["nbins", "range"]
        return self._suboptions(sub_options)

    def get(self, genotype):
        """Get the data for a given genotype from the latest iteration of the model.
        """
        return self.latest_iteration.get(genotype)

    def iterate(self, label, genotypes, phenotypes, stdeviations, nsamples, **options):
        """Construct a new iteration of the model.
        """
        for iteration in self.iterations.values():
            iteration.Group.attrs["latest"] = 0
        self.options.update(**options)
        Iteration = _Iteration(self, label, genotypes, phenotypes, stdeviations)
        # Make all other iterations have a latest attribute set to False (0)
        self.iterations[label] = Iteration
        # Run through fitting pipeline
        Iteration.sample(nsamples)
        Iteration.bin(**self.options_genotypes)
        Iteration.fit()
        Iteration.predict()
        self.latest_iteration = Iteration

    def bin(self, nbins):
        """Rebin data in latest iteration.
        """
        self.options.update(nbins=nbins)
        Iteration = self.latest_iteration
        Iteration.bin(**self.options_genotypes)

    def sample(self, nsamples):
        """Adds samples to the latest iteration of the predictor
        """
        Iteration = self.latest_iteration
        Iteration.sample(nsamples)

    def fit(self):
        """Fit the genotypes of the latest iteration.
        """
        Iteration = self.latest_iteration
        Iteration.fit()

    def predict(self):
        """Get a set of posterior predictions from the latest iteration"""
        Iteration = self.latest_iteration
        Iteration.predict()

    def get_priors(self):
        """"""
        # Check if an iteration exists
        if self.latest_iteration is None:
            return self.genotypes, self.phenotypes, self.stdeviations
        else:
            Iteration = self.latest_iteration
            #prior_genotypes = Iteration.Posteriors.genotypes
            #prior_phenotypes = Iteration.Posteriors.phenotypes
            #prior_stdeviations = Iteration.Posteriors.stdeviations
            prior_genotypes = _np.concatenate([Iteration.Priors.genotypes, Iteration.Posteriors.genotypes])
            prior_phenotypes = _np.concatenate([Iteration.Priors.phenotypes, Iteration.Posteriors.phenotypes])
            prior_stdeviations = _np.concatenate([Iteration.Priors.stdeviations, Iteration.Posteriors.stdeviations])
            return prior_genotypes, prior_phenotypes, prior_stdeviations

    def learn(self, **options):
        """Automagically learn from data and predict phenotypes.
        """
        self.options.update(**options)
        # Get previous posteriors if they exist.
        if hasattr(self.latest_iteration, "Posteriors"):
            new_predictions = self.latest_iteration.Posteriors.length
        else:
            new_predictions = len(self.genotypes)
        # Keep iterating the model until no predictions exist
        while new_predictions > 0:
            # Get priors for a new iteration
            prior_genotypes, prior_phenotypes, prior_stdeviations = self.get_priors()
            # Create a new iteration with new priors
            self.iterate(self._suggest_iteration_label(), prior_genotypes, prior_phenotypes,
                prior_stdeviations, 1000)
            # Track number of new posteriors found.
            new_predictions = self.latest_iteration.Posteriors.length
        # Gather the results and save them as separate datasets
        self.Results = _Results(self, *self.get_priors())
        # Print when finished.
        print("Finished: " + str(self.latest_iteration.Priors.length) + " known genotypes")
