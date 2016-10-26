import matplotlib.pyplot as _plt
import numpy as _np
import json as _json

import seqspace as _seqspace

from .iteration import Iteration as _Iteration
import h5py as _h5py

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
    def __init__(self, known_genotypes, known_phenotypes, known_stdeviations,
            fname="predictor.hdf5",
            **options
        ):
        # Construct
        self.__construct__(**options)
        self.GenotypePhenotypeMap = _seqspace.GenotypePhenotypeMap(
            known_genotype[0],
            known_genotypes,
            known_phenotypes,
            known_stdeviations
            **self.options_gpm)
        # Create an HDF5 for predictor class
        self.File = _h5py.File(fname, "a")
        #genotypes = known_genotypes.astype("S" + str(len(known_genotypes[0])))
        self.genotypes = known_genotypes#self.File.create_dataset("genotypes", data=genotypes)
        self.phenotypes = known_phenotypes#self.File.create_dataset("phenotypes", data=known_phenotypes)
        self.stdeviations = known_stdeviations#self.File.create_dataset("stdeviations", data=known_errors)

    def __construct__(self, **options):
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
        self.GenotypePhenotypeMap = _seqspace.GenotypePhenotypeMap(
            self.genotypes[0],
            self.genotypes,
            self.phenotypes,
            self.stdeviations,
            **self.options_gpm)
        items = list(self.File.keys())
        items.remove("genotypes")
        items.remove("phenotypes")
        items.remove("stdeviations")
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
    def from_gpm(cls, space, fname="predictor.h5", **options):
        """"""
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
        self.GenotypePhenotypeMap = space
        # Write out main datasets
        self.genotypes = space.genotypes
        self.phenotypes = space.phenotypes
        self.stdeviations = space.stdeviations
        return self

    def get(self, genotype):
        """Get the data for a given genotype from the latest iteration of the model.
        """
        return self.latest_iteration.get(genotype)

    def iterate(self,
            label,
            genotypes,
            phenotypes,
            stdeviations,
            nsamples,
            **options
        ):
        """
        """
        for iteration in self.iterations.values():
            iteration.Group.attrs["latest"] = 0
        self.options.update(**options)
        Iteration = _Iteration(self, label, genotypes, phenotypes, stdeviations)
        # Make all other iterations have a latest attribute set to False (0)
        self.iterations[label] = Iteration
        Iteration.sample(nsamples)
        Iteration.bin(**self.options_genotypes)
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
        pass

    def learn(self, **options):
        """Automagically learn from data and predict phenotypes.
        """
        self.options.update(**options)
