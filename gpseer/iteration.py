import seqspace as _seqspace
import numpy as _np
from .model import Model as _Model
from .genotype import Genotype as _Genotype
from .bayes import (Priors as _Priors,
                    Posteriors as _Posteriors)

class Iteration(object):
    """An object/API that manages a single iteration of predictions for a given
    genotype-phenotype map.
    """
    def __init__(self,
            Predictor,
            label,
            prior_genotypes,
            prior_phenotypes,
            prior_stdeviations,
        ):
        # Attach to main API
        self.Predictor = Predictor
        # Create a genotype_phenotype map.
        self.GenotypePhenotypeMap = _seqspace.GenotypePhenotypeMap(prior_genotypes[0],
            prior_genotypes,
            prior_phenotypes,
            stdeviations=prior_stdeviations)
        # Get name of iteration.
        self.label = label
        self.Group = self.Predictor.File.create_group(self.label)
        self.Group.attrs["latest"] = 1
        # Initialize datasets
        self.Priors = _Priors(self, prior_genotypes, prior_phenotypes, prior_stdeviations)
        # Create subgroups
        self.model_group = self.Group.create_group("Models")
        self.genotype_group = self.Group.create_group("Genotypes")
        # Initialize models
        self.Models = {}
        for genotype in self.GenotypePhenotypeMap.complete_genotypes:
            model = _Model(self, genotype, **self.Predictor.options_model)
            self.Models[genotype] = model
        # Initialize genotypes
        self.Genotypes = {}
        # Genotypes to fit
        for index, genotype in enumerate(self.GenotypePhenotypeMap.missing_genotypes):
            # Index is shifted by the amount of known genotypes
            geno = _Genotype(self, genotype, self.GenotypePhenotypeMap.n + index)
            self.Genotypes[genotype] = geno

    @classmethod
    def read(cls, Predictor, Group):
        """Read an iteration from an hdf5 file.
        """
        self = cls.__new__(cls)
        self.Predictor = Predictor
        self.label = Group.name.split("/")[-1]
        self.Group = Group
        # Read prior
        self.Priors = _Priors.read(self.Group["Priors"])
        # Create a genotype_phenotype map.
        self.GenotypePhenotypeMap = GenotypePhenotypeMap(self.Priors.genotypes[0],
            self.Priors.genotypes,
            self.Priors.phenotypes,
            errors=self.Priors.stdeviations)
        # Get sub groups
        self.model_group = self.Group["Models"]
        self.genotype_group = self.Group["Genotypes"]
        # Read members of groups
        self.Models = {}
        for key, model in self.model_group.items():
            self.Models[key] = _Model.read(self, model, **self.Predictor.options_model)
        self.Genotypes = {}
        for key, genotype in self.genotype_group.items():
            self.Genotypes[key] = _Genotype.read(self, genotype)
        return self

    def get(self, genotype):
        """Get the data for a given genotype.
        """
        return self.Genotypes[genotype].Dataset

    def bin(self, nbins=100,range=(0,100)):
        """Sweep through genotypes and bin data
        """
        for key, genotype in self.Genotypes.items():
            # Try to clear the dataset
            try: genotype.clear()
            except: pass
            genotype.bin(nbins, range)

    def bin_genotype(self, genotype, nbins=100,range=(0,100)):
        """Bin data of a given genotype
        """
        genotypex = self.Genotypes[genotype]
        try: genotypex.clear()
        except: pass
        genotypex.bin(nbins, range)

    def sample(self, nsamples):
        """Sweep through all models and add samples"""
        for key, model in self.Models.items():
            model.sample(nsamples)

    def sample_model(self, model_label, nsamples):
        """Add samples to a specific model."""
        modelx = self.Models[model_label]
        modelx.sample(nsamples)

    def fit(self):
        """Fits each prediction spectra with a multiple gaussian peaks
        """
        for key, Genotype in self.Genotypes.items():
            Genotype.fit()

    def predict(self):
        """Will attempt to create a set of posteriors from the unobserved genotypes.
        If a distribution is underdetermined (multiple peaks), it will skip that
        genotype and move on. The list of posteriors at the end represent the
        information gain from the model.
        """
        post_genotypes = []
        post_phenotypes = []
        post_stdeviations = []
        for key, Genotype in self.Genotypes.items():
            if Genotype.npeaks == 1:
                peak = Genotype.peaks[0]
                post_genotypes.append(key)
                post_phenotypes.append(peak[0])
                post_stdeviations.append(peak[2])
        # Set up numpy arrays
        self.Posteriors = _Posteriors(self,
            _np.array(post_genotypes),
            _np.array(post_phenotypes),
            _np.array(post_stdeviations)
        )
