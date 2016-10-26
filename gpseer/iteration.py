from .model import Model as _Model
from .genotype import Genotype as _Genotype

class Iteration(object):
    """A container object that holds a single iteration of the predictor.
    """
    def __init__(self,
            Predictor,
            label,
            known_genotypes,
            known_phenotypes,
            known_stdeviations,
        ):
        # Attach to main API
        self.Predictor = Predictor
        # Get name of iteration.
        self.label = label
        self.Group = self.Predictor.File.create_group(self.label)

        # Initialize datasets
        self.genotypes = known_genotypes
        self.phenotypes = known_phenotypes
        self.stdeviations = known_stdeviations

        # Create subgroups
        self.model_group = self.Group.create_group("Models")
        self.genotype_group = self.Group.create_group("Genotypes")

        # Initialize models
        self.Models = {}
        for genotype in self.genotypes:
            model = _Model(self, genotype, **self.Predictor.options_model)
            self.Models[genotype] = model

        # Initialize genotypes
        self.Genotypes = {}
        for i, genotype in enumerate(self.GenotypePhenotypeMap.complete_genotypes):
            geno = _Genotype(self, genotype, i)
            self.Genotypes[genotype] = geno

    @property
    def GenotypePhenotypeMap(self):
        return self.Predictor.GenotypePhenotypeMap

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
        self._genotypes = self.Group.create_dataset("genotypes",
            data=genotypes.astype("S" + str(len(genotypes[0]))))

    @phenotypes.setter
    def phenotypes(self, phenotypes):
        """Writes phenotypes to datasets"""
        self._phenotypes = self.Group.create_dataset("phenotypes", data=phenotypes)

    @stdeviations.setter
    def stdeviations(self, stdeviations):
        """Writes phenotypes to datasets"""
        self._stdeviations = self.Group.create_dataset("stdeviations", data=stdeviations)

    @classmethod
    def read(cls, file, path):
        """"""

    def get(self, genotype):
        """Get the data for a given genotype.
        """
        return self.Genotypes[genotype].Dataset

    def bin(self, nbins=100,range=(0,100)):
        """Sweep through genotypes and bin data
        """
        for key, genotype in self.Genotypes.items():
            genotype.bin(nbins, range)

    def bin_genotype(self, genotype, nbins=100,range=(0,100)):
        """Bin data of a given genotype
        """
        genotypex = self.Genotypes[genotype]
        genotypex.bin(nbins, range)

    def sample(self, nsamples):
        """Sweep through all models and add samples"""
        for key, model in self.Models.items():
            model.sample(nsamples)

    def sample_model(self, model_label, nsamples):
        """Add samples to a specific model."""
        modelx = self.Models[model_label]
        modelx.sample(nsamples)
