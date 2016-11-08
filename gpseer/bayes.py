

class BayesProbabilities(object):

    def __init__(self, Iteration, genotypes, phenotypes, stdeviations):
        self.Iteration = Iteration
        self.Group = self.Iteration.Group.create_group(self.group_name)
        self.genotypes = genotypes
        self.phenotypes = phenotypes
        self.stdeviations = stdeviations

    @property
    def group_name(self):
        raise Exception("""Must be implemented in a subclass.""")

    @classmethod
    def read(self, Iteration, Group):
        """Read a posterior dataset
        """
        self = cls.__new__(cls)
        self.Iteration = Iteration
        self.Group = self.Iteration[self.group_name]
        self._genotypes = self.Group["genotypes"]
        self._phenotypes = self.Group["phenotypes"]
        self._stdeviations = self.Group["stdeviations"]
        return self

    @property
    def length(self):
        return len(self.genotypes)

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
            data=genotypes.astype("S" + str(self.Iteration.GenotypePhenotypeMap.length)))

    @phenotypes.setter
    def phenotypes(self, phenotypes):
        """Writes phenotypes to datasets"""
        self._phenotypes = self.Group.create_dataset("phenotypes", data=phenotypes)

    @stdeviations.setter
    def stdeviations(self, stdeviations):
        """Writes phenotypes to datasets"""
        self._stdeviations = self.Group.create_dataset("stdeviations", data=stdeviations)


class Priors(BayesProbabilities):
    """An object that attempts to make prior predictions about a genotype-
    phenotype map.
    """
    @property
    def group_name(self):
        return "Priors"

class Posteriors(BayesProbabilities):
    """
    """
    @property
    def group_name(self):
        return "Posteriors"
