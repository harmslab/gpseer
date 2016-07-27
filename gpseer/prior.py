
class Prior(object):
    """An object that attempts to make prior predictions about a genotype-
    phenotype map.

    Parameters
    ----------
    gpm : seqspace.GenotypePhenotypeMap object
        a genotype-phenotype object to make prior observations
    model : epistasis.model object
        an epistasis model
    """
    def __init__(self, gpm, model):
        self.gpm = gpm
        self.model

    def rank(self, genotype):
        """Return a rank of reference states for given genotype by choosing the
        model that has the highest information content. Information content is
        defined as the number of coefficients need to determine a given genotype
        and the relative certainty of that genotype.
        """

    def information(self, genotype, reference):
        """Calculate the amount of information for a genotypes from a given
        reference state.
        """
