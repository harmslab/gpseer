import matplotlib.pyplot as plt
import seqspace

class Predictor(object):
    """Genotype-phenotype map predictor.

    Parameters
    ----------
    space : seqspace.GenotypePhenotypeMap object
        The genotype-phenotype map to fit and predict.
    """
    def __init__(self, space):
        if type(space) != seqspace.gpm.GenotypePhenotypeMap:
            raise Exception("""space must be a GenotypePhenotypeMap object""")
        self.space = space

    def add_model(self, modelclass, **kwargs):
        """Add an epistasis model to the predictor object. Also include any
        keyword arguments

        Parameters
        ----------
        """
        self.model = model
        self.kwargs = kwargs

    def add_prediction(self, predictions):
        """Add a prediction to
        """

    def sample(self):
        """Sample the genotype-phenotype map, and fit with epistasis model."""
        sample = self.space.sample()
        modeli = self.model.from_gpm(sample.get_gpm())
        modeli.fit()
        return modeli.epistasis.values()

    def sample_to_convergence(self, sample_size=50, rtol=1e-2):
        """Sample the phenotypes until predictions converge to a tolerance of rtol.
        """
