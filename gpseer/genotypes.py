import numpy as np

from .statistics import fit_peaks, lorentz

class GenotypeList(object):
    """Container object for genotypes.
    """
    def __init__(self, **kwargs):
        """
        """
        self._genotypes = {}
        self.add(**kwargs)

    def get(self, genotype):
        """Get a genotype.
        """
        return getattr(self, genotype)

    def add(self, **kwargs):
        """Add genotypes to list.
        """
        for genotype, obj in kwargs.items():
            setattr(self, genotype, obj)
            self._genotypes[genotype] = obj

    def rm(self, *genotypes):
        """Remove genotype from list"""
        for genotype in genotypes:
            delattr(self, genotype)
            del self._genotypes[genotype]


class Genotype(object):
    """Container object for Genotype prediction data.

    Parameters
    ----------
    genotype : str
        genotype
    missing : bool
        True is the genotype is missing from data, else False.
    predictions : gpseer.PredictionsFile object
        object linked to HDF5 file.
    """
    def __init__(self, genotype, missing, predictions):
        self.name = "g" + genotype
        self.genotype = genotype
        self.missing = missing
        self.predictions = predictions

    @property
    def references(self):
        return self.predictions.references

    def samples(self, reference=None):
        """Get all predictions from all references states for a given genotype.

        Parameters
        ----------
        genotype : string
            genotype to get statistics.
        """
        if reference is None:
            references = self.predictions.references
        else:
            references = ["r" + reference]

        for ref in references:
            ref = ref[1:]
            print(ref)
            genotypes = self.predictions.get(ref).genotypes
            mapping = dict(zip(genotypes, np.arange(len(genotypes))))
            index = mapping[self.genotype]

            data = self.predictions.get(ref).samples[:,index]
            # Concatenate to full set, unless this is the first reference.
            try:
                samples = np.concatenate((samples, data))
            except NameError:
                samples = data

        return samples

    def histogram(self, reference=None, bins=30, range=None, **kwargs):
        """bin data.
        """
        predictions = self.samples(reference=reference)
        return np.histogram(predictions, bins=bins, range=range, **kwargs)

    def fit_peaks(self, reference=None,
        cwtrange=None,
        peak_widths=np.arange(1,100),
        bins=30,
        function=lorentz,
        **kwargs):
        """Find peaks in the data.
        """
        data = self.histogram(reference, bins, range=cwtrange, normed=True, **kwargs)
        counts = data[0]
        values = data[1][1:]
        print(len(counts), len(values))
        return fit_peaks(values, counts, widths=peak_widths, function=function)
