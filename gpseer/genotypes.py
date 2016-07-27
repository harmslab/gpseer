import numpy as np

from .statistics import fit_peaks, gaussian

class GenotypeList(object):
    """Container object for a list of genotypes.
    """
    def __init__(self, **kwargs):
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

        Returns
        -------
        samples : 2d array
            array of many samples of predicted data for this genotype.
        """
        if reference is None:
            references = self.predictions.references
        else:
            references = ["r" + reference]

        for ref in references:
            ref = ref[1:]
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
        """Make histogram of all prediction for this genotype.

        Parameters
        ----------
        reference : string (default=None)
            reference states for epistasis models.
        bins : int
            number of bins to construct histogram
        range : tuple
            bounds for histogram
        kwargs are passed into numpy's histogram function

        Returns
        -------
        histdata : tuple of arrays
            (count, values)
        """
        samples = self.samples(reference=reference)
        self.histdata = np.histogram(samples, bins=bins, range=range, **kwargs)
        return self.histdata

    def fit_peaks(self, reference=None,
        cwtrange=None,
        peak_widths=np.arange(1,100),
        bins=30,
        function=gaussian,
        **kwargs):
        """Find peaks in the prediction distributions and return the statistics.

        Parameters
        ----------
        reference : str

        Returns
        -------
        peaks : list
            list of tuples. first element is a peak center, and second element
            is a peak width.
        """
        try:
            data = self.histdata
        except AttributeError:
            data = self.histogram(reference, bins, range=cwtrange, normed=True, **kwargs)
        counts = data[0]
        values = data[1][1:]
        self.peaks =  fit_peaks(values, counts, widths=peak_widths, function=function)
        return self.peaks
