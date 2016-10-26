import numpy as _np
from .statistics import fit_peaks, gaussian

class Genotype(object):
    """API interface to pull data from a series of models in a single iteration
    of the predictor class. Creates a `dataset` attribute which is a histogram
    of all predictions across that single iteration.

    Parameters
    ----------
    Interation : Iteration object
    """
    def __init__(self, Iteration, genotype, index):
        self.Iteration = Iteration
        self.Group = self.Iteration.genotype_group
        self.index = index
        self.label = genotype

    @classmethod
    def read(cls, dataset):
        """Read the genotype from an existing Iteration object.
        """
        genotype = cls(label)
        genotype._dataset = self.Iteration.Group[genotype.label]
        return genotype

    def bin(self, nbins=100,range=(0,100)):
        """Bin all data for this genotype, setting the dataset attribute of this
        genotype.

        Breaks the binning process into chunks to prevent memory overflow.

        Parameters
        ----------
        nbins : int
            number of bins to partition the data
        range : tuple
            range for histogram
        """
        dataset = _np.empty((nbins,2), dtype=float)
        for key, model in self.Iteration.Models.items():
            # Get data for a given genotype
            data = model.Dataset[:, self.index]
            heights, bins = _np.histogram(data, range=range, bins=nbins)
            dataset[:,0] += heights
        dataset[:,1] = bins[1:]
        # Write dataset to hdf5 file.
        self.Dataset = self.Group.create_dataset(self.label, data=dataset)

    def fit_peaks(self, reference=None,
        cwtrange=None,
        peak_widths=_np.arange(1,100),
        bins=30,
        function=gaussian,
        **kwargs):
        """Find peaks in the predicxtion distributions and return the statistics.

        Parameters
        ----------
        reference : str

        Returns
        -------
        peaks : list
            list of tuples. first element is a peak center, and second element
            is a peak width.
        """
        data = self.Dataset
        counts = data[0]
        values = data[1][1:]
        self.peaks =  fit_peaks(values, counts, widths=peak_widths, function=function)
        return self.peaks
