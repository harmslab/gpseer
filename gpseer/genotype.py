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
    def __init__(self, Iteration, genotype):
        self.Iteration = Iteration
        self.Group = self.Iteration.genotype_group
        self.genotype = genotype
        self.index = _np.where(self.Iteration.GenotypePhenotypeMap.complete_genotypes == genotype)

    @classmethod
    def read(cls, Iteration, Dataset):
        """Read the genotype from an existing Iteration object.
        """
        genotype = Dataset.name.split("/")[-1]
        self = cls(Iteration, genotype)
        self.Dataset = Dataset
        return self

    def clear(self):
        """Clear the Genotype Dataset from the H5Py
        """
        path = self.Dataset.name
        file = self.Dataset.file
        del file[path]
        del self.Dataset

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
        dataset = _np.zeros((nbins,2), dtype=float)
        for key, model in self.Iteration.Models.items():
            # Get data for a given genotype
            data = model.Dataset[:, self.index]
            heights, bins = _np.histogram(data, range=range, bins=nbins)
            dataset[:,0] += heights
        dataset[:,1] = bins[1:]
        # Write dataset to hdf5 file.
        self.Dataset = self.Group.create_dataset(self.genotype, data=dataset)

    def fit_peaks(self,
        cwtrange=None,
        peak_widths=_np.arange(1,100),
        bins=30,
        function=gaussian,
        **kwargs):
        """
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
        self.peaks = fit_peaks(values, counts, widths=peak_widths, function=function)
        return self.peaks
