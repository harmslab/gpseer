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
        self.genotype = genotype
        self._index = index

    @classmethod
    def read(cls, Iteration, Dataset):
        """Read the genotype from an existing Iteration object.
        """
        genotype = Dataset.name.split("/")[-1]
        self = cls(Iteration, genotype)
        self.Dataset = Dataset
        return self

    @property
    def index(self):
        """Get list of peaks in memory.
        """
        # Try returning the hdf5 dataset attribute
        try:
            return self.Dataset.attrs["index"]
        except:
            return self._index

    @index.setter
    def index(self, index):
        """Set the index of the genotype in model samples.
        """
        # Try setting attribute of hdf5 dataset otherwise just set attribute
        self.Dataset.attrs["index"] = index

    @property
    def npeaks(self):
        """Get the number of peaks in dataset."""
        return len(self.peaks)

    @property
    def peaks(self):
        """Get list of peaks in memory.
        """
        return self.Dataset.attrs["peaks"]

    @peaks.setter
    def peaks(self, peaks):
        """
        """
        self.Dataset.attrs["peaks"] = peaks

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
        self.index = self._index

    def fit(self):
        """
        Returns
        -------
        peaks : list
            list of tuples. first element is a peak center, and second element
            is a peak width.
        """
        data = self.Dataset
        counts = data[:,0]
        values = data[:,1]
        peaks = fit_peaks(values, counts)
        self.peaks = peaks
