import h5py
import numpy as np

class ReferenceDatasets(object):
    """Class for constructing pythonic API interface to phenotype prediction
    h5py Dataset object stored in HDF5 file. Assumes that the dataset is two
    dimensions (with unbound size).

    Parameters
    ----------
    name : str
        name of this object stored in parent object (i.e. 'REF0000')
    genotypes : array
        array of genotypes in the same order that samples will be added. Used
        to map genotypes to their predictions. NOT stored in memory, but a
        separate HDF5 dataset.
    predictions : PredictionsFile object
        Mapping object attached to a HDF5 File.
    """
    def __init__(self, name, genotypes, predictions):
        self.name = name
        self.predictions = predictions
        ###### Create the datasets on disk ######
        # Write genotypes to disk -- must be genotype
        self.predictions.File[self.name].create_dataset("genotypes",
            data=genotypes.astype("S" + str(len(genotypes[0]))))
        # Create a predictions datasets
        self.predictions.File[self.name].create_dataset("samples",
            shape=(0,len(genotypes)), maxshape=(None,len(genotypes)))

    def add_samples(self, data):
        """Add a sample of data to dataset. Adds more rows.

        Parameters
        ----------
        data : numpy array
            prediction samples to add to data array.
        """
        olddata = self.samples_
        oldshape = olddata.shape
        newshape = data.shape
        self._grow(nrow=data.shape[0])
        # Add the new data to old dataset
        olddata[oldshape[0]:oldshape[0]+newshape[0]] = data

    @property
    def genotypes_(self):
        """Get genotypes h5py dataset object."""
        return self.predictions.File[self.name]["genotypes"]

    @property
    def genotypes(self):
        """Get the genotypes for this dataset."""
        return self.genotypes_.value.astype(str)

    @property
    def samples_(self):
        """Get the data."""
        return self.predictions.File[self.name]["samples"]

    @property
    def samples(self):
        """Get sample values."""
        return self.samples_.value

    def _grow(self, nrow=0, ncol=0):
        """Increase the size of the h5map by the given number of rows
        and columns.

        Parameters
        ----------
        nrow : int
            number of rows to shrink the dataset by.
        ncol : int
            number of columns to shrink the dataset by.
        """
        data = self.samples_
        shape = data.shape
        data.resize((shape[0]+nrow, shape[1]+ncol))

    def _shrink(self, nrow=0, ncol=0):
        """Shrink the size of the h5map by the given number of rows
        and columns.

        Parameters
        ----------
        nrow : int
            number of rows to shrink the dataset by.
        ncol : int
            number of columns to shrink the dataset by.
        """
        data = self.samples_
        shape = data.shape
        data.resize((shape[0]-nrow, shape[1]-ncol))


class PredictionsFile(object):
    """An object to read and write to an HDF5 file via h5py's API. This is essentially
    a thin layer on top of h5py.

    Parameters
    ----------
    path : string
        path name to save prediction data.
    """
    def __init__(self, path, predictor):
        self.path = path
        self.predictor = predictor
        self.references = []
        # Initialize an HDF5 file.
        self.File = h5py.File(self.path, "w")

    def read(method):
        """Wrapper function for methods that need to read from file
        Opens file for reading and closes after.
        """
        def inner(self, *args, **kwargs):
            # Try to open file, unless already open
            try:
                self.File = h5py.File(self.path, "r")
            except OSError:
                pass
            output = method(self, *args, **kwargs)
            return output
        return inner

    def write(method):
        """Wrapper function for methods that need to write to file.
        Opens file for writing and closes after finished.
        """
        def inner(self, *args, **kwargs):
            # Try to open file, unless already open
            try:
                self.File = h5py.File(self.path, "a")
            except OSError:
                pass
            output = method(self, *args, **kwargs)
            return output
        return inner

    @read
    def get(self, reference):
        """Get the complete dataset, named by reference.
        """
        return getattr(self, "r"+reference)

    @write
    def add(self, reference, genotypes):
        """Add a new HDF5 group to the File. This group will contain two
        datasets, 1. genotypes and 2. samples.
        """
        # Expose API for that dataset
        name = "r" + reference
        self.File.create_group(name)
        new_ref = ReferenceDatasets(name, genotypes, self)

        # Attach to this object.
        setattr(self, name, new_ref)
        self.references.append(name)
