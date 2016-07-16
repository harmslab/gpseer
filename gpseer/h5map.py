import h5py
import numpy as np

class Reference(object):
    """Class for constructing pythonic API interface to phenotype prediction
    h5py Dataset object stored in HDF5 file. Assumes that the dataset is two
    dimensions (with unbound size).

    Parameters
    ----------
    h5map : H5Map object
        Mapping object attached to a predictions dataset
    """
    def __init__(self, name, genotypes, h5map):
        self.h5map = h5map
        self.name = name
        #print(genotypes, genotypes.dtype)
        # Create the dataset on disk
        #self.h5map._f[self.name].create_dataset("genotypes", data=genotypes, dtype=np.string_)
        self.h5map._f[self.name].create_dataset("samples", shape=(0,len(genotypes)), maxshape=(None,None))

    def add(self, data):
        """Add a sample of data to dataset. Adds more rows.
        """
        olddata = self.samples
        oldshape = olddata.shape
        newshape = data.shape
        self.grow(nrow=data.shape[0])
        # Add the new data to old dataset
        olddata[oldshape[0]:oldshape[0]+newshape[0]] = data

    @property
    def genotypes(self):
        """Get the genotypes for this dataset."""
        return self.h5map._f(self.name)["genotypes"]

    @property
    def samples(self):
        """Get the data."""
        return self.h5map._f[self.name]["samples"]

    def grow(self, nrow=0, ncol=0):
        """Increase the size of the h5map by the given number of rows
        and columns.

        Parameters
        ----------
        nrow : int
            number of rows to shrink the dataset by.
        ncol : int
            number of columns to shrink the dataset by.
        """
        data = self.samples
        shape = data.shape
        data.resize((shape[0]+nrow, shape[1]+ncol))

    def shrink(self, nrow=0, ncol=0):
        """Shrink the size of the h5map by the given number of rows
        and columns.

        Parameters
        ----------
        nrow : int
            number of rows to shrink the dataset by.
        ncol : int
            number of columns to shrink the dataset by.
        """
        data = self.samples
        shape = data.shape
        data.resize((shape[0]-nrow, shape[1]-ncol))


class H5Map(object):
    """Thin layer class to make h5py datasets more readable as API.

    Parameters
    ----------
    path : string
        path name to save prediction data.
    """
    def __init__(self, path, predictor):
        self.path = path
        self.predictor = predictor
        self.references = []
        self._f = h5py.File(self.path, "w")

    def read(method):
        """Wrapper function for methods that need to read from file
        Opens file for reading and closes after.
        """
        def inner(self, *args, **kwargs):
            # Try to open file, unless already open
            try:
                self._f = h5py.File(self.path, "r")
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
                self._f = h5py.File(self.path, "a")
            except OSError:
                pass
            output = method(self, *args, **kwargs)
            return output
        return inner

    @read
    def get(self, reference):
        """Get the complete dataset, named by reference.
        """
        return getattr(self, "ref"+reference)

    @write
    def add(self, reference, model):
        """Add a reference prediction to file.
        """
        # Expose API for that dataset
        name = "ref" + reference
        self._f.create_group(name)
        new_ref = Reference(name, model, self)
        setattr(self, name, new_ref)
        self.references.append(name)
