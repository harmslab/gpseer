import h5py

class Reference(object):
    """Class for constructing pythonic API interface to phenotype prediction
    h5py Dataset object stored in HDF5 file.

    Parameters
    ----------
    h5map : H5Map object
        Mapping object attached to a predictions dataset
    """
    def __init__(self, name, h5map):
        self.h5map
        self.name

    @property
    def data(self):
        """Get the data"""
        return self.h5map.get(name)

class H5Map(self):
    """Class that maps predictor objects to HDF5 tables stored to disk.

    Parameters
    ----------
    path : string
        path name to save prediction data.
    predictor : gpseer.predictor.Predictor object
        predictor object that stores data.
    """
    def __init__(self, path)#, predictor):
        self.path = path
        #self.predictor = predictor
        #self._gpm = self.predictor.gpm
        self._f = h5py.File(self.path, "w")

    def read(self, method, *args, **kwarg):
        """Wrapper function for methods that need to read from file
        Opens file for reading and closes after.
        """
        self._f = h5py.File(self.path, "r")
        def inner(self, method):
            output = method(self, *args, **kwargs)
            return output
        return inner

    def write(self, function):
        """Wrapper function for methods that need to write to file.
        Opens file for writing and closes after finished.
        """
        self._f = h5py.File(self.path, "w")
        def inner(self, method):
            output = method(self, *args, **kwargs)
            return output
        return inner

    @self.read
    def get(self, reference):
        """Get the complete dataset, named by reference.
        """
        return dataset = self._f[reference]

    @self.write
    def add(self, reference):
        """Add a reference prediction to file.
        """
        self._f.create_dataset(reference, maxshape=(None,None))
        new_ref = Reference(reference, self)
        setattr(self, reference, new_ref)
