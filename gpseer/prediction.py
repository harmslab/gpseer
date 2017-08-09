import h5py
import pickle
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt

class Prediction(object):
    """Object for analyzing a predictions HDF5 database.

    Parameters
    ----------
    path : str
        Path to HDF5 file with samples.

    Attributes
    ----------
    h : np.array
        A histogram of the predictions from the HDF5 file (stored by `histogram` method.)
    p : np.array
        percentiles stored by the `percentiles` method.
    """
    def __init__(self, path, chunks=1000):
        self.path = path
        # Create a Dask array from a genotype.
        f = h5py.File(path, "r")
        ds = f["/likelihood"]
        self.samples = da.from_array(ds, chunks=chunks)
        self.p = None
        self.h = None

    def peak(self):
        """Get the peak in the histogrammed data."""
        if self.h is None:
            raise Exception("Call `histogram` method before estimating the peak.")
        max_ = np.argmax(self.h)
        peak = self.bins[max_]
        return peak

    def percentile(self, percentiles):
        """Computes percentiles within the samples."""
        out = da.percentile(self.samples, percentiles)
        self.p = out.compute()
        return self.p

    def histogram(self, bins="auto", range=None):
        """Compute a histogram of the samples."""
        # Determine range max/min
        if range is None:
            min_ = da.min(self.samples)
            min_ = min_.compute()
            max_ = da.max(self.samples)
            max_ = max_.compute()
        else:
            min_ = range[0]
            max_ = range[1]

        # If no bins are given, use the Freedman Diaconis Estimator
        if bins == "auto":
            data = self.samples
            n = data.size
            machine =  da.percentile(data, (25,75))
            ends = machine.compute()
            IQR = ends[1] - ends[0]
            binsize = 2 * (IQR) / (n**(1/3))
            bins = np.arange(min_, max_, binsize)

        # Make sure bins are present
        if min_ == 0 and max_ == 0:
            self.h = np.array([len(self.samples)])
            self.bins = np.array([0])

        elif len(bins) == 0 :
            self.bins = np.array([0])
            self.h = np.array([0])

        else:
            # Calculate
            h, self.bins = da.histogram(self.samples, bins=bins, range=range)
            self.h = h.compute()
        return self.h, self.bins

    def snapshot(self):
        """Create a Snapshot of the prediction."""
        return Snapshot(self.h, self.bins, self.p)

    def plot(self):
        """Plot histogram."""
        fig, ax = plt.subplots()
        width = self.bins[1] - self.bins[0]
        ax.bar(self.bins[:-1], self.h, width=width)

class Snapshot(object):
    """A lightweight snapshot of the Predictions object.

    Parameters
    ----------
    path : str
        Path to HDF5 file with samples.

    Attributes
    ----------
    h : np.array
        A histogram of the predictions from the HDF5 file (stored by `histogram` method.)
    bins : np.array
        Bins for histogram of the predictions from the HDF5 file (stored by `histogram` method.)
    p : np.array
        percentiles stored by the `percentiles` method.
    """
    def __init__(self, h, bins, p=None):
        self.h = h
        self.bins = bins
        self.p = p

    def peak(self):
        """Get the peak in the histogrammed data."""
        if hasattr(self, "h") is False:
            raise Exception("Call `histogram` method before estimating the peak.")
        max_ = np.argmax(self.h)
        peak = self.bins[max_]
        return peak

    @classmethod
    def load(cls, fname):
        """Load a snapshot from file."""
        with open(fname, "rb") as f:
            data = pickle.load(f)
        h = data["h"]
        bins = data["bins"]
        p = data["p"]
        self = cls(h, bins, p)
        return self

    def pickle(self, fname):
        """Pickle Snapshot to file."""
        data = {"h":self.h, "p":self.p, "bins": self.bins}
        with open(fname, "wb") as f:
            pickle.dump(data, f)

    def plot(self):
        """Plot histogram."""
        fig, ax = plt.subplots()
        width = self.bins[1] - self.bins[0]
        ax.bar(self.bins[:-1], self.h, width=width)
