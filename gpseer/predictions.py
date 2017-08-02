import os
import pickle

import h5py
import dask.array as da
import pandas as pd

class GPPredictions(object):
    """
    """
    def __init__(self, db_dir):
        self._db_dir = db_dir

        # Read model
        with open(os.path.join(self._db_dir, "model.pickle"), "rb") as f:
            self.model = pickle.load(f)
        # Read model
        with open(os.path.join(self._db_dir, "gpm.pickle"), "rb") as f:
            self.gpm = pickle.load(f)

    def _daskify_likelihood(self, genotype, chunks=(1000,1000)):
        """Read a h5py dataset as a dask array"""
        path = os.path.join(self._db_dir, "likelihoods", genotype, "likelihoods.hdf5")
        f = h5py.File(path)
        ds = f["/likelihoods"]
        arr = da.from_array(ds, chunks=chunks)
        return arr

    def link_likelihoods(self):
        """Link a dask array to all genotypes."""
        genotypes = self.gpm.complete_genotypes
        self.likelihoods = {g: self._daskify_likelihood(g) for g in genotypes}

    def calc_histograms(self, bins, range):
        """Return histogram for all datasets."""
        self.histograms = {}
        for genotype, data in self.likelihoods.items():
            h, bins_ = da.histogram(data, bins=bins, range=range)
            self.histograms[genotype] = pd.Series(h.compute(), index=bins_[:-1])
        return self.histograms

    def calc_stats(self):
        """Calculate statistics."""
        cols = ["genotype", "peak", "lbound", "rbound"]
        data = {c: [] for c in cols}
        for genotype in self.gpm.complete_genotypes:
            # Get peak
            hist = self.histograms[genotype]
            peak = hist.idxmax()
            # Get bounds
            likelihood = self.likelihoods[genotype]
            bounds = da.percentile(likelihood.flatten(), (2.5,97.5)).compute()
            # Build dataframe.
            data["peak"].append(peak)
            data["lbound"].append(bounds[0])
            data["rbound"].append(bounds[1])
            data["genotype"].append(genotype)
        # To dataframe
        self.df = pd.DataFrame(data, columns=cols)
        return self.df

    def plot_histograms(self, ncols=3):
        """Plot histograms."""
        from .plot import plot_histogram
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        genotypes = self.gpm.complete_genotypes
        nplots = len(genotypes)
        nrows = int(nplots/ncols)
        if nplots%ncols != 0 :
            nrows += 1
        figsize = (ncols*3, nrows * 2)

        fig = plt.figure(figsize=figsize)

        # Main gridspec
        gs = gridspec.GridSpec(nrows, ncols)
        gs.update(hspace=0.5)

        for i in range(nrows):
            for j in range(ncols):
                # Get genotype to plot
                k = i + j
                genotype = genotypes[k]

                hist = self.histograms[genotype]
                gs_sub = gridspec.GridSpecFromSubplotSpec(3,32, subplot_spec=gs[i, j])
                plot_histogram(hist.index, hist, gridspec=gs_sub)

        return fig

    def snapshot(self):
        """"""
        return GPSnapshot(self.histogram, self.df)


class GPSnapshot(object):
    """
    """
    def __init__(self, histograms, df):
        self.histograms = histograms
        self.df = df

    def plot_histograms(self, genotypes=None, ncols=3):
        """Plot histograms."""
        # Imports for plotting
        from .plot import plot_histogram
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        genotypes = self.gpm.complete_genotypes
        nplots = len(genotypes)
        nrows = int(nplots/ncols)
        if nplots%ncols != 0 :
            nrows += 1
        figsize = (ncols*3, nrows * 2)

        plt.figure(figsize=figsize)

        # Main gridspec
        gs = gridspec.GridSpec(nrows, ncols)

        for i in range(nrows):
            for j in range(ncols):
                # Get genotype to plot
                k = i + j
                genotype = genotypes[k]

                hist = self.histograms[genotype]
                gs_sub = gridspec.GridSpecFromSubplotSpec(3,32, subplot_spec=gs[i, j])
                plot_histogram(hist.index, hist, gridspec=gs_sub)
