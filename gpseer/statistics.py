import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt

def lorentz(x, center, width):
    """A Lorentz distribution for fitting peaks.

    Parameters
    ----------
    x : numpy.array
        x values for distribution
    center : float
        the center of the distribution
    width : float
        full width at half maximum.

    Returns
    -------
    distribution : numpy array
        normalized gaussian distribution
    """
    distribution = (1 / np.pi) * (0.5 * width) / ((x - center)**2 + (0.5*width)**2)
    return distribution

def gaussian(x, center, width):
    """A Gaussian distribution for fitting peaks.

    Parameters
    ----------
    x : numpy.array
        x values for distribution
    center : float
        the center of the distribution
    width : float
        full width at half maximum.

    Returns
    -------
    distribution : numpy array
        normalized lorentzian distribution
    """
    distribution = (1 / np.sqrt(2 * np.pi * width**2)) * np.exp(-(x - center)**2/ (2*width**2))
    return distribution

def fit_peaks(xdata, ydata, function=lorentz, widths=np.arange(1,100)):
    """Find peaks in a dataset using continuous wave transform and fit with
    distribution function.

    Parameters
    ----------
    data : array
        1-D data to search for peaks
    widths :
        1-D array of widths to use for calculating the CWT matrix. In general,
        this range should cover the expected width of peaks of interest.
    """
    # Find peaks using
    indices = find_peaks_cwt(ydata, widths=widths)

    # Fit the peaks that were found with function.
    centers = xdata[indices]
    widths = np.ones(len(centers))
    peaks = []
    npeaks = len(indices)
    for i in range(npeaks):
        popt, pcov = curve_fit(function, xdata, ydata, p0=(centers[i], widths[i]))
        center = popt[0]
        width = popt[1]
        peaks.append((center, width))
    return peaks
