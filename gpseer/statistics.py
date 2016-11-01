import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt

def pearson(y_obs, y_pred):
    """Calculate pearson coefficient between two variables.
    """
    x = y_obs
    y = y_pred

    xbar = np.mean(y_obs)
    ybar = np.mean(y_pred)

    terms = np.zeros(len(x), dtype=float)

    for i in range(len(x)):
        terms[i] = (x[i] - xbar) * (y[i] - ybar)

    numerator = sum(terms)

    # calculate denominator
    xdenom = sum((x - xbar)**2)
    ydenom = sum((y - ybar)**2)
    denominator = np.sqrt(xdenom)*np.sqrt(ydenom)

    return numerator/denominator

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

def gaussian(x, center, amp, width):
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
    distribution = amp * np.exp(-(x - center)**2/ (2*width**2))
    return distribution

def multigaussian(x, *args):
    """Construct the sum of multiple distributions.
    Parameters
    ----------
    x : numpy.array
        x values for distribution
    args :
        first half must be the center of peaks, second half are widths.
    """
    args = np.array(args)
    if len(args)%3 != 0:
        raise Exception("""Number of args must be divisible by 3.""")
    npeaks = int(len(args) / 3)
    distribution = np.zeros(len(x))
    for i in range(0, len(args), 3):
        center = args[i]
        amp = args[i+1]
        width = args[i+2]
        distribution += gaussian(x, center, amp, width)
    return distribution

def fit_peaks(xdata, ydata, widths=np.arange(1,100)):
    """Find peaks in a dataset using continuous wave transform and fit with
    distribution function.
    Parameters
    ----------
    xdata : array
        1-D data to search for peaks
    ydata : array
        1-D data, height of peaks
    widths :
        1-D array of widths to use for calculating the CWT matrix. In general,
        this range should cover the expected width of peaks of interest.
    Returns
    -------
    peaks : 2d-array
        Each row is the [center, amplitude, and width] of a single gaussian peak.
    """
    # Find peaks using continuous wave tranform
    indices = find_peaks_cwt(ydata, widths=widths)
    npeaks = len(indices)
    # Attempt to fit npeaks with model
    attempts = 0
    score = 0
    finished = False
    while attempts < 5 and finished is False and score < 0.9:
        finished = False
        # Attempt guesses that are multiple orders of magnitude
        try:
            # Construct parameters for model with guesses
            p0 = np.ones(npeaks*3) * 0.001 * 10**attempts
            for i in range(npeaks):
                p0[3*i] = xdata[indices[i]]
                p0[3*i+1] = ydata[indices[i]]
            # Fit the data with multiple peaks
            popt, pcov = curve_fit(multigaussian, xdata, ydata, p0=p0)
            score = pearson(ydata, multigaussian(xdata, *popt))
            if score < 0.9:
                raise RuntimeError
            finished = True
        except RuntimeError:
            attempts += 1
    # If the last loop didn't finish, raise an error
    if finished is False:
        return []
    else:
        # Return parameters
        peaks = np.empty((npeaks, 3), dtype=float)
        for i in range(0, npeaks):
            peaks[i, :] = np.array([popt[3*i], popt[3*i+1], popt[3*i+2]])
        return peaks
