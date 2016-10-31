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

def multigaussian(x, *flattened_peak_data):
    """Returns a 1d array with multiple gaussian peaks c

    Parameters
    ----------
    x : numpy.array
        x values for distribution
    flattened_peak_data : (optional arguments)
        3-tuples with (center, amplitude, width) of each peak.
    """
    # Check to see if the data is flattened.
    try:
        if hasattr(flattened_peak_data[0], "__init__"):
            


    npeaks = len(peaks)
    distribution = np.zeros(len(x))
    for i in range(0, peaks):
        if len(peaks[i]) != 3:
            raise Exception("Each peak must have three arguments: (center, height, width)")
        distribution += gaussian(x, *peaks[i])
    return distribution

def fit_peaks(xdata, ydata, widths=np.arange(1,100)):
    """Detect, fit, and return parameters of multiple gaussian peaks in a 1d-array using continuous
    wave transform.

    Uses scipy's `find_peaks_cwt` function to find peaks in the array first. Then,
    uses scipy's `curve_fit` function to fit the peaks and estimate their amplitudes
    and widths.

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
    peaks : array
        list of parameters
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
                p0[3*i] = ydata[indices[i]]
                p0[3*i+1] = xdata[indices[i]]
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
        raise RuntimeError("""Optimal parameters not found.""")
    # Return parameters
    peaks = []
    for i in range(0,len(popt),3):
        peaks.append((popt[i], popt[i+1], popt[i+2]))
    return peaks
