import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from epistasis.models import (
    EpistasisPipeline,
    EpistasisLogisticRegression,
    EpistasisSpline,
    EpistasisLinearRegression
)

from epistasis.pyplot.nonlinear import plot_scale

from collections import OrderedDict

def plot_spline(model,prediction_df):
    """
    Plot the results of the spline fit portion of the epistasis model.

    model: epistasis pipeline model (fit)
    prediction_df: data frame containing predicted and measured values
    """

    # Find the spline model in this epistasis pipeline
    spline_model = None
    for m in model:
        if isinstance(m,EpistasisSpline):
            spline_model = m
            break

    if spline_model is None:
        err = "plot_spline can only be used for datasets using EpistasisSpline models\n"
        raise ValueError(err)

    # Grab data frame containing only values with measurements
    df = prediction_df[np.logical_not(prediction_df.measured.isnull())]

    # Get minimum and maximum values from the data sets
    min_value = np.min(df[["prediction","measured"]].min())
    max_value = np.max(df[["prediction","measured"]].max())
    five_pct = (max_value - min_value)*0.05

    # Grab any genotypes classified as dead
    has_threshold = False
    try:
        below = df[df.phenotype_class == "below"]
        if len(below) > 0:
            has_threshold = True
    except AttributeError:
        pass

    # Actually calculate the spline bits
    yobs = spline_model.gpm.phenotypes
    yadd = spline_model.Additive.predict()
    spline_xx = np.linspace(min(yadd), max(yadd),100)
    spline_yy = spline_model.minimizer.predict(spline_xx)

    missing_err_mask = np.array([s is None for s in spline_model.gpm.stdeviations],
                                 dtype=np.bool)
    has_err_mask = np.logical_not(missing_err_mask)
    yerr = spline_model.gpm.stdeviations[has_err_mask]

    fig, ax = plt.subplots()

    # Plot spline
    ax.errorbar(x=yadd[missing_err_mask],y=yobs[missing_err_mask],
                fmt="o",color="black",label="above")
    ax.errorbar(x=yadd[has_err_mask],y=yobs[has_err_mask],yerr=yerr,
                fmt="o",color="black",label="above")

    ax.plot(spline_xx,spline_yy,"-",color="red",linewidth=2)

    # Plot any classified-as-below
    if has_threshold:
        ax.errorbar(x=below.prediction,
                    y=below.measured,
                    #yerr=below.measured_err,
                    fmt="o",
                    color="gray",
                    label="below")

        # Create legend without duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels,handles))
        ax.legend(by_label.values(),by_label.keys(),
                  title="threshold category",loc="upper left",frameon=False)

    ax.set_xlabel("model")
    ax.set_ylabel("observed")
    ax.set_xlim((min_value - 3*five_pct ,max_value + 3*five_pct))
    ax.set_ylim((min_value - 3*five_pct ,max_value + 3*five_pct))

    ax.set_aspect('equal', 'box')

    if has_threshold:
        ax.set_title("spline and threshold fit to data")
        ax.text(max_value - 12*five_pct,min_value + 3*five_pct,"classifier threshold: {}".format(model[0].threshold) )
    else:
        ax.set_title("spline fit to data")

    ax.text(max_value - 12*five_pct,min_value + 2*five_pct,"order: {}".format(spline_model.k) )
    ax.text(max_value - 12*five_pct,min_value + 1*five_pct,"smoothness: {}".format(spline_model.s))


    plt.tight_layout()

    return fig, ax

def plot_correlation(model,prediction_df):
    """
    Plot the correlation between the predicted and measured phenotypes.

    model: epistasis pipeline model (fit)
    prediction_df: data frame containing predicted and measured values
    """

    # Grab data frame containing only values with measurements
    df = prediction_df[np.logical_not(prediction_df.measured.isnull())]

    # Get minimum and maximum values from the data sets
    min_value = np.min(df[["prediction","measured"]].min())
    max_value = np.max(df[["prediction","measured"]].max())
    five_pct = (max_value - min_value)*0.05

    fig = plt.figure(figsize=(5,7))

    heights = [4,1]
    spec = fig.add_gridspec(ncols=1,nrows=2,height_ratios=heights)

    ax = [fig.add_subplot(spec[0])]
    ax.append(fig.add_subplot(spec[1],sharex=ax[0]))
    plt.setp(ax[0].get_xticklabels(), visible=False)

    # Plot a 1-to-1 line
    ax[0].plot(np.array([min_value,max_value]),
               np.array([min_value,max_value]),
               color="gray",linestyle="dashed")

    # Plot a horizontal line
    ax[1].plot((min_value,max_value),(0,0),"--",color="gray")

    # Look for two phenotype classes.  Each class will be plotted as its own
    # color
    try:
        classes = set(df.phenotype_class)
        if len(classes) > 2:
            err = "this function assumes only two categories\n"
            raise ValueError(err)

        has_threshold = True
        dfs = []
        labels = ["below","above"]
        colors = ["gray","black"]
        for c in labels:
            dfs.append(df[df.phenotype_class == c])

    except AttributeError:
        has_threshold = False
        labels = [None]
        dfs = [df]
        colors = ["black"]

    for i in range(len(dfs)):

        # look for missing
        missing_meas_err_mask = dfs[i].measured_err.isnull()

        # Plot values with missing measured error bars
        df_no_meas_err = dfs[i][missing_meas_err_mask]
        if len(df_no_meas_err) > 0:

            # main plot
            ax[0].errorbar(x=df_no_meas_err.measured,
                           y=df_no_meas_err.prediction,
                           yerr=df_no_meas_err.prediction_err,
                           fmt="o",color=colors[i],
                           label=labels[i])

            # residual plot
            residual = df_no_meas_err.prediction - df_no_meas_err.measured

            ax[1].errorbar(x=df_no_meas_err.measured,
                           y=residual,
                           fmt="o",color=colors[i],
                           label=labels[i])


        # Plot values with measured error bars
        df_meas_err = dfs[i][np.logical_not(missing_meas_err_mask)]
        if len(df_meas_err) > 0:

            # Main plot
            ax[0].errorbar(x=df_meas_err.measured,
                           y=df_meas_err.prediction,
                           xerr=df_meas_err.measured_err,
                           yerr=df_meas_err.prediction_err,
                           fmt="o",color=colors[i],
                           label=labels[i])

            # residual plot
            residual = df_meas_err.prediction - df_meas_err.measured

            ax[1].errorbar(x=df_meas_err.measured,
                           y=residual,
                           xerr=df_meas_err.measured_err,
                           fmt="o",color=colors[i],
                           label=labels[i])


    # Grab R2
    predicted_err = (1 - model.score()) * np.mean(model.gpm.phenotypes)

    ax[0].set_ylabel("predicted values")
    ax[0].set_aspect('equal', 'box')
    ax[0].set_title("prediction quality for training genotypes")
    ax[0].text(max_value - 12*five_pct,min_value + 2*five_pct,"R^2 (train): {:.2f}".format(model.score()) )
    ax[0].text(max_value - 12*five_pct,min_value + 1*five_pct,"phenotype uncertainty: +/- {:.2f}".format(predicted_err) )

    if has_threshold:
        ax[0].legend(title="threshold category",loc="upper left",frameon=False)

    ax[1].set_ylabel("pred - meas")
    ax[1].set_xlabel("measured values")

    fig.tight_layout()

    return fig, ax

def plot_histograms(model,prediction_df):
    """
    Plot histograms of measured values, training set predictions, and test set
    predictions.

    model: epistasis pipeline model (fit)
    prediction_df: data frame containing predicted and measured values
    """

    # Construct bins for histograms
    min_value = np.min(prediction_df[["prediction","measured"]].min())
    max_value = np.max(prediction_df[["prediction","measured"]].max())
    num_values = round(np.sqrt(len(prediction_df["prediction"])),0).astype(int)
    bins = np.linspace(min_value*0.99,max_value*1.01,num_values)

    # Slice test and training sets out
    meas = prediction_df[np.logical_not(prediction_df.measured.isnull())]
    pred = prediction_df[prediction_df.measured.isnull()]

    # Generate histogram vectors
    meas_counts, _ = np.histogram(meas.measured,bins=bins)
    meas_pred_counts, _ = np.histogram(meas.prediction,bins=bins)
    pred_counts, _ = np.histogram(pred.prediction,bins=bins)

    # Create plots
    fig, ax = plt.subplots(3,1,figsize=(5,9),sharex=True)

    x_values = (bins[:-1] + bins[1:])/2
    width = bins[1] - bins[0]

    ax[0].bar(x_values,meas_counts,width=width,color="gray",edgecolor="black",linewidth=1)
    ax[1].bar(x_values,meas_pred_counts,width=width,color="gray",edgecolor="black",linewidth=1)
    ax[2].bar(x_values,pred_counts,width=width,color="gray",edgecolor="black",linewidth=1)

    ax[2].set_xlabel("value")
    ax[0].set_ylabel("counts")
    ax[1].set_ylabel("counts")
    ax[2].set_ylabel("counts")
    ax[0].set_title("histogram of measured values")
    ax[1].set_title("histogram of training set predictions")
    ax[2].set_title("histogram of test set predictions")

    return fig, ax




def plots_to_pdf(model,prediction_df,out_root):
    """
    Plot a collection of summary graphs for a prediction, writing them to pdf.

    model: EpistasisPipline object containing completed fit
    prediction_df: prediction_to_dataframe output, containing finalized dataframe
                  with predictions
    out_root: root name for all output pdfs
    """

    # First, see if we need to plot a spline
    for m in model:

        # If we see a spline...
        if isinstance(m,EpistasisSpline):
            plt, ax = plot_spline(model,prediction_df)
            plt.savefig("{}_spline-fit.pdf".format(out_root))
            break

    # Plot correlation between predicted and observed values for training set
    fig, ax = plot_correlation(model,prediction_df)
    fig.savefig("{}_correlation-plot.pdf".format(out_root))

    # Plot histograms of values for measured values, training set predictions,
    # and test set predictions
    fig, ax = plot_histograms(model,prediction_df)
    fig.savefig("{}_histograms.pdf".format(out_root))
