import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def plot_histogram(bins, ys, threshold=None, xlabel="", gridspec=None):
    """Plot a histogram"""
    # Prepare plot grid
    gs = gridspec
    if gs is None:
        gs = gridspec.GridSpec(3,32)
    #plt.figure(figsize=(4,2))
    plot_dead = plt.subplot(gs[:, :2])
    plot_alive = plt.subplot(gs[:, 2:])

    # ----------------------------------------------------
    # Alive block
    # ----------------------------------------------------

    width = bins[1] - bins[0]
    #height = frac_dead / width
    plot_dead.bar(left=0, height=0, width=width, color="gray")
    plot_dead.set_xlabel("below\ndetection")
    plot_dead.spines["left"].set_visible(False)
    plot_dead.spines["right"].set_visible(False)
    plot_dead.spines["top"].set_visible(False)
    plot_dead.set_yticks([])
    plot_dead.set_xticks([])

    #plot_dead.set_ylim(-0.002,.1)
    #plot_dead.set_aspect("box-forced")

    # ----------------------------------------------------
    # Alive block
    # ----------------------------------------------------

    plot_alive.bar(left=bins, height=ys, width=width, color="gray")
    #plot_alive.set_xticks(range(20,180,20))
    plot_alive.spines["right"].set_visible(False)
    plot_alive.spines["top"].set_visible(False)
    plot_alive.spines["left"].set_visible(False)
    plot_alive.set_yticks([])
    #plot_alive.spines["bottom"].set_bounds(5,160)
    plot_alive.set_xlabel(xlabel)
    #plot_alive.axis([0,160,-0.002,0.1])

    # ----------------------------------------------------
    # Threshold?
    # ----------------------------------------------------

    if threshold != None:
        plot_alive.vlines(threshold, 0, 0.1, linestyles="dotted")

    return gs, plot_dead, plot_alive
