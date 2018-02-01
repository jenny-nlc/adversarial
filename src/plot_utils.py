import numpy as np
from matplotlib import pyplot as plt


def var_fill_plot(ax, x, ys, label="", c='blue', alpha =0.3):
    """
    Plot x against ys, showing the mean as a line variance of y's with a fill
    """
    med = np.median(ys, axis=1)
    ax.plot(x, med, c=c, label=label + '(median)', linestyle='--')

    mean = np.mean(ys, axis=1)
    ax.plot(x, med, c=c, label=label + '(mean)')


    #fifth and 95th percentile
    lo = np.percentile(ys, 5, axis=1)
    hi = np.percentile(ys, 95, axis=1)
    ax.fill_between(x, lo, hi, color = c, alpha = alpha)

    lo = np.percentile(ys, 25, axis=1)
    hi = np.percentile(ys, 75, axis=1)
    ax.fill_between(x, lo, hi, color = c, alpha = alpha)
    return ax
