import numpy as np
from matplotlib import pyplot as plt


def var_fill_plot(ax, x, ys, label="", c='blue', alpha =0.4):
    """
    Plot x against ys, showing the mean as a line variance of y's with a fill
    """
    mean = ys.mean(axis=1)
    ax.plot(x, mean, c=c, label=label)
    std = ys.std(axis=1)
    ax.fill_between(x, mean - std, mean + std, color = c, alpha = alpha)
    return ax
