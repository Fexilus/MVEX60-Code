"""Plot an example of ODE solutions to a differential equation"""
import os.path

import numpy as np

import matplotlib.pyplot as plt


FIG_WIDTH = 9
FIG_HEIGHT = 6

def plot(save_path=None, file_name="generic-solutions.eps"):
    plt.rc("mathtext", fontset="cm")

    xlim = (-1, 1)
    ylim = (-8, 8)

    x_vec = np.linspace(*xlim, 100)
    y_vec = np.linspace(*ylim, 100)

    #xs, ys = np.meshgrid(x_vec, y_vec)
    # Matplotlib can't render transparent surfaces correctly since it
    # lacks a 3d-rendering backend. This workaround works for a flat
    # surface
    xs, ys = np.meshgrid(xlim, ylim)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for c in np.linspace(-8 * np.exp(1), 8 * np.exp(1), 14)[1:-2]:
        line_xs = x_vec
        line_ys = c * np.exp(x_vec)

        mask = (line_ys > ylim[0]) & (line_ys < ylim[1])

        line_xs = line_xs[mask]
        line_ys = line_ys[mask]

        ax.plot(line_xs, line_ys, color="black", zorder=4)
        ax.plot(line_xs, line_ys, color="black")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_aspect((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    if save_path:
        file_path = os.path.join(save_path, file_name)

        plt.savefig(file_path, format="eps", bbox_inches="tight")


if __name__ == "__main__":
    plot()
    plt.show()
