"""Plot an example of ODE solutions to a differential equation"""
import os.path

import numpy as np

import matplotlib.pyplot as plt

from symmetries.visualize.arrowpath import WithArrowStroke


FIG_WIDTH = 9
FIG_HEIGHT = 6

def plot(save_path=None, file_names=["rotation-field.eps",
                                     "rotation-representative.eps"]):
    plt.rc("mathtext", fontset="cm")

    xlim = (-1, 1)
    ylim = (-1, 1)

    x_vec = np.linspace(*xlim, 10)
    y_vec = np.linspace(*ylim, 10)

    xs, ys = np.meshgrid(x_vec, y_vec)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.quiver(xs, ys, -ys, xs)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_aspect((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    if save_path:
        file_path = os.path.join(save_path, file_names[0])

        plt.savefig(file_path, format="eps", bbox_inches="tight")

    inte_start = (0.3, -0.2)
    angles = np.linspace(0, 2, 30)

    inte_xs = inte_start[0] * np.cos(angles) - inte_start[1] * np.sin(angles)
    inte_ys = inte_start[0] * np.sin(angles) + inte_start[1] * np.cos(angles)

    ax.scatter(*inte_start, marker="o", color="C1", zorder=3)
    ax.scatter(inte_xs[-1], inte_ys[-1], marker="o", color="C1", zorder=3)
    ax.plot(inte_xs, inte_ys, lw=3,
            path_effects=[WithArrowStroke(spacing=20.0,scaling=10.0)])

    if save_path:
        file_path = os.path.join(save_path, file_names[1])

        plt.savefig(file_path, format="eps", bbox_inches="tight")


if __name__ == "__main__":
    plot()
    plt.show()
