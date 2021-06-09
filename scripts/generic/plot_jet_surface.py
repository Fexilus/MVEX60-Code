"""Plot an example of a jet surface corresponding to a differential
equation.
"""
import os.path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


FIG_WIDTH = 9
FIG_HEIGHT = 6

def plot(save_path=None, file_name="jet-surface.pdf", plot_projection=True,
         plot_lifts=True, plot_surface=True):
    plt.rc("mathtext", fontset="cm")

    xlim = (-1, 1)
    ylim = (-8, 8)
    zlim = (-16, 8)

    x_vec = np.linspace(*xlim, 100)
    y_vec = np.linspace(*ylim, 100)

    #xs, ys = np.meshgrid(x_vec, y_vec)
    # Matplotlib can't render transparent surfaces correctly since it
    # lacks a 3d-rendering backend. This workaround works for a flat
    # surface
    xs, ys = np.meshgrid(xlim, ylim)

    zs = ys

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"),
                           figsize=(FIG_WIDTH, FIG_HEIGHT))

    if plot_surface:
        ax.plot_surface(xs, ys, zs, antialiased=False,
                        color=to_rgba("C0", 0.85), edgecolors=to_rgba("C0", 0),
                        linewidths=0.0)

    for c in np.linspace(-8 * np.exp(1), 8 * np.exp(1), 14)[1:-2]:
        line_xs = x_vec
        line_ys = c * np.exp(x_vec)
        line_zs = line_ys

        mask = (line_ys > ylim[0]) & (line_ys < ylim[1])

        line_xs = line_xs[mask]
        line_ys = line_ys[mask]
        line_zs = line_zs[mask]

        if plot_lifts:
            ax.plot(line_xs, line_ys, line_zs, color="black", zorder=4)
        if plot_projection:
            ax.plot(line_xs, line_ys, zlim[0], color="black")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel(r"$y'$")

    ax.view_init(elev=12, azim=-68)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if save_path:
        name_head, _, name_tail = file_name.partition(".")

        if plot_projection and plot_lifts and not plot_surface:
            name_head += "-no-surface"
        if plot_projection and not plot_lifts and plot_surface:
            name_head += "-no-lifts"
        if plot_projection and not plot_lifts and not plot_surface:
            name_head += "-only-projection"
        if not plot_projection and plot_lifts and plot_surface:
            name_head += "-no-projection"
        if not plot_projection and plot_lifts and not plot_surface:
            name_head += "-only-lifts"
        if not plot_projection and not plot_lifts and plot_surface:
            name_head += "-only-surface"
        if not plot_projection and not plot_lifts and not plot_surface:
            name_head += "-only-axis"

        file_name = name_head + "." + name_tail

        file_path = os.path.join(save_path, file_name)

        # Create a bounding box of the axis that eliminates the vertical
        # white space.
        bbox = ax.get_position()
        bbox.x1 += 0.04
        bbox.y0 = 0.10
        bbox.y1 = 0.85
        # Convert the bounding box to inches
        bbox.x0 *= FIG_WIDTH
        bbox.x1 *= FIG_WIDTH
        bbox.y0 *= FIG_HEIGHT
        bbox.y1 *= FIG_HEIGHT

        plt.savefig(file_path, format="pdf", bbox_inches=bbox)


if __name__ == "__main__":
    plot()
    plt.show()
