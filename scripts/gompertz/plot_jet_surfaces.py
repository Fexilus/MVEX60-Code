"""Plot transformations on the jet surfaces corresponding to the
Gompertz models.

Used for the cover of the thesis
"""
import os.path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.interpolate import interp1d
from sympy import symbols

from symmetries import generator_on
from symmetries.visualize import plot_transformation_curves


FIG_WIDTH = 9
FIG_HEIGHT = 6


def plot(save_path=None, file_names=["classical-gompertz-surface.pdf",
                                     "autonomous-gompertz-surface.pdf"],
         plot_projection=False, plot_lifts=True, plot_surface=True):
    r"""Plot the jet surface of the classical and autonomous Gompertz
    models on the front page of the thesis.

    Args:
        save_path: Optional path to save the figures to.

        file_names: List of the names of the file name to save to if a
            path is given.

        plot_projection: Boolean value to control whether the projection
            of solution and transformation lines are plotted. (Not
            implemented)

        plot_lifts: Boolean value to control whether the lifts of
            solution and transformation lines are plotted.

        plot_surface: Boolean value to control whether the ODE surface
            if plotted.
    """
    plt.rc("mathtext", fontset="cm")

    tlim = (-2, 10)
    Wlim = (0.001, 4.2)
    dWdtlim = (-4, 2)

    t_vec = np.linspace(*tlim, 50)
    W_vec = np.linspace(*Wlim, 50)

    default_params = {"A": 3, "Ti": np.log(np.log(3)), "kG": 1}

    plot_classical_surface(default_params, t_vec, W_vec, tlim, Wlim, dWdtlim,
                           save_path=save_path, file_name=file_names[0],
                           plot_projection=plot_projection,
                           plot_lifts=plot_lifts,
                           plot_surface=plot_surface)

    plot_autonomous_surface(default_params, t_vec, W_vec, tlim, Wlim, dWdtlim,
                            save_path=save_path, file_name=file_names[1],
                            plot_projection=plot_projection,
                            plot_lifts=plot_lifts,
                            plot_surface=plot_surface)


def plot_classical_surface(default_params, t_vec, W_vec, tlim, Wlim, dWdtlim,
                           save_path=None,
                           file_name="classical-gompertz-surface.pdf",
                           plot_projection=False, plot_lifts=True,
                           plot_surface=True):
    r"""Plot the jet surface of the autonomous Gompertz model.

    Args:
        default_params: A dict containing the default parameters of the
            \(T_i\)-parametrized Gompertz model. The parameters are
            \lstinline{"A"}, \lstinline{"Ti"} and \lstinline{"kG"}.

        t_vec: numpy array of time values to evaluate at.

        W_vec: numpy array of size values to evaluate at.

        tlim: Two-tupple of limits for the time axis.

        Wlim: Two-tupple of limits for the size axis.

        dWdtlim: Two-tupple of limits for the size derivative axis.

        save_path: Optional path to save the figure to.

        file_name: Name of the file name to save to if a path is given.

        plot_projection: Boolean value to control whether the projection
            of solution and transformation lines are plotted. (Not
            implemented)

        plot_lifts: Boolean value to control whether the lifts of
            solution and transformation lines are plotted.

        plot_surface: Boolean value to control whether the ODE surface
            if plotted.
    """
    # Set up symbolic parameters
    t, W = symbols("t W")
    Generator = generator_on(((t,), (W,)))

    kG = default_params["kG"]
    Ti = default_params["Ti"]

    surf_ts, surf_Ws = np.meshgrid(t_vec, W_vec)
    surf_dWdts = kG * np.exp(-kG * (surf_ts - Ti)) * surf_Ws
    surf_ts, surf_Ws, surf_dWdts = crop_surface(surf_ts, surf_Ws, surf_dWdts,
                                                dWdtlim)

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"),
                           figsize=(FIG_WIDTH, FIG_HEIGHT))

    if plot_surface:
        ax.plot_surface(surf_ts, surf_Ws, surf_dWdts,
                        antialiased=False, color=to_rgba("C0", 0.85),
                        edgecolors=to_rgba("C0", 0), linewidths=0.0)

    def get_classical_lines(ts, A):
        line_ts = np.copy(ts)
        line_Ws = A * np.exp(-np.exp(-kG * (line_ts - Ti)))
        line_dWdts = kG * np.exp(-kG * (line_ts - Ti)) * line_Ws

        mask = (line_Ws > Wlim[0]) & (line_Ws < Wlim[1])

        line_ts = line_ts[mask]
        line_Ws = line_Ws[mask]
        line_dWdts = line_dWdts[mask]

        return line_ts, line_Ws, line_dWdts

    for A in (default_params["A"], default_params["A"] + 1):

        line_ts, line_Ws, line_dWdts = get_classical_lines(t_vec, A)

        if plot_lifts:
            ax.plot(line_ts, line_Ws, line_dWdts, color="black", zorder=4)
        if plot_projection:
            ax.plot(line_ts, line_Ws, dWdtlim[0], color="black")

    cla_generator = Generator(0, W)

    default_line = get_classical_lines(t_vec, default_params["A"])

    if plot_lifts:
        plot_transformation_curves(ax, np.stack(default_line, axis=1),
                                   cla_generator, default_params, tlim,
                                   (Wlim, dWdtlim),
                                   trans_max_len=np.log(4 / 3),
                                   plot_kwargs={"zorder": 4}, in2d=False,
                                   jet_space_order=1)
    if plot_projection:
        raise NotImplementedError("Projected transformations are not "
                                  "implemented.")

    ax.set_xlim(*tlim)
    ax.set_ylim(*Wlim)
    ax.set_zlim(*dWdtlim)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$W$")
    ax.set_zlabel(r"$W'$")

    ax.view_init(elev=34, azim=-53)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if save_path:
        # Create a bounding box of the axis that eliminates the vertical
        # white space.
        bbox = ax.get_position()
        bbox.x1 += 0.06
        bbox.y0 = -0.04
        bbox.y1 = 0.97
        # Convert the bounding box to inches
        bbox.x0 *= FIG_WIDTH
        bbox.x1 *= FIG_WIDTH
        bbox.y0 *= FIG_HEIGHT
        bbox.y1 *= FIG_HEIGHT

        file_path = os.path.join(save_path, file_name)

        plt.savefig(file_path, format="pdf", bbox_inches=bbox)


def plot_autonomous_surface(default_params, t_vec, W_vec, tlim, Wlim, dWdtlim,
                            save_path=None,
                            file_name="autonomous-gompertz-surface.pdf",
                            plot_projection=False, plot_lifts=True,
                            plot_surface=True):
    r"""Plot the jet surface of the autonomous Gompertz model.

    Args:
        default_params: A dict containing the default parameters of the
            \(T_i\)-parametrized Gompertz model. The parameters are
            \lstinline{"A"}, \lstinline{"Ti"} and \lstinline{"kG"}.

        t_vec: numpy array of time values to evaluate at.

        W_vec: numpy array of size values to evaluate at.

        tlim: Two-tupple of limits for the time axis.

        Wlim: Two-tupple of limits for the size axis.

        dWdtlim: Two-tupple of limits for the size derivative axis.

        save_path: Optional path to save the figure to.

        file_name: Name of the file name to save to if a path is given.

        plot_projection: Boolean value to control whether the projection
            of solution and transformation lines are plotted. (Not
            implemented)

        plot_lifts: Boolean value to control whether the lifts of
            solution and transformation lines are plotted.

        plot_surface: Boolean value to control whether the ODE surface
            if plotted.
    """
    # Set up symbolic parameters
    t, W = symbols("t W")
    Generator = generator_on(((t,), (W,)))

    kG = default_params["kG"]
    A = default_params["A"]

    surf_ts, surf_Ws = np.meshgrid(t_vec, W_vec)
    surf_dWdts = - kG * np.log(surf_Ws / A) * surf_Ws

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"),
                           figsize=(FIG_WIDTH, FIG_HEIGHT))

    if plot_surface:
        ax.plot_surface(surf_ts, surf_Ws, surf_dWdts, antialiased=False,
                        color=to_rgba("C0", 0.85), edgecolors=to_rgba("C0", 0),
                        linewidths=0.0)

    def get_autonomous_lines(ts, Ti):
        line_ts = np.copy(ts)
        line_Ws = A * np.exp(-np.exp(-kG * (line_ts - Ti)))
        line_dWdts = -kG * np.log(line_Ws / A) * line_Ws

        mask = (line_Ws > Wlim[0]) & (line_Ws < Wlim[1])

        line_ts = line_ts[mask]
        line_Ws = line_Ws[mask]
        line_dWdts = line_dWdts[mask]

        return line_ts, line_Ws, line_dWdts

    for Ti in (default_params["Ti"], default_params["Ti"] + 3):

        line_ts, line_Ws, line_dWdts = get_autonomous_lines(t_vec, Ti)

        if plot_lifts:
            ax.plot(line_ts, line_Ws, line_dWdts, color="black", zorder=4)
        if plot_projection:
            ax.plot(line_ts, line_Ws, dWdtlim[0], color="black")

    aut_generator = Generator(1, 0)

    default_line = get_autonomous_lines(t_vec, default_params["Ti"])

    if plot_lifts:
        plot_transformation_curves(ax, np.stack(default_line, axis=1),
                                   aut_generator, default_params, tlim,
                                   (Wlim, dWdtlim), trans_max_len=3,
                                   plot_kwargs={"zorder": 4}, in2d=False,
                                   jet_space_order=1)
    if plot_projection:
        raise NotImplementedError("Projected transformations are not "
                                  "implemented.")

    ax.set_xlim(*tlim)
    ax.set_ylim(*Wlim)
    ax.set_zlim(*dWdtlim)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$W$")
    ax.set_zlabel(r"$W'$")

    ax.view_init(elev=34, azim=-53)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if save_path:
        # Create a bounding box of the axis that eliminates the vertical
        # white space.
        bbox = ax.get_position()
        bbox.x1 += 0.06
        bbox.y0 = -0.04
        bbox.y1 = 0.97
        # Convert the bounding box to inches
        bbox.x0 *= FIG_WIDTH
        bbox.x1 *= FIG_WIDTH
        bbox.y0 *= FIG_HEIGHT
        bbox.y1 *= FIG_HEIGHT

        file_path = os.path.join(save_path, file_name)

        plt.savefig(file_path, format="pdf", bbox_inches=bbox)


def crop_surface(xs, ys, zs, zlim):
    """Crop the values sent to a surface plot by some limits in the z
    direction.

    Edge vertices are adjusted to create smooth cropping edges.

    Args:
        xs: Array of x values.

        ys: Array of y values.

        zs: Array of z values (the dimension to drop in).

        zlim: A two-tupple of the lower and upper limit in z direction

    Returns:
        The x, y and z arrays of values, with NaN inserted in the
        z-array to hide cropped vertices.
    """
    down_mask = zs > zlim[0]
    up_mask = zs < zlim[1]

    new_xs = np.copy(xs)
    new_ys = np.copy(ys)
    new_zs = np.copy(zs)
    # Cropped vertices are assigned NaN
    new_zs[np.logical_not(up_mask & down_mask)] = np.NaN

    for mask, lim in zip((down_mask, up_mask), zlim):
        # Find the cropped vertices at the edges of the visible surface
        left_edge_vertices = {(i + 1, j) for i, j
                              in zip(*np.where(mask[:-1,:] & ~mask[1:,:]))}
        right_edge_vertices = set(zip(*np.where(mask[1:,:] & ~mask[:-1,:])))
        front_edge_vertices = {(i, j + 1) for i, j
                               in zip(*np.where(mask[:,:-1] & ~mask[:,1:]))}
        back_edge_vertices = set(zip(*np.where(mask[:,1:] & ~mask[:,:-1])))

        # Move edge vertices to the interpolated edge of the cropping
        patched_vertices = []

        # The order of adjustment is arbitrary. Vertices with many edge
        # connections could be diagonally interpolated, but visual
        # results are not significantly improved.
        for i, j in front_edge_vertices:
            if (i, j) not in patched_vertices:
                new_xs[i, j] = interp1d(zs[i, j-1:j+1], xs[i, j-1:j+1])(lim)
                new_zs[i, j] = lim
                patched_vertices.append((i, j))

        for i, j in back_edge_vertices:
            if (i, j) not in patched_vertices:
                new_xs[i, j] = interp1d(zs[i, j:j+2], xs[i, j:j+2])(lim)
                new_zs[i, j] = lim
                patched_vertices.append((i, j))

        for i, j in right_edge_vertices:
            if (i, j) not in patched_vertices:
                new_ys[i, j] = interp1d(zs[i-1:i+1, j], ys[i-1:i+1, j])(lim)
                new_zs[i, j] = lim
                patched_vertices.append((i, j))

        for i, j in left_edge_vertices:
            if (i, j) not in patched_vertices:
                new_ys[i, j] = interp1d(zs[i:i+2, j], ys[i:i+2, j])(lim)
                new_zs[i, j] = lim
                patched_vertices.append((i, j))

    return new_xs, new_ys, new_zs


if __name__ == "__main__":
    plot()
    plt.show()
