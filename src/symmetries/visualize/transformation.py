from math import floor

import numpy as np

from scipy.integrate import ode

from .integralcurves import get_integral_curves
from .arrowpath import WithArrowStroke
from .utils import integrate_two_ways, get_spaced_points

from ..utils import iter_wrapper

def plot_transformation(generator, axs, diff_eq_rhs, init_val, tlim,
                        parameters=None, dt=0.1, ylim=None,
                        num_trans_points=10, trans_max_len=10,
                        arrow_stroke_arguments=None):
    """Plot transformation defined by generator of an ODE on axis."""

    # Process arguments
    axs = list(iter_wrapper(axs))

    if not parameters:
        parameters = {}

    arrow_stroke_arguments = arrow_stroke_arguments or {}

    # Plot the initial solution curve
    time_points, solut = plot_solution_curve(axs, diff_eq_rhs, init_val, tlim,
                                             dt=dt, ylim=ylim)

    # Calculate equally spaced points along the solution line
    tlim_diff = tlim[1] - tlim[0]

    if not ylim:
        ylim = (solut.min(axis=0), solut.max(axis=0))

    ylim = np.array(ylim, ndmin=2)
    ylim_diff = ylim[:, 1] - ylim[:, 0]

    solution_curve = np.concatenate((time_points, solut), axis=1)

    transformation_points = get_normed_spaced_points(solution_curve,
                                                     (tlim_diff, *ylim_diff),
                                                     num_trans_points)

    # Integrate the generator vector field in those points
    trans_curves = get_integral_curves(generator, transformation_points,
                                       parameters=parameters,
                                       boundry=(tlim, *ylim),
                                       max_len=trans_max_len)

    # Plot a new solution curve where the transformation lands.
    # The middle point is chosen as a starting point to ensure that the
    # integration of the new solution curve is stable
    center_trans_curves = trans_curves[floor(len(trans_curves) / 2)]
    center_trans_end_point = center_trans_curves[-1]

    time_points, solut = plot_solution_curve(axs, diff_eq_rhs,
                                             center_trans_end_point, tlim,
                                             dt=dt, ylim=ylim)

    # Set up arrow stroke effect
    arrow_stroke = WithArrowStroke(**arrow_stroke_arguments)

    # Plot the transformation curves
    for curve in trans_curves:
        curve = np.asarray(curve)
        for i, ax in enumerate(axs, start=1):
            ax.plot(curve[:,0], curve[:, i],
                    path_effects=[arrow_stroke],
                    color="black")


def plot_solution_curve(axs, diff_eq_rhs, init_val, tlim, dt=0.1, ylim=None):
    """ Plot the solution curve of an ODE."""

    # Process arguments
    axs = list(iter_wrapper(axs))

    # Set up numerical integrator
    integrator = ode(diff_eq_rhs).set_integrator('vode', method='adams')
    integrator.set_initial_value(init_val[1:], init_val[0])

    tlim_diff = tlim[1] - tlim[0]

    time_points, solut = integrate_two_ways(integrator, dt, max_len=tlim_diff,
                                            t_boundry=tlim, y_boundry=ylim)

    for i, ax in enumerate(axs):
        ax.plot(time_points, solut[:,i])

    return time_points, solut


def get_normed_spaced_points(curve, scales, num_points):
    """Get spaced points along a curve according to scaling."""

    norm_matrix = np.diag(scales)

    normed_curve = (np.linalg.inv(norm_matrix) @ curve.T).T

    normed_spaced_points = get_spaced_points(normed_curve, num_points)

    spaced_points = (norm_matrix @ normed_spaced_points.T).T

    return spaced_points
