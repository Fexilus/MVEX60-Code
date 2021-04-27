from math import floor

import numpy as np

from scipy.integrate import ode

from .integralcurves import get_integral_curves
from .arrowpath import WithArrowStroke
from .utils import integrate_two_ways, get_spaced_points

def plot_transformation(generator, ax, diff_eq_rhs, init_val, tlim,
                        parameters=None, dt=0.1, ylim=None):
    """Plot transformation defined by generator of an ODE on axis."""

    if not parameters:
        parameters = {}

    integrator = ode(diff_eq_rhs).set_integrator('vode', method='adams')
    integrator.set_initial_value(init_val[1], init_val[0])

    tlim_diff = tlim[1] - tlim[0]

    time_points, solut = integrate_two_ways(integrator, dt, max_len=tlim_diff,
                                            t_boundry=tlim, y_boundry=ylim)

    ax.plot(time_points, solut[:,0])

    if not ylim:
        ylim = (solut.min(), solut.max())

    ylim_diff = ylim[1] - ylim[0]

    solution_curve = np.concatenate((time_points, solut), axis=1)

    transformation_points = get_normed_spaced_points(solution_curve,
                                                     (tlim_diff, ylim_diff),
                                                     10)

    trans_curves = get_integral_curves(generator, transformation_points,
                                       parameters=parameters,
                                       boundry=(tlim, ylim))

    center_trans_curves = trans_curves[floor(len(trans_curves) / 2)]
    center_trans_end_point = center_trans_curves[-1]

    integrator.set_initial_value(center_trans_end_point[1],
                                 center_trans_end_point[0])

    time_points, solut = integrate_two_ways(integrator, dt, max_len=tlim_diff,
                                            t_boundry=tlim, y_boundry=ylim)

    ax.plot(time_points, solut)

    for curve in trans_curves:
        ax.plot(*(np.array(a) for a in list(zip(*curve))),
                path_effects=[WithArrowStroke(spacing=14)],
                color="black")


def get_normed_spaced_points(curve, scales, num_points):
    """Get spaced points along a curve according to scaling."""

    norm_matrix = np.diag(scales)

    normed_curve = (np.linalg.inv(norm_matrix) @ curve.T).T

    normed_spaced_points = get_spaced_points(normed_curve, num_points)

    spaced_points = (norm_matrix @ normed_spaced_points.T).T

    return spaced_points
