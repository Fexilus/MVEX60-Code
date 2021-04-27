"""Utility functions for the visualization code."""
import numpy as np

def in_ranges(vec, ranges):
    """Test if elements in vector are in their respective ranges.

    If ranges is None, allways return True.
    """
    if ranges is None:
        return True

    vec = np.array(vec)
    ranges = np.array(ranges, ndmin=2)

    return np.all(vec > ranges[:,0]) and np.all(vec < ranges[:,1])


def integrate_two_ways(integrator, dt, max_len, t_boundry=None,
                       y_boundry=None):
    """Integrate an ODE in both directions, given a step size.

    This function is useful when the ODE might become numerically
    unstable in either direction of the initial point, since the
    integration "starts" at the initial points in both directions.

    Args:
        integrator: A scipy integrator with set initial condition.

        dt: The step size of integration.

        max_len: The maximum independent value integrated to in both
            directions.

        y_boundry: An optional boundry for the dependent values where
            integration stops.
    """

    init_t = integrator.t
    init_y = integrator.y

    curve_forward_t = []
    curve_forward_y = []

    while (integrator.successful()
           and integrator.t <= init_t + max_len
           and in_ranges(integrator.t, t_boundry)
           and in_ranges(integrator.y, y_boundry)):
        integrator.integrate(integrator.t + dt)

        curve_forward_t.append(integrator.t)
        curve_forward_y.append(integrator.y)

    integrator.set_initial_value(init_y, init_t)

    curve_backward_t = []
    curve_backward_y = []

    while (integrator.successful()
           and integrator.t >= init_t - max_len
           and in_ranges(integrator.t, t_boundry)
           and in_ranges(integrator.y, y_boundry)):
        integrator.integrate(integrator.t - dt)

        curve_backward_t.append(integrator.t)
        curve_backward_y.append(integrator.y)

    curve_t = list(reversed(curve_backward_t)) + [init_t] + curve_forward_t
    curve_y = list(reversed(curve_backward_y)) + [init_y] + curve_forward_y

    # Reset initial values
    integrator.set_initial_value(init_y, init_t)

    return np.array(curve_t)[:,None], np.array(curve_y)


def integrate_forward(integrator, dt, max_len, t_boundry=None, y_boundry=None):
    """Integrate an ODE forward., given a step size.

    Exist to comply with the form of the two way integration function.

    Args:
        integrator: A scipy integrator with set initial condition.

        dt: The step size of integration.

        max_len: The maximum independent value integrated to.

        y_boundry: An optional boundry for the dependent values where
            integration stops.
    """

    init_t = integrator.t
    init_y = integrator.y

    curve_forward_t = []
    curve_forward_y = []

    while (integrator.successful()
           and integrator.t <= init_t + max_len
           and in_ranges(integrator.t, t_boundry)
           and in_ranges(integrator.y, y_boundry)):
        integrator.integrate(integrator.t + dt)

        curve_forward_t.append(integrator.t)
        curve_forward_y.append(integrator.y)

    curve_t = [init_t] + curve_forward_t
    curve_y = [init_y] + curve_forward_y

    # Reset initial values
    integrator.set_initial_value(init_y, init_t)

    return np.array(curve_t)[:,None], np.array(curve_y)


def get_spaced_points(curve, num_points):
    """Take a curve and return fairly spaced points of the curve."""

    if len(curve) <= num_points:
        return np.array(curve)

    curve_array = np.array(curve)
    curve_diffs = np.diff(curve_array, axis=0)
    curve_dists = np.linalg.norm(curve_diffs, axis=1)

    total_dist = np.sum(curve_dists)

    avg_dist = total_dist / (num_points + 1)

    final_points = []

    dist_error = 0

    for cur_point, dist in zip(curve_array[:-1, :], curve_dists):

        if dist / 2 + dist_error > avg_dist:
            final_points.append(cur_point)
            dist_error -= avg_dist

        dist_error += dist

    return np.array(final_points)
