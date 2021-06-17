"""Utility functions for the visualization code."""
import numpy as np

def in_ranges(vec, ranges, strict=False):
    """Test if elements in vector are in their respective ranges.

    If ranges is None, allways return True.
    """

    vec = np.array(vec, ndmin=1)

    if len(vec) > 1:
        mask = [_range is not None for _range in ranges]

        if not np.asarray(mask).any():
            return True

        vec = vec[mask]
        ranges = [_range for _range, m in zip(ranges, mask) if m]
    else:
        if ranges is None:
            return True

    ranges = np.array(ranges, ndmin=2)

    if strict:
        return np.all(vec >= ranges[:,0]) and np.all(vec <= ranges[:,1])
    else:
        return np.any(np.logical_and((vec >= ranges[:,0]),
                                     (vec <= ranges[:,1])))



def integrate_two_ways(integrator, dt, max_len, t_boundry=None,
                       y_boundry=None, strict=False):
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

    while (integrator.t <= init_t + max_len
           and in_ranges(integrator.t, t_boundry, strict=strict)
           and in_ranges(integrator.y, y_boundry, strict=strict)):
        integrator.integrate(integrator.t + dt)

        if integrator.successful():
            curve_forward_t.append(integrator.t)
            curve_forward_y.append(integrator.y)
        else:
            break

    integrator.set_initial_value(init_y, init_t)

    curve_backward_t = []
    curve_backward_y = []

    while (integrator.t >= init_t - max_len
           and in_ranges(integrator.t, t_boundry, strict=strict)
           and in_ranges(integrator.y, y_boundry, strict=strict)):
        integrator.integrate(integrator.t - dt)

        if integrator.successful():
            curve_backward_t.append(integrator.t)
            curve_backward_y.append(integrator.y)
        else:
            break

    curve_t = list(reversed(curve_backward_t)) + [init_t] + curve_forward_t
    curve_y = list(reversed(curve_backward_y)) + [init_y] + curve_forward_y

    # Reset initial values
    integrator.set_initial_value(init_y, init_t)

    return np.array(curve_t)[:,None], np.array(curve_y)


def integrate_forward(integrator, dt, max_len, t_boundry=None, y_boundry=None,
                      strict=False):
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

    while (integrator.t <= init_t + max_len
           and in_ranges(integrator.t, t_boundry, strict=strict)
           and in_ranges(integrator.y, y_boundry, strict=strict)):
        integrator.integrate(integrator.t + dt)

        if integrator.successful():
            curve_forward_t.append(integrator.t)
            curve_forward_y.append(integrator.y)
        else:
            break

    curve_t = [init_t] + curve_forward_t
    curve_y = [init_y] + curve_forward_y

    # Reset initial values
    integrator.set_initial_value(init_y, init_t)

    return np.array(curve_t)[:,None], np.array(curve_y)


def get_normed_spaced_points(curve, scales, num_points):
    """Get spaced points along a curve according to scaling."""

    norm_matrix = np.diag(scales)

    normed_curve = (np.linalg.inv(norm_matrix) @ curve.T).T

    normed_spaced_points = get_spaced_points(normed_curve, num_points)

    spaced_points = (norm_matrix @ normed_spaced_points.T).T

    return spaced_points


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


def get_spread(include_val, min_val, max_val, num_val):
    """Creates an equidistant spread of values including a specific
    value.
    """
    include_val = np.asarray(include_val)
    min_val = np.asarray(min_val)
    max_val = np.asarray(max_val)

    # Calculate step size and magnitude without include value
    apr_step_size = (max_val - min_val) / (num_val - 1)
    apr_step_mag = np.linalg.norm(apr_step_size)

    include_index = round(np.linalg.norm(include_val - min_val) / apr_step_mag)

    apr_min_val = include_val - include_index * apr_step_size
    apr_max_val = include_val + (num_val - 1 - include_index) * apr_step_size

    step_size = apr_step_size

    # Reduce step size in all dimensions with too low/high aproximative
    # values
    for i, min_diff in enumerate(min_val - apr_min_val):
        if min_diff > 0:
            step_size[i] -= min_diff / include_index

    for i, max_diff in enumerate(apr_max_val - max_val):
        if max_diff > 0:
            step_size[i] -= max_diff / (num_val - 1 - include_index)

    final_min_val = include_val - include_index * step_size
    final_max_val = include_val + (num_val - 1 - include_index) * step_size

    return np.linspace(final_min_val, final_max_val, num_val)
