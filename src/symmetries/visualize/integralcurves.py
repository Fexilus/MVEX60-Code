from itertools import chain

from sympy import lambdify
import numpy as np
from scipy.integrate import ode

from .utils import integrate_two_ways, integrate_forward

def get_integral_curves(generator, start_points, parameters=None, boundry=None,
                        max_len=10.0, two_sided=False):
    """Get integral curves of a generator in specific points."""

    coords = generator.total_space[0] + generator.total_space[1]

    if not parameters:
        parameters = {}

    param_syms, param_vals = zip(*parameters.items())

    vector_field = [lambdify(coords + list(param_syms), expr) for expr
                    in chain(generator.xis, generator.etas)]

    def diff_eq(t, y):
        return np.array([func(*y, *param_vals) for func in vector_field])

    ds = max_len / 100

    curves = []

    for point in start_points:
        integrator = ode(diff_eq).set_integrator('vode', method='adams')
        integrator.set_initial_value(point, 0)

        if two_sided:
            _, curve = integrate_two_ways(integrator, dt=ds, max_len=max_len,
                                          y_boundry=boundry)
        else:
            _, curve = integrate_forward(integrator, dt=ds, max_len=max_len,
                                         y_boundry=boundry)

        if integrator.get_return_code() < 0: # Ad hoc way to detect warnings
            return None

        curves.append(curve)

    return curves
