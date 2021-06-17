"""Calculation of integral curves of vector fields defined by
infinitesimal generators.
"""

from sympy import lambdify
import numpy as np
from scipy.integrate import ode

from .utils import integrate_two_ways, integrate_forward


def get_integral_curves(generator, start_points, parameters=None, boundry=None,
                        max_len=10.0, two_sided=False, jet_space_order=0,
                        strict=False):
    """Get integral curves of a generator in specific points."""

    coords = generator.get_jet_space_basis(jet_space_order)

    if not parameters:
        parameters = {}

    param_syms, param_vals = zip(*parameters.items())

    vector_field = [lambdify(coords + list(param_syms), expr) for expr
                    in generator.get_tangent_field(jet_space_order)]

    def diff_eq(_, y):
        return np.array([func(*y, *param_vals) for func in vector_field])

    ds = max_len / 100

    curves = []

    for point in start_points:
        integrator = ode(diff_eq).set_integrator('vode', method='adams')
        integrator.set_initial_value(point, 0)

        if two_sided:
            _, curve = integrate_two_ways(integrator, dt=ds, max_len=max_len,
                                          y_boundry=boundry, strict=strict)
        else:
            _, curve = integrate_forward(integrator, dt=ds, max_len=max_len,
                                         y_boundry=boundry, strict=strict)

        curves.append(curve)

    return curves
