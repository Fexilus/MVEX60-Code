from math import floor

from sympy import symbols, ln, exp, lambdify, sign

import numpy as np

from scipy.integrate import ode

import matplotlib.pyplot as plt

from symmetries.visualize.integralcurves import get_integral_curves
from symmetries.generator import generator_on, lie_bracket
from symmetries.jetspace import JetSpace
from symmetries.visualize.arrowpath import WithArrowStroke
from symmetries.visualize.utils import integrate_two_ways, get_spaced_points
from symmetries.visualize.transformation import plot_transformation


# Time
t = time = symbols("t", real=True)
# States
W = state = symbols("W", nonnegative=True)

# Jet space and derivative coordinate
jet_space = JetSpace(time, state, 1)

Wt = jet_space.fibres[W][(1,)]

# Parameters
kG, Ti, A = symbols('k_G T_i A')

# Differential equations
autonomous_equation = Wt + kG * W * ln(W / A)
autonomous_rhs = - kG * W * ln(W / A)

Generator = generator_on(([time], [state]))
# Generators
X_aut1 = Generator(1, 0)
X_aut2 = Generator(t, W * ln(W / A) * ln(abs(ln(W / A))))
X_aut3 = Generator(0, W * ln(W / A))
X_aut4 = Generator(exp(-kG * t), -kG * exp(-kG * t) * W * ln(W))
X_aut6 = Generator(0, exp(-kG * t) * W)
X_aut7 = - (lie_bracket(X_aut2, X_aut6) + X_aut6)

autonomous_generators = [X_aut1, X_aut2, X_aut3, X_aut4, X_aut6, X_aut7]

tlim = (-2, 10)
Wlim = (0, 3)

num_solution_lines = 11

params = {A: 3, kG: 1}

param_syms, param_vals = zip(*params.items())

coords = jet_space.original_total_space[0] + jet_space.original_total_space[1]

rhs_func = lambdify(coords + list(param_syms), autonomous_rhs)


def diff_eq(t, y):
    return rhs_func(t, *y, *param_vals)


for generator in autonomous_generators:
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_transformation(generator, ax, diff_eq, (0, 1), tlim=tlim, ylim=Wlim,
                        parameters=params)

plt.show()
