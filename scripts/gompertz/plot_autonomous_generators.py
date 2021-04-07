from math import floor

from sympy import symbols, ln, exp, lambdify, sign

import numpy as np

from scipy.integrate import ode

import matplotlib.pyplot as plt

from symmetries.visualize.integralcurves import get_integral_curves
from symmetries.generator import generator_on, lie_bracket
from symmetries.jetspace import JetSpace
from symmetries.visualize.arrowpath import WithArrowStroke
from symmetries.visualize.utils import integrate_two_ways, spaced_points


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

dt = 0.1

num_solution_lines = 11

params = {A: 3, kG: 1}

param_syms, param_vals = zip(*params.items())

coords = jet_space.original_total_space[0] + jet_space.original_total_space[1]

func = lambdify(coords + list(param_syms), autonomous_rhs)


def diff_eq(t, y):
    return func(t, *y, *param_vals)


def plot_transformation(generator, ax, differential_eq, init_val, parameters={}):

    integrator = ode(differential_eq).set_integrator('vode', method='adams')
    integrator.set_initial_value(init_val[1], init_val[0])

    time_points, solut = integrate_two_ways(integrator, dt, max_len=tlim[1] - tlim[0], t_boundry=tlim, y_boundry=Wlim)

    ax.plot(time_points, solut[:,0])

    normed_time_points = time_points / (tlim[1] - tlim[0])
    normed_solut = solut / (Wlim[1] - Wlim[0])

    normed_sol_curve = np.concatenate((normed_time_points, normed_solut), axis=1)

    normed_transformation_points = spaced_points(normed_sol_curve, 10)

    transformation_points = normed_transformation_points @ np.array([[tlim[1] - tlim[0], 0], [0, Wlim[1] - Wlim[0]]])

    int_curves = get_integral_curves(generator, transformation_points, parameters=params, boundry=(tlim, Wlim))

    center_int_curve = int_curves[floor(len(int_curves) / 2)]
    integrator.set_initial_value(center_int_curve[-1][1], center_int_curve[-1][0])

    time_points, solut = integrate_two_ways(integrator, dt, max_len=tlim[1] - tlim[0], t_boundry=tlim, y_boundry=Wlim)

    ax.plot(time_points, solut)

    for curve in int_curves:
        ax.plot(*(np.array(a) for a in list(zip(*curve))), path_effects=[WithArrowStroke(spacing=14)], color="black")


for generator in autonomous_generators:
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_transformation(generator, ax, diff_eq, (0, 1), params)

plt.show()
