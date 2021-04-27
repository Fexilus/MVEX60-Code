from sympy import symbols, ln, exp, lambdify

import matplotlib.pyplot as plt

from symmetries.generator import generator_on
from symmetries.jetspace import JetSpace
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
X_aut1 = Generator(exp(kG * t) * ln(W / A), 0)
X_aut2 = Generator(0, exp(-kG * t) * W)
X_aut3 = Generator(0, W * ln(W / A))
X_aut4 = Generator(1, 0)
X_aut5 = Generator(t, W * ln(W / A) * ln(abs(ln(W / A))))
X_aut6 = Generator(exp(-kG * t), -kG * exp(-kG * t) * W * ln(W)) 

generators = [X_aut1, X_aut2, X_aut3, X_aut4, X_aut5, X_aut6]

tlim = (-2, 10)
Wlim = (0, 3)

#num_solution_lines = 11
trans_max_lens = [10, 1, 2, 2, 1, 10]

params = {A: 3, kG: 1}

param_syms, param_vals = zip(*params.items())

coords = jet_space.original_total_space[0] + jet_space.original_total_space[1]

rhs_func = lambdify(coords + list(param_syms), autonomous_rhs)


def diff_eq(t, y):
    """The differential equation as a python function."""

    return rhs_func(t, *y, *param_vals)


fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)

iter_bundle = enumerate(zip(generators, axs.flat, trans_max_lens), start=1)
for i, (generator, ax, trans_max_len) in iter_bundle:
    plot_transformation(generator, ax, diff_eq, (0, 1), tlim=tlim, ylim=Wlim,
                        parameters=params, trans_max_len=trans_max_len)

    ax.set_title(f"$X_{{\\mathrm{{a}},{i}}}$")

plt.show()
