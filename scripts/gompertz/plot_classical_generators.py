"""
Plot transformations generated by symmetry generators of the classical
Gompertz model.
"""
import math
import os.path

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

# Parameters
kG, Ti = symbols("k_G T_i")

# Differential equations
classical_rhs = kG * exp(-kG * (t - Ti)) * W

Generator = generator_on(jet_space.original_total_space)
# Generators
X_cla1 = Generator(exp(kG * t), 0)
X_cla2 = Generator(W*exp(kG * t + exp(-kG * (t - Ti))), 0)
X_cla3 = Generator(0, exp(-exp(Ti * kG) * exp(-kG * t)))
X_cla4 = Generator(0, W)
X_cla5 = Generator(1, - kG * W * ln(W))

generators = [X_cla1, X_cla2, X_cla3, X_cla4, X_cla5]


def plot(save_path=None, file_names=["gompertz-classical-ansatz.eps",
                                     "gompertz-classical-param.eps"]):
    tlim = (-2, 10)
    Wlim = (0, 3)

    trans_max_lens = [0.15, 0.07, 0.6, 0.2, 1]

    params = {Ti: math.log(math.log(3)), kG: 1}

    param_syms, param_vals = zip(*params.items())

    coords = jet_space.original_total_space[0] + jet_space.original_total_space[1]

    rhs_func = lambdify(coords + list(param_syms), classical_rhs)


    def diff_eq(t, y):
        """The differential equation as a python function."""

        return rhs_func(t, *y, *param_vals)


    # Plot generators from ansatz
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 3))

    all_axs = axs.flat
    ansatz_iter_bundle = ((i, gen, max_len) for i, (gen, max_len)
                        in enumerate(zip(generators, trans_max_lens), start=1)
                        if i in [1, 2, 3])
    for i, gen, trans_max_len, ax in zip(*zip(*ansatz_iter_bundle), all_axs):
        plot_transformation(gen, ax, diff_eq, (0, 1), tlim=tlim, ylim=Wlim,
                            parameters=params, trans_max_len=trans_max_len)

        ax.set_title(f"$X_{{\\mathrm{{c}},{i}}}$")
        ax.set_aspect((tlim[1] - tlim[0]) / (Wlim[1] - Wlim[0]))
        ax.set_xlabel("t")
        ax.set_ylabel("W")

    for ax in all_axs:
        ax.set_axis_off()

    fig.tight_layout()

    if save_path:
        file_path = os.path.join(save_path, file_names[0])
        plt.savefig(file_path, format="eps",
                    bbox_inches="tight")

    # Plot generators from parameter independence
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 3))

    param_iter_bundle = ((i, gen, max_len) for i, (gen, max_len)
                        in enumerate(zip(generators, trans_max_lens), start=1)
                        if i in [1, 4, 5])
    all_axs = axs.flat
    for i, gen, trans_max_len, ax in zip(*zip(*param_iter_bundle), all_axs):
        plot_transformation(gen, ax, diff_eq, (0, 1), tlim=tlim, ylim=Wlim,
                            parameters=params, trans_max_len=trans_max_len)

        ax.set_title(f"$X_{{\\mathrm{{c}},{i}}}$")
        ax.set_aspect((tlim[1] - tlim[0]) / (Wlim[1] - Wlim[0]))
        ax.set_xlabel("t")
        ax.set_ylabel("W")

    for ax in all_axs:
        ax.set_axis_off()

    fig.tight_layout()

    if save_path:
        file_path = os.path.join(save_path, file_names[1])
        plt.savefig(file_path, format="eps",
                    bbox_inches="tight")


if __name__ == "__main__":
    plot()
    plt.show()
