"""Plot transformations generated by symmetry generators of the system
Gompertz model.
"""
import math
import os.path

from sympy import symbols, ln, exp, lambdify
import matplotlib.pyplot as plt

from symmetries import generator_on, JetSpace
from symmetries.visualize import plot_transformation


# Time
t = time = symbols("t", real=True)
# States
W, G = state = symbols("W G", nonnegative=True)

# Jet space and derivative coordinate
jet_space = JetSpace(time, state, 1)

# Parameters
kG = symbols("k_G")

# Differential equations
system_rhs = [W * G, -kG * G]

Generator = generator_on(jet_space.original_total_space)
# Generators
X_sys1 = Generator(1, (0, 0))
X_sys2 = Generator(- (exp(kG * t)) / kG, (0, + G * exp(kG * t)))
X_sys3 = Generator(exp(kG * t) * G, (0, 0))
X_sys4 = Generator(0, (W, 0))
X_sys5 = Generator(0, (- (W * exp(-kG * t)) / kG, exp(-kG * t)))
X_sys6 = Generator(0, (ln(W) * W, G))

generators = [X_sys1, X_sys2, X_sys3, X_sys4, X_sys5, X_sys6]


def plot(save_path=None, file_names=["gompertz-system-ansatz.eps",
                                     "gompertz-system-param.eps"],
         transformation_kw_args=None):

    transformation_kw_args = transformation_kw_args or {}

    tlim = (-2, 10)
    Wlim = (0, 3.1)
    Glim = (0, 3.1)

    trans_max_lens = [3, 0.8, 0.8, 0.3, 3, 0.2]

    params = {kG: 1}

    param_syms, param_vals = zip(*params.items())

    coords = (jet_space.original_total_space[0]
              + jet_space.original_total_space[1])

    rhs_func = lambdify(coords + list(param_syms), system_rhs)

    def diff_eq(t, y):
        """The differential equation as a python function."""
        return rhs_func(t, *y, *param_vals)

    # Plot generators from ansatz
    fig = plt.figure(constrained_layout=True, figsize=(12, 9))
    subfigs = fig.subfigures(3, 2).flat

    ansatz_iter_bundle = ((i, gen, max_len) for i, (gen, max_len)
                          in enumerate(zip(generators, trans_max_lens),
                                       start=1)
                          if i in [1, 2, 3, 4, 5])
    for i, gen, max_len, subfig in zip(*zip(*ansatz_iter_bundle), subfigs):
        axs = subfig.subplots(1, 2)
        plot_transformation(gen, axs, diff_eq, (0, 1, math.log(3)), tlim=tlim,
                            ylim=(Wlim, Glim), parameters=params,
                            trans_max_len=max_len, **transformation_kw_args)

        subfig.suptitle(f"$X_{{\\mathrm{{s}},{i}}}$")
        axs[0].set_xlim(tlim)
        axs[0].set_ylim(Wlim)
        axs[0].set_aspect((tlim[1] - tlim[0]) / (Wlim[1] - Wlim[0]))
        axs[0].set_xlabel("$t$")
        axs[0].set_ylabel("$W$")
        axs[1].set_xlim(tlim)
        axs[1].set_ylim(Glim)
        axs[1].set_aspect((tlim[1] - tlim[0]) / (Glim[1] - Glim[0]))
        axs[1].set_xlabel("$t$")
        axs[1].set_ylabel("$G$")

    if save_path:
        file_path = os.path.join(save_path, file_names[0])
        plt.savefig(file_path, format="eps")

    # Plot generators from parameter independence
    fig = plt.figure(constrained_layout=True, figsize=(6, 9))
    subfigs = fig.subfigures(3, 1)

    ansatz_iter_bundle = ((i, gen, max_len) for i, (gen, max_len)
                          in enumerate(zip(generators, trans_max_lens),
                                       start=1)
                          if i in [1, 4, 6])
    for i, gen, max_len, subfig in zip(*zip(*ansatz_iter_bundle), subfigs):
        axs = subfig.subplots(1, 2)
        plot_transformation(gen, axs, diff_eq, (0, 1, math.log(3)), tlim=tlim,
                            ylim=(Wlim, Glim), parameters=params,
                            trans_max_len=max_len, **transformation_kw_args)

        subfig.suptitle(f"$X_{{\\mathrm{{s}},{i}}}$")
        axs[0].set_xlim(tlim)
        axs[0].set_ylim(Wlim)
        axs[0].set_aspect((tlim[1] - tlim[0]) / (Wlim[1] - Wlim[0]))
        axs[0].set_xlabel("$t$")
        axs[0].set_ylabel("$W$")
        axs[1].set_xlim(tlim)
        axs[1].set_ylim(Glim)
        axs[1].set_aspect((tlim[1] - tlim[0]) / (Glim[1] - Glim[0]))
        axs[1].set_xlabel("$t$")
        axs[1].set_ylabel("$G$")

    if save_path:
        file_path = os.path.join(save_path, file_names[1])
        plt.savefig(file_path, format="eps")


if __name__ == "__main__":
    plot()
    plt.show()
