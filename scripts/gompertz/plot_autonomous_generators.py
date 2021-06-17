"""Plot transformations generated by symmetry generators of the
autonomous Gompertz model.
"""
import os.path

from sympy import symbols, ln, exp, lambdify
import matplotlib.pyplot as plt

from symmetries import generator_on, JetSpace
from symmetries.visualize import plot_transformation


# Time
t = time = symbols("t", real=True)
# States
W = state = symbols("W", nonnegative=True)

# Jet space and derivative coordinate
jet_space = JetSpace(time, state, 1)

# Parameters
kG, A = symbols("k_G A")

# Differential equation
autonomous_rhs = - kG * W * ln(W / A)

Generator = generator_on(jet_space.original_total_space)
# Generators
X_aut1 = Generator(exp(kG * t) * ln(W / A), 0)
X_aut2 = Generator(0, exp(-kG * t) * W)
X_aut3 = Generator(0, W * ln(W / A))
X_aut4 = Generator(1, 0)
X_aut5 = Generator(t, W * ln(W / A) * ln(abs(ln(W / A))))
X_aut6 = Generator(exp(-kG * t), -kG * exp(-kG * t) * W * ln(W))

generators = [X_aut1, X_aut2, X_aut3, X_aut4, X_aut5, X_aut6]


def plot(save_path=None, file_names=["gompertz-autonomous-ansatz.eps",
                                     "gompertz-autonomous-param.eps"],
         transformation_kw_args=None):

    transformation_kw_args = transformation_kw_args or {}

    tlim = (-2, 10)
    Wlim = (0, 3.1)

    #num_solution_lines = 11
    trans_max_lens = [10, 1, 2, 2, 1, 10]

    params = {A: 3, kG: 1}

    param_syms, param_vals = zip(*params.items())

    coords = (jet_space.original_total_space[0]
              + jet_space.original_total_space[1])

    rhs_func = lambdify(coords + list(param_syms), autonomous_rhs)

    def diff_eq(t, y):
        """The differential equation as a python function."""
        return rhs_func(t, *y, *param_vals)

    # Plot generators from ansatz
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 3))

    all_axs = axs.flat
    ansatz_iter_bundle = ((i, gen, max_len) for i, (gen, max_len)
                          in enumerate(zip(generators, trans_max_lens),
                                       start=1)
                          if i in [1, 2, 3])
    for i, gen, max_len, ax in zip(*zip(*ansatz_iter_bundle), all_axs):
        plot_transformation(gen, ax, diff_eq, (0, 1), tlim=tlim, ylim=Wlim,
                            parameters=params, trans_max_len=max_len,
                            **transformation_kw_args)

        ax.set_title(f"$X_{{\\mathrm{{a}},{i}}}$")
        ax.set_xlim(tlim)
        ax.set_ylim(Wlim)
        ax.set_aspect((tlim[1] - tlim[0]) / (Wlim[1] - Wlim[0]))
        ax.set_xlabel("$t$")
        ax.set_ylabel("$W$")

    for ax in all_axs:
        ax.set_axis_off()

    fig.tight_layout()

    if save_path:
        file_path = os.path.join(save_path, file_names[0])
        plt.savefig(file_path, format="eps",
                    bbox_inches="tight")

    # Plot generators from parameter independence
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(9, 6))

    param_iter_bundle = ((i, gen, max_len) for i, (gen, max_len)
                         in enumerate(zip(generators, trans_max_lens), start=1)
                         if i in [2, 3, 4, 5, 6])
    all_axs = axs.flat
    for i, gen, max_len, ax in zip(*zip(*param_iter_bundle), all_axs):
        plot_transformation(gen, ax, diff_eq, (0, 1), tlim=tlim, ylim=Wlim,
                            parameters=params, trans_max_len=max_len,
                            **transformation_kw_args)

        ax.set_title(f"$X_{{\\mathrm{{a}},{i}}}$")
        ax.set_xlim(tlim)
        ax.set_ylim(Wlim)
        ax.set_aspect((tlim[1] - tlim[0]) / (Wlim[1] - Wlim[0]))
        ax.set_xlabel("$t$")
        ax.set_ylabel("$W$")

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
