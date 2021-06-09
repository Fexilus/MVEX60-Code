"""Plot several solution lines of the Lotka-Volterra model."""
import os.path

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

from symmetries.visualize.utils import integrate_two_ways, get_spread


def plot(save_path=None, file_names=["lotka-volterra-solutions-varn.eps",
                                     "lotka-volterra-solutions-varp.eps"]):
    plt.rc("mathtext", fontset="cm")

    tlim = (-2, 10)
    Nlim = (0, 2)
    Plim = (0, 2)

    NUM_SOLUTION_LINES = 3
    include_init_val = (0, 1, 1)

    params = {"a": 1, "b": 2, "c": 2.5, "d": 1.5}


    def lotka_volterra_rhs(t, y, a=1, b=1, c=1, d=1):
        """The classical Gompertz model with \\(T_i\\)-parametrization."""

        N, P = y

        dNdt = a * N - b * N * P
        dPdt = c * N * P - d * P
        return np.array([dNdt, dPdt])


    integrator = ode(lambda t, y: lotka_volterra_rhs(t, y, **params))
    integrator.set_integrator('vode', method='adams')

    tlim_diff = tlim[1] - tlim[0]
    dt = tlim_diff / 100

    # Plot with varying initial N
    fig, axs = plt.subplots(1, 2)

    init_vals = get_spread(include_init_val, (0, (1 - Nlim[0]) / 2, 1),
                        (0, (1 + Nlim[1]) / 2, 1), NUM_SOLUTION_LINES)
    for init_val in init_vals:
        integrator.set_initial_value(init_val[1:], init_val[0])

        time_points, solut = integrate_two_ways(integrator, dt, max_len=tlim_diff,
                                                t_boundry=tlim,
                                                y_boundry=(Nlim, Plim))

        is_include_init_val = np.allclose(init_val, include_init_val)
        color = "black" if is_include_init_val else None
        zorder = 2 if is_include_init_val else 1

        axs[0].plot(time_points, solut[:, 0], color=color, zorder=zorder)
        axs[0].set_aspect((tlim[1] - tlim[0]) / (Nlim[1] - Nlim[0]))
        axs[0].set_xlabel("$t$")
        axs[0].set_ylabel("$N$")

        axs[1].plot(time_points, solut[:, 1], color=color, zorder=zorder)
        axs[1].set_aspect((tlim[1] - tlim[0]) / (Plim[1] - Plim[0]))
        axs[1].set_xlabel("$t$")
        axs[1].set_ylabel("$P$")

    fig.tight_layout()

    if save_path:
        file_path = os.path.join(save_path, file_names[0])
        plt.savefig(file_path, format="eps",
                    bbox_inches="tight")

    # Plot with varying initial P
    fig, axs = plt.subplots(1, 2)

    init_vals = get_spread(include_init_val, (0, 1, (1 - Plim[0]) / 2),
                        (0, 1, (1 + Plim[1]) / 2), NUM_SOLUTION_LINES)
    for init_val in init_vals:
        integrator.set_initial_value(init_val[1:], init_val[0])

        time_points, solut = integrate_two_ways(integrator, dt, max_len=tlim_diff,
                                                t_boundry=tlim, y_boundry=(Nlim, Plim))

        is_include_init_val = np.allclose(init_val, include_init_val)
        color = "black" if is_include_init_val else None
        zorder = 2 if is_include_init_val else 1

        axs[0].plot(time_points, solut[:, 0], color=color, zorder=zorder)
        axs[0].set_aspect((tlim[1] - tlim[0]) / (Nlim[1] - Nlim[0]))
        axs[0].set_xlabel("$t$")
        axs[0].set_ylabel("$N$")

        axs[1].plot(time_points, solut[:, 1], color=color, zorder=zorder)
        axs[1].set_aspect((tlim[1] - tlim[0]) / (Plim[1] - Plim[0]))
        axs[1].set_xlabel("$t$")
        axs[1].set_ylabel("$P$")

    fig.tight_layout()

    if save_path:
        file_path = os.path.join(save_path, file_names[1])
        plt.savefig(file_path, format="eps",
                    bbox_inches="tight")


if __name__ == "__main__":
    plot()
    plt.show()
