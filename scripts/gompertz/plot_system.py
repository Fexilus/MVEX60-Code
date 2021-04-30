"""Plot several solution lines to the autonomous Gompertz model."""

import numpy as np

from scipy.integrate import ode

import matplotlib.pyplot as plt

from symmetries.visualize.utils import integrate_two_ways, get_spread

tlim = (-2, 10)
Wlim = (0, 3)
Glim = (0, 3)

NUM_SOLUTION_LINES = 11
include_init_val = (0, 1, np.log(3))

params = {"kG": 1}


def system_rhs(t, y, kG=1,):
    """The classical Gompertz model with \\(T_i\\)-parametrization."""

    W, G = y

    dWdt = W * G
    dGdt = -kG * G
    return np.array([dWdt, dGdt])


integrator = ode(lambda t, y: system_rhs(t, y, **params))
integrator.set_integrator('vode', method='adams')

tlim_diff = tlim[1] - tlim[0]
dt = tlim_diff / 100

# Plot with varying initial W
fig, axs = plt.subplots(1, 2)

init_vals = get_spread(include_init_val, (0, Wlim[0], np.log(3)), (0, Wlim[1], np.log(3)),
                       NUM_SOLUTION_LINES)
for init_val in init_vals:
    integrator.set_initial_value(init_val[1:], init_val[0])

    time_points, solut = integrate_two_ways(integrator, dt, max_len=tlim_diff,
                                            t_boundry=tlim, y_boundry=(Wlim, Glim))

    is_include_init_val = np.allclose(init_val, include_init_val)
    color = "black" if is_include_init_val else "grey"

    axs[0].plot(time_points, solut[:, 0], color=color)
    axs[0].set_aspect((tlim[1] - tlim[0]) / (Wlim[1] - Wlim[0]))
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("W")

    axs[1].plot(time_points, solut[:, 1], color=color)
    axs[1].set_aspect((tlim[1] - tlim[0]) / (Glim[1] - Glim[0]))
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("G")

fig.tight_layout()

# Plot with varying initial G
fig, axs = plt.subplots(1, 2)

init_vals = get_spread(include_init_val, (0, 1, Glim[0]), (0, 1, Glim[1]),
                       NUM_SOLUTION_LINES)
for init_val in init_vals:
    integrator.set_initial_value(init_val[1:], init_val[0])

    time_points, solut = integrate_two_ways(integrator, dt, max_len=tlim_diff,
                                            t_boundry=tlim, y_boundry=(Wlim, Glim))

    is_include_init_val = np.allclose(init_val, include_init_val)
    color = "black" if is_include_init_val else "grey"

    axs[0].plot(time_points, solut[:, 0], color=color)
    axs[0].set_aspect((tlim[1] - tlim[0]) / (Wlim[1] - Wlim[0]))
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("W")

    axs[1].plot(time_points, solut[:, 1], color=color)
    axs[1].set_aspect((tlim[1] - tlim[0]) / (Glim[1] - Glim[0]))
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("G")

fig.tight_layout()

plt.show()