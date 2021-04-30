"""Plot several solution lines to the classical Gompertz model."""

import numpy as np

from scipy.integrate import ode

import matplotlib.pyplot as plt

from symmetries.visualize.utils import integrate_two_ways, get_spread

tlim = (-2, 10)
Wlim = (0, 3)

NUM_SOLUTION_LINES = 11
include_init_val = (0, 1)

params = {"A": 3, "kG": 1}


def autonomous_rhs(t, W, kG=1, A=1):
    """The autonomous Gompertz model."""

    dWdt = - kG * np.log(W / A) * W
    return dWdt


integrator = ode(lambda t, W: autonomous_rhs(t, W, **params))
integrator.set_integrator('vode', method='adams')

tlim_diff = tlim[1] - tlim[0]
dt = tlim_diff / 100

fig, ax = plt.subplots()

init_vals = get_spread(include_init_val, (0, Wlim[0]), (0, Wlim[1]),
                       NUM_SOLUTION_LINES)
for init_val in init_vals:
    integrator.set_initial_value(init_val[1], init_val[0])

    time_points, solut = integrate_two_ways(integrator, dt, max_len=tlim_diff,
                                            t_boundry=tlim, y_boundry=Wlim)

    is_include_init_val = np.allclose(init_val, include_init_val)
    color = "black" if is_include_init_val else "grey"

    ax.plot(time_points, solut, color=color)

ax.set_aspect((tlim[1] - tlim[0]) / (Wlim[1] - Wlim[0]))

plt.show()
