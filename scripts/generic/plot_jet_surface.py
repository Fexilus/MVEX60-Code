"""Plot an example of a jet surface corresponding to a differential equation"""
import numpy as np

import matplotlib.pyplot as plt

xlim = (-1, 1)
ylim = (-8, 8)
zlim = (-16, 8)

x_vec = np.linspace(*xlim, 100)
y_vec = np.linspace(*ylim, 100)

xs, ys = np.meshgrid(x_vec, y_vec)

zs = ys

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(9, 6))

ax.plot_surface(xs, ys, zs, antialiased=False)

for c in np.linspace(-8 * np.exp(1), 8 * np.exp(1), 14)[1:-2]:
    line_xs = x_vec
    line_ys = c * np.exp(x_vec)
    line_zs = line_ys

    mask = (line_ys > ylim[0]) & (line_ys < ylim[1])

    line_xs = line_xs[mask]
    line_ys = line_ys[mask]
    line_zs = line_zs[mask]

    ax.plot(line_xs, line_ys, line_zs, color="black", zorder=4)
    ax.plot(line_xs, line_ys, zlim[0], color="black")

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_zlim(*zlim)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$y'$")

ax.view_init(elev=12, azim=-68)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

fig.tight_layout()

plt.savefig("jet-surface.eps", format="eps", bbox_inches="tight")

plt.show()
