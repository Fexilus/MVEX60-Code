"""Plot the symmetries of an equilateral triangle."""
import os.path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Arc

from symmetries.visualize.arrowpath import WithArrowStroke


def plot(save_path=None, file_name="triangles.eps"):
    triangle1_vertices = np.array([[0, 0], [0.5, np.sqrt(0.75)], [1, 0]])
    triangle2_vertices = triangle1_vertices + np.array([1.2, 0])
    triangle3_vertices = triangle2_vertices + np.array([1.2, 0])

    fig, ax = plt.subplots(figsize=(6, 2))

    tri1 = plt.Polygon(triangle1_vertices, fill=False)
    tri2 = plt.Polygon(triangle2_vertices, fill=False)
    tri3 = plt.Polygon(triangle3_vertices, fill=False)

    ax.add_patch(tri1)
    ax.add_patch(tri2)
    ax.add_patch(tri3)

    points1 = np.stack([triangle1_vertices[0, :],
                        triangle2_vertices[2, :],
                        triangle3_vertices[1, :]])
    points2 = np.stack([triangle1_vertices[2, :],
                        triangle2_vertices[1, :],
                        triangle3_vertices[0, :]])
    points3 = np.stack([triangle1_vertices[1, :],
                        triangle2_vertices[0, :],
                        triangle3_vertices[2, :]])

    ax.scatter(points1[:, 0], points1[:, 1], marker="o", s=60, color="C0",
               zorder=3)
    ax.scatter(points2[:, 0], points2[:, 1], marker="s", s=60, color="C1",
               zorder=3)
    ax.scatter(points3[:, 0], points3[:, 1], marker="X", s=60, color="C2",
               zorder=3)

    arc2 = Arc(triangle2_vertices.sum(axis=0) / 3,
               width=np.sqrt(0.75) / 2,
               height=np.sqrt(0.75) / 2,
               theta1=5 * 30,
               theta2=9 * 30,
               path_effects=[WithArrowStroke(spacing=70.1, scaling=7)])
    arc3 = Arc(triangle3_vertices.sum(axis=0) / 3,
               width=np.sqrt(0.75) / 2,
               height=np.sqrt(0.75) / 2,
               theta1=5 * 30,
               theta2=13 * 30,
               path_effects=[WithArrowStroke(spacing=150.2, scaling=7)])

    ax.add_patch(arc2)
    ax.add_patch(arc3)

    ax.axis("equal")
    ax.axis("off")

    if save_path:
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path, format="eps",
                    bbox_inches="tight")


if __name__ == "__main__":
    plot()
    plt.show()
