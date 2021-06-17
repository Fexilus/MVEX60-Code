"""Plot the symmetries of an equilateral triangle."""
import os.path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def plot(save_path=None, file_name="triangles.pdf", vertex_labels=None):

    plt.rc("text", usetex=True)

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

    arc2 = FancyArrowPatch(triangle2_vertices[1], triangle2_vertices[0],
                           arrowstyle="->", mutation_scale=10,
                           connectionstyle="Angle3, angleA=180, angleB=120",
                           shrinkA=10, shrinkB=10)
    arc3 = FancyArrowPatch(triangle3_vertices[1], triangle3_vertices[2],
                           arrowstyle="->", mutation_scale=10,
                           connectionstyle="Angle3, angleA=180, angleB=240",
                           shrinkA=10, shrinkB=10)

    ax.add_patch(arc2)
    ax.add_patch(arc3)

    ax.text(*(triangle1_vertices[1] + np.array((0, 0.1))),
            "No rotation",
            horizontalalignment="center")
    ax.text(*(triangle2_vertices[1] + np.array((0, 0.1))),
            r"Rotation by $120^{\circ}$",
            horizontalalignment="center")
    ax.text(*(triangle3_vertices[1] + np.array((0, 0.1))),
            r"Rotation by $-120^{\circ}$",
            horizontalalignment="center")

    if vertex_labels:
        points_a = np.stack([triangle1_vertices[0, :],
                             triangle2_vertices[0, :],
                             triangle3_vertices[0, :]])
        points_b = np.stack([triangle1_vertices[1, :],
                             triangle2_vertices[1, :],
                             triangle3_vertices[1, :]])
        points_c = np.stack([triangle1_vertices[2, :],
                             triangle2_vertices[2, :],
                             triangle3_vertices[2, :]])

        for point in points_a:
            ax.text(*(point + (0.1, 0.05)), vertex_labels[0],
                    horizontalalignment="left")
        for point in points_b:
            ax.text(*(point + (0, -0.1)), vertex_labels[1],
                    horizontalalignment="center", verticalalignment="top")
        for point in points_c:
            ax.text(*(point + (-0.1, 0.05)), vertex_labels[2],
                    horizontalalignment="right")

    ax.axis("equal")
    ax.set(xlim=(0, 3.4), ylim=(0, np.sqrt(0.75)))
    ax.axis("off")

    fig.tight_layout()

    if save_path:
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path, format="pdf",
                    bbox_inches="tight")


if __name__ == "__main__":
    plot()
    plt.show()
