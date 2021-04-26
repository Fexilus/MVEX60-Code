"""Plot the symmetries of an equilateral triangle."""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Arc

from symmetries.visualize.arrowpath import WithArrowStroke

triangle1_vertecies = np.array([[0, 0], [0.5, np.sqrt(0.75)], [1, 0]])
triangle2_vertecies = triangle1_vertecies + np.array([1.2, 0])
triangle3_vertecies = triangle2_vertecies + np.array([1.2, 0])

fig, ax = plt.subplots()

tri1 = plt.Polygon(triangle1_vertecies, fill=False)
tri2 = plt.Polygon(triangle2_vertecies, fill=False)
tri3 = plt.Polygon(triangle3_vertecies, fill=False)

ax.add_patch(tri1)
ax.add_patch(tri2)
ax.add_patch(tri3)

points1 = np.stack([triangle1_vertecies[0,:],
                    triangle2_vertecies[1,:],
                    triangle3_vertecies[2,:]])
points2 = np.stack([triangle1_vertecies[2,:],
                    triangle2_vertecies[0,:],
                    triangle3_vertecies[1,:]])
points3 = np.stack([triangle1_vertecies[1,:],
                    triangle2_vertecies[2,:],
                    triangle3_vertecies[0,:]])

ax.scatter(points1[:,0], points1[:,1], marker="o", color="black")
ax.scatter(points2[:,0], points2[:,1], marker="s", color="black")
ax.scatter(points3[:,0], points3[:,1], marker="X", color="black")

arc2 = Arc(triangle2_vertecies.sum(axis=0) / 3,
           width=np.sqrt(0.75) / 2,
           height=np.sqrt(0.75) / 2,
           theta1=5 * 30,
           theta2=9 * 30,
           path_effects=[WithArrowStroke()])
arc3 = Arc(triangle3_vertecies.sum(axis=0) / 3,
           width=np.sqrt(0.75) / 2,
           height=np.sqrt(0.75) / 2,
           theta1=5 * 30,
           theta2=13 * 30,
           path_effects=[WithArrowStroke()])

ax.add_patch(arc2)
ax.add_patch(arc3)

ax.axis("equal")
ax.axis("off")

plt.show()
