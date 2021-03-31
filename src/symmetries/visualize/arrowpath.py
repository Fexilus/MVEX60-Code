from matplotlib.patheffects import AbstractPathEffect, _subclass_with_normal
from matplotlib.path import Path
import numpy as np



class ArrowStroke(AbstractPathEffect):
    """A line-based PathEffect which draws a path with arrows.

    Much of the code is reused from the TickedStroke class in matplotlib.
    """

    def __init__(self, offset=(0, 0), spacing=20.0, scaling=7.0, **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (0, 0)
            The (x, y) offset to apply to the path, in points.
        spacing : float, default: 10.0
            The spacing between ticks in points.
        **kwargs
            Extra keywords are stored and passed through to AbstractPathEffect.
        """
        super().__init__(offset)

        self._spacing = spacing
        self._scaling = scaling
        self._gc = kwargs


    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        """Draw the path with updated gc."""
        # Do not modify the input! Use copy instead.
        gc0 = renderer.new_gc()
        gc0.copy_properties(gc)

        gc0 = self._update_gc(gc0, self._gc)
        trans = affine + self._offset_transform(renderer)

        # Convert spacing parameter to pixels.
        spacing_px = renderer.points_to_pixels(self._spacing)

        width_scaling = self._scaling * 5.0 / 7.0

        # Transform before evaluation because to_polygons works at resolution
        # of one -- assuming it is working in pixel space.
        transpath = affine.transform_path(tpath)

        # Evaluate path to straight line segments that can be used to
        # construct line ticks.
        polys = transpath.to_polygons(closed_only=False)

        for p in polys:
            x = p[:, 0]
            y = p[:, 1]

            # Can not interpolate points or draw line if only one point in
            # polyline.
            if x.size < 2:
                continue

            # Find distance between points on the line
            ds = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])

            # Build parametric coordinate along curve
            s = np.concatenate(([0.0], np.cumsum(ds)))
            s_total = s[-1]

            num = int(np.ceil(s_total / spacing_px)) - 1
            # Pick parameter values for ticks.
            s_tick = np.linspace(spacing_px/2, s_total - spacing_px/2, num)

            # Find points along the parameterized curve
            x_tick = np.interp(s_tick, s, x)
            y_tick = np.interp(s_tick, s, y)

            # Find unit vectors in local direction of curve
            delta_s = self._spacing * .001
            u = (np.interp(s_tick + delta_s, s, x) - x_tick) / delta_s
            v = (np.interp(s_tick + delta_s, s, y) - y_tick) / delta_s

            # Normalize slope into unit slope vector.
            n = np.hypot(u, v)
            mask = n == 0
            n[mask] = 1.0

            uv = np.array([u / n, v / n]).T
            uv[mask] = np.array([0, 0]).T

            # Build arrow verticies

            x_right = x_tick + uv[:, 1] * width_scaling
            y_right = y_tick - uv[:, 0] * width_scaling

            x_pointy = x_tick + uv[:, 0] * self._scaling
            y_pointy = y_tick + uv[:, 1] * self._scaling

            x_left = x_tick - uv[:, 1] * width_scaling
            y_left = y_tick + uv[:, 0] * width_scaling

            # Interleave ticks to form Path vertices
            xyt = np.empty((num, 3, 2), dtype=x_tick.dtype)
            xyt[:, 0, 0] = x_right
            xyt[:, 1, 0] = x_pointy
            xyt[:, 2, 0] = x_left
            xyt[:, 0, 1] = y_right
            xyt[:, 1, 1] = y_pointy
            xyt[:, 2, 1] = y_left

            colors = np.repeat(np.repeat(np.array(gc0._rgb)[np.newaxis, np.newaxis, :], 3, axis=1), num, axis=0)

            # Transform back to data space during render
            renderer.draw_gouraud_triangles(gc0, xyt, colors, affine.inverted() + trans)

        gc0.restore()

WithArrowStroke = _subclass_with_normal(effect_class=ArrowStroke)
