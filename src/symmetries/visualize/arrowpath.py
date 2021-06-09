"""Path effect with arrows along lines."""
from matplotlib.patheffects import AbstractPathEffect, _subclass_with_normal
from matplotlib.path import Path
import numpy as np



class ArrowStroke(AbstractPathEffect):
    """A line-based PathEffect which draws a path with arrows.

    Much of the code is reused from the TickedStroke class in
    matplotlib.
    """

    def __init__(self, offset=(0, 0), spacing=10.0, scaling=4.0, **kwargs):
        """
        Parameters
        ----------
        offset : (float, float), default: (0, 0)
            The (x, y) offset to apply to the path, in points.
        spacing : float, default: 10.0
            The spacing between arrow bases, in points.
        scaling : float, default: 4.0
            The length of the arrows, in points.
        **kwargs
            Extra keywords are stored and passed through to
            AbstractPathEffect.
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

        # Set the offset from the begining of the line until first arrow
        spacing_begining = spacing_px / 2
        spacing_end = self._scaling

        width_scaling = self._scaling * 3.0 / 5.0

        # Transform before evaluation because to_polygons works at
        # a resolution of one -- assuming it is working in pixel space.
        transpath = affine.transform_path(tpath)

        # Evaluate path to straight line segments that can be used to
        # place arrows.
        polys = transpath.to_polygons(closed_only=False)

        for p in (p for p in polys if not np.any(np.isnan(p))):
            x = p[:, 0]
            y = p[:, 1]

            # Can not interpolate points or draw line if only one point 
            # in polyline.
            if x.size < 2:
                continue

            # Find distance between points on the line
            ds = np.hypot(x[1:] - x[:-1], y[1:] - y[:-1])

            # Build parametric coordinate along curve
            s = np.concatenate(([0.0], np.cumsum(ds)))
            s_total = s[-1]

            internal_space = s_total - spacing_begining - spacing_end
            num = int(np.floor(internal_space / spacing_px)) + 1

            if num > 0:
                # Pick parameter values for arrow bases.
                s_base = np.linspace(spacing_begining,
                                     spacing_begining + spacing_px * (num - 1),
                                     num)

                # Find points along the parameterized curve
                assert s_base[-1] <= s_total
                x_base = np.interp(s_base, s, x)
                y_base = np.interp(s_base, s, y)

                # Find unit vectors in local direction of curve
                # This is not optimal if the arrows are too large, as
                # they might go "out" from the curve
                delta_s = self._spacing * .001
                u = (np.interp(s_base + delta_s, s, x) - x_base) / delta_s
                v = (np.interp(s_base + delta_s, s, y) - y_base) / delta_s

                # Normalize slope into unit slope vector.
                n = np.hypot(u, v)
                mask = n == 0
                n[mask] = 1.0

                uv = np.array([u / n, v / n]).T
                uv[mask] = np.array([0, 0]).T

                # Build arrow verticies

                x_right = x_base + uv[:, 1] * width_scaling
                y_right = y_base - uv[:, 0] * width_scaling

                x_pointy = x_base + uv[:, 0] * self._scaling
                y_pointy = y_base + uv[:, 1] * self._scaling

                x_left = x_base - uv[:, 1] * width_scaling
                y_left = y_base + uv[:, 0] * width_scaling

                # Create vertex matrix
                xyt = np.empty((num, 4, 2), dtype=x_base.dtype)
                xyt[:, 0, 0] = x_right
                xyt[:, 1, 0] = x_pointy
                xyt[:, 2, 0] = x_left
                xyt[:, 3, 0] = x_right # Will be ignored
                xyt[:, 0, 1] = y_right
                xyt[:, 1, 1] = y_pointy
                xyt[:, 2, 1] = y_left
                xyt[:, 3, 1] = y_right # Will be ignored

                # TODO: This could at least be done per p
                tri_path = Path.make_compound_path_from_polys(xyt)

                # Transform back to data space during render
                renderer.draw_path(gc0, tri_path, affine.inverted() + trans,
                                   rgbFace=gc0._rgb)

        gc0.restore()

WithArrowStroke = _subclass_with_normal(effect_class=ArrowStroke)
