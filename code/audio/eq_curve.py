import numpy as np

class EqCurve:
    """
    Represents the Equalizer curve and performs interpolation.
    Points are represented as (x, y) tuples, where:
      - x: frequency normalized to [0.0, 1.0]
      - y: original model gain in range [0.0, 1.0]. Note that in our GUI,
           a higher visual position corresponds to smaller y value (i.e. model gain is 1.0 - y).
    """
    def __init__(self, points=None):
        if points is None:
            # Default points from EqWindow
            self.points = [(0.3, 0.5), (0.3, 0.2)]
        else:
            self.points = sorted(list(points), key=lambda p: p[0])

    def interpolate(self, f: float) -> float:
        """
        Interpolates the gain value for a given normalized frequency f in [0.0, 1.0].
        Returns the computed gain.
        """
        if not self.points:
            return 1.0

        # If f is before the first point, return the gain corresponding to the first point
        if f <= self.points[0][0]:
            return 1.0 - self.points[0][1]

        # If f is after the last point, return the gain corresponding to the last point
        if f >= self.points[-1][0]:
            return 1.0 - self.points[-1][1]

        # Otherwise, find the interval [points[i], points[i+1]] containing f
        for i in range(len(self.points) - 1):
            x1, y1_raw = self.points[i]
            x2, y2_raw = self.points[i+1]
            if x1 <= f <= x2:
                y1 = 1.0 - y1_raw
                y2 = 1.0 - y2_raw
                if x2 != x1:
                    t = (f - x1) / (x2 - x1)
                    return y1 + (y2 - y1) * t
                else:
                    return y1

        return 1.0 - self.points[-1][1]
