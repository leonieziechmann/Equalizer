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

    def interpolate(self, f: float, sample_rate: float = 44100) -> float:
        """
        Interpolates the gain value for a given normalized frequency f in [0.0, 1.0].
        Returns the computed gain as a linear multiplier.
        """
        if not self.points:
            return 1.0

        # Convert f (linear normalized to Nyquist) to frequency in Hz
        freq = f * (sample_rate / 2.0)

        # Convert frequency to log-spaced representation x in [0.0, 1.0]
        F_MIN = 20.0
        F_MAX = 20000.0
        if freq <= F_MIN:
            x = 0.0
        elif freq >= F_MAX:
            x = 1.0
        else:
            x = np.log10(freq / F_MIN) / np.log10(F_MAX / F_MIN)

        # Now interpolate y_raw (which is in [0.0, 1.0]) at x using the control points
        if x <= self.points[0][0]:
            y_raw = self.points[0][1]
        elif x >= self.points[-1][0]:
            y_raw = self.points[-1][1]
        else:
            # Find the interval [points[i], points[i+1]] containing x
            y_raw = self.points[-1][1]
            for i in range(len(self.points) - 1):
                x1, y1_raw = self.points[i]
                x2, y2_raw = self.points[i+1]
                if x1 <= x <= x2:
                    if x2 != x1:
                        t = (x - x1) / (x2 - x1)
                        y_raw = y1_raw + (y2_raw - y1_raw) * t
                    else:
                        y_raw = y1_raw
                    break

        # Convert interpolated y_raw to gain in dBFS
        # y_raw = 0.0 -> MAX_GAIN_DB (12.0)
        # y_raw = 1.0 -> MIN_GAIN_DB (-80.0)
        gain_db = (1.0 - y_raw) * 92.0 - 80.0

        # Convert dBFS to linear gain
        return float(10.0 ** (gain_db / 20.0))
