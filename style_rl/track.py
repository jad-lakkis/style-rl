"""Independent track geometry adapter for a future Open-Car-Dynamics environment.

Requires numpy, pandas, scipy. Loads the Assetto Corsa-derived monza_geometry.csv.
Progress is measured in geometric metres, not Assetto Corsa LapDist metres.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class MonzaTrack:
    def __init__(self, csv_path="ocd_data/monza_geometry.csv", k_nearest=12):
        df = pd.read_csv(csv_path)
        columns = ["x_center", "y_center", "x_left", "y_left", "x_right", "y_right"]
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        c = df[["x_center", "y_center"]].to_numpy(float)
        l = df[["x_left", "y_left"]].to_numpy(float)
        r = df[["x_right", "y_right"]].to_numpy(float)
        if len(c) < 4 or not all(np.isfinite(a).all() for a in [c, l, r]):
            raise ValueError("Track requires at least four finite points.")

        if np.linalg.norm(c[0] - c[-1]) < 1e-4:
            c, l, r = c[:-1], l[:-1], r[:-1]

        self.center, self.left, self.right = c, l, r
        self.d = np.roll(c, -1, axis=0) - c
        self.lengths = np.linalg.norm(self.d, axis=1)
        if np.any(self.lengths < 1e-6):
            raise ValueError("Consecutive duplicate centerline points found.")
        self.s_vertices = np.r_[0, np.cumsum(self.lengths)]
        self.length_m = float(self.s_vertices[-1])
        self.k = max(1, int(k_nearest))
        self.vertex_tree = cKDTree(c)

        # Verify original left/right conventions using segment perpendiculars.
        tangents = self.d / self.lengths[:, None]
        normals_left = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        wleft = np.einsum("ij,ij->i", l - c, normals_left)
        wright = -np.einsum("ij,ij->i", r - c, normals_left)
        if np.median(wleft) <= 0 or np.median(wright) <= 0:
            raise ValueError(
                "Boundary ordering does not match driving direction; "
                "check centerline order and left/right labels."
            )

    def query(self, x, y):
        """Return track-relative measurements for one Cartesian position.

        Segment search uses nearest vertices and their adjacent segments.
        For unusually long segments or self-near intersections, validate matching.
        """
        point = np.array([float(x), float(y)])
        n = len(self.center)
        _, vertex_idx = self.vertex_tree.query(point, k=min(self.k, n))
        vertex_idx = np.atleast_1d(vertex_idx).astype(int)
        candidates = np.unique(np.r_[vertex_idx, (vertex_idx - 1) % n])
        starts = self.center[candidates]
        vecs = self.d[candidates]
        lens2 = self.lengths[candidates] ** 2
        u = np.clip(np.einsum("ij,ij->i", point - starts, vecs) / lens2, 0, 1)
        projections = starts + u[:, None] * vecs
        distances2 = np.einsum("ij,ij->i", projections - point, projections - point)
        best = int(np.argmin(distances2))
        j = int(candidates[best])
        alpha = float(u[best])
        foot = projections[best]

        tangent = self.d[j] / self.lengths[j]
        normal_left = np.array([-tangent[1], tangent[0]])
        left = (1 - alpha) * self.left[j] + alpha * self.left[(j + 1) % n]
        right = (1 - alpha) * self.right[j] + alpha * self.right[(j + 1) % n]
        left_width = float(np.dot(left - foot, normal_left))
        right_width = float(-np.dot(right - foot, normal_left))
        lateral = float(np.dot(point - foot, normal_left))
        width_total = left_width + right_width
        if left_width <= 0 or right_width <= 0:
            raise ValueError(f"Invalid interpolated widths at segment {j}")

        # Normalized across-track coordinate: 0=left, 1=right.
        normalized = (left_width - lateral) / width_total
        s = (self.s_vertices[j] + alpha * self.lengths[j]) % self.length_m
        return {
            "progress_m": float(s),
            "lateral_m": lateral,
            "lateral_normalized": float(normalized),
            "left_width_m": left_width,
            "right_width_m": right_width,
            "heading_rad": float(np.arctan2(tangent[1], tangent[0])),
            "inside_track": bool(-right_width <= lateral <= left_width),
            "center_x": float(foot[0]),
            "center_y": float(foot[1]),
            "distance_to_center_m": float(np.sqrt(distances2[best])),
        }

    def plot(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))
        for coords, name, ls in [
            (self.center, "Centerline", "--"),
            (self.left, "Left boundary", "-"),
            (self.right, "Right boundary", "-"),
        ]:
            loop = np.vstack([coords, coords[0]])
            ax.plot(loop[:, 0], loop[:, 1], ls, label=name, linewidth=1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Monza track — standalone environment geometry")
        ax.legend()
        ax.grid(alpha=0.3)
        return ax


if __name__ == "__main__":
    track = MonzaTrack()
    print(f"Track length (geometry): {track.length_m:.2f} m")
    point = track.center[100]
    print("Example centerline query:", track.query(*point))
    track.plot()
    import matplotlib.pyplot as plt
    plt.show()
