"""Track geometry: where a car is on the road, and where the corners are.

A track is a CSV with the centre line and the two borders (columns x_center, y_center, x_left, y_left, x_right, y_right; metres; in driving direction,
one row per vertex). Progress is measured in metres along the centre line. 2D only: no elevation, no banking.

  Track.project(xy)    for every car position: progress along the lap, offset from the centre line (+ = left), and the distance to the left and right border
  Track.find_corners() corners found from the curvature alone (no hand-drawn zones), so the same code works on every circuit
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


class Track:
    def __init__(self, csv_path, k_nearest=12, ds=2.0, smooth_m=12.0):
        df = pd.read_csv(csv_path)
        missing = {"x_center", "y_center", "x_left", "y_left", "x_right", "y_right"} - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        c = df[["x_center", "y_center"]].to_numpy(float)
        l = df[["x_left", "y_left"]].to_numpy(float)
        r = df[["x_right", "y_right"]].to_numpy(float)
        if len(c) < 4 or not all(np.isfinite(a).all() for a in (c, l, r)):
            raise ValueError("A track needs at least four finite points.")
        if np.linalg.norm(c[0] - c[-1]) < 1e-4:          # the file repeats the first point at the end
            c, l, r = c[:-1], l[:-1], r[:-1]
        self.name = Path(csv_path).stem.replace("_geometry", "")
        self.center, self.left, self.right = c, l, r
        self.d = np.roll(c, -1, axis=0) - c
        self.lengths = np.linalg.norm(self.d, axis=1)
        if np.any(self.lengths < 1e-6):
            raise ValueError("Consecutive duplicate centre-line points.")
        self.s_vertices = np.r_[0, np.cumsum(self.lengths)]
        self.length_m = float(self.s_vertices[-1])
        self.k = max(1, int(k_nearest))
        self.vertex_tree = cKDTree(c)

        tangents = self.d / self.lengths[:, None]
        normals_left = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        if np.median(np.einsum("ij,ij->i", l - c, normals_left)) <= 0 or np.median(-np.einsum("ij,ij->i", r - c, normals_left)) <= 0:
            raise ValueError("Border order does not match the driving direction (left and right swapped?).")

        # regular grid along the lap (every ~2 m): borders, heading and signed curvature (+ = left turn)
        n_grid = int(round(self.length_m / ds))
        self.ds = self.length_m / n_grid
        self.s = np.arange(n_grid) * self.ds

        def on_grid(arr):
            ext = np.vstack([arr, arr[:1]])
            return np.column_stack([np.interp(self.s, self.s_vertices, ext[:, j]) for j in range(2)])
        self.center_grid, self.left_grid, self.right_grid = on_grid(c), on_grid(l), on_grid(r)
        smoothed = gaussian_filter1d(self.center_grid, smooth_m / self.ds, axis=0, mode="wrap")
        step = np.roll(smoothed, -1, axis=0) - np.roll(smoothed, 1, axis=0)
        heading = np.arctan2(step[:, 1], step[:, 0])
        dh = np.angle(np.exp(1j * (np.roll(heading, -1) - np.roll(heading, 1))))
        self.heading_grid = heading
        self.kappa = dh / (2 * self.ds)
        self.width_grid = np.linalg.norm(self.left_grid - self.right_grid, axis=1)

    def project(self, xy, s_hint=None, window_m=150.0):
        """Place every car position on the road.

        xy: (m, 2) positions. s_hint (m,): rough progress of each position (for example the lap distance of the game); it restricts the match to
        track segments within +-window_m, so a position is never matched to a different part of the circuit that happens to be close.
        Returns a dict of arrays: progress_m, lateral_m (+ = left of the centre line), left_width_m and right_width_m (road width on each side at that point).
        """
        xy = np.asarray(xy, float)
        n = len(self.center)
        _, idx = self.vertex_tree.query(xy, k=min(self.k, n))
        cand = np.concatenate([idx, (idx - 1) % n], axis=1)
        starts, vecs = self.center[cand], self.d[cand]
        l2 = self.lengths[cand] ** 2
        u = np.clip(np.einsum("mkj,mkj->mk", xy[:, None, :] - starts, vecs) / l2, 0, 1)
        proj = starts + u[..., None] * vecs
        d2 = ((proj - xy[:, None, :]) ** 2).sum(-1)
        if s_hint is not None:
            gap = np.abs((self.s_vertices[cand] - np.asarray(s_hint)[:, None] + self.length_m / 2) % self.length_m - self.length_m / 2)
            d2 = np.where(gap <= window_m, d2, np.inf)
        rows = np.arange(len(xy))
        b = np.argmin(d2, axis=1)
        j, alpha, foot = cand[rows, b], u[rows, b], proj[rows, b]
        tangent = self.d[j] / self.lengths[j][:, None]
        nl = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        a = alpha[:, None]
        left = (1 - a) * self.left[j] + a * self.left[(j + 1) % n]
        right = (1 - a) * self.right[j] + a * self.right[(j + 1) % n]
        return {
            "progress_m": (self.s_vertices[j] + alpha * self.lengths[j]) % self.length_m,
            "lateral_m": np.einsum("mj,mj->m", xy - foot, nl),
            "left_width_m": np.einsum("mj,mj->m", left - foot, nl),
            "right_width_m": -np.einsum("mj,mj->m", right - foot, nl),
        }

    def find_corners(self, kappa_enter=1 / 400, kappa_peak=1 / 200, min_len_m=25.0, merge_gap_m=60.0, min_angle_deg=20.0):
        """Cut the circuit into corners from the curvature alone.

        A corner is a run of same-sign curvature above kappa_enter (radius under 400 m) whose tightest point reaches kappa_peak (radius under 200 m), at least
        min_len_m long and turning at least min_angle_deg. Runs of the same sign closer than merge_gap_m are one corner (a chicane stays two corners).
        Returns a DataFrame sorted by apex: corner, s_start, s_apex, s_end (metres), length_m, angle_deg, r_min_m, direction (+1 left, -1 right).
        """
        k, n = self.kappa, len(self.kappa)
        a, sign = np.abs(k), np.sign(k)
        shift = int(np.argmin(a))                         # start the scan on a straight so that no corner wraps around the start
        ar, sr = np.roll(a, -shift), np.roll(sign, -shift)
        runs, i = [], 0
        while i < n:
            if ar[i] > kappa_enter:
                j = i
                while j + 1 < n and ar[j + 1] > kappa_enter and sr[j + 1] == sr[i]:
                    j += 1
                runs.append([i, j, sr[i]])
                i = j + 1
            else:
                i += 1
        merged = []
        for r in runs:
            if merged and merged[-1][2] == r[2] and (r[0] - merged[-1][1]) * self.ds <= merge_gap_m:
                merged[-1][1] = r[1]
            else:
                merged.append(list(r))
        rows = []
        for i0, i1, sg in merged:
            seg = ar[i0:i1 + 1]
            length = (i1 - i0 + 1) * self.ds
            angle = np.degrees(np.sum(seg) * self.ds)
            if seg.max() < kappa_peak or length < min_len_m or angle < min_angle_deg:
                continue
            ia = i0 + int(np.argmax(seg))
            rows.append({"s_start": ((i0 + shift) % n) * self.ds, "s_apex": ((ia + shift) % n) * self.ds, "s_end": ((i0 + shift) % n) * self.ds + length,
                         "length_m": length, "angle_deg": angle, "r_min_m": 1.0 / seg.max(), "direction": int(sg)})
        df = pd.DataFrame(rows).sort_values("s_apex").reset_index(drop=True)
        df.insert(0, "corner", np.arange(1, len(df) + 1))
        return df


def load_track(path):
    return Track(path)
