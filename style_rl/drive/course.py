"""Course: everything the driving environment needs to know about one circuit, on a regular station grid (stations = arc length along the centre line).

  centre / left / right / reference line  on a 1 m grid (the preview samples these)
  reference line xi_ref                   the minimum-curvature line `margin_m` inside both borders (a bounded least-squares problem)
  QSS speed profile                       the fastest speed the car's measured limits allow along xi_ref (point mass, forward-backward, friction ellipse); the same
                                          model as environment/reference_lap.py of the research repository. It is a CEILING for speeds, optimistic by a few percent.
  corners                                 geometry-only corner windows (for corner-focused starts and corner rewards)

The envelope file is the measured OCD car (data/ocd_envelope.json: lateral g, braking g, acceleration against speed). Replace it with another envelope (for example the
Assetto Corsa car) and the same course gives that car's ceiling.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import diags, identity, vstack
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from style_rl.track import MonzaTrack

G = 9.81
V_GRID = np.arange(0.0, 101.0, 1.0)


# ------------------------------------------------------------------ track on a station grid
class GridTrack(MonzaTrack):
    def __init__(self, csv_path, k_nearest=12, ds=2.0, smooth_m=12.0):
        super().__init__(csv_path, k_nearest)
        self.name = Path(csv_path).stem.replace("_geometry", "")
        n_grid = int(round(self.length_m / ds))
        self.ds = self.length_m / n_grid
        self.s = np.arange(n_grid) * self.ds
        sv = self.s_vertices

        def periodic(arr):
            ext = np.vstack([arr, arr[:1]])
            return np.column_stack([np.interp(self.s, sv, ext[:, j]) for j in range(2)])

        self.center_grid, self.left_grid, self.right_grid = periodic(self.center), periodic(self.left), periodic(self.right)
        sm = gaussian_filter1d(self.center_grid, smooth_m / self.ds, axis=0, mode="wrap")
        d = np.roll(sm, -1, axis=0) - np.roll(sm, 1, axis=0)
        self.heading_grid = np.arctan2(d[:, 1], d[:, 0])
        dh = np.angle(np.exp(1j * (np.roll(self.heading_grid, -1) - np.roll(self.heading_grid, 1))))
        self.kappa = dh / (2 * self.ds)                       # 1/m, + = turning left
        self.width_grid = np.linalg.norm(self.left_grid - self.right_grid, axis=1)

    def find_corners(self, kappa_enter=1 / 400, kappa_peak=1 / 200, min_len_m=25.0, merge_gap_m=60.0, min_angle_deg=20.0):
        """Corner windows from curvature alone: runs of same-sign curvature above kappa_enter whose peak reaches kappa_peak; runs closer than merge_gap_m merge."""
        k, n = self.kappa, len(self.kappa)
        a, sign = np.abs(k), np.sign(k)
        shift = int(np.argmin(a))                              # start the scan on a straight so that no run wraps
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
            rows.append({"s_start": ((i0 + shift) % n) * self.ds, "s_apex": ((ia + shift) % n) * self.ds, "length_m": length, "angle_deg": angle,
                         "r_min_m": 1.0 / seg.max(), "direction": int(sg)})
        df = pd.DataFrame(rows).sort_values("s_apex").reset_index(drop=True)
        df.insert(0, "corner", np.arange(1, len(df) + 1))
        return df


# ------------------------------------------------------------------ car limits and the QSS speed profile
@dataclass
class Envelope:
    name: str
    lat_g: np.ndarray      # maximum lateral acceleration (g) on V_GRID
    brk_g: np.ndarray      # maximum braking deceleration (g, drag included)
    acc_ms2: np.ndarray    # maximum net forward acceleration at full throttle
    v_cap: float = 95.0

    def ay(self, v):
        return np.interp(v, V_GRID, self.lat_g)

    def bx(self, v):
        return np.interp(v, V_GRID, self.brk_g)

    def ax(self, v):
        return np.interp(v, V_GRID, self.acc_ms2)

    def v_lat_max(self, kappa_abs):
        v = V_GRID[1:]
        cap = np.minimum.accumulate(G * self.ay(v) / v ** 2)   # curvature capacity, decreasing in v
        out = np.interp(kappa_abs, cap[::-1], v[::-1])
        return np.where(kappa_abs < cap[-1], self.v_cap, np.minimum(out, self.v_cap))


def load_envelope(path, name="ocd"):
    j = json.load(open(path))
    return Envelope(name, np.array(j["lat_g"]), np.array(j["brk_g"]), np.array(j["acc_ms2"]), v_cap=float(j["v_cap"]))


def _d2(n):
    d = diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format="lil")
    d[0, n - 1] = 1
    d[n - 1, 0] = 1
    return d.tocsr()


def _bounded_lsq(A, b, lo, hi, max_iter=60, ridge=1e-9):
    """min 0.5 ||A n - b||^2 with lo <= n <= hi: active-set loop on the sparse normal equations."""
    H = (A.T @ A + ridge * identity(A.shape[1])).tocsc()
    g0 = A.T @ b
    n = spsolve(H, g0)
    at_lo, at_hi = np.zeros(len(n), bool), np.zeros(len(n), bool)
    for _ in range(max_iter):
        at_lo |= n < lo
        at_hi |= n > hi
        fixed = at_lo | at_hi
        x = np.where(at_lo, lo, np.where(at_hi, hi, n))
        free = ~fixed
        if free.any():
            rhs = g0[free] - H[free][:, fixed] @ x[fixed]
            x[free] = spsolve(H[free][:, free].tocsc(), rhs)
        grad = H @ x - g0
        release_lo, release_hi = at_lo & (grad < -1e-9), at_hi & (grad > 1e-9)
        viol = (x[free] < lo[free] - 1e-9).any() or (x[free] > hi[free] + 1e-9).any()
        n = x
        if not viol and not release_lo.any() and not release_hi.any():
            break
        at_lo &= ~release_lo
        at_hi &= ~release_hi
    return np.clip(n, lo, hi)


def min_curvature_line(track, margin_m=1.0, step=2):
    """Lateral offset n (m, + = left of the centre line) on the 2 m grid of the line that minimises the squared second differences, `margin_m` inside both borders."""
    N, c, h = len(track.s), track.center_grid, track.heading_grid
    nl = np.column_stack([-np.sin(h), np.cos(h)])
    wl = np.einsum("ij,ij->i", track.left_grid - c, nl)
    wr = -np.einsum("ij,ij->i", track.right_grid - c, nl)
    hi, lo = wl - margin_m, -(wr - margin_m)
    mid = 0.5 * (hi + lo)
    hi, lo = np.maximum(hi, mid), np.minimum(lo, mid)
    idx = np.arange(0, N, step)
    D = _d2(len(idx))
    A = vstack([D @ diags(nl[idx, 0]), D @ diags(nl[idx, 1])]).tocsr()
    b = -np.concatenate([D @ c[idx, 0], D @ c[idx, 1]])
    n_c = _bounded_lsq(A, b, lo[idx], hi[idx])
    spl = CubicSpline(np.append(idx, N), np.append(n_c, n_c[0]), bc_type="periodic")
    return np.clip(spl(np.arange(N)), lo, hi), nl


def path_geometry(path, sigma_cells=3.0):
    """Curvature (1/m, + = left) and segment lengths of a closed polyline on a regular index grid."""
    p = np.column_stack([gaussian_filter1d(path[:, j], 1.5, mode="wrap") for j in range(2)])
    d1 = (np.roll(p, -1, 0) - np.roll(p, 1, 0)) / 2
    d2 = np.roll(p, -1, 0) - 2 * p + np.roll(p, 1, 0)
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum((d1 ** 2).sum(1), 1e-9) ** 1.5
    return gaussian_filter1d(kappa, sigma_cells, mode="wrap"), np.linalg.norm(np.roll(path, -1, 0) - path, axis=1)


def _ellipse(kappa_abs, v, env):
    ay = np.maximum(G * env.ay(v), 1e-6)
    return np.sqrt(np.clip(1.0 - (kappa_abs * v ** 2 / ay) ** 2, 0.0, 1.0))


def speed_profile(kappa, seg, env):
    """Forward-backward point-mass speed profile on a closed path under the speed-dependent g-g envelope (friction ellipse)."""
    N, ka = len(kappa), np.abs(kappa)
    v = env.v_lat_max(ka).copy()
    for _ in range(2):                                          # braking, backward, two laps so the wrap-around settles
        for i in range(N - 1, -1, -1):
            j = (i + 1) % N
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * seg[i] * G * env.bx(v[j]) * _ellipse(ka[j], v[j], env)))
    for _ in range(2):                                          # acceleration, forward
        for i in range(N):
            j = (i + 1) % N
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * seg[i] * max(env.ax(v[i]) * _ellipse(ka[i], v[i], env), 0.0)))
    t = seg / (0.5 * (v + np.roll(v, -1)))
    return v, float(t.sum())


# ------------------------------------------------------------------ the course
class Course:
    def __init__(self, track_csv, envelope_json, margin_m=1.0, grid_m=1.0):
        self.track = tr = GridTrack(track_csv)
        self.name, self.L = tr.name, tr.length_m
        self.envelope = load_envelope(envelope_json)
        n2, nl2 = min_curvature_line(tr, margin_m)
        path2 = tr.center_grid + nl2 * n2[:, None]
        kappa, seg = path_geometry(path2)
        v2, self.qss_lap_time_s = speed_profile(kappa, seg, self.envelope)
        # 1 m station grid (station i = arc length i * ds1 along the centre line)
        self.N = int(round(self.L / grid_m))
        self.ds = self.L / self.N
        s1 = np.arange(self.N) * self.ds

        def on_grid(arr):
            arr = np.asarray(arr, float)
            ext = np.concatenate([arr, arr[:1]], axis=0)
            x = np.append(tr.s, self.L)
            return np.column_stack([np.interp(s1, x, ext[:, j]) for j in range(ext.shape[1])]) if arr.ndim == 2 else np.interp(s1, x, ext)

        self.ref_xy, self.left_xy, self.right_xy, self.center_xy = on_grid(path2), on_grid(tr.left_grid), on_grid(tr.right_grid), on_grid(tr.center_grid)
        self.n_ref, self.v_qss = on_grid(n2), on_grid(v2)
        self._pts = np.stack([self.ref_xy, self.left_xy, self.right_xy], axis=1)          # (N, 3, 2)
        d = np.roll(self.ref_xy, -1, 0) - np.roll(self.ref_xy, 1, 0)
        self.ref_heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self.ref_heading = gaussian_filter1d(self.ref_heading, 3.0, mode="wrap")
        self._seg = np.roll(self.ref_xy, -1, 0) - self.ref_xy
        self._ref_tree = cKDTree(self.ref_xy)
        self.corners = tr.find_corners()

    # --- stations ------------------------------------------------------------------------------
    def wrap(self, s):
        return np.mod(s, self.L)

    def _interp(self, arr, s):
        i = np.asarray(self.wrap(s)) / self.ds
        i0 = np.floor(i).astype(int)
        f = (i - i0).reshape(i.shape + (1,) * (arr.ndim - 1))
        return arr[i0 % self.N] * (1 - f) + arr[(i0 + 1) % self.N] * f

    def at(self, s):
        """(ref, left, right) points at stations s (any shape) -> (..., 3, 2)."""
        return self._interp(self._pts, s)

    def ref_point(self, s):
        return self._interp(self.ref_xy, s)

    def ref_heading_at(self, s):
        h = self._interp(self.ref_heading[:, None], s)[..., 0]
        return h

    def qss_speed(self, s):
        return self._interp(self.v_qss[:, None], s)[..., 0]

    def qss_min_speed(self, s0, window_m):
        s = s0 + np.arange(0.0, window_m + self.ds, self.ds)
        return float(np.min(self.qss_speed(s)))

    # --- matching ---------------------------------------------------------------------------------
    def match_centre(self, x, y):
        """Station s (m), lateral offset (+ left), widths, inside flag and the centre line's heading at the car (MonzaTrack.query)."""
        return self.track.query(x, y)

    def match_ref(self, x, y):
        """Nearest point on the reference polyline: (station s, e = signed lateral offset [m, + left], heading of the line there [rad])."""
        p = np.array([x, y], float)
        _, i = self._ref_tree.query(p)
        best = None
        for j in ((i - 1) % self.N, i):
            a, v = self.ref_xy[j], self._seg[j]
            u = float(np.clip(np.dot(p - a, v) / max(np.dot(v, v), 1e-12), 0.0, 1.0))
            foot = a + u * v
            dist2 = float(np.sum((p - foot) ** 2))
            if best is None or dist2 < best[0]:
                best = (dist2, j, u, foot, v)
        _, j, u, foot, v = best
        tang = v / max(np.linalg.norm(v), 1e-12)
        e = float(tang[0] * (p[1] - foot[1]) - tang[1] * (p[0] - foot[0]))      # + left of the direction of travel
        return (j + u) * self.ds, e, float(np.arctan2(tang[1], tang[0]))

    def track_position(self, x, y, s):
        """-1 at the right edge ... +1 at the left edge, across the track at station s."""
        _, left, right = self.at(s)
        d = left - right
        p = np.array([x, y]) - right
        return float(2.0 * (p @ d) / (d @ d) - 1.0)

    def wheels_inside(self, x, y, yaw, half_base, half_track):
        """False if any wheel contact point lies outside the track edges."""
        q = self.track.query(x, y)
        h = q["heading_rad"]
        nl = np.array([-np.sin(h), np.cos(h)])
        c, s = np.cos(yaw), np.sin(yaw)
        for dx in (half_base, -half_base):
            for dy in (half_track, -half_track):
                w = np.array([c * dx - s * dy, s * dx + c * dy])
                lat = q["lateral_m"] + float(w @ nl)
                if lat < -q["right_width_m"] or lat > q["left_width_m"]:
                    return False
        return True
