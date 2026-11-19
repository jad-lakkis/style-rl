"""The fastest line for the car's measured limits: lateral offsets that minimise the QSS lap time (instead of the minimum-curvature line the environment uses today).

    python tools/optimise_line.py --margin 1.5 --out data/lines/monza_f317_m1.5.csv [--envelope data/ocd_envelope.json] [--track data/tracks/monza_geometry.csv]

QSS = the quasi-steady-state lap of style_rl.drive.course.speed_profile (point mass, forward-backward speed profile under the car's lateral g, braking g and full-throttle acceleration against
speed, friction ellipse). The line is a periodic cubic spline through knots every `--knot-m` metres, kept `--margin` m inside both borders; L-BFGS-B on the QSS lap time with a finite-difference
gradient (a numba copy of speed_profile, checked against the original, makes one lap time cost about 1 ms) and a small penalty on the second difference of the offsets.
The QSS is a point mass: a line it likes can ask for more than the car does, so treat the gain as an estimate of what is on offer, not as a promise.
Writes OUT.csv (s_m, offset_m on the 2 m grid, + = left of the centre line) and prints the lap times.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from style_rl.drive.course import G, V_GRID, GridTrack, load_envelope, min_curvature_line, path_geometry, speed_profile  # noqa: E402


@njit(cache=True)
def _lin(arr, v):
    if v <= 0.0:
        return arr[0]
    if v >= 100.0:
        return arr[100]
    i = int(v)
    f = v - i
    return arr[i] * (1.0 - f) + arr[i + 1] * f


@njit(cache=True)
def _ellipse(ka, v, ay):
    a = max(9.81 * _lin(ay, v), 1e-6)
    r = ka * v * v / a
    return np.sqrt(min(max(1.0 - r * r, 0.0), 1.0))


@njit(cache=True)
def qss_numba(ka, seg, ay, bx, ax, xp, fp, v_cap):
    """The same forward-backward speed profile as course.speed_profile; returns the speeds and the lap time."""
    n = ka.shape[0]
    v = np.empty(n)
    for i in range(n):
        if ka[i] < xp[0]:
            v[i] = v_cap
        else:
            v[i] = min(np.interp(ka[i], xp, fp), v_cap)
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            nv = np.sqrt(v[j] * v[j] + 2.0 * seg[i] * 9.81 * _lin(bx, v[j]) * _ellipse(ka[j], v[j], ay))
            if nv < v[i]:
                v[i] = nv
    for _ in range(2):
        for i in range(n):
            j = (i + 1) % n
            nv = np.sqrt(v[i] * v[i] + 2.0 * seg[i] * max(_lin(ax, v[i]) * _ellipse(ka[i], v[i], ay), 0.0))
            if nv < v[j]:
                v[j] = nv
    t = 0.0
    for i in range(n):
        t += seg[i] / (0.5 * (v[i] + v[(i + 1) % n]))
    return v, t


class LineProblem:
    def __init__(self, track, env, margin_m, knot_m, smooth=1e-3):
        self.gt, self.env = track, env
        n = len(track.s)
        self.N, self.c = n, track.center_grid
        h = track.heading_grid
        self.normals = np.column_stack([-np.sin(h), np.cos(h)])
        wl = np.einsum("ij,ij->i", track.left_grid - self.c, self.normals)
        wr = -np.einsum("ij,ij->i", track.right_grid - self.c, self.normals)
        hi, lo = wl - margin_m, -(wr - margin_m)
        mid = 0.5 * (hi + lo)
        self.hi, self.lo = np.maximum(hi, mid), np.minimum(lo, mid)
        self.idx = np.arange(0, n, max(int(round(knot_m / track.ds)), 1))
        self.grid = np.arange(n)
        v = V_GRID[1:]
        cap = np.minimum.accumulate(G * env.ay(v) / v ** 2)
        self.args = (np.ascontiguousarray(env.lat_g, float), np.ascontiguousarray(env.brk_g, float), np.ascontiguousarray(env.acc_ms2, float), cap[::-1].copy(), v[::-1].copy(), float(env.v_cap))
        self.smooth = smooth

    def offsets(self, x):
        spl = CubicSpline(np.append(self.idx, self.N), np.append(x, x[0]), bc_type="periodic")
        return np.clip(spl(self.grid), self.lo, self.hi)

    def lap(self, offsets):
        path = self.c + self.normals * offsets[:, None]
        kappa, seg = path_geometry(path)
        v, t = qss_numba(np.abs(kappa), seg, *self.args)
        return t, v

    def objective(self, x):
        t, _ = self.lap(self.offsets(x))
        d2 = np.roll(x, -1) - 2 * x + np.roll(x, 1)
        return t + self.smooth * float(d2 @ d2)

    def gradient(self, x, h=0.03):
        f0 = self.objective(x)
        g = np.empty_like(x)
        for k in range(len(x)):
            xk = x.copy()
            xk[k] += h
            g[k] = (self.objective(xk) - f0) / h
        return g

    def optimise(self, x0, max_iter=150, verbose=True):
        bounds = list(zip(self.lo[self.idx], self.hi[self.idx]))
        trace = []

        def cb(xk):
            trace.append(self.objective(xk))
            if verbose and len(trace) % 10 == 0:
                print(f"  iteration {len(trace):3d}: {trace[-1]:.3f} s", flush=True)

        res = minimize(self.objective, x0, jac=self.gradient, method="L-BFGS-B", bounds=bounds, callback=cb, options={"maxiter": max_iter, "ftol": 1e-9, "maxcor": 30})
        return res.x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default=str(ROOT / "data/tracks/monza_geometry.csv"))
    ap.add_argument("--envelope", default=str(ROOT / "data/ocd_envelope.json"))
    ap.add_argument("--margin", type=float, default=1.5, help="the line stays this far (m) inside both borders")
    ap.add_argument("--knot-m", type=float, default=20.0)
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gt, env = GridTrack(a.track), load_envelope(a.envelope)
    prob = LineProblem(gt, env, a.margin, a.knot_m)
    n0, nrm = min_curvature_line(gt, a.margin)
    # the numba profile must give the original lap time on the original line
    t_ref = speed_profile(*path_geometry(gt.center_grid + nrm * n0[:, None]), env)[1]
    t_num, _ = prob.lap(n0)
    print(f"minimum-curvature line, {a.margin:g} m from the edges: QSS {t_ref:.3f} s (numba copy {t_num:.3f} s)", flush=True)
    assert abs(t_ref - t_num) < 0.02, "the numba profile does not reproduce the original"
    x0 = n0[prob.idx]
    t0 = time.time()
    x = prob.optimise(x0, a.max_iter)
    n_opt = prob.offsets(x)
    t_opt = speed_profile(*path_geometry(gt.center_grid + nrm * n_opt[:, None]), env)[1]
    print(f"fastest line, {a.margin:g} m from the edges: QSS {t_opt:.3f} s ({t_opt - t_ref:+.3f} s against the minimum-curvature line), {time.time() - t0:.0f} s of optimisation", flush=True)
    d = np.abs(np.diff(np.append(n_opt, n_opt[0])))
    print(f"largest offset change per {gt.ds:.0f} m: {d.max():.2f} m; offsets {n_opt.min():.2f} .. {n_opt.max():.2f} m; difference to the minimum-curvature line: mean {np.mean(np.abs(n_opt - n0)):.2f} m, max {np.max(np.abs(n_opt - n0)):.2f} m")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, np.column_stack([gt.s, n_opt]), delimiter=",", header="s_m,offset_m", comments="", fmt="%.4f")
    print(f"written {out}")


if __name__ == "__main__":
    main()
