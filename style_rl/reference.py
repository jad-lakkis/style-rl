"""The reference driver: a simulated driver with a known style, on Monza, as a lap target (line, speed profile, pedals) plus its per-corner style numbers."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from style_rl.style import FEATURES, corner_features

ROOT = Path(__file__).resolve().parents[1]
DRIVERS = ["attacker", "smooth"]


class Reference:
    def __init__(self, driver, root=ROOT):
        tdir = Path(os.environ.get("STYLE_TARGETS") or Path(root) / "targets")   # STYLE_TARGETS: use another set of reference laps (e.g. a safety-margin variant)
        z = np.load(tdir / f"{driver}_monza.npz")
        self.driver = driver
        self.s, self.n, self.xy, self.v, self.thr, self.brk = z["s"], z["n"], z["xy"], z["v"], z["thr"], z["brk"]
        self.t = z["t"]
        self.length = float(z["length"])
        self.ds = float(self.s[1] - self.s[0])
        self.N = len(self.s)
        info = json.loads((tdir / "manifest.json").read_text())[driver]
        self.lap_time = float(info["lap_time_f317_model_s"])
        self.description = info["description"]
        self.knobs = info["knobs"]
        # heading and curvature of the reference line
        d = np.gradient(self.xy, axis=0)
        self.heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        self.curv = gaussian_filter1d(np.gradient(self.heading) / self.ds, 3.0, mode="wrap")
        corners = pd.read_csv(Path(root) / "data" / "monza_corners.csv")
        self.corners = corners[corners.angle_deg > 45].reset_index(drop=True)
        # style numbers of the reference at every corner where it brakes
        self.corner_ref = {}
        for c in self.corners.itertuples():
            f = corner_features(self.s, self.brk, self.thr, c.s_apex, self.length)
            if f[2] > 0.1:                                   # peak brake: the reference really brakes here
                self.corner_ref[int(c.corner)] = f

    def at(self, arr, progress):
        i = (progress / self.ds) % self.N
        i0 = int(np.floor(i))
        f = i - i0
        return arr[i0 % self.N] * (1 - f) + arr[(i0 + 1) % self.N] * f

    def at_angle(self, arr, progress):
        a0, a1 = arr[int(np.floor(progress / self.ds)) % self.N], arr[(int(np.floor(progress / self.ds)) + 1) % self.N]
        f = (progress / self.ds) % 1.0
        return a0 * (1 - f) + a1 * f

    def next_corner(self, progress):
        d = (self.corners.s_apex.to_numpy() - progress) % self.length
        k = int(np.argmin(d))
        return float(d[k]), float(self.corners.angle_deg.iloc[k])
