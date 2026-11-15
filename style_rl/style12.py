"""The generalised 12-number style (driver_style, see style_rl/dstyle) measured on the lap an agent is driving, and the distance to a target style.

RunningStyle collects the pedals, the steering and the position on the road step by step and returns the 12 numbers of the lap so far. A number that cannot be measured yet
(no brake zone, no corner so far) keeps its starting guess, and the estimate moves from the starting guess to the measured value as the lap advances:
    estimate = start + f * (measured - start),   f = distance driven / lap length    (f = 1 at the end of the lap: the plain 12 numbers of the lap)
so the style distance of the estimate changes smoothly during the lap and its payments add up to the distance of the whole lap.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from style_rl.dstyle.features import KEYS, style_vector
from style_rl.dstyle.track import Track

ROOT = Path(__file__).resolve().parents[1]
STEER_DEG_PER_UNIT = 0.15 * 57.2958 * 13.4      # OCD steering command 1.0 = 0.15 rad at the road wheels; x the steering ratio of the driver model (13.4) = steering wheel degrees
MIN_SAMPLES = 50


class StyleTargets:
    """targets/style12.json: the 12 numbers of each driver, of the neutral driver, and the spread of every number between real driver-track cells (the unit of the distance)."""

    def __init__(self, path=None):
        tdir = Path(os.environ.get("STYLE_TARGETS") or ROOT / "targets")
        d = json.load(open(path or tdir / "style12.json"))
        self.keys = d["keys"]
        assert self.keys == KEYS
        self.spread = np.array([d["spread"][k] for k in KEYS], float)
        self.drivers = {n: np.array([v[k] for k in KEYS], float) for n, v in d["drivers"].items()}

    def distance(self, a, b, clip=None):
        """Mean absolute difference of the 12 numbers, each in units of its spread (1 = as far apart as two typical real driver-track cells).
        clip: no single number counts for more than this many spread units (a car that is thrown about early in training has numbers far outside anything real)."""
        e = np.abs(np.asarray(a, float) - np.asarray(b, float)) / self.spread
        return float(np.mean(e if clip is None else np.minimum(e, clip)))


class RunningStyle:
    def __init__(self, track_csv, start_guess):
        self.track = Track(track_csv)
        self.corners = self.track.find_corners()
        self.L = float(self.track.length_m)
        self.start = np.asarray(start_guess, float)
        self.reset()

    def reset(self):
        self.t, self.thr, self.brk, self.steer, self.s, self.lat, self.lw, self.rw = ([] for _ in range(8))

    def add(self, time_s, throttle, brake, steer_unit, progress_m, lateral_m, left_width_m, right_width_m):
        self.t.append(time_s); self.thr.append(throttle); self.brk.append(brake); self.steer.append(steer_unit * STEER_DEG_PER_UNIT)
        self.s.append(progress_m); self.lat.append(lateral_m); self.lw.append(left_width_m); self.rw.append(right_width_m)

    def measure(self, covered_m=None):
        """The 12 numbers measured on the samples so far (nan where there is nothing to measure). covered_m: distance driven; the steering corrections per km are scaled to it."""
        if len(self.t) < MIN_SAMPLES:
            return np.full(len(KEYS), np.nan)
        lap = pd.DataFrame({"time_s": self.t, "throttle": self.thr, "brake": self.brk, "steer_deg": self.steer})
        q = {"progress_m": np.asarray(self.s), "lateral_m": np.asarray(self.lat), "left_width_m": np.asarray(self.lw), "right_width_m": np.asarray(self.rw)}
        m = style_vector(lap, self.track, corners=self.corners, q=q)
        if covered_m is not None and covered_m > 50.0:
            m["S_reversals_5deg_per_km"] *= self.L / covered_m        # style_vector divides by the length of the whole lap
        return np.array([m[k] for k in KEYS], float)

    def estimate(self, covered_m):
        meas = self.measure(covered_m)
        meas = np.where(np.isfinite(meas), meas, self.start)
        f = float(np.clip(covered_m / self.L, 0.0, 1.0))
        return self.start + f * (meas - self.start), meas
