"""Observation builder for the Assetto Corsa side: the same 159 numbers as the training environment (env.py), computed from AC telemetry packets and a reference file.
Pure NumPy, no OCD, no torch; this is what runs next to AC. Our own implementation of the recipe of the AC runner in the RL-SetupOptimization-Agent repository (rlsetup/assetto/driving.py),
whose packet field names and frames it keeps, so the same plugin packets feed either.

Packet fields (one telemetry sample, SI units, the planar frame (x, y) = (AC z, AC x) as in that runner):
    world_position_x, world_position_y, yaw, timestamp_ac, velocity_z, velocity_x,
    tyre_slip_ratio_rl, tyre_slip_ratio_rr, tyreContactHeading_{FL,FR,RL,RR}_{x,z}
Derived exactly as in training: body velocity by rotating the world velocity with yaw; yaw rate from the last two yaw samples; a_x = dv_x/dt - r v_y, a_y = dv_y/dt + r v_x with r and v at the
midpoint; the steering angle = front minus rear tyre heading (circular means), minus the neutral baseline; line errors by the nearest point on the reference polyline;
preview from the reference arrays (stations = rows, spacing ds).

    geom = ACGeometry.from_course(course)  ->  geom.save("monza_reference.npz")          # once, on the training machine
    geom = ACGeometry.load("monza_reference.npz"); obs = ACObservation(geom, baseline_angle_rad)
    o = obs.build(packet, last_action)       # None for the first samples (it needs previous samples), then a (159,) float32 vector
"""
import math
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import ObsConfig, OBS_DIM


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def number(packet, key):
    v = packet.get(key) if hasattr(packet, "get") else None
    if v is None or not np.isfinite(v):
        raise ValueError(f"missing or non-finite telemetry field: {key}")
    return float(v)


class ACGeometry:
    """Reference line, left and right edge on a regular station grid (row i = station i * ds), in the world frame of the telemetry."""

    def __init__(self, ref_xy, left_xy, right_xy, ds):
        self.ref, self.left, self.right = (np.asarray(a, float) for a in (ref_xy, left_xy, right_xy))
        if not (self.ref.shape == self.left.shape == self.right.shape and self.ref.ndim == 2 and self.ref.shape[1] == 2 and len(self.ref) >= 4):
            raise ValueError("reference, left and right must be matching N x 2 arrays")
        if not (np.isfinite(self.ref).all() and np.isfinite(self.left).all() and np.isfinite(self.right).all()):
            raise ValueError("non-finite geometry")
        self.ds, self.N = float(ds), len(self.ref)
        self.L = self.ds * self.N
        self.seg = np.roll(self.ref, -1, 0) - self.ref
        self.pts = np.stack([self.ref, self.left, self.right], axis=1)
        o = ObsConfig()
        self.preview = o.preview_horizon_m * (np.arange(1, o.n_preview + 1) / o.n_preview) ** o.preview_exponent
        try:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.ref)
        except ImportError:                                  # NumPy-only machines: a brute-force search over the 1 m grid is fast enough at 25 Hz
            self._tree = None

    @classmethod
    def from_course(cls, course):
        return cls(course.ref_xy, course.left_xy, course.right_xy, course.ds)

    def save(self, path):
        np.savez(path, ref_xy=self.ref, left_xy=self.left, right_xy=self.right, ds=self.ds)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["ref_xy"], z["left_xy"], z["right_xy"], float(z["ds"]))

    def at(self, station):
        """(ref, left, right) points at stations (m) -> (..., 3, 2)."""
        i = np.mod(np.asarray(station, float), self.L) / self.ds
        i0 = np.floor(i).astype(int)
        f = (i - i0)[..., None, None]
        return self.pts[i0 % self.N] * (1 - f) + self.pts[(i0 + 1) % self.N] * f

    def project(self, position):
        """(station [m], e = signed lateral offset [m, + left of the direction of travel], heading of the reference there [rad]) by the nearest point on the polyline."""
        p = np.asarray(position, float)
        if self._tree is not None:
            _, i = self._tree.query(p)
        else:
            i = int(np.argmin(np.sum((self.ref - p) ** 2, axis=1)))
        best = None
        for j in ((int(i) - 1) % self.N, int(i)):
            a, v = self.ref[j], self.seg[j]
            u = float(np.clip(np.dot(p - a, v) / max(np.dot(v, v), 1e-12), 0.0, 1.0))
            foot = a + u * v
            d2 = float(np.sum((p - foot) ** 2))
            if best is None or d2 < best[0]:
                best = (d2, j, u, foot, v)
        _, j, u, foot, v = best
        t = v / max(np.linalg.norm(v), 1e-12)
        return (j + u) * self.ds, float(t[0] * (p[1] - foot[1]) - t[1] * (p[0] - foot[0])), float(np.arctan2(t[1], t[0]))

    def track_position(self, position, station):
        _, left, right = self.at(station)
        d = left - right
        return float(2.0 * np.dot(np.asarray(position, float) - right, d) / np.dot(d, d) - 1.0)


def road_angle(packet, baseline):
    """Front minus rear circular mean of the tyre contact headings (positive left), minus the neutral baseline."""
    def axle(names):
        ang = []
        for n in names:
            x, z = number(packet, f"tyreContactHeading_{n}_x"), number(packet, f"tyreContactHeading_{n}_z")
            if math.hypot(x, z) < 1e-6:
                raise ValueError("degenerate tyre heading")
            ang.append(math.atan2(x, z))
        s, c = sum(math.sin(a) for a in ang), sum(math.cos(a) for a in ang)
        if math.hypot(s, c) < 1e-6:
            raise ValueError("ambiguous tyre heading mean")
        return math.atan2(s, c)
    return float(wrap(axle(("FL", "FR")) - axle(("RL", "RR")) - baseline))


class ACObservation:
    """Builds the 159-number observation from successive telemetry packets. The first samples are used to start the finite differences and the history; build() returns None until
    the history holds two real previous decisions, then a (159,) float32 vector per call. Call it once per decision with the action that is being applied/was applied last."""

    def __init__(self, geometry, baseline_angle_rad=0.0, cfg=ObsConfig()):
        self.g, self.baseline, self.cfg = geometry, float(baseline_angle_rad), cfg
        self.history = deque(maxlen=cfg.history - 1)
        self.prev_yaw = None            # (timestamp, yaw)
        self.prev_kin = None

    def build(self, packet, last_action):
        o = self.cfg
        pos = np.array([number(packet, "world_position_x"), number(packet, "world_position_y")])
        yaw, stamp = number(packet, "yaw"), number(packet, "timestamp_ac")
        c, s = math.cos(yaw), math.sin(yaw)
        wz, wx = number(packet, "velocity_z"), number(packet, "velocity_x")
        vx, vy = c * wz + s * wx, -s * wz + c * wx
        if self.prev_yaw is None:
            self.prev_yaw = (stamp, yaw)
            return None
        dt = stamp - self.prev_yaw[0]
        if dt <= 0:
            raise ValueError("non-advancing timestamp")
        rate = wrap(yaw - self.prev_yaw[1]) / dt
        self.prev_yaw = (stamp, yaw)
        kin = np.array([vx, vy, rate])
        if self.prev_kin is None:
            self.prev_kin = kin
            return None
        mid = (kin + self.prev_kin) / 2
        ax, ay = (kin[:2] - self.prev_kin[:2]) / dt + np.array([-mid[2] * mid[1], mid[2] * mid[0]])
        self.prev_kin = kin
        clip = lambda x: float(np.clip(o.slip_gain * x, -o.slip_clip, o.slip_clip))
        ego = np.array([math.hypot(vx, vy) / o.speed_scale_mps, vx / o.speed_scale_mps, vy / o.vy_scale_mps, rate / o.yaw_rate_scale_radps, ax / o.accel_scale_mps2, ay / o.accel_scale_mps2,
                        road_angle(packet, self.baseline) / o.steer_scale_rad, clip(number(packet, "tyre_slip_ratio_rl")), clip(number(packet, "tyre_slip_ratio_rr"))])
        station, e, heading = self.g.project(pos)
        pts = self.g.at(station + self.g.preview) - pos
        local = np.stack([c * pts[..., 0] + s * pts[..., 1], -s * pts[..., 0] + c * pts[..., 1]], axis=-1)
        preview = (local / np.maximum(self.g.preview, o.preview_min_scale_m)[:, None, None]).ravel()
        result = None
        if len(self.history) == o.history - 1:
            raw = np.concatenate([ego, [e / o.e_scale_m, wrap(yaw - heading) / o.heading_scale_rad, self.g.track_position(pos, station)], preview, np.asarray(last_action, float), *self.history])
            if raw.shape != (OBS_DIM,) or not np.isfinite(raw).all():
                raise ValueError("invalid observation")
            result = np.clip(raw, -o.clip, o.clip).astype(np.float32)
        self.history.appendleft(np.r_[ego, np.asarray(last_action, float)])
        return result
