"""Gymnasium environment: drive Monza in OCD with the style of a reference driver.

The agent sends three numbers per control step (0.04 s): steering, throttle, brake.
  residual mode (default)  the base driver (style_rl.base_driver: reference line + reference speed and pedals) already drives the lap; the agent adds a correction of
                           +-res_steer to the steering and +-res_pedal to the pedals. An untrained agent therefore drives at about the reference pace and style.
  pure mode                the agent's three numbers are the commands (steering, throttle, brake), no help.
Reward per step: progress along the lap, small penalties for leaving the track, drifting from the reference line, jerky actions and being much slower than the reference.
At the end of every corner where the reference brakes: a style term, minus the scaled difference between the agent's five corner numbers (style_rl.style) and the reference's.
Full-lap mode (evaluation): a bonus for completing the lap and a term for the lap time against the reference's.
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from style_rl.base_driver import BaseDriver
from style_rl.reference import Reference
from style_rl.style import POST, PRE, corner_features, style_error

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_W = dict(prog=0.02, off=5.0, off_end=20.0, line=0.01, act=0.05, slow=0.01, style=1.5, lap=50.0, time=5.0, stall=10.0)


def make_backend(name, track_csv=None):
    track_csv = track_csv or ROOT / "data/tracks/monza_geometry.csv"
    if name == "ocd":
        from style_rl.ocd_env import OCDMonzaEnv
        return OCDMonzaEnv(track_path=track_csv, dt=0.04)
    from style_rl.mock_env import MockOCDEnv
    return MockOCDEnv(track_csv, dt=0.04)


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class StyleRacingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, driver="attacker", backend="mock", mode="residual", episode_steps=700, full_lap=False, res_steer=0.35, res_pedal=0.35, weights=None, seed=0):
        super().__init__()
        self.ref = Reference(driver)
        self.env = make_backend(backend)
        self.mode, self.full_lap, self.episode_steps = mode, full_lap, int(episode_steps)
        self.res_steer, self.res_pedal = res_steer, res_pedal
        self.w = {**DEFAULT_W, **(weights or {})}
        self.base = BaseDriver(self.ref, max_steering_rad=self.env.max_steering_rad)
        self.dt = float(self.env.control_dt)
        self.L = float(self.ref.length)
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(27,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _observe(self, o):
        r, p, v = self.ref, o["progress_m"], o["speed"]
        bs, bt, bb, v_t = self.base.control(o)
        self._base = (bs, bt, bb, v_t)
        e_lat = o["lateral_m"] - r.at(r.n, p)
        head_err = wrap(o["yaw"] - r.at_angle(r.heading, p))
        slip = np.arctan2(o["vy"], max(o["vx"], 1.0))
        d_c, ang_c = r.next_corner(p)
        x = [v / 60, slip / 0.2, o["yaw_rate"] / 1.0, e_lat / 4, head_err / 0.5, float(o["inside_track"]), v_t / 60,
             r.at(r.v, p + 30) / 60, r.at(r.v, p + 80) / 60, r.at(r.v, p + 150) / 60, (v - v_t) / 10,
             r.at(r.curv, p + 10) * 100, r.at(r.curv, p + 30) * 100, r.at(r.curv, p + 60) * 100, r.at(r.curv, p + 120) * 100,
             self.last[0], self.last[1], self.last[2], r.at(r.thr, p), r.at(r.brk, p), bs, bt, bb,
             np.sin(2 * np.pi * p / self.L), np.cos(2 * np.pi * p / self.L), min(d_c / 300.0, 1.0), ang_c / 90.0]
        return np.clip(np.asarray(x, np.float32), -5.0, 5.0)

    def _apply(self, a):
        a = np.clip(np.asarray(a, float), -1.0, 1.0)
        bs, bt, bb, _ = self._base
        if self.mode == "residual":
            steer, thr, brk = np.clip(bs + self.res_steer * a[0], -1, 1), np.clip(bt + self.res_pedal * a[1], 0, 1), np.clip(bb + self.res_pedal * a[2], 0, 1)
        else:
            steer, thr, brk = a[0], np.clip(a[1], 0, 1), np.clip(a[2], 0, 1)
        if brk > 0.05:
            thr = 0.0
        return float(steer), float(thr), float(brk)

    def _corner_events(self):
        """Style reward for every corner whose window (330 m before to 250 m after the apex) has just been completed and was fully driven by this episode."""
        total, n = 0.0, 0
        P = np.asarray(self.log_p)
        for c, f_ref in self.ref.corner_ref.items():
            a = float(self.ref.corners[self.ref.corners.corner == c].s_apex.iloc[0])
            k = int(np.floor((self.prog_unw - POST - a) / self.L))
            apex = a + k * self.L
            if (c, k) in self.done_corners or apex - PRE < self.start or self.prog_unw < apex + POST:
                continue
            self.done_corners.add((c, k))
            m = (P >= apex - PRE) & (P <= apex + POST)
            f = corner_features(P[m], np.asarray(self.log_b)[m], np.asarray(self.log_t)[m], apex, self.L)
            err = style_error(f, f_ref)
            self.style_errors.append(err)
            self.corner_log.append({"corner": c, **dict(zip(["brake_start", "brake_end", "peak_brake", "mean_brake", "throttle_pt"], f)), "style_error": err})
            total += -self.w["style"] * min(err, 8.0)
            n += 1
        return total, n

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        p0 = 0.0 if self.full_lap else float(self.rng.uniform(0, self.L))
        v0 = float(self.ref.at(self.ref.v, p0)) * (1.0 if self.full_lap else float(self.rng.uniform(0.85, 1.0)))
        n_pts = len(self.env.track.center)
        o = self.env.reset(start_idx=int(p0 / (self.L / n_pts)) % n_pts, speed=v0)
        self.o = o
        self.prog_unw = self.start = float(o["progress_m"])
        self.last_p, self.t0 = self.prog_unw, float(o["time_s"])
        self.last = np.zeros(3)
        self.steps, self.off_steps, self.stall_steps = 0, 0, 0
        self.log_p, self.log_b, self.log_t, self.log_s, self.log_v = [], [], [], [], []
        self.done_corners, self.style_errors, self.corner_log = set(), [], []
        return self._observe(o), {}

    def step(self, action):
        steer, thr, brk = self._apply(action)
        prev = self.last.copy()
        o = self.env.step_normalized(steer, thr, brk)
        self.o = o
        self.steps += 1
        d = float(o["progress_m"]) - self.last_p
        if d < -self.L / 2:
            d += self.L
        elif d > self.L / 2:
            d -= self.L
        self.last_p = float(o["progress_m"])
        self.prog_unw += d
        self.last = np.array([steer, thr, brk])
        self.log_p.append(self.prog_unw)
        self.log_b.append(brk)
        self.log_t.append(thr)
        self.log_s.append(steer)
        self.log_v.append(o["speed"])
        w, r = self.w, 0.0
        r += w["prog"] * d
        e_lat = o["lateral_m"] - self.ref.at(self.ref.n, o["progress_m"])
        r -= w["line"] * min(e_lat ** 2, 25.0)
        r -= w["act"] * float(np.abs(self.last - prev).sum())
        v_ref = self.ref.at(self.ref.v, o["progress_m"])
        r -= w["slow"] * max(0.0, v_ref - 3.0 - o["speed"])
        terminated, truncated, info = False, False, {}
        if not o["inside_track"]:
            r -= w["off"]
            self.off_steps += 1
            if self.off_steps * self.dt > 1.0:
                r -= w["off_end"]
                terminated, info["end"] = True, "off_track"
        else:
            self.off_steps = 0
        self.stall_steps = self.stall_steps + 1 if o["speed"] < 2.0 else 0
        if self.stall_steps * self.dt > 3.0:
            r -= w["stall"]
            terminated, info["end"] = True, "stalled"
        rs, n = self._corner_events()
        r += rs
        if self.full_lap and self.prog_unw - self.start >= self.L:
            lap_time = float(o["time_s"]) - self.t0
            r += w["lap"] + w["time"] * (self.ref.lap_time - lap_time)
            terminated, info["end"], info["lap_time"] = True, "lap", lap_time
        if not terminated and ((not self.full_lap and self.steps >= self.episode_steps) or (self.full_lap and o["time_s"] - self.t0 > 300.0)):
            truncated, info["end"] = True, "time"
        info.update({"progress": self.prog_unw - self.start, "speed": o["speed"], "style_error": float(np.mean(self.style_errors)) if self.style_errors else np.nan})
        return self._observe(o), float(r), terminated, truncated, info
