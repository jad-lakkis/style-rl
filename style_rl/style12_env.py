"""Gymnasium environment: one FULL LAP of Monza in OCD, trained to match a target 12-number style while getting faster.

What the agent sees: the car state, the track's own (driver-independent) line ahead (the neutral line: minimum-curvature line, its speed profile and curvature, and the command a
pure-pursuit driver on that line would give), the TARGET 12 numbers, the agent's running 12 numbers of this lap and the lap fraction. It never sees the driver's own line or pedals.
What it does: steering, throttle, brake (pure mode). Two options change the action space (both off by default, so the existing runs are unchanged):
  pedal_axis  two numbers (steering, ONE pedal axis: above 0 throttle, below 0 brake) instead of three. With two pedals, each clipped at 0, half of every pedal's range does nothing and has no gradient.
  residual    (implies pedal_axis) the pure-pursuit base driver steers and sets the pedals, the agent adds a bounded correction: steering + res_steer x a[0], pedal axis + res_pedal x a[1].
              The car then drives the lap by itself (about the neutral lap) and the agent only learns what to change for the style and for speed.
What it is paid:
  time    every step: c_time x (what the neutral lap would need for the distance just driven - the time the step took). Over the lap this adds up to c_time x (neutral lap time - lap time):
          faster than the neutral lap is rewarded, slower is punished, and leaving the track gives no way out of it.
  style   every `check_every` steps: -lam x (change of D), D = distance of the running 12 numbers to the target (units: spread between real driver-track cells). The payments add up to
          -lam x (D at the end of the lap - D at the start), i.e. the style of the WHOLE lap, not of a sector.
          lam is a Lagrange multiplier: after every finished lap  lam <- clip(lam + eta x (D_end - d_max), 0, lam_max). While the lap's style is further from the target than the
          margin d_max the style weighs more, once it is inside, it weighs less and the agent may spend the room on speed.
  off     -off per step outside the track, the episode ends after 1 s outside (-off_end); a stall ends it too (-stall); a small penalty for jerky pedals.
Optional helpers for an agent that has to learn to drive from nothing (all off by default, so the runs without them are unchanged):
  line    -line_w x (distance of the car from the track's own line, a smooth penalty that grows linearly beyond line_delta metres) every step; fades to line_floor of its value over line_decay steps.
  prog    +prog_w per metre driven. Every finished lap covers the same distance, so it changes nothing for a finished lap; it only makes a crash cost the progress that was still to come.
  sdense  -style_dense x lam x (distance D of the 12 numbers measured so far to the target) x (share of the lap driven in this step): the style is paid all along the lap, not only through the change of D.
The lap library: every finished lap is returned in info["lap"] (time, position, pedals, ...) together with its lap time and 12 numbers.
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from style_rl.base_driver import BaseDriver
from style_rl.reference import Reference
from style_rl.style12 import RunningStyle, StyleTargets
from style_rl.style_env import BASE_KW, make_backend, wrap

ROOT = Path(__file__).resolve().parents[1]
# penalties: leaving the track costs about 45 in total (1 per step outside for 1 s + 20), a stall 80. The first version had off 5 / stall 10: stopping the car (-34) was cheaper than a crash (-175)
# and every agent learned to stall once the pull towards the pure-pursuit driver faded.
DEFAULT_W = dict(time=1.0, off=1.0, off_end=20.0, act=0.005, stall=80.0)


class Style12Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, driver="attacker", backend="mock", follow="neutral", weights=None, d_max=0.3, lam0=3.0, lam_eta=0.3, lam_max=30.0, check_every=25, rand_start=0.5, seed=0, targets=None, hide_base=False,
                 line_w=0.0, line_delta=1.0, line_cap=6.0, line_decay=0, line_floor=0.4, prog_w=0.0, style_dense=0.0, style_clip=4.0, start_speed_lo=0.85,
                 residual=False, res_steer=0.1, res_pedal=0.2, pedal_axis=False):
        super().__init__()
        self.residual = residual                    # True: the pure-pursuit driver steers and sets the pedals, the agent adds a bounded correction (+-res_steer on the steering, +-res_pedal on the pedal axis)
        self.res_steer, self.res_pedal = res_steer, res_pedal
        self.pedal_axis = bool(pedal_axis or residual)   # ONE pedal axis instead of throttle + brake: above 0 throttle, below 0 brake. (Two pedals clipped at 0 have a dead half where nothing happens and no gradient.)
        self.hide_base = hide_base      # True: the pure-pursuit command is not part of the observation (the agent has to learn to drive by itself; the track's line and speeds are still given)
        self.line_w, self.line_delta, self.line_cap, self.line_decay, self.line_floor = line_w, line_delta, line_cap, int(line_decay), line_floor
        self.prog_w, self.style_dense, self.style_clip, self.start_speed_lo = prog_w, style_dense, style_clip, start_speed_lo
        self.total_steps = 0            # steps taken by this environment so far (the line penalty fades with it); the training script sets it when a run is resumed
        self.comp_names = ("time", "style", "off", "jerk", "stall") + (("line",) if line_w > 0 else ()) + (("prog",) if prog_w > 0 else ()) + (("sdense",) if style_dense > 0 else ())
        self.driver_name = driver
        self.tg = targets or StyleTargets()
        self.target = self.tg.drivers[driver]
        self.prior = self.tg.drivers["neutral"]
        self.neutral = Reference("neutral")                    # the track's own line and speeds: what the agent sees
        self.driver_ref = Reference(driver)                    # the driver's lap in the simulator (pure pursuit): only for the comparison lap, never an input
        self.env = make_backend(backend)
        self.follow = follow
        ref_base = self.driver_ref if follow == "driver" else self.neutral
        self.base = BaseDriver(ref_base, max_steering_rad=self.env.max_steering_rad, **(BASE_KW.get(driver, {}) if follow == "driver" else {}))
        self.w = {**DEFAULT_W, **(weights or {})}
        self.d_max, self.lam, self.lam_eta, self.lam_max, self.check_every, self.rand_start = d_max, lam0, lam_eta, lam_max, int(check_every), rand_start
        self.dt = float(self.env.control_dt)
        self.L = float(self.neutral.length)
        self.rng = np.random.default_rng(seed)
        self.style = RunningStyle(Path(__file__).resolve().parents[1] / "data/tracks/monza_geometry.csv", self.prior)
        r = self.neutral
        self.t_lap = float(r.t[-1] + r.ds / max(r.v[-1], 1.0))
        self.target_z = np.clip((self.target - self.prior) / self.tg.spread, -5, 5).astype(np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(27 + 12 + 12 + 2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2 if self.pedal_axis else 3,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def t_ref(self, p):
        """Time the neutral lap needs to reach distance p (unwrapped), seconds."""
        k, x = divmod(p, self.L)
        return k * self.t_lap + float(np.interp(x, self.neutral.s, self.neutral.t))

    def _observe(self, o):
        r, p, v = self.neutral, o["progress_m"], o["speed"]
        bs, bt, bb, v_t = self.base.control(o)
        self._base = (bs, bt, bb, v_t)
        e_lat = o["lateral_m"] - r.at(r.n, p)
        head_err = wrap(o["yaw"] - r.at_angle(r.heading, p))
        slip = np.arctan2(o["vy"], max(o["vx"], 1.0))
        d_c, ang_c = r.next_corner(p)
        x = [v / 60, slip / 0.2, o["yaw_rate"] / 1.0, e_lat / 4, head_err / 0.5, float(o["inside_track"]), v_t / 60,
             r.at(r.v, p + 30) / 60, r.at(r.v, p + 80) / 60, r.at(r.v, p + 150) / 60, (v - v_t) / 10,
             r.at(r.curv, p + 10) * 100, r.at(r.curv, p + 30) * 100, r.at(r.curv, p + 60) * 100, r.at(r.curv, p + 120) * 100,
             self.last[0], self.last[1], self.last[2], r.at(r.thr, p), r.at(r.brk, p), *((0.0, 0.0, 0.0) if self.hide_base else (bs, bt, bb)),
             np.sin(2 * np.pi * p / self.L), np.cos(2 * np.pi * p / self.L), min(d_c / 300.0, 1.0), ang_c / 90.0]
        run_z = (self.est - self.prior) / self.tg.spread
        frac = np.clip((self.prog_unw - self.start_m) / self.L, 0.0, 1.0)
        return np.clip(np.concatenate([np.asarray(x, np.float32), self.target_z, run_z.astype(np.float32), [frac, self.D_prev / 2.0]]), -5.0, 5.0).astype(np.float32)

    def base_action(self):
        """The action that reproduces the pure-pursuit base driver in this environment's action space (residual: no correction)."""
        bs, bt, bb = self._base[:3]
        if self.residual:
            return np.zeros(2, np.float32)
        return np.array([bs, bt - bb] if self.pedal_axis else [bs, bt, bb], np.float32)

    def _apply(self, a):
        a = np.clip(np.asarray(a, float), -1.0, 1.0)
        if self.pedal_axis:
            if self.residual:
                bs, bt, bb = self._base[:3]
                steer, u = float(np.clip(bs + self.res_steer * a[0], -1.0, 1.0)), float(np.clip(bt - bb + self.res_pedal * a[1], -1.0, 1.0))
            else:
                steer, u = float(a[0]), float(a[1])
            thr, brk = max(u, 0.0), max(-u, 0.0)
        else:
            steer, thr, brk = float(a[0]), float(np.clip(a[1], 0, 1)), float(np.clip(a[2], 0, 1))
        if brk <= 0.05:
            brk = 0.0          # the simulator cuts the throttle for ANY brake value above 0: a pedal at 0.01 must not do that (and 0.05 is where the style numbers count the brake as on)
        else:
            thr = 0.0
        return steer, thr, brk

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        start = (options or {}).get("start")
        if start is None:
            start = float(self.rng.uniform(0, self.L)) if self.rng.random() < self.rand_start else 0.0
        p0 = float(start)
        r = self.neutral
        v0 = float(r.at(r.v, p0)) * (1.0 if p0 == 0.0 else float(self.rng.uniform(self.start_speed_lo, 1.0)))
        n_pts = len(self.env.track.center)
        pose = (float(r.at(r.xy[:, 0], p0)), float(r.at(r.xy[:, 1], p0)), float(r.at_angle(r.heading, p0)))
        o = self.env.reset(start_idx=int(p0 / (self.L / n_pts)) % n_pts, speed=v0, pose=pose)
        self.o = o
        self.prog_unw = self.start_m = float(o["progress_m"])
        self.last_p, self.t0 = self.prog_unw, float(o["time_s"])
        self.tref0 = self.t_ref(self.start_m)
        self.last = np.zeros(3)
        self.steps, self.off_steps, self.stall_steps = 0, 0, 0
        self.style.reset()
        self.est, self.meas = self.prior.copy(), self.prior.copy()
        self.D_prev = self.D0 = self.tg.distance(self.est, self.target)
        self.D_dense = self.tg.distance(self.meas, self.target, self.style_clip)
        self.log = {k: [] for k in ("t", "x", "y", "speed", "lateral", "steer", "throttle", "brake", "progress", "inside")}
        self.ep = {"off_steps": 0, "comp": dict.fromkeys(self.comp_names, 0.0)}
        return self._observe(o), {}

    def step(self, action):
        steer, thr, brk = self._apply(action)
        prev = self.last.copy()
        p_before = self.prog_unw
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
        covered = self.prog_unw - self.start_m
        lg = self.log
        for k, v in (("t", o["time_s"] - self.t0), ("x", o["x"]), ("y", o["y"]), ("speed", o["speed"]), ("lateral", o["lateral_m"]), ("steer", steer), ("throttle", thr), ("brake", brk), ("progress", covered), ("inside", o["inside_track"])):
            lg[k].append(v)
        self.style.add(o["time_s"], thr, brk, steer, o["progress_m"], o["lateral_m"], o["left_width_m"], o["right_width_m"])
        w = self.w
        c = dict.fromkeys(self.comp_names, 0.0)
        c["time"] = w["time"] * ((self.t_ref(self.prog_unw) - self.t_ref(p_before)) - self.dt)
        c["jerk"] = -w["act"] * float(np.abs(self.last - prev).sum())
        self.total_steps += 1
        dd = float(np.clip(d, -5.0, 8.0))
        if self.line_w > 0:
            fade = 1.0 if self.line_decay <= 0 else max(self.line_floor, 1.0 - self.total_steps / self.line_decay)
            e = min(abs(float(o["lateral_m"]) - float(self.neutral.at(self.neutral.n, o["progress_m"]))), self.line_cap)
            c["line"] = -self.line_w * fade * (e * e / (2.0 * self.line_delta) if e < self.line_delta else e - 0.5 * self.line_delta)
        if self.prog_w > 0:
            c["prog"] = self.prog_w * dd
        if self.style_dense > 0:
            c["sdense"] = -self.style_dense * self.lam * self.D_dense * dd / self.L
        terminated, truncated, info = False, False, {}
        if not o["inside_track"]:
            c["off"] -= w["off"]
            self.off_steps += 1
            self.ep["off_steps"] += 1
            if self.off_steps * self.dt > 1.0:
                c["off"] -= w["off_end"]
                terminated, info["end"] = True, "off_track"
        else:
            self.off_steps = 0
        self.stall_steps = self.stall_steps + 1 if o["speed"] < 2.0 else 0
        if self.stall_steps * self.dt > 3.0:
            c["stall"] = -w["stall"]
            terminated, info["end"] = True, "stalled"
        done_lap = covered >= self.L
        if not terminated and (self.steps % self.check_every == 0 or done_lap):
            self.est, self.meas = self.style.estimate(min(covered, self.L) if done_lap else covered)
            D = self.tg.distance(self.est, self.target)
            c["style"] = -self.lam * (D - self.D_prev)
            self.D_prev = D
            self.D_dense = self.tg.distance(self.meas, self.target, self.style_clip)
        if done_lap and not terminated:
            lap_time = float(o["time_s"]) - self.t0
            terminated, info["end"], info["lap_time"] = True, "lap", lap_time
            self.lam = float(np.clip(self.lam + self.lam_eta * (self.D_prev - self.d_max), 0.0, self.lam_max))     # every finished lap moves the multiplier
        if not terminated and o["time_s"] - self.t0 > 300.0:
            truncated, info["end"] = True, "time"
        r = float(sum(c.values()))
        for k, v in c.items():
            self.ep["comp"][k] += v
        if terminated or truncated:
            end = info.get("end", "?")
            info["ep_stats"] = {"end": end, "length": self.steps, "start_m": self.start_m, "progress_m": covered, "lap_time": info.get("lap_time", np.nan), "D_end": self.D_prev, "D_start": self.D0,
                                "lam": self.lam, "off_share": self.ep["off_steps"] / max(self.steps, 1), **{f"r_{k}": v for k, v in self.ep["comp"].items()}}
            if end == "lap":
                info["lap"] = {"arrays": {k: np.asarray(v, np.float32) for k, v in lg.items()}, "numbers": self.meas.copy(), "lap_time": info["lap_time"], "D": self.D_prev, "start_m": self.start_m}
        info.update({"progress": covered, "speed": o["speed"], "D": self.D_prev})
        return self._observe(o), r, terminated, truncated, info
