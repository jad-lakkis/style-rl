"""DrivingEnv: the portable driving environment (gymnasium).

Episode      training: the car is placed on the reference line at a random station with a random start speed (a fraction of the QSS ceiling), the scripted driver drives
             `settle_s` seconds (not part of the episode), then control passes to the agent for at most `max_episode_s` seconds. Evaluation: the same, but 200 m before the
             start line, and the agent drives `eval_laps` timed laps; lap 2 is the flying lap that is reported.
Action       3 numbers in [-1, 1]: steering (+-1 = +-steer_lock_rad, + = left), throttle, brake (pedal = clip((a + 0.5) / 1.5, 0, 1); a <= -0.5 releases). Both pedals act as commanded (no brake priority), as in the original environment and in Assetto Corsa.
Observation  159 numbers, the frozen contract in config.py:
               ego (9)     speed, v_x, v_y, yaw rate, a_x, a_y (body frame, from the last decision), actual steer angle, rear slip ratios RL, RR
               line (3)    lateral offset e to the reference line, heading error to it, position across the track (-1 right edge .. +1 left edge)
               preview     20 stations at d_k = 300 (k/20)^1.5 m ahead: reference line, left edge, right edge in the car frame (x forward, y left), divided by max(d_k, 10)
               last action (3), and the (ego, action) of the two decisions before
Blend        set_blend(w, fraction): in training the executed action is (1 - w) * the agent's action + w * the scripted driver's action (the driver aims at `fraction` of the QSS
             ceiling). w is annealed from ~1 to 0 for the 'pure pursuit at the start, pure RL at the end' variant; w = 0 is the pure agent. Evaluation always runs the agent alone.
             info['executed_action'] is the action that was executed; the replay buffer must store it (StagedSAC does).
Reward       time_pbrs (config.RewardConfig): progress - time - terminal penalty + potential shaping -|e|. Terminal failures: a wheel off the track, wrong way, non-finite state.
             A stall (speed < 3 m/s for 2 s) is a truncation. The time limit is a truncation.
"""
import math
from collections import deque
from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig, OBS_DIM
from . import randomize
from .demo import DemoDriver

EGO = 9
TERMINAL = ("off_track", "nonfinite", "wrong_way")


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class DrivingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, course, backend, cfg=EnvConfig(), evaluation=False):
        super().__init__()
        self.course, self.backend, self.cfg = course, backend, cfg
        if abs(backend.control_dt - cfg.dt_s) > 1e-9:
            raise ValueError("the backend's control step must equal EnvConfig.dt_s")
        o = cfg.obs
        k = np.arange(1, o.n_preview + 1)
        self.preview_d = o.preview_horizon_m * (k / o.n_preview) ** o.preview_exponent
        self.preview_scale = np.maximum(self.preview_d, o.preview_min_scale_m)[:, None, None]
        n = EGO + 3 + 6 * o.n_preview + 3 + (o.history - 1) * (EGO + 3)
        assert n == OBS_DIM, n
        self.observation_space = spaces.Box(-o.clip, o.clip, (n,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.demo = DemoDriver(course, cfg)
        self.teacher_w, self.teacher_f = 0.0, 0.85
        ns = randomize.NOISE_STD                       # observation noise in the units of the scaled inputs (used only when a randomised episode asks for it)
        self._noise_ego = np.array([ns["speed"] / o.speed_scale_mps, ns["v_x"] / o.speed_scale_mps, ns["v_y"] / o.vy_scale_mps, ns["yaw_rate"] / o.yaw_rate_scale_radps,
                                    ns["a_x"] / o.accel_scale_mps2, ns["a_y"] / o.accel_scale_mps2, ns["steer_actual"] / o.steer_scale_rad, ns["slip"] * o.slip_gain, ns["slip"] * o.slip_gain])
        self._noise_line = np.array([ns["e"] / o.e_scale_m, ns["heading"] / o.heading_scale_rad, ns["track_position"]])
        self._noise_preview = np.repeat(ns["preview_m"] / self.preview_scale[:, 0, 0], 6)
        self.dr = randomize.sample(randomize.RandomizeConfig(), None)
        self._pending = None
        self.evaluation = bool(evaluation)
        self._eval_default = self.evaluation
        self.state = None

    # --- controls ----------------------------------------------------------------------------------
    def set_blend(self, teacher_weight, fraction=None):
        self.teacher_w = float(np.clip(teacher_weight, 0.0, 1.0))
        if fraction is not None:
            self.teacher_f = float(fraction)

    def _pedal(self, a):
        return float(np.clip((a + 0.5) / 1.5, 0.0, 1.0)) if self.cfg.pedal_deadzone else float((a + 1.0) / 2.0)

    def _controls(self, action):
        return float(action[0]), self._pedal(action[1]), self._pedal(action[2])

    # --- geometry ------------------------------------------------------------------------------------
    def _geometry(self):
        st, c = self.state, self.course
        self.q = c.match_centre(st["x"], st["y"])
        self.s = float(self.q["progress_m"])
        _, self.e, h = c.match_ref(st["x"], st["y"])
        self.heading_err = wrap(st["yaw"] - h)

    def _inside(self):
        st, c, cfg = self.state, self.course, self.cfg
        return c.wheels_inside(st["x"], st["y"], st["yaw"], cfg.wheel_half_base_m, cfg.wheel_half_track_m)

    # --- observation ------------------------------------------------------------------------------------
    def _ego(self):
        st, o = self.state, self.cfg.obs
        cur = (st["vx"], st["vy"], st["yaw_rate"])
        if self._prev_kin is None:
            ax = ay = 0.0
        else:
            p = self._prev_kin
            vx, vy, r = (p[0] + cur[0]) / 2, (p[1] + cur[1]) / 2, (p[2] + cur[2]) / 2
            ax, ay = (cur[0] - p[0]) / self.cfg.dt_s - r * vy, (cur[1] - p[1]) / self.cfg.dt_s + r * vx
        self._next_kin = cur
        slip = lambda x: min(o.slip_clip, max(-o.slip_clip, o.slip_gain * x))
        return np.array([math.hypot(st["vx"], st["vy"]) / o.speed_scale_mps, st["vx"] / o.speed_scale_mps, st["vy"] / o.vy_scale_mps, st["yaw_rate"] / o.yaw_rate_scale_radps,
                         ax / o.accel_scale_mps2, ay / o.accel_scale_mps2, st["steer_actual"] / o.steer_scale_rad, slip(st["slip_rl"]), slip(st["slip_rr"])])

    def _push(self, ego):
        self._hist.appendleft(np.concatenate([ego, self.last_action]))
        self._prev_kin = self._next_kin

    def _observation(self):
        st, o = self.state, self.cfg.obs
        ego = self._ego()
        noise = self.dr["obs_noise"]
        if noise > 0.0:
            ego = ego + noise * self._noise_ego * self.np_random.standard_normal(EGO)
        pts = self.course.at(self.s + self.preview_d) - np.array([st["x"], st["y"]])          # (N, 3, 2): reference, left, right
        c, s = math.cos(st["yaw"]), math.sin(st["yaw"])
        local = np.stack([c * pts[..., 0] + s * pts[..., 1], -s * pts[..., 0] + c * pts[..., 1]], axis=-1)
        preview = (local / self.preview_scale).ravel()
        pos = self.course.track_position(st["x"], st["y"], self.s)
        line = np.array([self.e / o.e_scale_m, self.heading_err / o.heading_scale_rad, pos])
        if noise > 0.0:
            line = line + noise * self._noise_line * self.np_random.standard_normal(3)
            preview = preview + noise * self._noise_preview * self.np_random.standard_normal(preview.size)
        raw = np.concatenate([ego, line, preview, self.last_action, *self._hist])
        self._push(ego)
        return np.clip(raw, -o.clip, o.clip).astype(np.float32)

    # --- reset ----------------------------------------------------------------------------------------------
    def _sample_start(self, options):
        c, cfg = self.course, self.cfg
        if self.evaluation:
            s0, f = (-cfg.eval_run_in_m) % c.L, cfg.eval_speed_fraction
        else:
            s0, f = float(self.np_random.uniform(0.0, c.L)), float(self.np_random.uniform(*cfg.start_speed_fraction))
        s0 = float(options.get("start_s", s0)) % c.L
        if "start_speed" in options:
            return s0, f, float(options["start_speed"])
        window = f * float(c.qss_speed(s0)) * cfg.settle_s + cfg.settle_window_margin_m
        return s0, f, f * c.qss_min_speed(s0, window)

    def _start_and_settle(self, s0, v0):
        c, cfg = self.course, self.cfg
        x, y = c.ref_point(s0)
        self.state = st = self.backend.reset(float(x), float(y), float(c.ref_heading_at(s0)), float(v0))
        if not self._inside():
            return False
        self._geometry()
        self.last_action = np.array([0.0, -1.0, -1.0], dtype=np.float32)          # nothing pressed
        self._prev_kin = None
        placed = np.concatenate([self._ego(), self.last_action])
        self._hist = deque([placed] * (cfg.obs.history - 1), maxlen=cfg.obs.history - 1)
        self.demo.reset()
        for _ in range(cfg.decisions(cfg.settle_s)):
            self._push(self._ego())
            action = self.demo.act(st, self.s, v0)
            st = self.backend.step(*self._controls(action))
            self.state, self.last_action = st, action
            if not all(map(math.isfinite, st.values())):
                return False
            self._geometry()
            if not self._inside():
                return False
        return True

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = dict(options or {})
        self.evaluation = bool(options.get("evaluation", self._eval_default))
        rnd = self.cfg.randomize
        on = rnd.enabled and (not self.evaluation or self.cfg.randomize_eval)
        self.dr = randomize.sample(rnd if on else replace(rnd, enabled=False), self.np_random)
        self.backend.set_randomization(self.dr)
        for attempt in range(1, self.cfg.max_start_attempts + 1):
            s0, f, v0 = self._sample_start(options)
            if self._start_and_settle(s0, v0):
                break
            if self.evaluation or "start_s" in options:
                raise RuntimeError(f"the start at station {s0:.1f} m leaves the track during the settle phase")
        else:
            raise RuntimeError(f"no start settled on the track in {self.cfg.max_start_attempts} attempts")
        # hand-over: the episode counters start here
        self.t0 = self.state["time_s"]
        self.t, self._dist, self._stall_s, self.done = 0.0, 0.0, 0.0, False
        self._pending = self.last_action.copy()
        self._phi = -self.cfg.reward.shaping_k * abs(self.e)
        self.max_time_s = self.cfg.eval_max_s if self.evaluation else self.cfg.max_episode_s
        L = self.course.L
        if self.evaluation:
            d_line = float((L - self.s) % L)
            self._lines = [d_line + k * L for k in range(self.cfg.eval_laps + 1)]
            self._crossings = [0.0] if d_line == 0.0 else []
        else:
            self._lines, self._crossings = [], []
        obs = self._observation()
        info = {"start": {"station_m": s0, "fraction": f, "speed_mps": v0, "attempts": attempt}, "randomization": dict(self.dr), "e_ref": self.e, "speed": self.state["speed"], "evaluation": self.evaluation}
        return obs, info

    # --- step ---------------------------------------------------------------------------------------------------
    def step(self, action):
        if self.done:
            raise RuntimeError("the episode has ended; call reset()")
        cfg, rw, L = self.cfg, self.cfg.reward, self.course.L
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be 3 finite numbers")
        executed = action
        if self.teacher_w > 0.0 and not self.evaluation:
            a_t = self.demo.act(self.state, self.s, self.demo.lap_target(self.s, self.teacher_f, self.state["speed"]))
            executed = ((1.0 - self.teacher_w) * action + self.teacher_w * a_t).astype(np.float32)
        prev_s, d0, t0 = self.s, self._dist, self.t
        applied = executed
        if self.dr["action_delay"]:                       # the action reaches the car one decision late
            applied, self._pending = self._pending, executed
        st = self.backend.step(*self._controls(applied))
        self.state = st
        dt = float(cfg.dt_s)
        terminated = truncated = False
        cause = trunc_cause = None
        ds = 0.0
        if not all(map(math.isfinite, st.values())):
            cause = "nonfinite"
        else:
            self._geometry()
            ds = (self.s - prev_s + L / 2) % L - L / 2
            self._dist += ds
            self.t += dt
            if not self._inside():
                cause = "off_track"
            else:
                self._stall_s = self._stall_s + dt if st["speed"] < cfg.stall_speed_mps else 0.0
                if self._stall_s > cfg.stall_time_s:
                    truncated, trunc_cause = True, "stall"
                elif abs(self.heading_err) > cfg.wrong_way_rad:
                    cause = "wrong_way"
        self.last_action = executed
        if self.evaluation and cause != "nonfinite":
            j = len(self._crossings)
            if j < len(self._lines) and d0 < self._lines[j] <= self._dist:
                phi = (self._lines[j] - d0) / max(self._dist - d0, 1e-12)
                self._crossings.append(float(t0 + phi * (self.t - t0)))
            if cause is None and len(self._crossings) == len(self._lines):
                cause = "lap_complete"
                terminated = True
        r = rw.a * ds - rw.b * dt
        if cause in TERMINAL:
            terminated = True
            r -= rw.penalty
        phi_next = 0.0 if cause in TERMINAL else -rw.shaping_k * abs(self.e)
        r += rw.gamma * phi_next - self._phi
        self._phi = phi_next
        if not (terminated or truncated) and self.t >= self.max_time_s - 1e-9:
            truncated, trunc_cause = True, "time_limit"
        self.done = terminated or truncated
        obs = np.zeros(OBS_DIM, np.float32) if cause == "nonfinite" else self._observation()
        info = {"progress_m": self._dist, "time_s": self.t, "termination": cause if terminated else None, "truncation": trunc_cause,
                "e_ref": self.e if cause != "nonfinite" else math.nan, "heading_error": self.heading_err if cause != "nonfinite" else math.nan, "speed": st["speed"],
                "station_m": self.s, "executed_action": executed}
        if self.evaluation:
            c = self._crossings
            info["lap_times_s"] = [c[i + 1] - c[i] for i in range(len(c) - 1)]
            info["line_crossings_s"] = list(c)
        return obs, float(r), bool(terminated), bool(truncated), info
