"""The scripted driver (the teacher): pure pursuit on the reference line plus a speed PI on the pedals, with traction control.

It does three jobs: it settles the car before every episode (a few seconds at the start speed), it drives the demonstration laps the network first copies, and it is the base of
the comparison laps. It outputs ENV ACTIONS ([-1, 1]^3, the inverse of the env's pedal mapping), so the same object feeds the environment and the demonstration data.
Pure pursuit: steering = atan(2 L sin(alpha) / d) towards the reference-line point `look0 + look_gain * v` metres ahead.
Speed PI: pedal = kp * err + ki * integral(err dt), err = target speed - speed; positive = throttle, negative = brake (capped).
Traction control: throttle *= max(0, 1 - (rear slip - start) / span).
"""
import numpy as np


def pedal_to_action(pedal, deadzone=True, released=-0.8, full_at=1.0):
    """The env action that gives `pedal` (see config.action_from_pedal; full_at = EnvConfig.pedal_full_at)."""
    if deadzone:
        return -0.5 + (full_at + 0.5) * pedal if pedal > 0.0 else released
    return 2.0 * pedal - 1.0


class DemoDriver:
    """wheelbase, tc_start and tc_span default to the car's numbers in the EnvConfig (wheelbase_m, tc_start, tc_span)."""

    def __init__(self, course, cfg, kp=0.4, ki=0.2, integral_limit=5.0, brake_cap=1.0, look0=3.0, look_gain=0.35, wheelbase=None, tc_start=None, tc_span=None,
                 traction_control=True):
        self.course, self.cfg = course, cfg
        self.kp, self.ki, self.i_lim, self.brake_cap = kp, ki, integral_limit, brake_cap
        self.look0, self.look_gain = look0, look_gain
        self.wb = cfg.wheelbase_m if wheelbase is None else wheelbase
        self.tc_start = cfg.tc_start if tc_start is None else tc_start
        self.tc_span = cfg.tc_span if tc_span is None else tc_span
        self.tc = traction_control
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def act(self, state, station, v_target):
        v = float(state["speed"])
        look = self.look0 + self.look_gain * v
        tx, ty = self.course.ref_point(station + look)
        dx, dy = tx - state["x"], ty - state["y"]
        c, s = np.cos(state["yaw"]), np.sin(state["yaw"])
        alpha = np.arctan2(-s * dx + c * dy, c * dx + s * dy)
        delta = np.arctan2(2.0 * self.wb * np.sin(alpha), max(np.hypot(dx, dy), 1.0))
        steer = float(np.clip(delta / self.cfg.steer_lock_rad, -1.0, 1.0))
        err = float(v_target) - v
        self.integral = float(np.clip(self.integral + err * self.cfg.dt_s, -self.i_lim, self.i_lim))
        u = self.kp * err + self.ki * self.integral
        throttle, brake = float(np.clip(u, 0.0, 1.0)), float(np.clip(-u, 0.0, self.brake_cap))
        if self.tc and throttle > 0.0:
            slip = max(abs(state.get("slip_rl", 0.0)), abs(state.get("slip_rr", 0.0)))
            throttle *= max(0.0, 1.0 - (slip - self.tc_start) / self.tc_span)
        dz, full = self.cfg.pedal_deadzone, self.cfg.pedal_full_at
        return np.array([steer, pedal_to_action(throttle, dz, full_at=full), pedal_to_action(brake, dz, full_at=full)], dtype=np.float32)

    def lap_target(self, station, fraction, v):
        """Speed target on a lap: fraction of the QSS ceiling half a second ahead."""
        return fraction * float(self.course.qss_speed(station + 0.5 * v))
