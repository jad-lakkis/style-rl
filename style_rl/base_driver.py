"""The base driver: follows the reference line and speed with steering, throttle and brake (pure pursuit + feed-forward pedals).

In residual mode the RL agent adds a small correction on top of it, so an untrained agent already drives the lap at about the reference pace and style; in pure mode it is not used.
The reference pedals are used as they are while the speed error stays inside +-1 m/s; outside the band a proportional correction is added, and well over the target speed the brake takes over.
"""
import numpy as np


class BaseDriver:
    def __init__(self, ref, wheelbase=2.8, max_steering_rad=0.15, kp_thr=0.18, kp_brk=0.30, preview_s=0.35, look0=3.0, look_gain=0.3, band=1.0):
        self.r, self.wb, self.dmax = ref, wheelbase, max_steering_rad
        self.kp_thr, self.kp_brk, self.preview, self.look0, self.look_gain, self.band = kp_thr, kp_brk, preview_s, look0, look_gain, band

    def control(self, obs):
        r = self.r
        v, prog = float(obs["speed"]), float(obs["progress_m"])
        look = self.look0 + self.look_gain * v
        px, py = r.at(r.xy[:, 0], prog + look), r.at(r.xy[:, 1], prog + look)
        dx, dy = px - obs["x"], py - obs["y"]
        alpha = np.arctan2(dy, dx) - obs["yaw"]
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        delta = np.arctan2(2.0 * self.wb * np.sin(alpha), max(np.hypot(dx, dy), 1.0))
        steering = float(np.clip(delta, -self.dmax, self.dmax) / self.dmax)

        v_t = r.at(r.v, prog + v * self.preview)
        ff_thr, ff_brk = r.at(r.thr, prog), r.at(r.brk, prog)
        e = v_t - v
        b = self.band
        if ff_brk > 0.02:
            corr = self.kp_brk * (-e - b) if e < -b else (-self.kp_brk * (e - b) if e > b else 0.0)
            throttle, brake = 0.0, float(np.clip(ff_brk + corr, 0.0, 1.0))
        else:
            core = ff_thr + (self.kp_thr * (e - b) if e > b else (self.kp_thr * (e + b) if e < -b else 0.0))
            throttle, brake = float(np.clip(core, 0.0, 1.0)), 0.0
            if e < -3.0:
                throttle, brake = 0.0, float(np.clip(self.kp_brk * (-e - 3.0), 0.0, 1.0))
        return steering, throttle, brake, float(v_t)
