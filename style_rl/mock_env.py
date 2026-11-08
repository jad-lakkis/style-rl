"""A STAND-IN for the OCD environment, with the same interface. NOT OCD physics: a kinematic bicycle with the F317 power/braking/grip envelope (data/f317_envelope.npz).
Used to test the code on a machine without OCD (smoke tests, unit tests). Never report results from it as OCD results."""
from pathlib import Path

import numpy as np

from style_rl.track import MonzaTrack

ROOT = Path(__file__).resolve().parents[1]
G = 9.81


class MockOCDEnv:
    def __init__(self, track_path=None, dt=0.04):
        self.track = MonzaTrack(track_path or ROOT / "data/tracks/monza_geometry.csv")
        z = np.load(ROOT / "data/f317_envelope.npz")
        self._v, self._ax, self._bx, self._ay, self.kd = z["v"], z["ax"], z["bx"], z["ay"], float(z["kd"])
        self.control_dt, self.max_steering_rad, self.wheelbase = float(dt), 0.15, 2.8
        self.time = 0.0
        self.x = self.y = self.yaw = self.v = self.yaw_rate = 0.0

    def reset(self, start_idx=100, speed=0.0, pose=None):
        self.x, self.y = (float(a) for a in self.track.center[start_idx])
        self.yaw = float(self.track.query(self.x, self.y)["heading_rad"])
        if pose is not None:
            self.x, self.y, self.yaw = (float(a) for a in pose)
        self.v, self.yaw_rate, self.time = float(speed), 0.0, 0.0
        return self.observe()

    def observe(self):
        return {"time_s": self.time, "x": self.x, "y": self.y, "vx": self.v, "vy": 0.0, "speed": self.v, "yaw": self.yaw, "yaw_rate": self.yaw_rate, **self.track.query(self.x, self.y)}

    def step_normalized(self, steering, throttle, brake):
        steering = float(np.clip(steering, -1, 1)) * self.max_steering_rad
        throttle, brake = float(np.clip(throttle, 0, 1)), float(np.clip(brake, 0, 1))
        if brake > 0:
            throttle = 0.0
        h = 0.01
        for _ in range(int(round(self.control_dt / h))):
            a = throttle * np.interp(self.v, self._v, self._ax) - brake * G * np.interp(self.v, self._v, self._bx) - (0.0 if throttle > 0 else self.kd * self.v ** 2)
            self.v = max(self.v + a * h, 0.0)
            w = self.v * np.tan(steering) / self.wheelbase
            w_lim = G * np.interp(self.v, self._v, self._ay) / max(self.v, 1.0)
            self.yaw_rate = float(np.clip(w, -w_lim, w_lim))
            self.yaw += self.yaw_rate * h
            self.x += self.v * np.cos(self.yaw) * h
            self.y += self.v * np.sin(self.yaw) * h
            self.time += h
        return self.observe()
