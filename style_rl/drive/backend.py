"""The car simulators behind the driving environment, one interface.

    backend.reset(x, y, yaw, speed) -> state
    backend.step(steer, throttle, brake) -> state          steer in [-1, 1] (= +-steer_lock_rad), pedals in [0, 1]
    state: dict with time_s, x, y, yaw, vx, vy (body frame), speed, yaw_rate, steer_actual (road-wheel angle, rad), slip_rl, slip_rr (rear slip ratios)

`mock` is a kinematic stand-in with the F317 limits, for tests on a machine without OCD. NOT OCD physics: never report its results as OCD results.
`ocd` is Open-Car-Dynamics through style_rl.ocd_env (Linux only). The steering actuator of OCD is a PT1 (T = 0.06 s) with a rate limit of 0.5 rad/s and a lock of 0.3 rad;
the actual steer angle is tracked here with the same law until it is read from the simulator state (TODO: verify against the OCD output on the pod).
The rear slip ratios are read from the OCD wheel speeds when the output has them (TODO: field names, verify on the pod); otherwise they are 0 and a warning is printed once.
"""
import warnings

import numpy as np

T_PT1_S, RATE_MAX_RADPS = 0.06, 0.5


class _Steer:
    """Shadow of OCD's steering actuator: PT1 plus a rate limit, angle in rad."""

    def __init__(self, dt, lock):
        self.dt, self.lock, self.angle = dt, lock, 0.0

    def reset(self, angle=0.0):
        self.angle = angle

    def step(self, command_norm):
        target = float(np.clip(command_norm, -1.0, 1.0)) * self.lock
        nxt = self.angle + (1.0 - np.exp(-self.dt / T_PT1_S)) * (target - self.angle)
        self.angle += float(np.clip(nxt - self.angle, -RATE_MAX_RADPS * self.dt, RATE_MAX_RADPS * self.dt))
        return self.angle


class MockBackend:
    def __init__(self, track_csv, steer_lock_rad=0.30, dt=0.04):
        from style_rl.mock_env import MockOCDEnv
        self.env = MockOCDEnv(track_csv, dt=dt)
        self.env.max_steering_rad = steer_lock_rad
        self.control_dt = float(self.env.control_dt)
        self._steer = _Steer(self.control_dt, steer_lock_rad)

    def _state(self, o):
        return {"time_s": o["time_s"], "x": o["x"], "y": o["y"], "yaw": o["yaw"], "vx": o["vx"], "vy": o["vy"], "speed": o["speed"], "yaw_rate": o["yaw_rate"],
                "steer_actual": self._steer.angle, "slip_rl": 0.0, "slip_rr": 0.0}

    def reset(self, x, y, yaw, speed):
        self._steer.reset()
        return self._state(self.env.reset(start_idx=0, speed=speed, pose=(x, y, yaw)))

    def step(self, steer, throttle, brake):
        self._steer.step(steer)
        return self._state(self.env.step_normalized(steer, throttle, brake))


class OCDBackend:
    _warned = False

    def __init__(self, track_csv, steer_lock_rad=0.30, dt=0.04):
        from style_rl.ocd_env import OCDMonzaEnv
        self.env = OCDMonzaEnv(track_path=track_csv, dt=dt)
        self.env.max_steering_rad = steer_lock_rad          # steering action +-1 = +-0.3 rad (OCD's actuator lock); the old wrapper default of 0.15 rad is NOT used here
        self.control_dt = float(self.env.control_dt)
        self.rolling_radius_m = float(self.env.params["vehicle_dynamics_double_track.tire.rolling_radius_m.rear"])
        self._steer = _Steer(self.control_dt, steer_lock_rad)

    def _slip(self):
        """Rear slip ratios (RL, RR) from the wheel speeds of the OCD output; (0, 0) if the output does not expose them."""
        out = self.env.vehicle.get_vehicle_model_output()
        for path in ("wheel_angular_velocity_radps", "wheel_speed_radps", "wheel_rotational_speed_radps"):
            w = getattr(getattr(out, "vehicle_dynamics_output", out), path, None)
            if w is not None:
                break
        if w is None:
            if not OCDBackend._warned:
                warnings.warn("OCD output has no wheel speed field under the known names: rear slip ratios are set to 0 (verify on the pod)")
                OCDBackend._warned = True
            return 0.0, 0.0
        v = max(float(self.env.observe()["speed"]), 1.0)
        rl, rr = float(getattr(w, "rear_left", getattr(w, "rl", 0.0))), float(getattr(w, "rear_right", getattr(w, "rr", 0.0)))
        return (rl * self.rolling_radius_m - v) / v, (rr * self.rolling_radius_m - v) / v

    def _state(self, o):
        sl, sr = self._slip()
        return {"time_s": o["time_s"], "x": o["x"], "y": o["y"], "yaw": o["yaw"], "vx": o["vx"], "vy": o["vy"], "speed": o["speed"], "yaw_rate": o["yaw_rate"],
                "steer_actual": self._steer.angle, "slip_rl": sl, "slip_rr": sr}

    def reset(self, x, y, yaw, speed):
        self._steer.reset()
        return self._state(self.env.reset(start_idx=0, speed=speed, pose=(x, y, yaw)))

    def step(self, steer, throttle, brake):
        self._steer.step(steer)
        return self._state(self.env.step_normalized(steer, throttle, brake))


def make_backend(name, track_csv, steer_lock_rad=0.30, dt=0.04):
    if name == "ocd":
        return OCDBackend(track_csv, steer_lock_rad, dt)
    if name == "mock":
        return MockBackend(track_csv, steer_lock_rad, dt)
    raise ValueError(f"unknown backend {name!r}")
