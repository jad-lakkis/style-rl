"""The car simulators behind the driving environment, one interface.

    backend.reset(x, y, yaw, speed) -> state
    backend.step(steer, throttle, brake) -> state          steer in [-1, 1] (= +-steer_lock_rad), pedals in [0, 1]
    state: dict with time_s, x, y, yaw, vx, vy (body frame), speed, yaw_rate, steer_actual (road-wheel angle, rad), slip_rl, slip_rr (rear slip ratios)

`mock` is a kinematic stand-in with the F317 limits, for tests on a machine without OCD. NOT OCD physics: never report its results as OCD results.
`ocd` is Open-Car-Dynamics through style_rl.ocd_env (Linux only; the Docker image of docker/ has it). From the OCD output (checked with scripts/ocd_inspect.py):
  steer_actual   the mean of steering_angle_per_wheel_rad.front_left / front_right (the actuator is a PT1, T = 0.06 s, rate limit 0.5 rad/s, lock 0.3 rad)
  slip_rl, rr    vehicle_dynamics_output.tire_longitudinal_slip_per_wheel.rear_left / rear_right: the tyre model's own slip ratio, the counterpart of Assetto Corsa's tyre slip ratio
The mock tracks the steering with the same PT1 law and has no slip.
"""
import numpy as np

T_PT1_S, RATE_MAX_RADPS = 0.06, 0.5
GRIP_KEYS = tuple(f"vehicle_dynamics_double_track.z.tire_model.{w}.{k}" for w in ("front_left", "front_right", "rear_left", "rear_right") for k in ("LMUX", "LMUY"))


class _Steer:
    """Shadow of OCD's steering actuator: PT1 plus a rate limit, angle in rad."""

    def __init__(self, dt, lock):
        self.dt, self.lock, self.angle, self.T = dt, lock, 0.0, T_PT1_S

    def reset(self, angle=0.0):
        self.angle = angle

    def step(self, command_norm):
        target = float(np.clip(command_norm, -1.0, 1.0)) * self.lock
        nxt = self.angle + (1.0 - np.exp(-self.dt / self.T)) * (target - self.angle)
        self.angle += float(np.clip(nxt - self.angle, -RATE_MAX_RADPS * self.dt, RATE_MAX_RADPS * self.dt))
        return self.angle


class MockBackend:
    def __init__(self, track_csv, steer_lock_rad=0.30, dt=0.04):
        from style_rl.mock_env import MockOCDEnv
        self.env = MockOCDEnv(track_csv, dt=dt)
        self.env.max_steering_rad = steer_lock_rad
        self.control_dt = float(self.env.control_dt)
        self._steer = _Steer(self.control_dt, steer_lock_rad)
        e = self.env
        self._nominal = (e._ax.copy(), e._bx.copy(), e._ay.copy(), e.kd)

    def set_randomization(self, d):
        """Domain randomisation of the stand-in car (see randomize.py)."""
        e, (ax, bx, ay, kd) = self.env, self._nominal
        e._ax, e._bx, e._ay, e.kd = ax * d["drive"] / d["mass"], bx * d["brake"] / d["mass"], ay * d["grip"], kd * d["aero"]
        self._steer.T = d["steer_lag_s"] or T_PT1_S

    def _state(self, o):
        return {"time_s": o["time_s"], "x": o["x"], "y": o["y"], "yaw": o["yaw"], "vx": o["vx"], "vy": o["vy"], "speed": o["speed"], "yaw_rate": o["yaw_rate"],
                "steer_actual": self._steer.angle, "slip_rl": 0.0, "slip_rr": 0.0}

    def reset(self, x, y, yaw, speed):
        self._steer.reset()
        return self._state(self.env.reset(start_idx=0, speed=speed, pose=(x, y, yaw)))

    def step(self, steer, throttle, brake):
        self._steer.step(steer)
        return self._state(self.env.step_normalized(steer, throttle, brake, brake_priority=False))


class OCDBackend:
    _warned = False

    def __init__(self, track_csv, steer_lock_rad=0.30, dt=0.04):
        from style_rl.ocd_env import OCDMonzaEnv
        self.env = OCDMonzaEnv(track_path=track_csv, dt=dt)
        self.env.max_steering_rad = steer_lock_rad          # steering action +-1 = +-0.3 rad (OCD's actuator lock); the old wrapper default of 0.15 rad is NOT used here
        self.control_dt = float(self.env.control_dt)
        self._nominal = dict(self.env.params)

    def set_randomization(self, d):
        """Domain randomisation of the OCD car (see randomize.py): takes effect at the next reset (the parameters build the vehicle); the torque scales act at once."""
        p = dict(self._nominal)
        for k in GRIP_KEYS:
            p[k] = self._nominal[k] * d["grip"]
        for k in ("vehicle_dynamics_double_track.aerodynamics.c_l", "vehicle_dynamics_double_track.aerodynamics.c_d"):
            p[k] = self._nominal[k] * d["aero"]
        p["vehicle_dynamics_double_track.mass_vehicle_kg"] = self._nominal["vehicle_dynamics_double_track.mass_vehicle_kg"] * d["mass"]
        if d["steer_lag_s"]:
            p["steering_actuator.T_PT1"] = d["steer_lag_s"]
        self.env.params = p
        self.env.drive_scale, self.env.brake_scale = d["drive"], d["brake"]

    def _wheels(self):
        """(rear-left slip ratio, rear-right slip ratio, actual road-wheel steering angle [rad]) from the OCD output."""
        out = self.env.vehicle.get_vehicle_model_output()
        slip = out.vehicle_dynamics_output.tire_longitudinal_slip_per_wheel
        ang = out.steering_angle_per_wheel_rad
        return float(slip.rear_left), float(slip.rear_right), 0.5 * (float(ang.front_left) + float(ang.front_right))

    def _state(self, o):
        sl, sr, steer = self._wheels()
        return {"time_s": o["time_s"], "x": o["x"], "y": o["y"], "yaw": o["yaw"], "vx": o["vx"], "vy": o["vy"], "speed": o["speed"], "yaw_rate": o["yaw_rate"],
                "steer_actual": steer, "slip_rl": sl, "slip_rr": sr}

    def reset(self, x, y, yaw, speed):
        return self._state(self.env.reset(start_idx=0, speed=speed, pose=(x, y, yaw)))

    def step(self, steer, throttle, brake):
        return self._state(self.env.step_normalized(steer, throttle, brake, brake_priority=False))


def make_backend(name, track_csv, steer_lock_rad=0.30, dt=0.04):
    if name == "ocd":
        return OCDBackend(track_csv, steer_lock_rad, dt)
    if name == "mock":
        return MockBackend(track_csv, steer_lock_rad, dt)
    raise ValueError(f"unknown backend {name!r}")
