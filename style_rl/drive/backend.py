"""The car simulators behind the driving environment, one interface.

    backend.reset(x, y, yaw, speed) -> state
    backend.step(steer, throttle, brake) -> state          steer in [-1, 1] (= +-steer_lock_rad), pedals in [0, 1]
    state: dict with time_s, x, y, yaw, vx, vy (body frame), speed, yaw_rate, steer_actual (road-wheel angle, rad), slip_rl, slip_rr (rear slip ratios)

`mock` is a kinematic stand-in with the F317 limits, for tests on a machine without OCD. NOT OCD physics: never report its results as OCD results.
`ocd` is Open-Car-Dynamics through style_rl.ocd_env (Linux only; the Docker image of docker/ has it). From the OCD output (checked with scripts/ocd_inspect.py):
  steer_actual   the mean of steering_angle_per_wheel_rad.front_left / front_right (the actuator is a PT1, T = 0.06 s, rate limit 0.5 rad/s, lock 0.3 rad)
  slip_rl, rr    vehicle_dynamics_output.tire_longitudinal_slip_per_wheel.rear_left / rear_right: the tyre model's own slip ratio, the counterpart of Assetto Corsa's tyre slip ratio
The mock tracks the steering with the same PT1 law and has no slip.
`ocdlap` / `ocdlap:sf70h_f1[:setup]` is the F1 car of Ibrahim Khaled's ocd_lap package (his patched OCD build, F1 engine and gearbox), see OcdLapBackend.
"""
import math

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


class OcdLapBackend:
    """The F1 cars of Ibrahim Khaled's `ocd_lap` package (his private repository `ocd_first_lap`, branch sf70h-f1; we only READ it): his patched Open-Car-Dynamics build (`ocd_native`), the
    F1 engine with its automatic gearbox (python, at the 2 ms physics rate) and a setup of the car. The electric motor is off unless asked for.

    Where it is found: the folder named by the environment variable OCD_FIRST_LAP_DIR (or `repo`), a BUILD COPY of the repository: ocd_lap/, the built ocd_native module and
    vendor/Open-Car-Dynamics (the pinned commit with his patches); see runpod/setup_ocdlap.sh. None of his files are in this repository.
    Name: "ocdlap", "ocdlap:<car>", "ocdlap:<car>:<setup>" or with options "ocdlap:ac_sf70h?turbo=ac&ers=0".
      cars  sf70h_f1, sf70h_tum   his own (setups default, p1_TUM_f1)
            ac_sf70h              the Assetto Corsa Ferrari SF70H rebuilt from its data files (style_rl.accar): needs the replica json written by scripts/ac_car_to_ocd.py
                                  (path in AC_REPLICA_JSON, default data/ac_sf70h/replica.json); options turbo = ac | steady | linear (see accar/powertrain.py), ers = 0 | 1
    One decision = control_dt / 2 ms physics steps with the pedals held. No domain randomisation.

    Setup p1_TUM_f1 = the setup of his F1 runs (rlsetup/driving/setups.py of his public RL repository): the car's own default setup (heave springs on, brake bias 0.58) with the p1_TUM knobs
    roll_stiffness_front_share 0.4533, aero_front_share 0.4508, lsd_lock_drive 1.0, and heave_damping_ratio 0.3, a PLACEHOLDER value that cuts the 4.5 Hz porpoising of the car about tenfold
    (without heave damping the ride mode grows with speed and the car's settle collapses above about 75 m/s). The setup "default" is the car's own, undamped one.
    """

    SETUPS = {"default": {}, "p1_TUM_f1": {"roll_stiffness_front_share": 0.4533, "aero_front_share": 0.4508, "lsd_lock_drive": 1.0, "heave_damping_ratio": 0.3}}

    def __init__(self, track_csv, steer_lock_rad=0.30, dt=0.04, car="sf70h_f1", ers=False, repo=None, setup="p1_TUM_f1", options=None):
        import os
        import sys
        from pathlib import Path
        root = repo or os.environ.get("OCD_FIRST_LAP_DIR")
        if not root or not (Path(root) / "ocd_lap").is_dir():
            raise RuntimeError("OCD_FIRST_LAP_DIR must name the build copy of ocd_first_lap (with ocd_lap/, the built ocd_native module and vendor/Open-Car-Dynamics)")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from ocd_lap import cars
        from ocd_lap.backend import DT, Inputs, OCDBackend, StartPose
        self._inputs, self._pose = Inputs, StartPose
        opts = dict(options or {})
        self.car = car
        if car == "ac_sf70h":
            import json
            from style_rl.accar.ocdlap_glue import build
            path = opts.get("replica") or os.environ.get("AC_REPLICA_JSON") or str(Path(__file__).resolve().parents[2] / "data/ac_sf70h/replica.json")
            replica = json.loads(Path(path).read_text())
            self.backend, self.setup, self.spec = build(replica, turbo=opts.get("turbo", "ac"), ers=str(opts.get("ers", int(bool(ers)))).lower() in ("1", "true", "yes"))
            self.setup_name = "ac_default"
        else:
            self.spec = cars.get(car)
            if setup not in self.SETUPS:
                raise ValueError(f"unknown setup {setup!r}; choose from {sorted(self.SETUPS)}")
            self.setup_name = setup
            self.setup = cars.default_setup(car)
            if self.SETUPS[setup]:
                self.setup = self.setup.replace(**self.SETUPS[setup])
            self.backend = OCDBackend(powertrain="engine", car=car, ers=ers)
        if steer_lock_rad > self.spec.steer_max_rad + 1e-9:
            raise ValueError(f"steering lock {steer_lock_rad} rad is above the car's actuator limit {self.spec.steer_max_rad} rad")
        self.steer_lock = float(steer_lock_rad)
        self.control_dt = float(dt)
        self.n_sub = int(round(dt / DT))
        if self.n_sub < 1 or abs(self.n_sub * DT - dt) > 1e-9:
            raise ValueError(f"the control step {dt} s is not a whole number of {DT} s physics steps")
        self._t = 0.0

    def set_randomization(self, d):
        neutral = d["grip"] == d["brake"] == d["drive"] == d["aero"] == d["mass"] == d["slip_scale"] == 1.0 and d["steer_lag_s"] is None
        if not neutral:
            raise NotImplementedError("the F1 backend has no domain randomisation")

    def _state(self, s):
        return {"time_s": self._t, "x": s["x"], "y": s["y"], "yaw": s["yaw"], "vx": s["vx"], "vy": s["vy"], "speed": math.hypot(s["vx"], s["vy"]), "yaw_rate": s["yaw_rate"],
                "steer_actual": s["steer_actual"], "slip_rl": s["slip_rl"], "slip_rr": s["slip_rr"]}

    def reset(self, x, y, yaw, speed):
        self._t = 0.0
        return self._state(self.backend.reset(self.setup, self._pose(float(x), float(y), float(yaw), float(speed))))

    def step(self, steer, throttle, brake):
        clip = lambda v: min(max(float(v), 0.0), 1.0)
        inputs = self._inputs(float(steer) * self.steer_lock, clip(throttle), clip(brake))
        s = self.backend.step(inputs, steps=self.n_sub)
        self._t += self.control_dt
        return self._state(s)


def make_backend(name, track_csv, steer_lock_rad=0.30, dt=0.04):
    if name == "ocd":
        return OCDBackend(track_csv, steer_lock_rad, dt)
    if name == "mock":
        return MockBackend(track_csv, steer_lock_rad, dt)
    if name == "ocdlap" or name.startswith("ocdlap:"):
        head, _, query = name.partition("?")
        parts = head.split(":")
        options = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
        return OcdLapBackend(track_csv, steer_lock_rad, dt, car=parts[1] if len(parts) > 1 else "sf70h_f1", setup=parts[2] if len(parts) > 2 else "p1_TUM_f1", options=options)
    raise ValueError(f"unknown backend {name!r}")
