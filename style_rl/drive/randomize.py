"""Domain randomisation: every training episode runs on a slightly different car and with slightly different sensors, so that the policy does not learn the exact numbers of one OCD
parameter set. The point is Assetto Corsa: its car is not OCD's car (grip, brakes, power, aero and steering lag differ), its telemetry arrives at 25 Hz over UDP, and the observation
is partly DERIVED from it (accelerations from differenced velocities, yaw rate from heading differences), which is noisy.

Per episode (reset):  grip (tyre friction scale), brake torque, drive torque, aero (downforce and drag), mass, steering actuator time constant, slip-ratio scale, a one-decision action delay (with a probability).
Per decision:         observation noise on the derived quantities and on the line errors (scale `obs_noise`; 0 = none).
Evaluation episodes are not randomised unless asked for (`randomize_eval`), so lap times stay comparable.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RandomizeConfig:
    enabled: bool = False
    grip: tuple = (0.85, 1.15)            # tyre friction scale (LMUX and LMUY of every wheel)
    brake: tuple = (0.85, 1.15)           # brake torque scale
    drive: tuple = (0.92, 1.08)           # drive torque scale
    aero: tuple = (0.90, 1.10)            # downforce and drag scale
    mass: tuple = (0.97, 1.03)
    steer_lag_s: tuple = (0.04, 0.10)     # PT1 time constant of the steering actuator (nominal 0.06 s)
    slip_scale: tuple = (0.7, 1.3)        # scale of the slip ratios the agent sees (Assetto Corsa's tyre slip ratio is not necessarily defined like OCD's)
    action_delay_prob: float = 0.3        # probability that an episode runs with its actions applied one decision late
    obs_noise: float = 1.0                # scale of the observation noise


# raw standard deviations of the observation noise, in the units of the quantities (divided by the observation scales inside the environment)
NOISE_STD = {"speed": 0.05, "v_x": 0.05, "v_y": 0.05, "yaw_rate": 0.004, "a_x": 0.3, "a_y": 0.3, "steer_actual": 0.002, "slip": 0.005, "e": 0.03, "heading": 0.003,
             "track_position": 0.003, "preview_m": 0.02}


def sample(cfg, rng):
    """One episode's randomisation as a plain dict (nominal values when the config is disabled)."""
    if not cfg.enabled:
        return {"grip": 1.0, "brake": 1.0, "drive": 1.0, "aero": 1.0, "mass": 1.0, "steer_lag_s": None, "slip_scale": 1.0, "action_delay": False, "obs_noise": 0.0}
    u = lambda r: float(rng.uniform(*r))
    return {"grip": u(cfg.grip), "brake": u(cfg.brake), "drive": u(cfg.drive), "aero": u(cfg.aero), "mass": u(cfg.mass), "steer_lag_s": u(cfg.steer_lag_s), "slip_scale": u(cfg.slip_scale),
            "action_delay": bool(rng.uniform() < cfg.action_delay_prob), "obs_noise": float(cfg.obs_noise)}
