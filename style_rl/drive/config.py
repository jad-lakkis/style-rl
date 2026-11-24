"""Settings of the portable driving environment (style_rl.drive).

The observation block is a FROZEN CONTRACT: the same layout and scales as the actor input of the Assetto Corsa runner in the RL-SetupOptimization-Agent repository
(rlsetup/assetto/actor.py, SourceObservation), 159 numbers. An actor trained here is then a drop-in for that runner, and any change of these numbers makes every trained actor useless.
Our own implementation of the design in that repository (reward, start-and-settle episodes, preview of the line and the edges); no code is copied.
"""
import math
from dataclasses import dataclass, field

from .randomize import RandomizeConfig

OBS_DIM = 159                  # 9 ego + 3 line errors + 20 x 6 preview + 3 last action + 2 x (9 ego + 3 action) history


@dataclass(frozen=True)
class ObsConfig:
    n_preview: int = 20                  # stations ahead on the centre line
    preview_horizon_m: float = 300.      # d_k = horizon * (k / N) ** exponent, k = 1..N
    preview_exponent: float = 1.5
    preview_min_scale_m: float = 10.     # station k is divided by max(d_k, this)
    history: int = 3                     # decisions in the ego + action stack, including the current one
    speed_scale_mps: float = 80.
    vy_scale_mps: float = 5.
    accel_scale_mps2: float = 30.
    yaw_rate_scale_radps: float = 1.
    steer_scale_rad: float = .6          # the actual steer angle is divided by this (the car's lock is 0.3, so the input stays in +-0.5)
    heading_scale_rad: float = .5
    slip_gain: float = 5.
    slip_clip: float = 5.
    e_scale_m: float = 5.
    clip: float = 10.


@dataclass(frozen=True)
class RewardConfig:
    """time_pbrs: per decision  a ds - b dt - [terminal failure] penalty  +  gamma Phi(s') - Phi(s),  Phi = -shaping_k |e|  (e = lateral offset to the reference line).
    Phi(s') = 0 after a terminal failure (off track, wrong way). A stall is a truncation, not a failure."""
    a: float = 1.                        # per metre of progress
    b: float = 1.                        # per second
    gamma: float = .992                  # also the shaping discount (it must be the learner's discount)
    penalty: float = 100.                # terminal failure
    shaping_k: float = 1.                # [1/m]
    # Smoothness controls (generic, the same for every driver; 0 = off = the reward of runs a1 and a2). Per decision the reward loses
    #   smooth_w * ( smooth_steer * |steering jerk|  +  smooth_straight * straightness * |steering|  +  smooth_pedal * (|throttle jerk| + |brake jerk|) )
    # jerk = second difference of the executed command over three decisions (a(t) - 2 a(t-1) + a(t-2)): a steady ramp costs nothing, a flick or a chatter costs a lot; the pedals are the applied
    # pedal values (0..1, after the dead zone), the steering is the action (+-1 = full lock). straightness = 1 where the reference line stays below straight_kappa for the next straight_ahead_m
    # metres, falling to 0 where it reaches it (so a turn-in is never penalised). The first two decisions of an episode are not charged (the history comes from the scripted driver).
    smooth_w: float = 0.
    smooth_steer: float = 1.
    smooth_straight: float = 0.5
    smooth_pedal: float = 0.25
    straight_kappa: float = 1. / 800.    # [1/m]; Monza's gentlest bend (Curva Grande) is about 1/430
    straight_ahead_m: float = 80.


@dataclass(frozen=True)
class EnvConfig:
    dt_s: float = 0.04                   # one decision = the backend's control step (25 Hz)
    steer_lock_rad: float = 0.30         # steering action +-1 = +-this road-wheel angle (OCD's actuator limit)
    pedal_deadzone: bool = True          # pedal = clip((a + 0.5) / (pedal_full_at + 0.5), 0, 1) for throttle and brake alike; released = a <= -0.5
    pedal_full_at: float = 1.0           # the action at which a pedal is fully pressed. 1.0 = the original mapping: full pedal only at a = 1, the very edge of the tanh range, which run a1 never reached (its
                                         # throttle sat at 0.82-0.89 on the straights, raw action at most 0.988). 0.7 = full pedal for every a >= 0.7, so the policy does not have to sit at the edge
    max_episode_s: float = 60.           # training episodes are cut here (random start, not a whole lap)
    settle_s: float = 2.                 # the scripted driver drives this long before the hand-over (not part of the episode)
    start_speed_fraction: tuple = (.3, .7)    # f ~ U: the start speed is f * (lowest QSS speed over the settle window)
    settle_window_margin_m: float = 50.
    max_start_attempts: int = 20
    stall_speed_mps: float = 3.
    stall_time_s: float = 2.
    wrong_way_rad: float = math.pi / 2
    eval_run_in_m: float = 200.          # evaluation starts this far before the line, at fraction eval_speed_fraction
    eval_speed_fraction: float = .5
    eval_laps: int = 2                   # timed laps; lap 2 is the flying lap that is reported
    eval_max_s: float = 900.
    wheel_half_base_m: float = 1.4       # wheel contact points for the off-track test (a wheel outside the track = off track, as in Assetto Corsa)
    wheel_half_track_m: float = 0.8
    margin_m: float = 1.5                # the reference line stays this far inside both borders (a wheel is 0.95 m from the car centre: 1.0 m leaves no room for tracking error)
    train_inset_m: float = 0.0           # training episodes only: the track counts this much narrower on both sides for the off-track test, a safety buffer (evaluation always uses the true edges)
    wheelbase_m: float = 2.8             # the scripted driver's pure-pursuit wheelbase (Dallara F317 2.8 m, SF70H 3.6 m)
    tc_start: float = 0.08               # the scripted driver's traction control: throttle *= max(0, 1 - (rear slip - tc_start) / tc_span); SF70H: 0.06 / 0.03 (the F1 tyre peaks at slip 0.075-0.08)
    tc_span: float = 0.12
    demo_throttle_gamma: float = 0.0         # the scripted driver plans a torque fraction u and presses the pedal that gives it: torque(p) = p (1 + B p^gamma) / (1 + B) with B = demo_throttle_boost (the AC turbo: gamma 5.75, B 3.4); gamma 0 = pedal = u
    demo_throttle_boost: float = 0.0
    demo_brake_cap: float = 1.0          # the scripted driver never presses the brake pedal harder than this (the F1 car with 15 kN m of brakes locks its wheels above about 0.3)
    ref_line: str = ""                   # csv of the reference line (s_m, offset_m on the 2 m grid, tools/optimise_line.py); "" = the minimum-curvature line margin_m inside the borders
    grid_m: float = 1.0
    randomize: RandomizeConfig = field(default_factory=RandomizeConfig)    # domain randomisation (see randomize.py); off by default
    randomize_eval: bool = False         # randomise evaluation episodes too (off: lap times stay comparable)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)

    def __post_init__(self):
        if not -0.5 < self.pedal_full_at <= 1.0:
            raise ValueError("pedal_full_at must be in (-0.5, 1]")
        if self.train_inset_m < 0.0:
            raise ValueError("train_inset_m must not be negative")

    def decisions(self, seconds):
        n = seconds / self.dt_s
        if abs(n - round(n)) > 1e-9:
            raise ValueError(f"{seconds} s is not a whole number of decisions of {self.dt_s} s")
        return int(round(n))


def pedal_from_action(a, cfg):
    """Throttle or brake pedal (0..1) for an action a in [-1, 1]: clip((a + 0.5) / (pedal_full_at + 0.5), 0, 1); without the dead zone (a + 1) / 2."""
    if not cfg.pedal_deadzone:
        return (float(a) + 1.0) / 2.0
    return min(max((float(a) + 0.5) / (cfg.pedal_full_at + 0.5), 0.0), 1.0)


def action_from_pedal(pedal, cfg, released=-0.8):
    """The action that gives `pedal` (the inverse of pedal_from_action); no pedal is `released`."""
    if not cfg.pedal_deadzone:
        return 2.0 * pedal - 1.0
    return -0.5 + (cfg.pedal_full_at + 0.5) * pedal if pedal > 0.0 else released
