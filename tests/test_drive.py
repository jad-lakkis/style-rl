"""Tests of the portable driving environment and the staged SAC (style_rl.drive), on the stand-in car (NOT OCD physics).   pytest tests/test_drive.py"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from style_rl.drive.backend import make_backend
from style_rl.drive.config import OBS_DIM, EnvConfig, RewardConfig
from style_rl.drive.course import Course
from style_rl.drive.env import DrivingEnv

ROOT = Path(__file__).resolve().parents[1]
TRACK, ENVELOPE = ROOT / "data/tracks/monza_geometry.csv", ROOT / "data/ocd_envelope.json"


@pytest.fixture(scope="module")
def course():
    return Course(TRACK, ENVELOPE, margin_m=EnvConfig().margin_m)


def make_env(course, cfg=EnvConfig(), evaluation=False):
    return DrivingEnv(course, make_backend("mock", TRACK, cfg.steer_lock_rad, cfg.dt_s), cfg, evaluation=evaluation)


def drive(env, steps, fraction=0.7):
    for _ in range(steps):
        a = env.demo.act(env.state, env.s, env.demo.lap_target(env.s, fraction, env.state["speed"]))
        obs, r, term, trunc, info = env.step(a)
        yield obs, r, term, trunc, info
        if term or trunc:
            return


def test_course_reference_line_is_inside_the_track(course):
    c = course
    assert c.ref_xy.shape == (c.N, 2) and np.isfinite(c.v_qss).all()
    margin = np.minimum(np.linalg.norm(c.ref_xy - c.left_xy, axis=1), np.linalg.norm(c.ref_xy - c.right_xy, axis=1))
    assert margin.min() > 1.0
    assert 105.0 < c.qss_lap_time_s < 125.0


def test_observation_contract(course):
    env = make_env(course)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32 and np.isfinite(obs).all()
    assert np.abs(obs).max() <= EnvConfig().obs.clip
    # layout: ego 0-8, line errors 9-11, preview 12-131, last action 132-134, history 135-158
    assert np.allclose(obs[132:135], env.last_action)
    for o, *_ in drive(env, 30):
        assert o.shape == (OBS_DIM,)
    # the previous decision's ego block is the first history block
    prev_ego = o[:9].copy()
    o2, *_ = next(drive(env, 1))
    assert np.allclose(o2[135:144], prev_ego, atol=1e-6)


def test_reward_is_progress_minus_time_without_shaping(course):
    cfg = replace(EnvConfig(), reward=RewardConfig(shaping_k=0.0))
    env = make_env(course, cfg)
    env.reset(seed=5)
    total = 0.0
    n = 0
    for _, r, term, trunc, info in drive(env, 200):
        total, n = total + r, n + 1
    assert not term
    assert total == pytest.approx(info["progress_m"] - n * cfg.dt_s, abs=1e-6)


def test_random_start_episodes_run_to_the_time_limit(course):
    env = make_env(course)
    for seed in range(3):
        env.reset(seed=seed)
        for _, _, term, trunc, info in drive(env, 5000, fraction=0.5):
            pass
        assert trunc and info["truncation"] == "time_limit" and not term
        assert info["time_s"] == pytest.approx(EnvConfig().max_episode_s, abs=1e-6)


def test_leaving_the_track_is_a_terminal_failure_with_penalty(course):
    env = make_env(course)
    env.reset(seed=1)
    steer_hard = np.array([1.0, 2.0, -1.0], np.float32)        # full left lock, full throttle
    for k in range(3000):
        _, r, term, trunc, info = env.step(steer_hard)
        if term or trunc:
            break
    assert term and info["termination"] == "off_track"
    assert r < -EnvConfig().reward.penalty + 5.0                 # the penalty dominates the last reward


def test_evaluation_lap_times(course):
    env = make_env(course, evaluation=True)
    env.reset(seed=0)
    for _, _, term, trunc, info in drive(env, 20000, fraction=0.9):
        pass
    assert term and info["termination"] == "lap_complete"
    t = info["lap_times_s"]
    assert len(t) == 2 and abs(t[0] - t[1]) < 0.5
    assert 100.0 < t[1] < 140.0


def test_frozen_actor_does_not_change_and_released_actor_does(course):
    import torch as th
    from stable_baselines3.common.vec_env import DummyVecEnv
    from style_rl.drive import agent as ag
    venv = DummyVecEnv([lambda: make_env(course)])
    model = ag.build_model(venv, hidden=(16, 16, 16), batch_size=64, buffer_size=4000, n_steps=3, gradient_steps=1)
    model.set_frozen(True, True)
    a0 = [p.detach().clone() for p in model.actor.parameters()]
    c0 = [p.detach().clone() for p in model.critic.parameters()]
    alpha0 = model.alpha
    model.learn(total_timesteps=400)
    assert all(th.equal(a, b) for a, b in zip(a0, model.actor.parameters())), "frozen actor changed"
    assert model.alpha == alpha0, "frozen entropy coefficient changed"
    assert any(not th.equal(a, b) for a, b in zip(c0, model.critic.parameters())), "critic did not train"
    model.set_frozen(False, False)
    model.learn(total_timesteps=300, reset_num_timesteps=False)
    assert any(not th.equal(a, b) for a, b in zip(a0, model.actor.parameters())), "released actor did not change"
    snap = ag.snapshot_state(model)
    before = [p.detach().clone() for p in model.actor.parameters()]
    model.learn(total_timesteps=100, reset_num_timesteps=False)
    ag.restore_state(model, snap)
    assert all(th.equal(a, b) for a, b in zip(before, model.actor.parameters())), "restore did not bring the actor back"


def test_prefill_closes_every_column(course):
    from stable_baselines3.common.buffers import NStepReplayBuffer
    from style_rl.drive.demos import DemoConfig, assemble, prefill_replay_buffer, run_episode
    cfg = DemoConfig(n_transitions=1, workers=1, seed=7)
    env = make_env(course, replace(EnvConfig(), max_episode_s=8.0))
    eps = [run_episode(env, cfg, k) for k in range(4)]
    d = assemble(eps)
    env0 = make_env(course)
    rb = NStepReplayBuffer(2000, env0.observation_space, env0.action_space, n_envs=2, n_steps=3, gamma=0.992, handle_timeout_termination=True)
    info = prefill_replay_buffer(rb, d)
    T = info["rows"]
    assert rb.pos == T and T > 0
    assert rb.dones[T - 1].astype(bool).all()                      # the last row of every column is closed ...
    assert rb.timeouts[T - 1].astype(bool).all()                   # ... as a time-limit cut (these 8 s episodes end by the time limit, not by a failure)
    assert np.isfinite(rb.rewards[:T]).all()


def test_both_pedals_act_as_commanded(course):
    """No brake priority: throttle and brake together give a net effect between throttle alone and brake alone."""
    def speed_after(thr_a, brk_a):
        env = make_env(course)
        env.reset(seed=2, options={"start_s": 1500.0, "start_speed": 30.0})
        for _ in range(25):
            env.step(np.array([0.0, thr_a, brk_a], np.float32))
        return env.state["speed"]
    throttle_only, both, brake_only = speed_after(1.0, -1.0), speed_after(1.0, 0.0), speed_after(-1.0, 0.0)
    assert throttle_only > both > brake_only


def test_portable_actor_matches_sb3_and_the_network_contract(course, tmp_path):
    """The exported tensors, run by the NumPy-only runtime, give the SB3 actor's deterministic action; the ten tensors have the names and shapes of the AC runner's actor.
    Set RLSETUP_REPO to a checkout of the RL-SetupOptimization-Agent repository to also run its own tensor validator on them."""
    import os
    import sys
    import torch as th
    from stable_baselines3.common.vec_env import DummyVecEnv
    from style_rl.drive import agent as ag
    from style_rl.drive.portable import KEYS, PortableActor, export_actor
    venv = DummyVecEnv([lambda: make_env(course)])
    model = ag.build_model(venv, buffer_size=1000, batch_size=64)
    env = make_env(course)
    obs, _ = env.reset(seed=1)
    seen = [obs]
    for _ in range(150):
        obs, _, term, trunc, _ = env.step(model.predict(obs, deterministic=False)[0])
        seen.append(obs)
        if term or trunc:
            obs, _ = env.reset(seed=2)
    bundle = export_actor(model, tmp_path / "bundle", meta={"test": True}, course=course)
    actor = PortableActor(bundle)
    for o in seen:
        assert np.allclose(actor.predict(o), model.predict(o, deterministic=True)[0], atol=1e-5)
    assert actor.steer_lock_rad == EnvConfig().steer_lock_rad
    with pytest.raises(FileExistsError):
        export_actor(model, tmp_path / "bundle")
    repo = os.environ.get("RLSETUP_REPO")
    if repo:
        sys.dont_write_bytecode = True
        sys.path.insert(0, repo)
        from rlsetup.assetto.actor import _validate_weights
        _validate_weights({k: th.from_numpy(np.load(bundle / "actor.npz")[k].copy()) for k in KEYS})


def test_blend_executes_the_mixture_and_evaluation_ignores_it(course):
    def twin():
        e = make_env(course)
        e.reset(seed=11)
        return e

    policy = np.array([0.4, -0.2, 0.6], np.float32)
    ref = twin()
    teacher = ref.demo.act(ref.state, ref.s, ref.demo.lap_target(ref.s, 0.8, ref.state["speed"]))
    for w, expected in ((0.0, policy), (1.0, teacher), (0.25, 0.75 * policy + 0.25 * teacher)):
        e = twin()
        e.set_blend(w, 0.8)
        *_, info = e.step(policy)
        assert np.allclose(info["executed_action"], expected, atol=1e-6), w
        assert np.allclose(e.last_action, expected, atol=1e-6)
    ev = make_env(course, evaluation=True)
    ev.set_blend(1.0, 0.8)
    ev.reset(seed=0)
    *_, info = ev.step(policy)
    assert np.allclose(info["executed_action"], policy)


def test_staged_sac_stores_the_executed_action(course):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
    from style_rl.drive import agent as ag

    class Rec(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.executed = []

        def step(self, a):
            out = self.env.step(a)
            self.executed.append(np.array(out[4]["executed_action"]))
            return out

    rec = Rec(make_env(course))
    venv = DummyVecEnv([lambda: rec])
    venv.env_method("set_blend", 1.0, 0.8)
    model = ag.build_model(venv, hidden=(16, 16, 16), batch_size=32, buffer_size=2000, n_steps=3, gradient_steps=1)
    model.learn(total_timesteps=60)
    n = model.replay_buffer.pos
    stored = model.replay_buffer.actions[:n, 0, :]
    assert n == 60 and np.allclose(stored, np.stack(rec.executed[:n]), atol=1e-6)
    assert np.abs(stored[:, 0]).max() > 0.0            # the teacher steers; the untrained agent's own steering is ~0


def _nominal_dr(**kw):
    d = {"grip": 1.0, "brake": 1.0, "drive": 1.0, "aero": 1.0, "mass": 1.0, "steer_lag_s": None, "slip_scale": 1.0, "action_delay": False, "obs_noise": 0.0}
    d.update(kw)
    return d


def test_randomization_is_reproducible_and_skips_evaluation(course):
    from style_rl.drive.randomize import RandomizeConfig
    cfg = replace(EnvConfig(), randomize=RandomizeConfig(enabled=True))
    infos = []
    for seed in (5, 5, 6):
        _, info = make_env(course, cfg).reset(seed=seed)
        infos.append(info["randomization"])
    assert infos[0] == infos[1] and infos[0] != infos[2]
    assert 0.85 <= infos[0]["grip"] <= 1.15 and 0.04 <= infos[0]["steer_lag_s"] <= 0.10
    _, ev = make_env(course, cfg, evaluation=True).reset(seed=5)
    assert ev["randomization"] == _nominal_dr()


def test_backend_randomization_changes_the_car(course):
    b = make_backend("mock", TRACK, 0.3, 0.04)
    x, y = course.ref_point(1500.0)
    yaw = float(course.ref_heading_at(1500.0))

    def speed_after(steer_thr_brk, **kw):
        b.set_randomization(_nominal_dr(**kw))
        b.reset(float(x), float(y), yaw, 40.0)
        for _ in range(25):
            st = b.step(*steer_thr_brk)
        return st["speed"]
    assert speed_after((0, 0, 1), brake=0.5) > speed_after((0, 0, 1)) > speed_after((0, 0, 1), brake=1.5)       # weaker brakes stop later
    assert speed_after((0, 1, 0), drive=1.1) > speed_after((0, 1, 0)) > speed_after((0, 1, 0), drive=0.9)
    assert speed_after((0, 1, 0), mass=1.03) < speed_after((0, 1, 0))


def test_observation_noise_only_noise_and_deterministic(course):
    from style_rl.drive.randomize import RandomizeConfig
    base = dict(enabled=True, grip=(1, 1), brake=(1, 1), drive=(1, 1), aero=(1, 1), mass=(1, 1), steer_lag_s=(0.06, 0.06), action_delay_prob=0.0)
    quiet, loud = RandomizeConfig(obs_noise=0.0, **base), RandomizeConfig(obs_noise=1.0, **base)         # same draws, so the same start: only the noise differs
    o_quiet, _ = make_env(course, replace(EnvConfig(), randomize=quiet)).reset(seed=7)
    o_a, _ = make_env(course, replace(EnvConfig(), randomize=loud)).reset(seed=7)
    o_b, _ = make_env(course, replace(EnvConfig(), randomize=loud)).reset(seed=7)
    assert np.array_equal(o_a, o_b)                                # seeded, reproducible
    diff = np.abs(o_a - o_quiet).max()
    assert 0.0 < diff < 0.2                                        # small, in scaled units


def test_action_delay_applies_the_previous_action(course):
    from style_rl.drive.randomize import RandomizeConfig
    delay = RandomizeConfig(enabled=True, grip=(1, 1), brake=(1, 1), drive=(1, 1), aero=(1, 1), mass=(1, 1), steer_lag_s=(0.06, 0.06), action_delay_prob=1.0, obs_noise=0.0)
    env = make_env(course, replace(EnvConfig(), randomize=delay))
    env.reset(seed=3)
    applied = []
    orig = env.backend.step
    env.backend.step = lambda s, t, b: (applied.append((s, t, b)), orig(s, t, b))[1]
    a1, a2 = np.array([0.3, 0.5, -1.0], np.float32), np.array([-0.3, -1.0, 0.5], np.float32)
    env.step(a1)
    env.step(a2)
    assert applied[0] != env._controls(a1)                       # the first action has not arrived yet
    assert applied[1] == env._controls(a1)                       # the second step applies the first action
    assert np.allclose(env.last_action, a2)                      # the observation shows the action that was chosen


def test_four_tracks_the_scripted_driver_stays_on_every_one(course):
    names = ("monza", "barcelona", "red_bull_ring", "silverstone")
    for name in names:
        csv = ROOT / f"data/tracks/{name}_geometry.csv"
        c = Course(csv, ENVELOPE, margin_m=EnvConfig().margin_m)
        env = DrivingEnv(c, make_backend("mock", csv, 0.3, 0.04), EnvConfig())
        for seed in range(3):
            env.reset(seed=seed)
            for _, _, term, trunc, info in drive(env, 1000, fraction=0.7):
                pass
            assert not term, (name, seed, info["termination"], info["progress_m"])


def test_multi_track_vector_env_and_demos(course):
    from style_rl.drive.demos import DemoConfig, collect_episodes
    from style_rl.drive.vec import build_vec_env
    tracks = [str(ROOT / "data/tracks/monza_geometry.csv"), str(ROOT / "data/tracks/barcelona_geometry.csv")]
    venv = build_vec_env(4, tracks, ENVELOPE, "mock", EnvConfig(), subprocess=False)
    assert [c.name for c in venv.get_attr("course")] == ["monza", "barcelona", "monza", "barcelona"]
    eps = collect_episodes(DemoConfig(n_transitions=250, workers=1), (tracks, str(ENVELOPE), "mock", replace(EnvConfig(), max_episode_s=4.0)), "spawn")
    assert len(eps) >= 2 and all(len(e["actions"]) > 0 for e in eps)


def test_ac_observation_builder_reproduces_the_environment_observation(course):
    """Feed the states of an episode to the AC-side builder as telemetry packets (the field names and frames of the AC plugin): it must give the observation the environment gave."""
    import math
    from style_rl.drive.ac_obs import ACGeometry, ACObservation

    def packet(st):
        yaw, d = st["yaw"], st["steer_actual"]
        c, s = math.cos(yaw), math.sin(yaw)
        p = {"world_position_x": st["x"], "world_position_y": st["y"], "yaw": yaw, "timestamp_ac": st["time_s"],
             "velocity_z": c * st["vx"] - s * st["vy"], "velocity_x": s * st["vx"] + c * st["vy"],       # the world velocity in the planar frame (x, y) = (AC z, AC x)
             "tyre_slip_ratio_rl": st["slip_rl"], "tyre_slip_ratio_rr": st["slip_rr"]}
        for w, a in (("FL", yaw + d), ("FR", yaw + d), ("RL", yaw), ("RR", yaw)):
            p[f"tyreContactHeading_{w}_z"], p[f"tyreContactHeading_{w}_x"] = math.cos(a), math.sin(a)
        return p

    env = make_env(course)
    env.reset(seed=4)
    ac = ACObservation(ACGeometry.from_course(course))
    ac.build(packet(env.state), env.last_action)
    compared = 0
    for obs, _, term, trunc, _ in drive(env, 60, fraction=0.7):
        theirs = ac.build(packet(env.state), env.last_action)
        if theirs is not None:
            assert np.abs(theirs - obs).max() < 1e-3, np.argmax(np.abs(theirs - obs))
            compared += 1
    assert compared >= 50


# ---------------------------------------------------------------------------------------------------------------- pedal mapping, training inset, car presets, the F1 backend
def test_default_pedal_mapping_is_the_original_one(course):
    env = make_env(course)
    for a in np.linspace(-1.0, 1.0, 41):
        assert env._pedal(a) == pytest.approx(float(np.clip((a + 0.5) / 1.5, 0.0, 1.0)))
    assert env._pedal(1.0) == 1.0 and env._pedal(0.9) < 1.0


def test_pedal_full_at_gives_full_pedal_inside_the_range(course):
    from style_rl.drive.config import action_from_pedal, pedal_from_action
    from style_rl.drive.demo import pedal_to_action
    cfg = replace(EnvConfig(), pedal_full_at=0.7)
    env = make_env(course, cfg)
    assert env._pedal(0.7) == 1.0 and env._pedal(0.95) == 1.0 and env._pedal(-0.5) == 0.0 and env._pedal(-0.9) == 0.0
    assert env._pedal(0.1) == pytest.approx(0.5)                      # (0.1 + 0.5) / 1.2
    for p in (0.05, 0.3, 0.77, 1.0):                                  # the scripted driver's action gives back the pedal it meant, for both mappings
        for c in (EnvConfig(), cfg):
            assert pedal_from_action(action_from_pedal(p, c), c) == pytest.approx(p)
            assert pedal_from_action(pedal_to_action(p, c.pedal_deadzone, full_at=c.pedal_full_at), c) == pytest.approx(p)
    assert action_from_pedal(0.0, cfg) == -0.8 and pedal_from_action(-0.8, cfg) == 0.0
    with pytest.raises(ValueError):
        replace(EnvConfig(), pedal_full_at=1.2)


def test_demo_driver_takes_the_car_numbers_from_the_config(course):
    env = make_env(course, replace(EnvConfig(), wheelbase_m=3.6, tc_start=0.06, tc_span=0.03))
    assert (env.demo.wb, env.demo.tc_start, env.demo.tc_span) == (3.6, 0.06, 0.03)
    default = make_env(course)
    assert (default.demo.wb, default.demo.tc_start, default.demo.tc_span) == (2.8, 0.08, 0.12)


def test_demo_driver_caps_the_brake_pedal_at_the_cars_value(course):
    for cap in (1.0, 0.3):
        env = make_env(course, replace(EnvConfig(), demo_brake_cap=cap))
        env.reset(seed=1)
        assert env.demo.brake_cap == cap
        a = env.demo.act(env.state, env.s, 0.0)                            # a target of 0 m/s: the speed PI asks for the hardest braking
        assert env._pedal(a[2]) == pytest.approx(cap, abs=1e-6) and env._pedal(a[1]) == 0.0


def test_the_training_inset_is_a_training_only_buffer(course):
    s = course.L - 400.0                                                  # on the main straight
    centre = course._interp(course.center_xy, s)
    left = course._interp(course.left_xy, s)
    n_left = (left - centre) / np.linalg.norm(left - centre)
    q = course.track.query(float(centre[0]), float(centre[1]))
    off = q["left_width_m"] - 0.8 - 0.1                                   # the left wheels are 0.1 m inside the true edge
    x, y = centre + off * n_left
    assert course.wheels_inside(x, y, q["heading_rad"], 1.4, 0.8)
    assert not course.wheels_inside(x, y, q["heading_rad"], 1.4, 0.8, inset=0.2)
    env = make_env(course, replace(EnvConfig(), train_inset_m=0.2))
    env.state = {"x": float(x), "y": float(y), "yaw": float(q["heading_rad"])}
    env.evaluation = False
    assert not env._inside()
    env.evaluation = True
    assert env._inside()


def test_car_presets():
    from style_rl.drive.cars import PRESETS, apply_preset
    assert apply_preset("f317", EnvConfig()) == EnvConfig()
    f1 = apply_preset("sf70h_f1", EnvConfig())
    assert (f1.wheelbase_m, f1.wheel_half_base_m, f1.tc_start, f1.tc_span) == (3.6, 1.8, 0.06, 0.03) and PRESETS["sf70h_f1"].demo_fraction == (0.5, 0.8)
    assert PRESETS["sf70h_f1"].backend == "ocdlap:sf70h_f1:p1_TUM_f1" and f1.steer_lock_rad == 0.30
    assert f1.demo_brake_cap == 0.3 and EnvConfig().demo_brake_cap == 1.0 and PRESETS["sf70h_f1"].train == {"gate_required": False} and PRESETS["f317"].train == {}
    with pytest.raises(ValueError):
        apply_preset("nope", EnvConfig())


STAND_IN_OCD_LAP = """
import math
from dataclasses import dataclass
DT = .002
@dataclass(frozen=True)
class StartPose:
    x: float = 0.
    y: float = 0.
    yaw: float = 0.
    speed: float = 5.
@dataclass(frozen=True)
class Inputs:
    steering_rad: float
    throttle: float
    brake: float
class OCDBackend:
    made = []
    def __init__(self, *, powertrain='direct', max_drive_torque_Nm=270., max_brake_torque_Nm=None, overrides=None, car='av21', ers=None):
        OCDBackend.made.append(dict(powertrain=powertrain, car=car, ers=ers))
    def _state(self):
        s = self.s
        return dict(x=s['x'], y=s['y'], yaw=s['yaw'], vx=s['vx'], vy=0., yaw_rate=0., steer_actual=0.1, slip_rl=0.01, slip_rr=0.02)
    def reset(self, setup, start_pose):
        self.setup = setup
        self.s = dict(x=start_pose.x, y=start_pose.y, yaw=start_pose.yaw, vx=start_pose.speed)
        return self._state()
    def step(self, inputs, steps=1):
        self.last = (inputs, steps)
        for _ in range(steps):
            self.s['x'] += self.s['vx'] * math.cos(self.s['yaw']) * DT
            self.s['y'] += self.s['vx'] * math.sin(self.s['yaw']) * DT
        return self._state()
"""
STAND_IN_CARS = """
import dataclasses
from dataclasses import dataclass
@dataclass(frozen=True)
class Spec:
    name: str
    steer_max_rad: float = .3
@dataclass(frozen=True)
class FakeSetup:
    heave_damping_ratio: float = 0.
    roll_stiffness_front_share: float = .61
    aero_front_share: float = .45
    lsd_lock_drive: float = 0.
    brake_bias_front: float = .58
    def replace(self, **changes):
        return dataclasses.replace(self, **changes)
def get(car):
    return Spec(car)
def default_setup(car):
    return FakeSetup()
"""
NEUTRAL = {"grip": 1.0, "brake": 1.0, "drive": 1.0, "aero": 1.0, "mass": 1.0, "steer_lag_s": None, "slip_scale": 1.0, "action_delay": False, "obs_noise": 0.0}


def test_the_f1_backend_adapter_on_a_stand_in_for_ocd_lap(tmp_path, monkeypatch):
    """The adapter calls the OCDBackend of the ocd_lap package as documented there (engine powertrain, ERS off, the car's default setup, 20 physics steps per decision). The stand-in package is a few
    lines written here, NOT the real code."""
    import sys
    pkg = tmp_path / "ocd_lap"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "backend.py").write_text(STAND_IN_OCD_LAP)
    (pkg / "cars.py").write_text(STAND_IN_CARS)
    gone = lambda: [m for m in sys.modules if m == "ocd_lap" or m.startswith("ocd_lap.")]
    for name in gone():
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setenv("OCD_FIRST_LAP_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "path", [str(tmp_path)] + sys.path)
    try:
        b = make_backend("ocdlap:sf70h_f1", TRACK, 0.3, 0.04)
        assert b.control_dt == 0.04 and b.n_sub == 20
        s0 = b.reset(1.0, 2.0, 0.5, 30.0)
        assert set(s0) == {"time_s", "x", "y", "yaw", "vx", "vy", "speed", "yaw_rate", "steer_actual", "slip_rl", "slip_rr"} and s0["time_s"] == 0.0 and s0["speed"] == 30.0
        assert b.setup_name == "p1_TUM_f1"                                   # the setup of his F1 runs: heave damping 0.3 (the car porpoises without it) and the p1_TUM knobs, brake bias kept
        assert (b.backend.setup.heave_damping_ratio, b.backend.setup.roll_stiffness_front_share, b.backend.setup.aero_front_share, b.backend.setup.lsd_lock_drive,
                b.backend.setup.brake_bias_front) == (0.3, 0.4533, 0.4508, 1.0, 0.58)
        assert make_backend("ocdlap:sf70h_f1:default", TRACK, 0.3, 0.04).setup.heave_damping_ratio == 0.0
        with pytest.raises(ValueError):
            make_backend("ocdlap:sf70h_f1:nope", TRACK, 0.3, 0.04)
        s1 = b.step(0.5, 1.3, -0.2)                                       # steering 0.5 of the 0.3 rad lock, pedals clipped to [0, 1]
        inputs, steps = b.backend.last
        assert (inputs.steering_rad, inputs.throttle, inputs.brake, steps) == (pytest.approx(0.15), 1.0, 0.0, 20)
        assert s1["time_s"] == pytest.approx(0.04) and s1["x"] > s0["x"]
        from ocd_lap.backend import OCDBackend
        assert OCDBackend.made[-1] == {"powertrain": "engine", "car": "sf70h_f1", "ers": False}
        b.set_randomization(NEUTRAL)
        with pytest.raises(NotImplementedError):
            b.set_randomization({**NEUTRAL, "grip": 0.9})
        with pytest.raises(ValueError):                                   # above the car's actuator limit (0.3 rad in the stand-in)
            make_backend("ocdlap:sf70h_f1", TRACK, 0.5, 0.04)
        with pytest.raises(ValueError):                                   # not a whole number of 2 ms physics steps
            make_backend("ocdlap:sf70h_f1", TRACK, 0.3, 0.0405)
    finally:
        for name in gone():
            del sys.modules[name]


def test_the_f1_backend_needs_its_folder(monkeypatch):
    monkeypatch.delenv("OCD_FIRST_LAP_DIR", raising=False)
    with pytest.raises(RuntimeError, match="OCD_FIRST_LAP_DIR"):
        make_backend("ocdlap", TRACK, 0.3, 0.04)


def test_the_reference_line_can_come_from_a_file(tmp_path):
    from style_rl.drive.course import GridTrack, min_curvature_line
    gt = GridTrack(TRACK)
    n10, _ = min_curvature_line(gt, 1.0)
    csv = tmp_path / "line.csv"
    np.savetxt(csv, np.column_stack([gt.s, n10]), delimiter=",", header="s_m,offset_m", comments="", fmt="%.6f")
    from_file = Course(TRACK, ENVELOPE, margin_m=1.5, line_csv=csv)                  # the margin argument is ignored when a line is given
    reference = Course(TRACK, ENVELOPE, margin_m=1.0)
    assert from_file.qss_lap_time_s == pytest.approx(reference.qss_lap_time_s, abs=0.02)
    assert from_file.qss_lap_time_s < Course(TRACK, ENVELOPE, margin_m=1.5).qss_lap_time_s - 0.5
    assert np.abs(from_file.ref_xy - reference.ref_xy).max() < 0.05
    bad = tmp_path / "bad.csv"
    np.savetxt(bad, np.column_stack([gt.s[:-5], n10[:-5]]), delimiter=",", header="s_m,offset_m", comments="", fmt="%.6f")
    with pytest.raises(ValueError):
        Course(TRACK, ENVELOPE, line_csv=bad)
    env = make_env(from_file, replace(EnvConfig(), ref_line=str(csv)))
    assert env.cfg.ref_line == str(csv)


# ---------------------------------------------------------------------------------------------------------------- failure-focused starts, reliability over start stations, the gated best
class TeacherAsModel:
    """The scripted driver behind the model interface (predict) that the evaluation functions use. NOT a trained model."""

    def __init__(self, env, fraction=0.8, alpha=0.05):
        self.env, self.fraction, self.alpha = env.unwrapped, fraction, alpha

    def predict(self, obs, deterministic=True):
        e = self.env
        return e.demo.act(e.state, e.s, e.demo.lap_target(e.s, self.fraction, e.state["speed"])), None

    def save(self, path):
        open(path, "wb").write(b"model")

    def save_replay_buffer(self, path):
        open(path, "wb").write(b"buffer")


def test_focus_starts_come_before_the_failure_places(course):
    env = make_env(course)
    env.reset(seed=1)
    bins = 100
    w = np.zeros(bins)
    w[int(2050.0 / course.L * bins)] = 1.0                                   # all the weight on the bin around station 2050 m
    env.set_focus(w, 1.0)
    L, width = course.L, course.L / bins
    for _ in range(40):
        s0, f, v0 = env._sample_start({})
        assert 2050.0 - width - 600.0 - 1e-6 <= s0 <= 2050.0 + width - 150.0 + 1e-6
        assert 0.3 <= f <= 0.7
    env.set_focus(w, 0.0)                                                    # probability 0: back to uniform starts
    starts = [env._sample_start({})[0] for _ in range(60)]
    assert max(starts) - min(starts) > 0.5 * L
    env.set_focus(None, 1.0)
    assert env.focus_w is None
    with pytest.raises(ValueError):
        env.set_focus(np.zeros(5), 0.5)


def test_focus_starts_leave_the_random_stream_alone_when_off(course):
    a, b = make_env(course), make_env(course)
    a.reset(seed=5)
    b.reset(seed=5)
    assert [a._sample_start({})[0] for _ in range(5)] == [b._sample_start({})[0] for _ in range(5)]


def test_focus_callback_counts_failures_and_gives_weights(course):
    from style_rl.drive.train import FocusCallback, TrainConfig
    cfg = replace(TrainConfig(), focus_prob=0.3, focus_min_failures=2.0)
    cb = FocusCallback(cfg, course.L, run=None)
    assert cb.weights() is None
    infos = [{"termination": "off_track", "station_m": 2060.0}, {"termination": "off_track", "station_m": 2070.0}, {"termination": "wrong_way", "station_m": 2065.0},
             {"termination": None, "station_m": 100.0}, {"termination": "nonfinite", "station_m": 900.0}]
    cb.locals = {"infos": infos}
    cb.model = type("M", (), {"num_timesteps": 10**9})()                     # past `next`, so the weights are computed (training_env is only called when there are weights)
    sent = {}
    cb.__class__.training_env = property(lambda self: type("V", (), {"env_method": lambda _s, name, w, p: sent.update(name=name, w=np.asarray(w), p=p)})())
    cb.__class__.logger = property(lambda self: type("Lg", (), {"record": lambda *_a, **_k: None})())
    cb.run = type("R", (), {"event": lambda *_a, **_k: None})()
    try:
        cb._on_step()
    finally:
        del cb.__class__.training_env, cb.__class__.logger
    assert cb.counts.sum() == pytest.approx(3.0 * cfg.focus_decay)           # three failures counted (the other two ended otherwise), then forgotten a little
    assert sent["name"] == "set_focus" and sent["p"] == 0.3
    centre_station = (np.average(np.arange(len(sent["w"])), weights=sent["w"]) + 0.5) * course.L / len(sent["w"])
    assert abs(centre_station - 2065.0) < 30.0                                  # the weight is centred on the three failures, smoothed over a few bins


def test_reliability_runs_laps_from_other_stations(course):
    from style_rl.drive.evaluate import reliability, varied_starts
    assert varied_starts(1000.0, 4) == [125.0, 375.0, 625.0, 875.0]
    env = make_env(course, evaluation=True)
    out = reliability(TeacherAsModel(env), env, 2)
    assert out["of"] == 2 and out["clean"] == 2 and len(out["lap2_s"]) == 2 and out["ends"] == ["lap_complete"] * 2
    assert all(100.0 < t < 200.0 for t in out["lap2_s"])


def test_the_eval_callback_saves_the_best_gated_checkpoint(course, tmp_path):
    import json
    from style_rl.drive.train import EvalCallback, Run, TrainConfig
    run = Run(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    cfg = replace(TrainConfig(), reliability_starts=2, eval_stochastic_laps=1)
    env = make_env(course, evaluation=True)
    cb = EvalCallback(cfg, {course.name: env}, run)
    cb.model = TeacherAsModel(env)
    cb.evaluate(1000)
    rec = json.loads((tmp_path / "evals.jsonl").read_text().splitlines()[-1])
    assert rec["det_clean"] and rec["gated"] is True and rec["reliability"][course.name]["clean"] == 2
    assert (tmp_path / "checkpoints" / "best_gated.zip").exists() and (tmp_path / "checkpoints" / "best.zip").exists()
    st = json.loads((tmp_path / "status.json").read_text())
    assert st["best_gated_score_lap2_over_qss"] == pytest.approx(st["best_score_lap2_over_qss"]) and st["best_gated_score_lap2_over_qss"] > 1.0
    # a resumed run keeps what the earlier sessions reached
    again = EvalCallback(cfg, {course.name: env}, run)
    assert again.best == pytest.approx(st["best_score_lap2_over_qss"]) and again.best_gated == pytest.approx(st["best_gated_score_lap2_over_qss"])
