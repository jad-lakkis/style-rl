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
    d = {"grip": 1.0, "brake": 1.0, "drive": 1.0, "aero": 1.0, "mass": 1.0, "steer_lag_s": None, "action_delay": False, "obs_noise": 0.0}
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
