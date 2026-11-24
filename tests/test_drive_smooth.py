"""Tests of the smoothness controls of the driving environment (style_rl.drive, RewardConfig.smooth_*), on the stand-in car (NOT OCD physics).   pytest tests/test_drive_smooth.py"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from style_rl.drive.backend import make_backend
from style_rl.drive.config import EnvConfig, RewardConfig
from style_rl.drive.course import Course
from style_rl.drive.demos import DemoConfig, dataset_key
from style_rl.drive.env import DrivingEnv
from style_rl.drive.evaluate import smoothness

ROOT = Path(__file__).resolve().parents[1]
TRACK, ENVELOPE = ROOT / "data/tracks/monza_geometry.csv", ROOT / "data/ocd_envelope.json"


@pytest.fixture(scope="module")
def course():
    return Course(TRACK, ENVELOPE, margin_m=EnvConfig().margin_m)


def make_env(course, **reward):
    cfg = replace(EnvConfig(), reward=replace(RewardConfig(shaping_k=0.0), **reward))
    return DrivingEnv(course, make_backend("mock", TRACK, cfg.steer_lock_rad, cfg.dt_s), cfg)


def run(env, actions):
    """Step the given actions; returns the list of (reward, smooth_pen) and the ended flag."""
    out = []
    for a in actions:
        _, r, term, trunc, info = env.step(np.asarray(a, np.float32))
        out.append((r, info["smooth_pen"]))
        if term or trunc:
            break
    return out


COAST = (0.0, -1.0, -1.0)                      # no pedal pressed; the steering is the first number


def acts(steer):
    return [(s, -1.0, -1.0) for s in steer]


def test_off_by_default_and_the_other_weights_do_nothing_when_smooth_w_is_zero(course):
    a = make_env(course)
    b = make_env(course, smooth_steer=7.0, smooth_straight=3.0, smooth_pedal=9.0)         # smooth_w stays 0
    rng = np.random.default_rng(0)
    steer = rng.uniform(-0.2, 0.2, 40)
    a.reset(seed=4); b.reset(seed=4)
    ra, rb = run(a, acts(steer)), run(b, acts(steer))
    assert [x[1] for x in ra] == [0.0] * len(ra)
    assert [x[0] for x in ra] == [x[0] for x in rb]


def test_a_steady_ramp_costs_nothing_and_a_chatter_costs_its_jerk(course):
    env = make_env(course, smooth_w=1.0, smooth_straight=0.0, smooth_pedal=0.0)
    env.reset(seed=4)
    ramp = run(env, acts([0.002 * k for k in range(30)]))
    assert max(p for _, p in ramp) < 1e-5                                          # second difference of a linear ramp = 0
    env = make_env(course, smooth_w=1.0, smooth_straight=0.0, smooth_pedal=0.0)
    env.reset(seed=4)
    chatter = run(env, acts([0.05 * (-1) ** k for k in range(30)]))
    # decisions 0 and 1 are not charged (their history comes from the scripted driver); from decision 2 on |a(t) - 2 a(t-1) + a(t-2)| = 0.05 * 4 = 0.2
    assert [p for _, p in chatter[:2]] == [0.0, 0.0]
    assert len(chatter) >= 10 and all(p == pytest.approx(0.2, abs=1e-6) for _, p in chatter[2:])


def test_the_penalty_is_subtracted_from_the_reward(course):
    base = make_env(course)
    pen = make_env(course, smooth_w=0.5)
    base.reset(seed=9); pen.reset(seed=9)
    steer = [0.05 * (-1) ** k for k in range(20)]
    rb, rp = run(base, acts(steer)), run(pen, acts(steer))
    for (r0, _), (r1, p1) in zip(rb, rp):
        assert r1 == pytest.approx(r0 - p1, abs=1e-6)


def test_pedal_jerk_is_measured_on_the_applied_pedals_not_on_the_raw_action(course):
    # a throttle action that flicks between -0.9 and -0.6 never leaves the dead zone (pedal = 0 in both): no cost
    env = make_env(course, smooth_w=1.0, smooth_steer=0.0, smooth_straight=0.0, smooth_pedal=1.0)
    env.reset(seed=4)
    quiet = run(env, [(0.0, -0.9 if k % 2 else -0.6, -1.0) for k in range(20)])
    assert len(quiet) >= 10 and max(p for _, p in quiet) == 0.0
    env = make_env(course, smooth_w=1.0, smooth_steer=0.0, smooth_straight=0.0, smooth_pedal=1.0)
    env.reset(seed=4)
    flick = run(env, [(0.0, 1.0 if k % 2 else -1.0, -1.0) for k in range(20)])          # throttle pedal 0 <-> 1 (full at a = 1.0): |1 - 2 * 0 + 1| = 2
    assert len(flick) >= 10 and all(p == pytest.approx(2.0, abs=1e-6) for _, p in flick[2:])


def test_straightness_grid(course):
    env = make_env(course)
    g, ds = env._straight, env._straight_ds
    assert g.min() == 0.0 and g.max() > 0.9
    assert 0.2 < float((g > 0.5).mean()) < 0.8                                        # a good part of Monza is straight, a good part is not
    for c in course.track.find_corners().itertuples():
        assert g[int(c.s_apex / ds) % len(g)] == 0.0                                  # a corner's apex is never a straight
    # the longest run of g > 0.5 is the start straight (about 800 m; the centre line's own curvature noise keeps g below 1 even there)
    ones = np.concatenate([[0], (g > 0.5).astype(int), [0]])
    edges = np.flatnonzero(np.diff(ones))
    assert (edges[1::2] - edges[::2]).max() * ds > 500.0


def test_steering_away_from_straight_is_charged_on_the_straights_only(course):
    env = make_env(course, smooth_w=1.0, smooth_steer=0.0, smooth_straight=1.0, smooth_pedal=0.0)
    seen = []
    for s0 in (float(np.argmax(env._straight) * env._straight_ds), float(course.track.find_corners().s_apex.iloc[0])):
        env.reset(seed=2, options={"start_s": s0})
        steps = run(env, acts([0.3] * 5))
        sg = float(env._straight[int(env.s / env._straight_ds) % len(env._straight)])
        assert steps[-1][1] == pytest.approx(1.0 * sg * 0.3, abs=1e-6)                   # |steering| x straightness at the station the car reached
        seen.append(sg)
    assert seen[0] > 0.9                                                              # the start on the straightest place is on a straight: the steering is charged almost in full


def test_dataset_key_ignores_the_smoothness_controls_but_not_the_rest(course):
    dc = DemoConfig()
    a = dataset_key(dc, (str(TRACK), str(ENVELOPE), "mock", EnvConfig()))
    b = dataset_key(dc, (str(TRACK), str(ENVELOPE), "mock", replace(EnvConfig(), reward=replace(RewardConfig(), smooth_w=1.6, smooth_steer=2.0, straight_kappa=0.01))))
    c = dataset_key(dc, (str(TRACK), str(ENVELOPE), "mock", replace(EnvConfig(), reward=replace(RewardConfig(), penalty=50.0))))
    assert a == b and a != c


def test_smoothness_metrics_of_a_trace(course):
    env = make_env(course)
    n = 200
    s = np.linspace(0.0, 400.0, n)
    flat = {"steer": np.zeros(n), "throttle": np.full(n, 0.5), "brake": np.zeros(n), "station_m": s}
    m = smoothness(flat, env)
    assert m["steer_change"] == 0.0 and m["steer_jerk"] == 0.0 and m["pedal_jerk"] == 0.0 and m["both_pedals"] == 0.0
    chat = dict(flat, steer=0.05 * (-1.0) ** np.arange(n), brake=np.where(np.arange(n) % 2, 0.5, 0.0))
    m = smoothness(chat, env)
    assert m["steer_change"] == pytest.approx(0.1, abs=1e-9) and m["steer_jerk"] == pytest.approx(0.2, abs=1e-9)
    assert m["pedal_jerk"] == pytest.approx(1.0, abs=1e-9) and m["both_pedals"] == pytest.approx(0.5, abs=0.01)
    assert smoothness({k: v[:3] for k, v in flat.items()}, env) is None
