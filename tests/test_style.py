import numpy as np

from style_rl.reference import Reference
from style_rl.style import corner_features, style_error
from style_rl.style_env import StyleRacingEnv


def test_corner_features_known_pedals():
    L = 1000.0
    p = np.arange(0, L, 2.0)
    apex = 500.0
    brake = np.where((p > apex - 120) & (p < apex - 20), 0.8, 0.0)
    thr = np.where(p > apex + 40, 1.0, 0.0)
    f = corner_features(p, brake, thr, apex, L)
    assert abs(f[0] - 118) <= 2 and abs(f[1] - 22) <= 2 and abs(f[2] - 0.8) < 1e-6 and abs(f[4] - 42) <= 2
    assert style_error(f, f) == 0.0


def test_reference_has_different_styles():
    a, s = Reference("attacker"), Reference("smooth")
    common = set(a.corner_ref) & set(s.corner_ref)
    assert common
    assert np.mean([a.corner_ref[c][2] - s.corner_ref[c][2] for c in common]) > 0.1      # the attacker brakes harder


def test_env_runs_on_the_stand_in():
    env = StyleRacingEnv(driver="attacker", backend="mock", episode_steps=50)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (27,) and np.isfinite(obs).all()
    for _ in range(50):
        obs, r, term, trunc, info = env.step(np.zeros(3, np.float32))
        assert np.isfinite(r) and np.isfinite(obs).all()
        if term or trunc:
            break
