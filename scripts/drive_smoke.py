"""Smoke test of the portable driving environment (style_rl.drive) on the stand-in car:   python scripts/drive_smoke.py [--backend mock|ocd] [--fraction 0.9]

Builds the course (reference line + QSS ceiling), checks the observation contract, lets the scripted driver play random-start episodes and a flying-lap evaluation."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.drive.backend import make_backend  # noqa: E402
from style_rl.drive.config import OBS_DIM, EnvConfig  # noqa: E402
from style_rl.drive.course import Course  # noqa: E402
from style_rl.drive.env import DrivingEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def play(env, fraction, options=None, seed=0, max_steps=100000):
    obs, info = env.reset(seed=seed, options=options)
    ret, n, lap = 0.0, 0, None
    while True:
        a = env.demo.act(env.state, env.s, env.demo.lap_target(env.s, fraction, env.state["speed"]))
        obs, r, term, trunc, info = env.step(a)
        ret, n = ret + r, n + 1
        if term or trunc or n >= max_steps:
            return obs, ret, n, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--track", default=str(ROOT / "data/tracks/monza_geometry.csv"))
    ap.add_argument("--fraction", type=float, default=0.9, help="the scripted driver aims at this fraction of the QSS ceiling")
    ap.add_argument("--episodes", type=int, default=6)
    a = ap.parse_args()

    t0 = time.time()
    cfg = EnvConfig()
    course = Course(a.track, ROOT / "data/ocd_envelope.json", margin_m=cfg.margin_m)
    print(f"course {course.name}: {course.L:.1f} m, {len(course.corners)} corners, QSS ceiling {course.qss_lap_time_s:.2f} s, built in {time.time() - t0:.1f} s")
    backend = make_backend(a.backend, a.track, cfg.steer_lock_rad, cfg.dt_s)
    env = DrivingEnv(course, backend, cfg)
    obs, info = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32, obs.shape
    print(f"observation {obs.shape}, clipped values {int(np.sum(np.abs(obs) >= cfg.obs.clip))}, start {info['start']}")
    print("ego", np.round(obs[:9], 3), "\nline", np.round(obs[9:12], 3), "\nfirst preview station (ref, left, right x/y)", np.round(obs[12:18], 3), "\nlast station", np.round(obs[12 + 114:12 + 120], 3))

    print(f"\nrandom-start episodes, scripted driver at {a.fraction:.0%} of the ceiling")
    for k in range(a.episodes):
        t = time.time()
        _, ret, n, info = play(env, a.fraction, seed=10 + k)
        print(f"  ep {k}: {n} decisions, {info['progress_m']:7.1f} m, return {ret:8.1f}, end {info['termination'] or info['truncation']:10s}, {n / (time.time() - t):.0f} decisions/s")

    print("\nflying-lap evaluation (2 timed laps)")
    t = time.time()
    _, ret, n, info = play(env, a.fraction, options={"evaluation": True}, seed=0)
    print(f"  end {info['termination'] or info['truncation']}, lap times {np.round(info.get('lap_times_s', []), 2)}, {n} decisions in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
