"""How fast does the SAC learner run on this machine?   python3 scripts/bench_learner.py [--threads 1 2 4] [--batches 512 256]

Fills a replay buffer with random transitions and times gradient steps of the driving agent (3x256, LayerNorm critics, 3-step returns) for each torch thread count and batch size.
The learner, not the simulator, limits the training speed: one vector step costs (n_envs x environment time) + gradient_steps x (1 / gradient steps per second)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.drive import agent as ag  # noqa: E402
from style_rl.drive.config import EnvConfig  # noqa: E402
from style_rl.drive.vec import build_vec_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--batches", type=int, nargs="+", default=[512, 256])
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args()
    venv = build_vec_env(1, ROOT / "data/tracks/monza_geometry.csv", ROOT / "data/ocd_envelope.json", "mock", EnvConfig(), subprocess=False)
    for batch in a.batches:
        model = ag.build_model(venv, batch_size=batch, buffer_size=20000, gradient_steps=1)
        rb = model.replay_buffer
        rng = np.random.default_rng(0)
        for _ in range(3000):
            rb.add(rng.normal(size=(1, 159)).astype(np.float32), rng.normal(size=(1, 159)).astype(np.float32), rng.uniform(-1, 1, size=(1, 3)).astype(np.float32),
                   np.array([1.0], np.float32), np.array([0.0], np.float32), [{}])
        model.set_logger(__import__("stable_baselines3.common.logger", fromlist=["configure"]).configure(None, []))
        for t in a.threads:
            th.set_num_threads(t)
            model.train(10, batch)                                   # warm up
            t0 = time.perf_counter()
            model.train(a.steps, batch)
            dt = time.perf_counter() - t0
            print(f"batch {batch:4d}  torch threads {t}:  {a.steps / dt:6.1f} gradient steps/s", flush=True)


if __name__ == "__main__":
    main()
