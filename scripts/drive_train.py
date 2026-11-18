"""Train the RL-only driver (style_rl.drive):

    python scripts/drive_train.py --backend ocd --out runs/drive/monza_a --max-hours 5
    python scripts/drive_train.py --backend mock --smoke --out /tmp/drive_smoke        # a few minutes on any machine; NOT OCD physics

Resume a stopped run with the same command plus --resume."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.drive.config import EnvConfig  # noqa: E402
from style_rl.drive.demos import DemoConfig  # noqa: E402
from style_rl.drive.train import TrainConfig, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="ocd", choices=["ocd", "mock"])
    ap.add_argument("--track", default=str(ROOT / "data/tracks/monza_geometry.csv"))
    ap.add_argument("--envelope", default=str(ROOT / "data/ocd_envelope.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--gradient-steps", type=int, default=None, help="per vector step; default n_envs / 2 (updates per environment step 0.5)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--demo-transitions", type=int, default=200_000)
    ap.add_argument("--demo-workers", type=int, default=4)
    ap.add_argument("--demo-dir", default=str(ROOT / "data/demos"))
    ap.add_argument("--eval-interval", type=int, default=50_000)
    ap.add_argument("--eval-stochastic", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny settings to test the pipeline")
    ap.add_argument("--no-gate", action="store_true", help="continue even if the behaviour-cloned copy cannot finish a clean lap (tests only)")
    a = ap.parse_args()
    cfg = TrainConfig(out=a.out, track_csv=a.track, envelope_json=a.envelope, backend=a.backend, seed=a.seed, n_envs=a.n_envs, gradient_steps=a.gradient_steps or max(a.n_envs // 2, 1),
                      device=a.device, total_steps=a.steps, max_hours=a.max_hours, eval_interval=a.eval_interval, eval_stochastic_laps=a.eval_stochastic, demo_dir=a.demo_dir,
                      demo=DemoConfig(n_transitions=a.demo_transitions, workers=a.demo_workers))
    if a.smoke:
        cfg = replace(cfg, total_steps=6000, demo=DemoConfig(n_transitions=4000, workers=1), bc_epochs=3, offline_blocks=2, offline_steps_per_block=200, online_warmup_steps=2000,
                      guard_interval=2000, guard_window=4000, guard_n_starts=3, guard_probe_s=10.0, eval_interval=3000, eval_stochastic_laps=1, checkpoint_interval=3000,
                      buffer_size=50_000, batch_size=128, hidden=(64, 64, 64))
    if a.no_gate:
        cfg = replace(cfg, gate_required=False)
    run(cfg, resume=a.resume)


if __name__ == "__main__":
    main()
