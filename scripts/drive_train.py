"""Train the RL-only driver (style_rl.drive):

    python scripts/drive_train.py --backend ocd --out runs/drive/monza_a --max-hours 5
    python scripts/drive_train.py --backend mock --smoke --out /tmp/drive_smoke        # a few minutes on any machine; NOT OCD physics

Resume a stopped run with the same command plus --resume."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.drive.cars import PRESETS, apply_preset  # noqa: E402
from style_rl.drive.config import EnvConfig  # noqa: E402
from style_rl.drive.demos import DemoConfig  # noqa: E402
from style_rl.drive.randomize import RandomizeConfig  # noqa: E402
from style_rl.drive.train import TrainConfig, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--car", default="f317", choices=sorted(PRESETS), help="f317: Dallara F317 in OCD (run a1); sf70h_f1: the F1 car of ocd_lap (needs OCD_FIRST_LAP_DIR, data/f1/ocd_envelope_sf70h_f1.json). Sets the simulator, "
                    "the limits file, the scripted driver's wheelbase and traction control, the wheel positions and the demonstration speed")
    ap.add_argument("--backend", default=None, help="ocd, mock (a kinematic stand-in, NOT OCD physics) or ocdlap:<car>; default: the car's")
    ap.add_argument("--track", default=str(ROOT / "data/tracks/monza_geometry.csv"))
    ap.add_argument("--tracks", default="", help="comma-separated track names (data/tracks/<name>_geometry.csv): one policy trained on all of them, e.g. monza,barcelona,red_bull_ring,silverstone")
    ap.add_argument("--envelope", default=None, help="the car's limits (json from scripts/measure_ocd_envelope.py); default: the car's")
    ap.add_argument("--pedal-full-at", type=float, default=None, help="the action at which a pedal is fully pressed (default 1.0 = the original mapping, run a1; 0.7 = full pedal for every action >= 0.7)")
    ap.add_argument("--train-inset-m", type=float, default=None, help="training only: the track counts this much narrower on both sides for the off-track test (a safety buffer); evaluation uses the true edges")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--gradient-steps", type=int, default=None, help="per vector step; default n_envs / 2 (updates per environment step 0.5)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1, help="CPU threads of the learner (1 is the replica; 2-3 is a little faster on a machine with spare cores)")
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--buffer-size", type=int, default=1_000_000, help="replay buffer (transitions); 1M is ~1.3 GB in memory and on disk at every save, use ~200000 on a small machine")
    ap.add_argument("--demo-transitions", type=int, default=200_000)
    ap.add_argument("--demo-workers", type=int, default=4)
    ap.add_argument("--demo-dir", default=str(ROOT / "data/demos"))
    ap.add_argument("--eval-interval", type=int, default=50_000)
    ap.add_argument("--eval-stochastic", type=int, default=5)
    ap.add_argument("--init", default="bc", choices=["bc", "bias"], help="bc: copy the scripted driver first; bias: no demonstrations")
    ap.add_argument("--blend-steps", type=int, default=0, help="> 0: the scripted driver assists the executed action, its weight falling linearly from --blend-w0 to 0 over this many steps")
    ap.add_argument("--blend-w0", type=float, default=0.9)
    ap.add_argument("--blend-fraction", type=float, default=0.85)
    ap.add_argument("--randomize", action="store_true", help="domain randomisation of the car and the sensors in every training episode (for the transfer to Assetto Corsa)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny settings to test the pipeline")
    ap.add_argument("--no-gate", action="store_true", help="continue even if the behaviour-cloned copy cannot finish a clean lap (tests only)")
    a = ap.parse_args()
    tracks = tuple(str(ROOT / f"data/tracks/{t.strip()}_geometry.csv") for t in a.tracks.split(",") if t.strip())
    preset = PRESETS[a.car]
    env = apply_preset(a.car, EnvConfig())
    if a.pedal_full_at is not None:
        env = replace(env, pedal_full_at=a.pedal_full_at)
    if a.train_inset_m is not None:
        env = replace(env, train_inset_m=a.train_inset_m)
    cfg = TrainConfig(out=a.out, track_csv=a.track, tracks=tracks, envelope_json=a.envelope or str(ROOT / preset.envelope), backend=a.backend or preset.backend, seed=a.seed, n_envs=a.n_envs,
                      gradient_steps=a.gradient_steps or max(a.n_envs // 2, 1), device=a.device, torch_threads=a.torch_threads, total_steps=a.steps, buffer_size=a.buffer_size, max_hours=a.max_hours,
                      eval_interval=a.eval_interval, eval_stochastic_laps=a.eval_stochastic, demo_dir=a.demo_dir,
                      demo=DemoConfig(n_transitions=a.demo_transitions, workers=a.demo_workers, fraction_range=preset.demo_fraction), init_mode=a.init, blend_steps=a.blend_steps, blend_w0=a.blend_w0,
                      blend_fraction=a.blend_fraction, env=env)
    if a.smoke:
        cfg = replace(cfg, total_steps=6000, demo=DemoConfig(n_transitions=4000, workers=1), bc_epochs=3, offline_blocks=2, offline_steps_per_block=200, online_warmup_steps=2000,
                      guard_interval=2000, guard_window=4000, guard_n_starts=3, guard_probe_s=10.0, eval_interval=3000, eval_stochastic_laps=1, checkpoint_interval=3000,
                      buffer_size=50_000, batch_size=128, hidden=(64, 64, 64))
    if a.randomize:
        cfg = replace(cfg, env=replace(cfg.env, randomize=RandomizeConfig(enabled=True)))
    if a.no_gate:
        cfg = replace(cfg, gate_required=False)
    run(cfg, resume=a.resume)


if __name__ == "__main__":
    main()
