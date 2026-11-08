"""Train SAC to drive Monza with the style of a reference driver.

    python scripts/train.py --driver attacker --backend ocd --out runs/attacker --max-hours 9.5
    python scripts/train.py --driver smooth   --backend ocd --out runs/smooth   --max-hours 9.5
    python scripts/train.py --driver attacker --backend mock --steps 5000 --out runs/test          # test run without OCD (stand-in vehicle, not OCD physics)

Resumable: if <out>/latest.zip exists the run continues from it (and from latest_buffer.pkl). Writes <out>/progress.csv (SB3 log), <out>/eval.csv (a full-lap evaluation every --eval-every steps),
<out>/latest.zip (overwritten), <out>/checkpoints/step_<n>.zip (one per evaluation, all kept, so earlier and later versions can be compared), <out>/best.zip (lowest lap time + style error among completed laps),
<out>/best_keep_style.zip (the fastest lap that is faster than the base driver AND keeps the style within --style-tol; <out>/records.csv lists every new record), <out>/laps/step_<n>.csv and
<out>/laps/base_driver.csv (the original driver model alone in the simulator: the baseline the agent is compared with).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.evaluate import run_lap  # noqa: E402
from style_rl.style_env import StyleRacingEnv  # noqa: E402


class Periodic(BaseCallback):
    def __init__(self, args, out, eval_env, t_start, base):
        super().__init__()
        self.args, self.out, self.eval_env, self.t0, self.base = args, Path(out), eval_env, t_start, base
        self.best = np.inf
        b_err = base["style_error"]
        self.style_tol = args.style_tol if args.style_tol > 0 else (1.25 * b_err + 0.3 if np.isfinite(b_err) else 2.0)
        self.best_fast = base["lap_time"] if base["completed"] else np.inf
        rec = self.out / "records.csv"
        if rec.is_file():
            self.best_fast = min(self.best_fast, float(pd.read_csv(rec).lap_time.min()))
        self.last_eval, self.last_save, self.last_print = 0, 0, 0
        self.t_print = time.time()

    def _on_step(self):
        n = self.num_timesteps
        if n - self.last_print >= 1000:
            now = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] step {n}: {1000 / (now - self.t_print):.0f} steps/s, elapsed {(now - self.t0) / 3600:.2f} h", flush=True)
            self.last_print, self.t_print = n, now
        if n - self.last_save >= self.args.save_every:
            self.model.save(self.out / "latest")
            self.model.save_replay_buffer(self.out / "latest_buffer")
            self.last_save = n
        if n - self.last_eval >= self.args.eval_every:
            self.last_eval = n
            res, df, corners = run_lap(self.eval_env, self.model)
            (self.out / "laps").mkdir(exist_ok=True)
            (self.out / "checkpoints").mkdir(exist_ok=True)
            df.to_csv(self.out / "laps" / f"step_{n}.csv", index=False)
            self.model.save(self.out / "checkpoints" / f"step_{n}")
            ref_t, base_t = self.eval_env.ref.lap_time, self.base["lap_time"]
            row = {"step": n, "hours": (time.time() - self.t0) / 3600, **res, "reference_lap_time": ref_t, "base_lap_time": base_t, "base_style_error": self.base["style_error"],
                   "delta_vs_reference": res["lap_time"] - ref_t, "delta_vs_base": res["lap_time"] - base_t}
            pd.DataFrame([row]).to_csv(self.out / "eval.csv", mode="a", header=not (self.out / "eval.csv").exists(), index=False)
            print(f"  EVAL step {n}: {row['end']}, lap {res['lap_time']:.1f} s (reference {row['reference_lap_time']:.1f}), style error {res['style_error']:.2f}, off-track {100 * res['off_track_share']:.1f}%", flush=True)
            if res["completed"] and np.isfinite(res["style_error"]) and res["style_error"] <= self.style_tol and res["lap_time"] < self.best_fast:
                self.best_fast = res["lap_time"]
                self.model.save(self.out / "best_keep_style")
                pd.DataFrame([{"step": n, "lap_time": res["lap_time"], "style_error": res["style_error"], "delta_vs_base": row["delta_vs_base"], "delta_vs_reference": row["delta_vs_reference"],
                               "checkpoint": f"checkpoints/step_{n}.zip"}]).to_csv(self.out / "records.csv", mode="a", header=not (self.out / "records.csv").exists(), index=False)
                print(f"  NEW RECORD at step {n}: lap {res['lap_time']:.2f} s ({row['delta_vs_base']:+.2f} s against the base driver), style error {res['style_error']:.2f} <= {self.style_tol:.2f}", flush=True)
            if res["completed"]:
                score = res["lap_time"] + 10.0 * (res["style_error"] if np.isfinite(res["style_error"]) else 5.0)
                if score < self.best:
                    self.best = score
                    self.model.save(self.out / "best")
        return (time.time() - self.t0) < self.args.max_hours * 3600


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--driver", choices=["attacker", "smooth"], required=True)
    ap.add_argument("--backend", choices=["ocd", "mock"], default="ocd")
    ap.add_argument("--mode", choices=["residual", "pure"], default="residual")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--max-hours", type=float, default=9.5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--buffer", type=int, default=300_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--learning-starts", type=int, default=5000)
    ap.add_argument("--train-freq", type=int, default=0, help="environment steps per update; 0 = 1 on a GPU, 2 on a CPU")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--episode-steps", type=int, default=700)
    ap.add_argument("--res-steer", type=float, default=0.35)
    ap.add_argument("--res-pedal", type=float, default=0.35)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--style-tol", type=float, default=0.0, help="largest style error for which a faster lap counts as a record; 0 = 1.25 x the base driver's style error + 0.3")
    ap.add_argument("--save-every", type=int, default=10_000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.train_freq == 0:
        args.train_freq = 1 if device == "cuda" else 2
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=1))
    kw = dict(driver=args.driver, backend=args.backend, mode=args.mode, res_steer=args.res_steer, res_pedal=args.res_pedal)
    env = Monitor(StyleRacingEnv(episode_steps=args.episode_steps, full_lap=False, seed=args.seed, **kw))
    eval_env = StyleRacingEnv(full_lap=True, **kw)
    latest = args.out / "latest.zip"
    if latest.is_file():
        print("resuming from", latest, flush=True)
        model = SAC.load(str(latest), env=env, device=device)
        if (args.out / "latest_buffer.pkl").is_file():
            model.load_replay_buffer(str(args.out / "latest_buffer"))
    else:
        model = SAC("MlpPolicy", env, learning_rate=args.lr, buffer_size=args.buffer, learning_starts=args.learning_starts, batch_size=args.batch_size, tau=0.005, gamma=args.gamma,
                    train_freq=args.train_freq, gradient_steps=1, ent_coef="auto", policy_kwargs={"net_arch": [args.hidden, args.hidden]}, device=device, seed=args.seed, verbose=0)
    model.set_logger(configure(str(args.out), ["csv", "stdout"]))
    n_par = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"{args.driver} ({args.mode}, {args.backend}): reference lap {eval_env.ref.lap_time:.1f} s; network {args.hidden}x{args.hidden}, {n_par:,} parameters (actor + 2 critics + 2 target critics), device {device}, one update per {args.train_freq} step(s)", flush=True)
    res, df0, _ = run_lap(eval_env, None)
    print(f"base driver alone (zero residual): {res['end']}, lap {res['lap_time']:.1f} s, style error {res['style_error']:.2f}, off-track {100 * res['off_track_share']:.1f}%", flush=True)
    (args.out / "laps").mkdir(exist_ok=True)
    df0.to_csv(args.out / "laps" / "base_driver.csv", index=False)
    (args.out / "baseline.json").write_text(json.dumps({k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in {**res, "reference_lap_time": eval_env.ref.lap_time}.items()}, indent=1, default=float))
    cb = Periodic(args, args.out, eval_env, time.time(), res)
    remaining = max(args.steps - model.num_timesteps, 1)
    model.learn(total_timesteps=remaining, callback=cb, reset_num_timesteps=False, log_interval=20)
    model.save(args.out / "latest")
    model.save_replay_buffer(args.out / "latest_buffer")
    print("finished", flush=True)


if __name__ == "__main__":
    main()
