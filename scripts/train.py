"""Train SAC to drive Monza with the style of a reference driver.

    python scripts/train.py --driver attacker --backend ocd --out runs/attacker --max-hours 9.5
    python scripts/train.py --driver smooth   --backend ocd --out runs/smooth   --max-hours 9.5
    python scripts/train.py --driver attacker --backend mock --steps 5000 --out runs/test          # test run without OCD (stand-in vehicle, not OCD physics)

Resumable: if <out>/latest.zip exists the run continues from it (and from latest_buffer.pkl). Writes <out>/progress.csv (SB3 log), <out>/eval.csv (a full-lap evaluation every --eval-every steps),
<out>/latest.zip (overwritten), <out>/checkpoints/step_<n>.zip (one per evaluation, all kept, so earlier and later versions can be compared), <out>/best.zip (lowest lap time + style error among completed laps),
<out>/best_keep_style.zip (the fastest lap that is faster than the base driver AND keeps the style within --style-tol; <out>/records.csv lists every new record), <out>/laps/step_<n>.csv and
<out>/laps/base_driver.csv (the original driver model alone in the simulator: the baseline the agent is compared with).
Everything is logged: progress.csv (SAC losses, entropy coefficient, fps, mean return ... every 20 episodes) and the TensorBoard files (same numbers, if tensorboard is installed),
episodes.csv (one row per training episode: return split into its reward terms, how it ended, share of the time inside the track, line error, style error and the share of matched corners),
eval.csv (full-lap evaluation: lap time, deltas, style error, matched corners, off-track share, pedal/steering statistics, policy entropy, Q value, entropy coefficient),
corners/step_<n>.csv (the five style numbers of the agent against the reference at every corner), system.txt (disk and memory).
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.evaluate import compare_with_reference, run_lap  # noqa: E402
from style_rl.style_env import STYLE_GOOD, STYLE_OK, StyleRacingEnv  # noqa: E402


BASE_CMD = slice(20, 23)        # the base driver's suggested steering/throttle/brake inside the 27 observation numbers (style_env._observe)


class BCSAC(SAC):
    """SAC whose actor loss also pulls the action towards the base driver's command (the observation carries it), with a weight that fades linearly to zero over `bc_decay` steps.
    The Q term is normalised by the mean |Q| (as in TD3+BC), so one weight works at every reward scale. Used by the pure agents after their warm start: without the pull, the first SAC
    updates unlearn the copied behaviour; with it the policy can improve on the base driver step by step."""

    def __init__(self, *args, bc_weight=5.0, bc_decay=400_000, ent_max=0.1, **kwargs):
        self.bc_weight, self.bc_decay, self.ent_max = bc_weight, bc_decay, ent_max
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps, batch_size=64):
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer] + ([self.ent_coef_optimizer] if self.ent_coef_optimizer is not None else [])
        self._update_learning_rate(optimizers)
        w_bc = self.bc_weight * max(0.0, 1.0 - self.num_timesteps / self.bc_decay)
        ent_coef_losses, ent_coefs, actor_losses, critic_losses, bc_losses = [], [], [], [], []
        for gradient_step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = data.discounts if data.discounts is not None else self.gamma
            actions_pi, log_prob = self.actor.action_log_prob(data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef = torch.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            ent_coef_losses.append(ent_coef_loss.item())
            ent_coefs.append(ent_coef.item())
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            with torch.no_grad():
                self.log_ent_coef.clamp_(max=math.log(self.ent_max))      # the entropy coefficient may not run away (it did when the pull acted on sampled actions: 3e7)
                next_actions, next_log_prob = self.actor.action_log_prob(data.next_observations)
                next_q = torch.cat(self.critic_target(data.next_observations, next_actions), dim=1).min(dim=1, keepdim=True)[0] - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = data.rewards + (1 - data.dones) * discounts * next_q
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in self.critic(data.observations, data.actions))
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            min_q_pi = torch.cat(self.critic(data.observations, actions_pi), dim=1).min(dim=1, keepdim=True)[0]
            rl_loss = (ent_coef * log_prob - min_q_pi).mean()
            mean_actions, _, _ = self.actor.get_action_dist_params(data.observations)
            bc_loss = F.mse_loss(torch.tanh(mean_actions), data.observations[:, BASE_CMD])     # pull the MEAN action (not the sampled one: that would shrink the spread)
            actor_loss = rl_loss / (min_q_pi.abs().mean().detach() + 1e-6) + w_bc * bc_loss if w_bc > 0 else rl_loss
            actor_losses.append(actor_loss.item())
            bc_losses.append(bc_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/bc_weight", w_bc)


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
        self.ep_rows, self.recent = [], []

    def _log_episodes(self):
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            st = info.get("ep_stats") if done else None
            if st is None:
                continue
            row = {"step": self.num_timesteps, "hours": (time.time() - self.t0) / 3600, **st}
            self.ep_rows.append(row)
            self.recent = (self.recent + [row])[-50:]
            r = pd.DataFrame(self.recent)
            for k in ("inside_share", "rms_line_error", "mean_speed", "progress_m", "length", "corners", "style_error", "style_ok_share", "style_good_share",
                      "r_progress", "r_line", "r_jerk", "r_slow", "r_off_track", "r_stall", "r_style", "r_lap"):
                if k in r and r[k].notna().any():
                    self.logger.record(f"episode/{k}", float(r[k].mean()))
            self.logger.record("episode/off_track_end_share", float((r.end == "off_track").mean()))
            self.logger.record("episode/stalled_end_share", float((r.end == "stalled").mean()))
        if len(self.ep_rows) >= 25:
            self._flush_episodes()

    def _flush_episodes(self):
        if self.ep_rows:
            pd.DataFrame(self.ep_rows).to_csv(self.out / "episodes.csv", mode="a", header=not (self.out / "episodes.csv").exists(), index=False)
            self.ep_rows = []

    def _on_training_end(self):
        self._flush_episodes()

    def _policy_stats(self):
        try:
            rb = self.model.replay_buffer
            if rb.size() < 1024:
                return {}
            data = rb.sample(1024)
            with torch.no_grad():
                _, log_std, _ = self.model.actor.get_action_dist_params(data.observations)
                ent = float((log_std + 0.5 * (1.0 + np.log(2 * np.pi))).sum(dim=1).mean())
                q = float(torch.cat(self.model.critic(data.observations, self.model.actor(data.observations, deterministic=True)), dim=1).min(dim=1)[0].mean())
            out = {"policy_entropy": ent, "q_value": q, "mean_log_std": float(log_std.mean())}
            if hasattr(self.model, "log_ent_coef") and self.model.log_ent_coef is not None:
                out["ent_coef"] = float(torch.exp(self.model.log_ent_coef.detach()))
            return out
        except Exception as ex:                                  # logging must never stop the training
            return {"policy_stats_error": str(ex)[:80]}

    def _on_step(self):
        n = self.num_timesteps
        self._log_episodes()
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
            self._flush_episodes()
            res, df, corners = run_lap(self.eval_env, self.model)
            (self.out / "laps").mkdir(exist_ok=True)
            (self.out / "checkpoints").mkdir(exist_ok=True)
            df.to_csv(self.out / "laps" / f"step_{n}.csv", index=False)
            self.model.save(self.out / "checkpoints" / f"step_{n}")
            ref_t, base_t = self.eval_env.ref.lap_time, self.base["lap_time"]
            extra = {"corners_matched": int(len(corners)), "style_ok_share": float((corners.style_error <= STYLE_OK).mean()) if len(corners) else np.nan,
                     "style_good_share": float((corners.style_error <= STYLE_GOOD).mean()) if len(corners) else np.nan,
                     "steer_activity": float(np.abs(np.diff(df.steer)).mean()), "mean_brake": float(df.brake.mean()), "brake_share": float((df.brake > 0.05).mean()),
                     "full_throttle_share": float((df.throttle > 0.95).mean()), "mean_speed": float(df.speed.mean()), "max_abs_lateral": float(df.lateral.abs().max()), **self._policy_stats()}
            if len(corners):
                (self.out / "corners").mkdir(exist_ok=True)
                compare_with_reference(self.eval_env, corners).to_csv(self.out / "corners" / f"step_{n}.csv", index=False)
            row = {"step": n, "hours": (time.time() - self.t0) / 3600, **res, "reference_lap_time": ref_t, "base_lap_time": base_t, "base_style_error": self.base["style_error"],
                   "delta_vs_reference": res["lap_time"] - ref_t, "delta_vs_base": res["lap_time"] - base_t, **extra}
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


def warm_start(model, kw, args, n_steps=40_000, bc_steps=4000, critic_steps=3000):
    """Pure mode only: instead of random actions, start from a policy that copies the base driver.
    1. the base driver drives the training environment (with action noise in most episodes, so that the data also covers states off the perfect line); every step goes into the replay buffer,
    2. the actor is trained by supervised learning to output the base driver's three commands at the visited states,
    3. the critics are trained alone on that data for a while (the actor is fixed), so that the first SAC updates do not destroy the copied behaviour.
    The agent still outputs only steering, throttle and brake."""
    env = StyleRacingEnv(episode_steps=args.episode_steps, full_lap=False, seed=args.seed + 1, **kw)
    rng = np.random.default_rng(args.seed + 1)
    obs, _ = env.reset(seed=args.seed + 1)
    sigma = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
    O, A = [], []
    t0 = time.time()
    for t in range(n_steps):
        bs, bt, bb, _ = env._base                                  # the base driver's command at the current observation
        target = np.array([bs, bt, bb], np.float32)
        a = np.clip(target + sigma * rng.standard_normal(3), -1.0, 1.0).astype(np.float32)
        nobs, r, term, trunc, info = env.step(a)
        model.replay_buffer.add(obs[None], nobs[None], a[None], np.array([r], np.float32), np.array([float(term or trunc)], np.float32), [{"TimeLimit.truncated": bool(trunc and not term)}])
        O.append(obs)
        A.append(target)
        obs = nobs
        if term or trunc:
            obs, _ = env.reset()
            sigma = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
    print(f"warm start: {n_steps} base-driver steps collected in {time.time() - t0:.0f} s", flush=True)
    dev = model.device
    O, A = torch.as_tensor(np.array(O), device=dev), torch.as_tensor(np.array(A), device=dev).clamp(-0.98, 0.98)
    opt = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    for i in range(bc_steps):
        idx = torch.randint(0, len(O), (512,), device=dev)
        mean_actions, _, _ = model.actor.get_action_dist_params(O[idx])
        loss = F.mse_loss(torch.tanh(mean_actions), A[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 1000 == 0 or i == bc_steps - 1:
            print(f"warm start: behaviour cloning step {i}, mean squared error {loss.item():.5f}", flush=True)
    with torch.no_grad():                                            # little exploration noise to begin with
        model.actor.log_std.bias.fill_(-2.0)
        model.actor.log_std.weight.mul_(0.1)
    ent = torch.exp(model.log_ent_coef.detach())
    for i in range(critic_steps):
        data = model.replay_buffer.sample(256)
        with torch.no_grad():
            na, nlp = model.actor.action_log_prob(data.next_observations)
            nq = torch.cat(model.critic_target(data.next_observations, na), dim=1).min(dim=1, keepdim=True)[0] - ent * nlp.reshape(-1, 1)
            tq = data.rewards + (1 - data.dones) * model.gamma * nq
        loss = 0.5 * sum(F.mse_loss(q, tq) for q in model.critic(data.observations, data.actions))
        model.critic.optimizer.zero_grad()
        loss.backward()
        model.critic.optimizer.step()
        if i % 2 == 0:
            polyak_update(model.critic.parameters(), model.critic_target.parameters(), model.tau)
        if i % 1000 == 0 or i == critic_steps - 1:
            print(f"warm start: critic step {i}, loss {loss.item():.3f}", flush=True)


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
    ap.add_argument("--warm-start", action="store_true", help="pure mode: start from a policy that copies the base driver (behaviour cloning + critic warm-up) instead of random actions")
    ap.add_argument("--bc-weight", type=float, default=0.0, help="weight of the pull towards the base driver's command in the actor loss (0 = plain SAC); fades to 0 over --bc-decay steps")
    ap.add_argument("--bc-decay", type=int, default=400_000)
    ap.add_argument("--ent-max", type=float, default=0.1, help="upper limit of the entropy coefficient (BC-anchored SAC only)")
    ap.add_argument("--ent-init", type=float, default=0.0, help="initial entropy coefficient (0 = SB3 default 1.0); a warm-started policy needs a small one")
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
        model = (BCSAC if args.bc_weight > 0 else SAC).load(str(latest), env=env, device=device)
        if (args.out / "latest_buffer.pkl").is_file():
            model.load_replay_buffer(str(args.out / "latest_buffer"))
    else:
        algo = BCSAC if args.bc_weight > 0 else SAC
        extra = dict(bc_weight=args.bc_weight, bc_decay=args.bc_decay, ent_max=args.ent_max) if args.bc_weight > 0 else {}
        model = algo("MlpPolicy", env, **extra, learning_rate=args.lr, buffer_size=args.buffer, learning_starts=args.learning_starts, batch_size=args.batch_size, tau=0.005, gamma=args.gamma,
                    train_freq=args.train_freq, gradient_steps=1, ent_coef=(f"auto_{args.ent_init}" if args.ent_init > 0 else "auto"), policy_kwargs={"net_arch": [args.hidden, args.hidden]}, device=device, seed=args.seed, verbose=0)
    formats = ["csv", "stdout"]
    try:
        import tensorboard  # noqa: F401
        formats.append("tensorboard")
    except ImportError:
        print("tensorboard is not installed: only progress.csv is written", flush=True)
    model.set_logger(configure(str(args.out), formats))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except Exception:
        commit = "?"
    du = shutil.disk_usage(args.out)
    (args.out / "system.txt").write_text(f"commit {commit}" + chr(10) + f"disk: {du.total / 1e9:.1f} GB total, {du.free / 1e9:.1f} GB free at the start" + chr(10) + f"device {device}, torch {torch.__version__}" + chr(10))
    n_par = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"{args.driver} ({args.mode}, {args.backend}): reference lap {eval_env.ref.lap_time:.1f} s; network {args.hidden}x{args.hidden}, {n_par:,} parameters (actor + 2 critics + 2 target critics), device {device}, one update per {args.train_freq} step(s)", flush=True)
    base_env = eval_env if args.mode == "residual" else StyleRacingEnv(full_lap=True, **{**kw, "mode": "residual"})   # the original driver: the base driver alone (for a pure agent, which has no base driver, a separate residual-mode environment)
    res, df0, _ = run_lap(base_env, None)
    if args.warm_start and args.mode == "pure" and not latest.is_file():
        warm_start(model, kw, args)
        ws, _, wc = run_lap(eval_env, model)
        print(f"after the warm start (before any SAC update): {ws['end']}, lap {ws['lap_time']:.1f} s, style error {ws['style_error']:.2f}, off-track {100 * ws['off_track_share']:.1f}%", flush=True)
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
