"""Train SAC on FULL laps of Monza in OCD to match a target 12-number style and get faster (style_rl/style12_env.py has the reward).

    python scripts/train12.py --driver attacker --backend ocd --out runs12/attacker --max-hours 4
    python scripts/train12.py --driver smooth   --backend mock --steps 6000 --out /tmp/t12      # test without OCD (stand-in car)

The agent's three numbers are steering, throttle, brake. It starts as a copy of the neutral base driver (a pure-pursuit driver on the track's own line, not on the driver's line) and is pulled
towards it while the pull fades (BC-anchored SAC from scripts/train.py).
Writes into --out:
  baseline_pure_pursuit_driver.csv / baseline_neutral.csv   the pure-pursuit lap on the driver's own mapped lap (the driver's lap in the simulator) and on the neutral line, with baselines.json (lap time, 12 numbers, distance D)
  library.csv        EVERY finished lap (training laps and evaluation laps): lap time, distance D of its 12 numbers to the target, start, share off the track, the 12 numbers, multiplier, file
  laps/              the trajectory of every finished lap (training: lap_<n>.npz, evaluation: eval_<step>.csv)
  episodes.csv, progress.csv, eval.csv, checkpoints/, latest.zip, best_keep_style.zip (fastest evaluation lap with D <= d_max), state.json (multiplier, lap counter)
"""
import argparse
import json
import shutil
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
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import polyak_update

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style_rl.dstyle.features import KEYS  # noqa: E402
from style_rl.style12_env import Style12Env  # noqa: E402
from train import BCSAC  # noqa: E402


def run_lap12(env, policy=None, start=0.0, max_steps=9000):
    """One full lap from `start`. policy None = the base driver of the environment (pure pursuit, its command is the action)."""
    obs, _ = env.reset(seed=0, options={"start": start})
    info = {}
    for _ in range(max_steps):
        a = env.base_action() if policy is None else policy.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    lg = {k: np.asarray(v) for k, v in env.log.items()}
    df = pd.DataFrame(lg)
    st = info.get("ep_stats", {})
    numbers = env.meas.copy()
    res = {"end": st.get("end", info.get("end", "?")), "completed": info.get("end") == "lap", "lap_time": info.get("lap_time", np.nan), "D": float(env.D_prev), "off_share": float((~df.inside.astype(bool)).mean()),
           "distance_m": float(df.progress.iloc[-1]) if len(df) else 0.0}
    return res, df, numbers


class Library:
    """Every finished lap goes in here."""

    def __init__(self, out, start_id=0):
        self.out, self.n = Path(out), start_id
        (self.out / "laps").mkdir(exist_ok=True)

    def add(self, kind, step, hours, lap_time, D, start_m, off_share, numbers, lam, arrays=None, df=None, tag=""):
        self.n += 1
        name = f"laps/{kind}_{tag or self.n}"
        if df is not None:
            df.to_csv(self.out / f"{name}.csv", index=False)
            name += ".csv"
        elif arrays is not None:
            np.savez_compressed(self.out / f"{name}.npz", **arrays)
            name += ".npz"
        row = {"id": self.n, "kind": kind, "step": step, "hours": hours, "lap_time": lap_time, "D": D, "start_m": start_m, "off_share": off_share, "lam": lam, "file": name, **dict(zip(KEYS, np.asarray(numbers, float)))}
        pd.DataFrame([row]).to_csv(self.out / "library.csv", mode="a", header=not (self.out / "library.csv").exists(), index=False)


class Periodic(BaseCallback):
    def __init__(self, args, out, eval_env, t_start, baselines, lib):
        super().__init__()
        self.args, self.out, self.eval_env, self.t0, self.base, self.lib = args, Path(out), eval_env, t_start, baselines, lib
        self.last_eval = self.last_save = self.last_print = 0
        self.t_print = time.time()
        self.ep_rows = []
        rec = self.out / "records.csv"
        self.best_fast = float(pd.read_csv(rec).lap_time.min()) if rec.is_file() else np.inf

    def _flush(self):
        if self.ep_rows:
            pd.DataFrame(self.ep_rows).to_csv(self.out / "episodes.csv", mode="a", header=not (self.out / "episodes.csv").exists(), index=False)
            self.ep_rows = []

    def _on_training_end(self):
        self._flush()

    def _on_step(self):
        n = self.num_timesteps
        hours = (time.time() - self.t0) / 3600
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if not done or "ep_stats" not in info:
                continue
            st = info["ep_stats"]
            self.ep_rows.append({"step": n, "hours": hours, **st})
            if "lap" in info:
                lap = info["lap"]
                self.lib.add("train", n, hours, lap["lap_time"], lap["D"], lap["start_m"], st["off_share"], lap["numbers"], st["lam"], arrays=lap["arrays"])
            self.logger.record("episode/D_end", st["D_end"])
            self.logger.record("episode/lam", st["lam"])
            self.logger.record("episode/off_share", st["off_share"])
            self.logger.record("episode/progress_m", st["progress_m"])
            self.logger.record("episode/r_time", st["r_time"])
            self.logger.record("episode/r_style", st["r_style"])
        if len(self.ep_rows) >= 20:
            self._flush()
        if n - self.last_print >= 1000:
            now = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] step {n}: {1000 / (now - self.t_print):.0f} steps/s, elapsed {hours:.2f} h", flush=True)
            self.last_print, self.t_print = n, now
        if n - self.last_save >= self.args.save_every:
            self.model.save(self.out / "latest")
            self.model.save_replay_buffer(self.out / "latest_buffer")
            (self.out / "state.json").write_text(json.dumps({"lam": self.training_env.envs[0].unwrapped.lam, "library_n": self.lib.n}))
            self.last_save = n
        if n - self.last_eval >= self.args.eval_every:
            self.last_eval = n
            self._flush()
            self.eval_env.lam = self.training_env.envs[0].unwrapped.lam
            res, df, numbers = run_lap12(self.eval_env, self.model)
            (self.out / "checkpoints").mkdir(exist_ok=True)
            self.model.save(self.out / "checkpoints" / f"step_{n}")
            self.lib.add("eval", n, hours, res["lap_time"], res["D"], 0.0, res["off_share"], numbers, self.eval_env.lam, df=df, tag=f"step_{n}")
            row = {"step": n, "hours": hours, **res, "lam": self.eval_env.lam, "pure_pursuit_lap_time": self.base["pure_pursuit_driver"]["lap_time"], "pure_pursuit_D": self.base["pure_pursuit_driver"]["D"],
                   "neutral_lap_time": self.base["neutral"]["lap_time"], "neutral_D": self.base["neutral"]["D"], "mean_speed": float(df.speed.mean()), "mean_brake": float(df.brake.mean()), "full_throttle_share": float((df.throttle > 0.95).mean())}
            pd.DataFrame([row]).to_csv(self.out / "eval.csv", mode="a", header=not (self.out / "eval.csv").exists(), index=False)
            print(f"  EVAL step {n}: {res['end']}, lap {res['lap_time']:.2f} s (pure pursuit {self.base['pure_pursuit_driver']['lap_time']:.2f}), D {res['D']:.2f} (pure pursuit {self.base['pure_pursuit_driver']['D']:.2f}, margin {self.args.d_max}), "
                  f"off-track {100 * res['off_share']:.1f}%, lam {self.eval_env.lam:.1f}", flush=True)
            if res["completed"] and res["D"] <= self.args.d_max and res["lap_time"] < self.best_fast:
                self.best_fast = res["lap_time"]
                self.model.save(self.out / "best_keep_style")
                pd.DataFrame([{"step": n, "lap_time": res["lap_time"], "D": res["D"], "checkpoint": f"checkpoints/step_{n}.zip"}]).to_csv(self.out / "records.csv", mode="a", header=not (self.out / "records.csv").exists(), index=False)
                print(f"  NEW RECORD: lap {res['lap_time']:.2f} s with D {res['D']:.2f} <= {self.args.d_max}", flush=True)
        return (time.time() - self.t0) < self.args.max_hours * 3600


def warm_start(model, env_kw, args, n_steps=40_000, bc_steps=4000, critic_steps=3000):
    """Start from a copy of the neutral base driver: it drives the training environment (with action noise in most episodes), every step goes into the replay buffer, the actor is trained to
    output its command, then the critics are trained alone for a while. (As in scripts/train.py, with full-lap episodes.)"""
    env = Style12Env(seed=args.seed + 1, **env_kw)
    rng = np.random.default_rng(args.seed + 1)
    obs, _ = env.reset(seed=args.seed + 1)
    sigma = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
    O, A = [], []
    t0 = time.time()
    for t in range(n_steps):
        target = np.array(env._base[:3], np.float32)
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
    with torch.no_grad():                                    # the policy's own noise starts (almost) at zero: white noise written into the pedals and the wheel is measured as style (see main)
        model.actor.log_std.bias.fill_(args.init_log_std)
        model.actor.log_std.weight.zero_()
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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--max-hours", type=float, default=4.0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--train-freq", type=int, default=0, help="environment steps per update; 0 = 1 on a GPU, 2 on a CPU")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--save-every", type=int, default=10_000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-max", type=float, default=0.3, help="style margin: largest distance D (spread units) of the lap's 12 numbers to the target that the multiplier tolerates")
    ap.add_argument("--lam0", type=float, default=3.0)
    ap.add_argument("--lam-eta", type=float, default=0.2)
    ap.add_argument("--lam-max", type=float, default=20.0)
    ap.add_argument("--rand-start", type=float, default=0.5, help="share of training laps that start at a random point of the lap instead of the line")
    ap.add_argument("--bc-weight", type=float, default=5.0)
    ap.add_argument("--bc-decay", type=int, default=500_000)
    ap.add_argument("--ent-max", type=float, default=0.1)
    ap.add_argument("--ent-init", type=float, default=1e-4, help="initial entropy coefficient; tiny: the exploration is the correlated noise below, not the policy's own white noise")
    ap.add_argument("--target-entropy", type=float, default=-30.0, help="very low: the policy's own noise is not wanted (white noise per step shows up in hand speed and steering corrections)")
    ap.add_argument("--init-log-std", type=float, default=-7.0, help="log of the policy's own noise at the start")
    ap.add_argument("--ou-steer", type=float, default=0.03, help="exploration: Ornstein-Uhlenbeck noise (slow, correlated), standard deviation of the steering command")
    ap.add_argument("--ou-throttle", type=float, default=0.05)
    ap.add_argument("--ou-brake", type=float, default=0.02, help="small: brake noise above 0.05 counts as a brake zone and cuts the throttle")
    ap.add_argument("--ou-tau", type=float, default=20.0, help="correlation time of the exploration noise in steps (20 steps = 0.8 s)")
    ap.add_argument("--warm-steps", type=int, default=40_000)
    ap.add_argument("--no-warm-start", action="store_true", help="start from a random network instead of a copy of the neutral base driver (use with --bc-weight 0, a normal entropy coefficient and the OU noise at 0)")
    ap.add_argument("--hide-base-command", action="store_true", help="do not put the pure-pursuit command into the observation")
    ap.add_argument("--line-w", type=float, default=0.0, help="penalty per step for being off the track's own line (see style12_env.py); 0 = none")
    ap.add_argument("--line-delta", type=float, default=1.0, help="metres from the line below which the line penalty is quadratic (smooth), linear beyond")
    ap.add_argument("--line-decay", type=int, default=0, help="steps over which the line penalty fades to --line-floor of its value (0 = constant)")
    ap.add_argument("--line-floor", type=float, default=0.4)
    ap.add_argument("--prog-w", type=float, default=0.0, help="reward per metre driven (the same for every finished lap; a crash loses the progress still to come)")
    ap.add_argument("--style-dense", type=float, default=0.0, help="weight of the style penalty paid all along the lap: -style_dense x lam x D x (share of the lap driven)")
    ap.add_argument("--style-clip", type=float, default=4.0, help="no single number counts for more than this many spread units in that penalty")
    ap.add_argument("--start-speed-lo", type=float, default=0.85, help="a training lap that starts at a random point starts at between this share and 1.0 of the line's speed there")
    ap.add_argument("--init-log-std-scratch", action="store_true", help="with --no-warm-start: also start the policy's own noise at --init-log-std (tiny), so that the exploration is the correlated noise")
    ap.add_argument("--init-throttle", type=float, default=0.0, help="with --no-warm-start: the untrained policy's mean throttle (a bias, so that the car drives forward from the first step)")
    ap.add_argument("--pedal-axis", action="store_true", help="the agent's action is (steering, one pedal axis: above 0 throttle, below 0 brake) instead of (steering, throttle, brake); needs --no-warm-start")
    ap.add_argument("--residual", action="store_true", help="the pure-pursuit driver steers and sets the pedals, the agent adds a bounded correction (implies --pedal-axis; needs --no-warm-start --bc-weight 0)")
    ap.add_argument("--res-steer", type=float, default=0.1, help="largest steering correction of the agent (steering is -1..1)")
    ap.add_argument("--res-pedal", type=float, default=0.2, help="largest correction of the agent on the pedal axis (-1..1)")
    ap.add_argument("--steer-auth", type=float, default=0.0, help="steering action 1.0 = the wheel angle for this many m/s^2 of lateral acceleration at the current speed (never above full lock); 0 = off (1.0 = full lock). Try 45")
    ap.add_argument("--speed-w", type=float, default=0.0, help="weight of -min(|speed - the line's speed here| / 10, 3) per step: follow the line's speed profile (stops crawling)")
    ap.add_argument("--style-w", type=float, default=1.0, help="0 = no style payments (stage 1 of the curriculum: line and speed only), 1 = as before")
    ap.add_argument("--w-time", type=float, default=1.0, help="weight of the lap-time term (0 in stage 1 of the curriculum)")
    ap.add_argument("--ped-w", type=float, default=0.0, help="weight of -|pedal axis - the track line's pedal axis here| per step: brake where the line brakes (a dense training aid, fades with --ped-decay)")
    ap.add_argument("--steer-w", type=float, default=0.0, help="weight of -|steering - the pure-pursuit steering on the track line here| per step (fades with --ped-decay)")
    ap.add_argument("--ped-decay", type=int, default=0, help="steps over which --ped-w fades linearly to 0 (0 = no fade)")
    args = ap.parse_args()
    if (args.residual or args.pedal_axis) and not args.no_warm_start:
        ap.error("--residual / --pedal-axis have two action numbers; the warm start copies a three-number base command: use --no-warm-start --bc-weight 0")
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.train_freq == 0:
        args.train_freq = 1 if device == "cuda" else 2
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=1))
    state_f = args.out / "state.json"
    state = json.loads(state_f.read_text()) if state_f.is_file() else {}
    env_kw = dict(driver=args.driver, backend=args.backend, d_max=args.d_max, lam0=state.get("lam", args.lam0), lam_eta=args.lam_eta, lam_max=args.lam_max, rand_start=args.rand_start, hide_base=args.hide_base_command,
                  line_w=args.line_w, line_delta=args.line_delta, line_decay=args.line_decay, line_floor=args.line_floor, prog_w=args.prog_w, style_dense=args.style_dense, style_clip=args.style_clip,
                  start_speed_lo=args.start_speed_lo, residual=args.residual, res_steer=args.res_steer, res_pedal=args.res_pedal, pedal_axis=args.pedal_axis,
                  steer_auth=args.steer_auth, speed_w=args.speed_w, style_w=args.style_w, ped_w=args.ped_w, ped_decay=args.ped_decay, steer_w=args.steer_w, weights=({"time": args.w_time} if args.w_time != 1.0 else None))
    env = Monitor(Style12Env(seed=args.seed, **env_kw))
    eval_env = Style12Env(seed=args.seed + 7, **{**env_kw, "rand_start": 0.0})
    lib = Library(args.out, state.get("library_n", 0))
    latest = args.out / "latest.zip"
    if latest.is_file():
        print("resuming from", latest, flush=True)
        model = BCSAC.load(str(latest), env=env, device=device)
        if (args.out / "latest_buffer.pkl").is_file():
            model.load_replay_buffer(str(args.out / "latest_buffer"))
        env.unwrapped.total_steps = model.num_timesteps          # the line penalty fades with the steps taken so far
    else:
        model = BCSAC("MlpPolicy", env, bc_weight=args.bc_weight, bc_decay=args.bc_decay, ent_max=args.ent_max, learning_rate=args.lr, buffer_size=args.buffer, learning_starts=0, batch_size=args.batch_size,
                      tau=0.005, gamma=args.gamma, train_freq=args.train_freq, gradient_steps=1, ent_coef=f"auto_{args.ent_init}", target_entropy=args.target_entropy,
                      policy_kwargs={"net_arch": [args.hidden, args.hidden]}, device=device, seed=args.seed, verbose=0)
        if args.no_warm_start:
            with torch.no_grad():
                if args.init_log_std_scratch:
                    model.actor.log_std.bias.fill_(args.init_log_std)
                    model.actor.log_std.weight.zero_()
                if args.init_throttle > 0:
                    model.actor.mu.bias[1] = float(np.arctanh(min(args.init_throttle, 0.95)))      # the mean action is tanh(mu): this is the throttle of the untrained policy
    theta = 1.0 / args.ou_tau       # exploration noise: Ornstein-Uhlenbeck, stationary standard deviation = the --ou-* values, correlation time --ou-tau steps
    stat = np.array([args.ou_steer, args.ou_throttle] if env.action_space.shape[0] == 2 else [args.ou_steer, args.ou_throttle, args.ou_brake])     # with one pedal axis --ou-throttle is the noise of that axis
    model.action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(len(stat)), sigma=stat * np.sqrt(2 * theta), theta=theta, dt=1.0)
    formats = ["csv", "stdout"]
    try:
        import tensorboard  # noqa: F401
        formats.append("tensorboard")
    except ImportError:
        pass
    model.set_logger(configure(str(args.out), formats))
    du = shutil.disk_usage(args.out)
    (args.out / "system.txt").write_text(f"device {device}, torch {torch.__version__}, {du.free / 1e9:.1f} GB free at the start\n")
    tg = eval_env.tg
    print(f"{args.driver} ({args.backend}): target numbers {dict(zip(KEYS, np.round(eval_env.target, 3)))}", flush=True)
    print(f"distance of the neutral driver's numbers to the target: {tg.distance(tg.drivers['neutral'], eval_env.target):.2f} (margin {args.d_max}); network {args.hidden}x{args.hidden}, device {device}, one update per {args.train_freq} step(s)", flush=True)

    # the comparison laps: the driver's lap in the simulator (pure pursuit on the driver's own mapped lap) and the neutral base driver (pure pursuit on the track's line)
    if not (args.out / "baselines.json").is_file():
        base = {}
        for name, follow in (("pure_pursuit_driver", "driver"), ("neutral", "neutral")):
            e = Style12Env(**{**env_kw, "follow": follow, "rand_start": 0.0, "steer_auth": 0.0})      # the comparison laps are the base driver itself: its steering is not scaled
            res, df, numbers = run_lap12(e, None)
            df.to_csv(args.out / f"baseline_{name}.csv", index=False)
            base[name] = {**res, "numbers": dict(zip(KEYS, map(float, numbers)))}
            print(f"baseline {name}: {res['end']}, lap {res['lap_time']:.2f} s, D {res['D']:.2f}, off-track {100 * res['off_share']:.1f}%", flush=True)
        (args.out / "baselines.json").write_text(json.dumps(base, indent=1, default=float))
    base = json.loads((args.out / "baselines.json").read_text())
    if not latest.is_file() and not args.no_warm_start:
        warm_start(model, env_kw, args, n_steps=args.warm_steps)
        ws, _, _ = run_lap12(eval_env, model)
        print(f"after the warm start: {ws['end']}, lap {ws['lap_time']:.2f} s, D {ws['D']:.2f}, off-track {100 * ws['off_share']:.1f}%", flush=True)
    cb = Periodic(args, args.out, eval_env, time.time(), base, lib)
    model.learn(total_timesteps=max(args.steps - model.num_timesteps, 1), callback=cb, reset_num_timesteps=False, log_interval=20)
    model.save(args.out / "latest")
    model.save_replay_buffer(args.out / "latest_buffer")
    print("finished", flush=True)


if __name__ == "__main__":
    main()
