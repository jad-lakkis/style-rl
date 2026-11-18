"""Training pipeline of the RL-only driver (style_rl.drive).

    scripted teacher -> demonstrations -> behaviour cloning (the network copies the teacher) -> gate (the copy must finish a clean lap)
    -> critic warm-up on the demonstrations with the actor frozen -> online SAC with the actor still frozen for `online_warmup_steps` -> the actor is released
    -> a collapse guard probes the policy every `guard_interval` steps for `guard_window` steps and rolls back to the last good state if driving collapses.

After the release only the network drives. Steps are transitions summed over the parallel environments. Run folder `out/`:
  config.json, status.json, events.jsonl, evals.jsonl, progress.csv (SB3 logger), bc/report.json, checkpoints/{last,best,step_N}.zip, replay_buffer.pkl, laps/eval_step_N.npz
Resume with `resume=True`: the last checkpoint and replay buffer are loaded and the staged start is skipped.
"""
import json
import os
import signal
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure

from . import agent as ag
from .bc import train_bc
from .config import EnvConfig
from .demos import DemoConfig, ensure_demos, prefill_replay_buffer
from .evaluate import flying_lap, make_probe_starts, probe
from .vec import DEFAULT_START_METHOD, build_vec_env, make_env


@dataclass
class TrainConfig:
    out: str = "runs/drive/run"
    track_csv: str = "data/tracks/monza_geometry.csv"
    tracks: tuple = ()                       # several track files: environment i drives track i % n, one policy learns them all; empty = track_csv only
    envelope_json: str = "data/ocd_envelope.json"
    backend: str = "ocd"
    seed: int = 0
    n_envs: int = 4
    gradient_steps: int = 2                  # per vector step: n_envs 4, gradient_steps 2 = updates per environment step 0.5
    torch_threads: int = 1
    device: str = "cpu"
    total_steps: int = 2_000_000
    max_hours: float = None
    hidden: tuple = (256, 256, 256)
    lr: float = 3e-4
    batch_size: int = 512
    buffer_size: int = 1_000_000
    n_steps: int = 3
    ent_init: float = 0.1
    demo: DemoConfig = field(default_factory=DemoConfig)
    bc_epochs: int = 30
    gate_required: bool = True
    offline_blocks: int = 20
    offline_steps_per_block: int = 1000
    online_warmup_steps: int = 20_000
    guard_n_starts: int = 8
    guard_probe_s: float = 30.0
    guard_interval: int = 10_000
    guard_window: int = 200_000
    guard_threshold: float = 0.5
    guard_rollback_lr: float = 1e-4
    guard_max_rollbacks: int = 1
    init_mode: str = "bc"                    # "bc": copy the scripted driver first (staged start) | "bias": no demonstrations, the network starts from its output bias, nothing frozen
    blend_steps: int = 0                     # > 0: the scripted driver assists the executed action with weight w(t) = blend_w0 * (1 - t / blend_steps): pure pursuit at the start, the agent alone at blend_steps
    blend_w0: float = 0.9
    blend_fraction: float = 0.85             # the assisting driver aims at this fraction of the QSS ceiling
    blend_update: int = 1000
    eval_interval: int = 50_000
    eval_stochastic_laps: int = 5
    eval_seed: int = 0
    checkpoint_interval: int = 100_000
    demo_dir: str = "data/demos"
    env: EnvConfig = field(default_factory=EnvConfig)


# ------------------------------------------------------------------ small helpers
def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")


def save_atomic(model, path):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.zip")
    model.save(str(tmp))
    os.replace(tmp, path)


def save_buffer_atomic(model, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    model.save_replay_buffer(str(tmp))
    os.replace(tmp, path)


class Run:
    def __init__(self, out):
        self.out = Path(out)
        self.t0 = time.time()

    def hours(self):
        return (time.time() - self.t0) / 3600.0

    def event(self, name, **kw):
        rec = {"event": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "hours": round(self.hours(), 4), **kw}
        append_jsonl(self.out / "events.jsonl", rec)
        print(f"[{time.strftime('%H:%M:%S')}] {name} " + " ".join(f"{k}={v}" for k, v in kw.items() if not isinstance(v, (list, dict))), flush=True)

    def status(self, **kw):
        p = self.out / "status.json"
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        p.write_text(json.dumps(cur, indent=1, default=float))


# ------------------------------------------------------------------ callbacks
class PhaseCallback(BaseCallback):
    """Releases actor and entropy coefficient at `unfreeze_step`."""

    def __init__(self, unfreeze_step, run):
        super().__init__()
        self.unfreeze_step, self.run = unfreeze_step, run

    def _on_step(self):
        if self.model.freeze_actor and self.model.num_timesteps >= self.unfreeze_step:
            self.model.set_frozen(False, False)
            self.run.event("unfreeze", step=self.model.num_timesteps)
        return True


class CollapseGuard(BaseCallback):
    """From the release to release + window, every `interval` steps: probe progress >= threshold x baseline keeps a snapshot of the model; below, the last snapshot is restored with a lower
    actor learning rate (a rollback). After `max_rollbacks` rollbacks the next collapse stops the run."""

    def __init__(self, probe_fn, baseline, cfg, run):
        super().__init__()
        self.probe_fn, self.baseline, self.cfg, self.run = probe_fn, float(baseline), cfg, run
        self.snapshot = self.start = self.next_probe = None
        self.rollbacks = 0
        self.collapsed = False

    def _arm(self, n):
        self.start, self.snapshot = n, ag.snapshot_state(self.model)
        self.next_probe = n + self.cfg.guard_interval

    def _on_step(self):
        n, m = self.model.num_timesteps, self.model
        if self.snapshot is None:
            if not m.freeze_actor:
                self._arm(n)
            return True
        if n >= self.start + self.cfg.guard_window or n < self.next_probe:
            return True
        while self.next_probe <= n:
            self.next_probe += self.cfg.guard_interval
        progress = float(self.probe_fn(m))
        ok = progress >= self.cfg.guard_threshold * self.baseline
        self.logger.record("guard/probe_progress_m", progress)
        append_jsonl(self.run.out / "guard.jsonl", {"step": n, "mean_progress_m": progress, "baseline_m": self.baseline, "ok": ok})
        if ok:
            self.snapshot = ag.snapshot_state(m)
            return True
        if self.rollbacks >= self.cfg.guard_max_rollbacks:
            self.collapsed = True
            self.run.event("collapsed", step=n, progress_m=progress, baseline_m=self.baseline)
            self.run.status(state="collapsed")
            return False
        ag.restore_state(m, self.snapshot)
        m.actor_lr = self.cfg.guard_rollback_lr
        self.rollbacks += 1
        self.run.event("rollback", step=n, progress_m=progress, baseline_m=self.baseline, rollbacks=self.rollbacks)
        return True


class EvalCallback(BaseCallback):
    """Every `eval_interval` steps (and before the first online step): deterministic flying lap + stochastic laps; saves last.zip and the replay buffer; best.zip for the best clean
    deterministic lap 2; step_N.zip every `checkpoint_interval`."""

    def __init__(self, cfg, eval_envs, run):
        super().__init__()
        self.cfg, self.eval_envs, self.run = cfg, eval_envs, run          # eval_envs: {track name: DrivingEnv}
        self.best = float("inf")
        self.next_eval = 0
        self.next_ckpt = cfg.checkpoint_interval

    def _on_step(self):
        n = self.model.num_timesteps
        if n >= self.next_eval:
            self.evaluate(n)
            self.next_eval = (n // self.cfg.eval_interval + 1) * self.cfg.eval_interval
        if n >= self.next_ckpt:
            save_atomic(self.model, self.run.out / "checkpoints" / f"step_{n}.zip")
            self.next_ckpt = (n // self.cfg.checkpoint_interval + 1) * self.cfg.checkpoint_interval
        return True

    def evaluate(self, n):
        t = time.time()
        names = list(self.eval_envs)
        dets = {k: flying_lap(self.model, e, deterministic=True, seed=self.cfg.eval_seed, trace=True) for k, e in self.eval_envs.items()}
        det = dets[names[0]]                                                # the first track is the primary one (the top-level fields)
        sto = [flying_lap(self.model, self.eval_envs[names[k % len(names)]], deterministic=False, seed=self.cfg.eval_seed + k) for k in range(self.cfg.eval_stochastic_laps)]
        clean_sto = [r for r in sto if r.clean]
        score = float(np.mean([d.lap2_s / self.eval_envs[k].course.qss_lap_time_s for k, d in dets.items()])) if all(d.clean for d in dets.values()) else float("inf")   # lap 2 over the QSS ceiling, mean over tracks
        rec = {"step": n, "hours": round(self.run.hours(), 4), "det_clean": det.clean, "det_end": det.termination, "det_lap1_s": det.lap_times_s[0] if det.lap_times_s else None,
               "det_lap2_s": det.lap2_s, "sto_clean": len(clean_sto), "sto_laps": len(sto), "sto_lap2_mean_s": float(np.mean([r.lap2_s for r in clean_sto])) if clean_sto else None,
               "tracks": {k: {"clean": d.clean, "lap2_s": d.lap2_s, "end": d.termination} for k, d in dets.items()}, "score": None if score == float("inf") else score,
               "alpha": self.model.alpha, "eval_wall_s": round(time.time() - t, 1)}
        append_jsonl(self.run.out / "evals.jsonl", rec)
        (self.run.out / "laps").mkdir(exist_ok=True)
        for k, d in dets.items():
            tag = f"eval_step_{n}.npz" if len(dets) == 1 else f"eval_step_{n}_{k}.npz"
            np.savez_compressed(self.run.out / "laps" / tag, **d.trace, lap_times_s=np.asarray(d.lap_times_s), clean=d.clean)
        save_atomic(self.model, self.run.out / "checkpoints" / "last.zip")
        save_buffer_atomic(self.model, self.run.out / "replay_buffer.pkl")
        new = ""
        if score < self.best:
            self.best = score
            save_atomic(self.model, self.run.out / "checkpoints" / "best.zip")
            new = " NEW BEST"
        self.run.status(state="running", step=n, best_score_lap2_over_qss=None if self.best == float("inf") else self.best)
        per = ", ".join(f"{k} {d.lap2_s if d.clean else d.termination}" for k, d in dets.items())
        print(f"  EVAL step {n}: det lap2 [{per}], stochastic clean {len(clean_sto)}/{len(sto)}, {rec['eval_wall_s']:.0f} s{new}", flush=True)


class BlendCallback(BaseCallback):
    """Anneals the scripted driver's weight in the executed action from blend_w0 to 0 over blend_steps steps (all environments)."""

    def __init__(self, cfg, run):
        super().__init__()
        self.cfg, self.run, self.next, self.last_w = cfg, run, 0, None

    def weight(self, n):
        return max(0.0, self.cfg.blend_w0 * (1.0 - n / self.cfg.blend_steps))

    def _set(self, n):
        w = self.weight(n)
        self.training_env.env_method("set_blend", w, self.cfg.blend_fraction)
        self.logger.record("blend/teacher_weight", w)
        if self.last_w is None or (w == 0.0 and self.last_w > 0.0):
            self.run.event("blend", step=n, teacher_weight=round(w, 4))
        self.last_w = w

    def _on_training_start(self):
        self._set(self.model.num_timesteps)

    def _on_step(self):
        n = self.model.num_timesteps
        if n >= self.next:
            self._set(n)
            self.next = n + self.cfg.blend_update
        return True


class WallClock(BaseCallback):
    """Ends the training cleanly (the run saves everything and exits with code 0) after `hours`, or as soon as the file out/STOP exists."""

    def __init__(self, hours, run):
        super().__init__()
        self.hours, self.run, self.k = hours, run, 0

    def _on_step(self):
        self.k += 1
        if self.hours is not None and self.run.hours() >= self.hours:
            self.run.event("wall_clock_limit", hours=self.hours)
            return False
        if self.k % 50 == 0 and (self.run.out / "STOP").exists():
            self.run.event("stop_file")
            return False
        return True


# ------------------------------------------------------------------ the pipeline
def run(cfg, resume=False, log=print):
    out = Path(cfg.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    th.set_num_threads(cfg.torch_threads)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1, default=str))
    r = Run(out)
    (out / "STOP").unlink(missing_ok=True)                       # a stop file of an earlier session must not stop this one
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))     # a terminate signal ends the run through the `finally` below, which saves the model and the buffer
    tracks = list(cfg.tracks) or [cfg.track_csv]
    env_args = (tracks, cfg.envelope_json, cfg.backend, cfg.env)
    vec = build_vec_env(cfg.n_envs, tracks, cfg.envelope_json, cfg.backend, cfg.env, subprocess=cfg.n_envs > 1)
    eval_envs = {}
    for t in tracks:
        e = make_env(0, t, cfg.envelope_json, cfg.backend, cfg.env)
        eval_envs[e.course.name] = e
    probe_env = make_env(0, tracks[0], cfg.envelope_json, cfg.backend, replace(cfg.env, max_episode_s=cfg.guard_probe_s))
    model = ag.build_model(vec, hidden=cfg.hidden, lr=cfg.lr, batch_size=cfg.batch_size, buffer_size=cfg.buffer_size, gamma=cfg.env.reward.gamma, n_steps=cfg.n_steps,
                           gradient_steps=cfg.gradient_steps, ent_init=cfg.ent_init, seed=cfg.seed, device=cfg.device)
    model.set_logger(configure(str(out), ["csv"]))
    starts = make_probe_starts(probe_env, cfg.guard_n_starts)
    last = out / "checkpoints" / "last.zip"
    if resume and last.exists():
        model = ag.StagedSAC.load(str(last), env=vec, device=cfg.device)
        model.set_logger(configure(str(out), ["csv"]))
        if (out / "replay_buffer.pkl").exists():
            model.load_replay_buffer(str(out / "replay_buffer.pkl"))
        baseline = float(json.loads((out / "status.json").read_text()).get("bc_baseline_m", 1000.0))
        r.event("resume", step=model.num_timesteps, frozen=model.freeze_actor)
    elif cfg.init_mode == "bias":
        ag.init_output_bias(model.actor)
        baseline = probe(model, probe_env, starts)["mean_progress_m"]
        r.status(state="running", init_mode="bias", bc_baseline_m=baseline)
        r.event("init_bias_only", baseline_m=baseline)
    else:
        ag.init_output_bias(model.actor)
        demos, summary = ensure_demos(cfg.demo, env_args, cfg.demo_dir, start_method=DEFAULT_START_METHOD, log=log)
        r.status(state="bc", demos=summary)
        report = train_bc(model.actor, demos, seed=cfg.seed, max_epochs=cfg.bc_epochs, log=log)
        (out / "bc").mkdir(exist_ok=True)
        (out / "bc" / "report.json").write_text(json.dumps(report, indent=1))
        names = list(eval_envs)
        gates = {k: flying_lap(model, e, deterministic=True, seed=cfg.eval_seed) for k, e in eval_envs.items()}
        sto = [flying_lap(model, eval_envs[names[k % len(names)]], deterministic=False, seed=cfg.eval_seed + k) for k in range(3)]
        gate_ok = all(g.clean for g in gates.values())
        r.event("bc_gate", clean=gate_ok, per_track={k: (g.lap2_s if g.clean else g.termination) for k, g in gates.items()}, stochastic_clean=sum(x.clean for x in sto))
        if not gate_ok and cfg.gate_required:
            r.status(state="bc_gate_failed")
            return model
        baseline = probe(model, probe_env, starts)["mean_progress_m"]
        r.status(state="warmup", bc_baseline_m=baseline)
        model.set_frozen(True, True)
        info = prefill_replay_buffer(model.replay_buffer, demos)
        r.event("prefill", **info)
        for b in range(cfg.offline_blocks):
            model.train(cfg.offline_steps_per_block, cfg.batch_size)
            cl = model.logger.name_to_value.get("train/critic_loss")
            append_jsonl(out / "warmup.jsonl", {"block": b, "critic_loss": cl})
        r.event("offline_critic_done", blocks=cfg.offline_blocks, critic_loss=model.logger.name_to_value.get("train/critic_loss"))
    r.status(state="running")
    cbs = [PhaseCallback(cfg.online_warmup_steps, r), CollapseGuard(lambda m: probe(m, probe_env, starts)["mean_progress_m"], baseline, cfg, r), EvalCallback(cfg, eval_envs, r),
           WallClock(cfg.max_hours, r)]
    if cfg.blend_steps > 0:
        cbs.append(BlendCallback(cfg, r))
    try:
        model.learn(total_timesteps=max(cfg.total_steps - model.num_timesteps, 1), callback=CallbackList(cbs), reset_num_timesteps=False, log_interval=50)
    finally:
        save_atomic(model, last)
        save_buffer_atomic(model, out / "replay_buffer.pkl")
    if json.loads((out / "status.json").read_text()).get("state") != "collapsed":
        r.status(state="done")
    return model
