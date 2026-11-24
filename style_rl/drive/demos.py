"""Demonstration data for the first copy of the scripted driver. No torch.

An episode is one training episode (random start, settle, up to 60 s) driven by the scripted driver aiming at a fraction f ~ U(fraction_range) of the QSS ceiling, with Gaussian noise
added to every action (the NOISY action is executed and recorded, the clean one is kept next to it): the noise makes the data cover small deviations from the line, which is what the
copy needs in order to recover from its own mistakes. Whole episodes are collected until n_transitions is reached.

Gate: at least min_clean_fraction of the episodes must end without a failure (off track, wrong way, stall); otherwise the data would teach the network to fail.

npz arrays   obs (sum(T_i + 1), D): each episode's T_i + 1 observations, the last being the true terminal / truncation observation;  actions, actions_clean (N, 3);  rewards (N,);
             terminated, truncated (N,);  ep_start_obs, ep_start_tr (E,): first row of episode e in obs / in the transition arrays;  f_demo, s0 (E,);  ep_cause (E,).
"""
import hashlib
import json
import multiprocessing
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FAILURES = ("off_track", "nonfinite", "wrong_way", "stall")


@dataclass(frozen=True)
class DemoConfig:
    n_transitions: int = 200_000
    fraction_range: tuple = (0.35, 0.65)
    noise_std: tuple = (0.05, 0.1, 0.1)          # per decision, per action dimension (steering, throttle, brake)
    min_clean_fraction: float = 0.8
    seed: int = 12345
    workers: int = 4


class DemoGateFailed(RuntimeError):
    pass


def run_episode(env, cfg, ep):
    rng = np.random.default_rng([cfg.seed, ep])
    f = float(rng.uniform(*cfg.fraction_range))
    std = np.asarray(cfg.noise_std, float)
    obs, info = env.reset(seed=cfg.seed + ep)
    s0 = float(info["start"]["station_m"])
    env.demo.reset()
    observations, actions, clean_actions, rewards, terms, truncs = [obs], [], [], [], [], []
    while True:
        clean = env.demo.act(env.state, env.s, env.demo.lap_target(env.s, f, env.state["speed"]))
        a = np.clip(clean + rng.normal(0.0, std), -1.0, 1.0).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        observations.append(obs)
        actions.append(a)
        clean_actions.append(np.asarray(clean, np.float32))
        rewards.append(r)
        terms.append(term)
        truncs.append(trunc)
        if term or trunc:
            break
    return {"obs": np.asarray(observations, np.float32), "actions": np.asarray(actions, np.float32), "actions_clean": np.asarray(clean_actions, np.float32),
            "rewards": np.asarray(rewards, np.float64), "terminated": np.asarray(terms, bool), "truncated": np.asarray(truncs, bool), "f_demo": f, "s0": s0,
            "cause": info["termination"] or info["truncation"] or "none"}


_WORKER = {}


def _env_for(store, track, rest):
    from .vec import make_env
    if track not in store:
        store[track] = make_env(0, track, *rest)
    return store[track]


def _init_worker(rest):
    _WORKER["rest"], _WORKER["envs"] = rest, {}


def _worker_episode(args):
    cfg_dict, ep, track = args
    cfg = DemoConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg_dict.items()})
    return run_episode(_env_for(_WORKER["envs"], track, _WORKER["rest"]), cfg, ep)


def collect_episodes(cfg, env_args, start_method):
    """env_args = (tracks, envelope_json, backend, env_cfg). Episodes 0, 1, ... in order until >= n_transitions, episode e on track e % n_tracks (the data does not depend on the number of workers)."""
    from .vec import as_tracks
    tracks, rest = as_tracks(env_args[0]), tuple(env_args[1:])
    episodes, total = [], 0
    if cfg.workers <= 1:
        store, ep = {}, 0
        while total < cfg.n_transitions:
            episodes.append(run_episode(_env_for(store, tracks[ep % len(tracks)], rest), cfg, ep))
            total += len(episodes[-1]["actions"])
            ep += 1
        return episodes
    ctx = multiprocessing.get_context(start_method)
    d = asdict(cfg)
    with ctx.Pool(cfg.workers, initializer=_init_worker, initargs=(rest,)) as pool:
        pending, ep = deque(), 0
        while total < cfg.n_transitions:
            while len(pending) < 2 * cfg.workers:
                pending.append(pool.apply_async(_worker_episode, ((d, ep, tracks[ep % len(tracks)]),)))
                ep += 1
            episodes.append(pending.popleft().get())
            total += len(episodes[-1]["actions"])
        pool.terminate()
    return episodes


def summarize(episodes):
    causes = Counter(e["cause"] for e in episodes)
    n = sum(len(e["actions"]) for e in episodes)
    failed = sum(causes[c] for c in FAILURES)
    return {"episodes": len(episodes), "transitions": n, "causes": dict(sorted(causes.items())), "failures": failed, "clean_fraction": 1.0 - failed / max(len(episodes), 1),
            "mean_reward_per_decision": float(sum(e["rewards"].sum() for e in episodes) / max(n, 1)), "mean_episode_decisions": n / max(len(episodes), 1)}


def assemble(episodes):
    lens = np.array([len(e["actions"]) for e in episodes], np.int64)
    ep_start_tr = np.r_[0, np.cumsum(lens)[:-1]].astype(np.int64)
    out = {k: np.concatenate([e[k] for e in episodes]) for k in ("obs", "actions", "actions_clean", "rewards", "terminated", "truncated")}
    out.update(ep_start_tr=ep_start_tr, ep_start_obs=ep_start_tr + np.arange(len(episodes), dtype=np.int64), f_demo=np.array([e["f_demo"] for e in episodes]),
               s0=np.array([e["s0"] for e in episodes]), ep_cause=np.array([e["cause"] for e in episodes], dtype="U16"))
    return out


def dataset_key(cfg, env_args):
    from .vec import as_tracks
    tracks, envelope_json, backend, env_cfg = as_tracks(env_args[0]), env_args[1], env_args[2], env_args[3]
    h = hashlib.sha256()
    env_d = asdict(env_cfg)
    for k in [k for k in env_d["reward"] if k.startswith(("smooth_", "straight_"))]:
        env_d["reward"].pop(k)              # the smoothness controls do not change the scripted driver: a run with them reuses the demonstrations of a run without (same key as before they existed)
    h.update(json.dumps({"demo": {k: v for k, v in asdict(cfg).items() if k != "workers"}, "env": env_d, "backend": backend}, sort_keys=True, default=str).encode())
    for p in (*tracks, envelope_json):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def ensure_demos(cfg, env_args, demo_dir, *, start_method, log=print):
    """Collect (or load the cached) dataset; raises DemoGateFailed if too few episodes are clean. Returns (arrays dict, summary)."""
    key = dataset_key(cfg, env_args)
    path = Path(demo_dir) / f"demos_{key[:12]}.npz"
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        summary = json.loads(meta.read_text())
        log(f"demos: cached {path.name}: {summary['transitions']} transitions, clean {summary['clean_fraction']:.2f}")
        return load_demos(path), summary
    log(f"demos: collecting {cfg.n_transitions} transitions with {cfg.workers} workers")
    episodes = collect_episodes(cfg, env_args, start_method)
    summary = summarize(episodes)
    log(f"demos: {summary['episodes']} episodes, {summary['transitions']} transitions, clean {summary['clean_fraction']:.2f}, causes {summary['causes']}")
    if summary["clean_fraction"] < cfg.min_clean_fraction:
        raise DemoGateFailed(f"clean fraction {summary['clean_fraction']:.2f} < {cfg.min_clean_fraction}")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = assemble(episodes)
    np.savez_compressed(path, **arrays)
    meta.write_text(json.dumps(summary, indent=1))
    return arrays, summary


def load_demos(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def with_index(d):
    """Adds obs_index (row of each transition's observation in d['obs']), episode (episode of each transition) and ep_start arrays used by BC."""
    n_ep = len(d["ep_start_tr"])
    lens = np.diff(np.r_[d["ep_start_tr"], len(d["actions"])])
    episode = np.repeat(np.arange(n_ep), lens)
    t = np.arange(len(d["actions"])) - d["ep_start_tr"][episode]
    out = dict(d)
    out.update(episode=episode, obs_index=d["ep_start_obs"][episode] + t)
    return out


def prefill_replay_buffer(rb, d):
    """Write the demonstration transitions into an SB3 replay buffer. Episodes are dealt round-robin into the buffer's n_envs columns, each column concatenated and all trimmed
    to the shortest; the last row of every column is closed (done, time-limit truncation) so that an n-step return never runs from a demonstration row into an online row."""
    n_env = rb.n_envs
    d = with_index(d)
    n_ep = len(d["ep_start_tr"])
    cols = [[] for _ in range(n_env)]
    for e in range(n_ep):
        cols[e % n_env].append(e)
    rows = []
    for eps in cols:
        o, no, a, r, tm, dn, tr = [], [], [], [], [], [], []
        for e in eps:
            idx = np.flatnonzero(d["episode"] == e)
            oi = d["obs_index"][idx]
            o.append(d["obs"][oi])
            no.append(d["obs"][oi + 1])
            a.append(d["actions"][idx])
            r.append(d["rewards"][idx])
            term, trunc = d["terminated"][idx], d["truncated"][idx]
            tm.append(term)
            dn.append(term | trunc)
            tr.append(trunc & ~term)
        rows.append([np.concatenate(x) for x in (o, no, a, r, dn, tr, tm)])
    T = min(len(c[2]) for c in rows)
    for c in rows:                                                    # close the last row of every column: done, and a time-limit truncation unless it is a true termination
        c[4][T - 1] = True
        c[5][T - 1] = not c[6][T - 1]
    for t in range(T):
        rb.add(np.stack([c[0][t] for c in rows]), np.stack([c[1][t] for c in rows]), np.stack([c[2][t] for c in rows]), np.array([c[3][t] for c in rows], np.float32),
               np.array([c[4][t] for c in rows], np.float32), [{"TimeLimit.truncated": bool(c[5][t])} for c in rows])
    return {"rows": T, "transitions": T * n_env}
