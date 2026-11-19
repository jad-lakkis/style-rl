"""Vector environment of DrivingEnvs. This module must NOT import torch: the worker factories are unpickled in fresh processes."""
import sys
from functools import partial

from .config import EnvConfig

DEFAULT_START_METHOD = "spawn" if sys.platform == "win32" else "forkserver"


def make_env(rank, track_csv, envelope_json, backend, cfg):
    """One DrivingEnv (its own Course and simulator). Module level, so that partial(make_env, ...) pickles by reference."""
    from .backend import make_backend
    from .course import Course
    from .env import DrivingEnv
    course = Course(track_csv, envelope_json, margin_m=cfg.margin_m, grid_m=cfg.grid_m, line_csv=cfg.ref_line or None)
    return DrivingEnv(course, make_backend(backend, track_csv, cfg.steer_lock_rad, cfg.dt_s), cfg)


def as_tracks(tracks):
    return [str(tracks)] if isinstance(tracks, (str, bytes)) or hasattr(tracks, "__fspath__") else [str(t) for t in tracks]


def env_fns(n_envs, tracks, envelope_json, backend, cfg=EnvConfig()):
    """One environment per rank; with several tracks the rank-i environment drives track i % n_tracks (so one policy learns them all at once)."""
    ts = as_tracks(tracks)
    return [partial(make_env, i, ts[i % len(ts)], str(envelope_json), backend, cfg) for i in range(n_envs)]


def build_vec_env(n_envs, tracks, envelope_json, backend, cfg=EnvConfig(), *, start_method=DEFAULT_START_METHOD, subprocess=True):
    """VecMonitor(SubprocVecEnv) of n_envs environments (DummyVecEnv if not `subprocess`). Call from inside `if __name__ == '__main__':`."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    fns = env_fns(n_envs, tracks, envelope_json, backend, cfg)
    return VecMonitor(SubprocVecEnv(fns, start_method=start_method) if subprocess else DummyVecEnv(fns))


def base_venv(vec_env):
    while hasattr(vec_env, "venv"):
        vec_env = vec_env.venv
    return vec_env
