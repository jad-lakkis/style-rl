"""Portable actor: the trained actor as ten float32 tensors plus a manifest, and a NumPy-only runtime for the machine that runs Assetto Corsa (no torch, no OCD).

Network contract (the same as the actor of the Assetto Corsa runner in the RL-SetupOptimization-Agent repository, so the tensors have the same names and shapes):
    159 -> 256 -> 256 -> 256 (ReLU) -> mu (3) and log_std (3); deterministic action = tanh(mu); stochastic = tanh(mu + exp(clip(log_std, -20, 2)) * N(0, 1)).
Tensors: latent_pi.{0,2,4}.{weight,bias}, mu.{weight,bias}, log_std.{weight,bias}.
Action contract: [steering, throttle, brake] in [-1, 1]; steering +-1 = +-steer_lock_rad road-wheel angle (0.3 rad for our car, NOT the 0.6 rad that his AC runner assumes),
pedal = clip((a + 0.5) / (pedal_full_at + 0.5), 0, 1) for throttle and brake, both pedals act as commanded (pedal_full_at is in the manifest; 1.0 = the original divisor 1.5 of his runner,
a bundle with 0.7 needs a runner that reads it: his divisor would give 0.8 of the pedal where this gives 1.0).
Observation contract: style_rl.drive.config.ObsConfig / env.py docstring (159 numbers); `OBS_COLUMNS` below names them in order.

    export_actor(model, "bundles/monza_a", meta={...})        # directory with actor.npz, actor.pt (tensors only), manifest.json
    actor = PortableActor("bundles/monza_a"); action = actor.predict(obs)
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import EnvConfig, ObsConfig, OBS_DIM

KEYS = ("latent_pi.0.weight", "latent_pi.0.bias", "latent_pi.2.weight", "latent_pi.2.bias", "latent_pi.4.weight", "latent_pi.4.bias", "mu.weight", "mu.bias", "log_std.weight", "log_std.bias")
HIDDEN = (256, 256, 256)
SCHEMA, VERSION = "style_rl.portable-actor", 1


def obs_columns(o=ObsConfig()):
    ego = ["speed", "v_x", "v_y", "yaw_rate", "a_x", "a_y", "steer_actual", "slip_rl", "slip_rr"]
    act = ["steering_action", "throttle_action", "brake_action"]
    cols = ego + ["e_ref", "heading_error_ref", "track_position"]
    cols += [f"preview_{k}.{p}.{a}" for k in range(1, o.n_preview + 1) for p in ("xi_ref", "left_edge", "right_edge") for a in ("x", "y")]
    cols += act
    cols += [f"previous_{k}.{n}" for k in range(1, o.history) for n in ego + act]
    return cols


OBS_COLUMNS = obs_columns()
assert len(OBS_COLUMNS) == OBS_DIM


def _expected_shapes(hidden=HIDDEN):
    sizes = (OBS_DIM, *hidden)
    shapes = {}
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        shapes[f"latent_pi.{2 * i}.weight"], shapes[f"latent_pi.{2 * i}.bias"] = (b, a), (b,)
    for head in ("mu", "log_std"):
        shapes[f"{head}.weight"], shapes[f"{head}.bias"] = (3, sizes[-1]), (3,)
    return shapes


def actor_tensors(model):
    """The ten actor tensors of an SB3 SAC model as float32 NumPy arrays; raises if the network is not the portable one."""
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.actor.state_dict().items() if k in KEYS}
    shapes = _expected_shapes()
    if set(sd) != set(shapes) or any(tuple(sd[k].shape) != shapes[k] for k in shapes):
        raise ValueError(f"the actor is not the portable network {OBS_DIM}-{'-'.join(map(str, HIDDEN))}-3")
    if not all(np.isfinite(v).all() for v in sd.values()):
        raise ValueError("non-finite actor tensor")
    return sd


def export_actor(model, out_dir, *, meta=None, env_cfg=None, course=None):
    """Writes actor.npz, actor.pt and manifest.json into out_dir (must not hold an exported actor yet). `meta` is stored as given (run name, steps, track, backend, git commit...)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already holds an exported actor")
    sd = actor_tensors(model)
    np.savez(out / "actor.npz", **sd)
    try:
        import torch
        torch.save({k: torch.from_numpy(v.copy()) for k, v in sd.items()}, out / "actor.pt")
    except ImportError:
        pass
    cfg = env_cfg or EnvConfig()
    digest = hashlib.sha256()
    for k in KEYS:
        digest.update(sd[k].tobytes())
    manifest = {
        "schema": SCHEMA, "version": VERSION,
        "network": {"observation_dim": OBS_DIM, "action_dim": 3, "hidden": list(HIDDEN), "activation": "ReLU", "log_std_min": -20.0, "log_std_max": 2.0},
        "weights": {"file": "actor.npz", "tensors": list(KEYS), "sha256": digest.hexdigest(), "dtype": "float32"},
        "observation": {"shape": [OBS_DIM], "scales": asdict(cfg.obs), "columns": OBS_COLUMNS, "final_clip": [-cfg.obs.clip, cfg.obs.clip],
                        "frame": "x forward, y left; yaw and steering positive left",
                        "acceleration": "a_x = dv_x/dt - r*v_y; a_y = dv_y/dt + r*v_x; midpoint r, v over the last decision",
                        "lateral": "e_ref positive left of the reference line; heading error relative to the reference line; track_position -1 right edge, +1 left edge",
                        "preview": "stations d_k = 300*(k/20)^1.5 m along the centre line; reference line, left edge, right edge as x/y in the car frame / max(d_k, 10 m)",
                        "history": "the two previous decisions' ego + last-action blocks, newest first; at placement they hold the ego state and [0, -1, -1]",
                        "normalization": "fixed scales; rear slip clip(5*slip, -5, 5); final clip; no running normalization"},
        "action": {"names": ["steering", "throttle", "brake"], "bounds": [-1.0, 1.0], "steer_lock_rad": cfg.steer_lock_rad, "steering_positive": "left",
                   "pedal_formula": f"clip((a+0.5)/{cfg.pedal_full_at + 0.5:g}, 0, 1)", "pedal_full_at": cfg.pedal_full_at, "both_pedals_act": True, "decision_hz": 1.0 / cfg.dt_s},
        "reward": asdict(cfg.reward),
        "meta": meta or {},
    }
    if course is not None:
        manifest["course"] = {"name": course.name, "length_m": course.L, "margin_m": cfg.margin_m, "qss_lap_time_s": course.qss_lap_time_s}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return out


class PortableActor:
    """NumPy-only runtime of an exported actor."""

    def __init__(self, bundle):
        b = Path(bundle)
        self.manifest = json.loads((b / "manifest.json").read_text())
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("not a style_rl portable actor")
        z = np.load(b / "actor.npz")
        self.w = {k: z[k].astype(np.float32) for k in KEYS}
        shapes = _expected_shapes()
        for k, s in shapes.items():
            if tuple(self.w[k].shape) != s:
                raise ValueError(f"tensor {k}: shape {self.w[k].shape}, expected {s}")
        self.steer_lock_rad = float(self.manifest["action"]["steer_lock_rad"])

    def mean_log_std(self, obs):
        x = np.asarray(obs, np.float32).reshape(-1)
        if x.shape != (OBS_DIM,) or not np.isfinite(x).all():
            raise ValueError("observation must be 159 finite numbers")
        for i in (0, 2, 4):
            x = np.maximum(self.w[f"latent_pi.{i}.weight"] @ x + self.w[f"latent_pi.{i}.bias"], 0.0)
        return self.w["mu.weight"] @ x + self.w["mu.bias"], np.clip(self.w["log_std.weight"] @ x + self.w["log_std.bias"], -20.0, 2.0)

    def predict(self, obs, deterministic=True, rng=None):
        mean, log_std = self.mean_log_std(obs)
        if deterministic:
            return np.tanh(mean).astype(np.float32)
        rng = rng or np.random.default_rng()
        return np.tanh(mean + np.exp(log_std) * rng.standard_normal(3)).astype(np.float32)

    def controls(self, action):
        """(steering road-wheel angle [rad], throttle pedal 0..1, brake pedal 0..1) for an action."""
        a = np.asarray(action, float)
        full_at = float(self.manifest["action"].get("pedal_full_at", 1.0))          # bundles exported before this field existed use the original mapping
        pedals = np.clip((a[1:] + 0.5) / (full_at + 0.5), 0.0, 1.0)
        return float(a[0]) * self.steer_lock_rad, float(pedals[0]), float(pedals[1])
