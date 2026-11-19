"""Evaluation of a SAC model on the driving environment: the flying lap (reported lap time) and the probe episodes (collapse guard)."""
from dataclasses import dataclass, field

import numpy as np
import torch as th


@dataclass
class LapReport:
    clean: bool                     # all timed laps finished, no failure, no truncation
    lap_times_s: list
    termination: str
    truncation: str
    decisions: int
    trace: dict = field(repr=False, default_factory=dict)

    @property
    def lap2_s(self):
        return self.lap_times_s[1] if len(self.lap_times_s) >= 2 else None


def flying_lap(model, env, *, deterministic, seed=0, trace=False, start_s=None, start_speed=None):
    """One evaluation episode (evaluation mode: run-in, then eval_laps timed laps; lap 2 is the flying lap). start_s / start_speed: another start station (m) and speed (m/s) than the standard
    run-in 200 m before the line (the car then drives on to the line, and the two timed laps follow)."""
    e = env.unwrapped
    if not deterministic:
        th.manual_seed(int(seed))
    options = {"evaluation": True}
    if start_s is not None:
        options["start_s"] = float(start_s)
    if start_speed is not None:
        options["start_speed"] = float(start_speed)
    obs, _ = env.reset(seed=int(seed), options=options)
    rec = {k: [] for k in ("station_m", "x", "y", "speed", "e", "steer", "throttle", "brake", "time_s")} if trace else {}
    n = 0
    while True:
        a, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(a)
        n += 1
        if trace:
            for k, v in (("station_m", info["station_m"]), ("x", e.state["x"]), ("y", e.state["y"]), ("speed", info["speed"]), ("e", info["e_ref"]), ("steer", float(a[0])),
                         ("throttle", e._pedal(a[1])), ("brake", e._pedal(a[2])), ("time_s", info["time_s"])):
                rec[k].append(v)
        if term or trunc:
            break
    times = list(info.get("lap_times_s", []))
    clean = info["termination"] == "lap_complete" and len(times) == e.cfg.eval_laps
    return LapReport(clean, times, str(info["termination"]), str(info["truncation"]), n, {k: np.asarray(v) for k, v in rec.items()})


def varied_starts(length_m, n, offset_fraction=0.5):
    """n start stations spread evenly around the lap, the first at `offset_fraction` of a spacing (never at the line itself, which the standard evaluation start is near)."""
    return [((k + offset_fraction) * length_m / n) % length_m for k in range(n)]


def reliability(model, env, n_starts, seed=0):
    """Deterministic flying laps (two timed laps each) from n_starts different stations around the lap. A start whose settle phase leaves the track by the scripted driver's fault is skipped.
    Returns {"of": starts used, "clean": clean ones, "lap2_s": [lap 2 of the clean ones], "ends": [how each ended], "starts_m": [...]}."""
    e = env.unwrapped
    out = {"of": 0, "clean": 0, "lap2_s": [], "ends": [], "starts_m": []}
    for k, s0 in enumerate(varied_starts(float(e.course.L), n_starts)):
        try:
            rep = flying_lap(model, env, deterministic=True, seed=int(seed) + 100 + k, start_s=s0)
        except RuntimeError:
            continue
        out["of"] += 1
        out["starts_m"].append(round(s0, 1))
        out["ends"].append(rep.termination)
        if rep.clean:
            out["clean"] += 1
            out["lap2_s"].append(rep.lap2_s)
    return out


def make_probe_starts(env, n_starts=8, offset_m=37.0, shift_m=50.0, max_shifts=10, seed_base=1000):
    """Evenly spaced starts around the lap; a start whose settle phase fails is shifted."""
    L = float(env.unwrapped.course.L)
    starts = []
    for k in range(n_starts):
        s = k * L / n_starts + offset_m
        for shift in range(max_shifts + 1):
            try:
                env.reset(seed=seed_base + k, options={"start_s": s % L})
                break
            except RuntimeError:
                s += shift_m
        else:
            raise RuntimeError(f"probe start {k}: no start that settles on the track")
        starts.append({"k": k, "s": float(s % L), "seed": seed_base + k})
    return starts


def probe(model, env, starts):
    """Deterministic probe episodes (env in training mode, short max_episode_s) from `starts`: progress driven [m] and the end cause of each."""
    progress, causes = [], []
    for st in starts:
        obs, _ = env.reset(seed=st["seed"], options={"start_s": st["s"]})
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        progress.append(float(info["progress_m"]))
        causes.append(info["termination"] or info["truncation"])
    return {"progress_m": progress, "causes": causes, "mean_progress_m": float(np.mean(progress))}
