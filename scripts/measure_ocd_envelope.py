"""Measure what the OCD car can do, as the limits a driver model needs (about 1 minute):   python scripts/measure_ocd_envelope.py [--backend ocd] [--out data/ocd_envelope.json]

On the straight of Monza, for a grid of speeds:
  acceleration  full throttle, net forward acceleration (m/s2, drag included)
  braking       best of several brake levels (a hard brake can lock the rear), deceleration in g, drag included
  cornering     constant steering (three sizes) with the speed held, lateral acceleration in g, only samples with a small side-slip angle
  drag          coasting deceleration, fitted as kd * v^2
Writes a JSON with the curves on a 1 m/s grid from 0 to 100 m/s (the format of environment.reference_lap.Envelope in the research repository) and prints the same
lines as ENV_* so they can be read from a log.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.style_env import make_backend  # noqa: E402

G = 9.81
DT = 0.04
V_GRID = np.arange(0.0, 101.0, 1.0)
SPEEDS = np.arange(10.0, 86.0, 5.0)


def run(env, v0, steer, policy, n):
    o = env.reset(start_idx=100, speed=v0)
    out = [o]
    for k in range(n):
        thr, brk = policy(o, k)
        o = env.step_normalized(steer, thr, brk)
        out.append(o)
    return out


def accel(env, v):
    out = run(env, v, 0.0, lambda o, k: (1.0, 0.0), 15)
    return (out[15]["speed"] - out[5]["speed"]) / (10 * DT)


def brake(env, v):
    best = 0.0
    for level in (0.5, 0.7, 0.85, 1.0):
        out = run(env, v, 0.0, lambda o, k: (0.0, level), 16)
        a = (out[4]["speed"] - out[14]["speed"]) / (10 * DT) if v >= 15 else (out[3]["speed"] - out[8]["speed"]) / (5 * DT)
        best = max(best, a)
    return best / G


def coast(env, v):
    out = run(env, v, 0.0, lambda o, k: (0.0, 0.0), 15)
    return (out[3]["speed"] - out[13]["speed"]) / (10 * DT)


def corner_samples(env, v):
    pts = []
    for steer in (0.4, 0.7, 1.0):
        out = run(env, v, steer, lambda o, k: (float(np.clip(0.45 + 0.3 * (v - o["speed"]), 0.0, 1.0)), float(np.clip(0.3 * (o["speed"] - v - 1.0), 0.0, 1.0))), 75)
        tail = out[-10:]
        sp = np.mean([o["speed"] for o in tail])
        slip = np.max([abs(o["vy"]) / max(o["vx"], 1.0) for o in tail])
        ay = np.mean([abs(o["yaw_rate"]) * o["speed"] for o in tail]) / G
        if abs(sp - v) < 0.2 * v and slip < 0.15:
            pts.append((sp, ay))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ocd", "mock"], default="ocd")
    ap.add_argument("--out", type=Path, default=Path("data/ocd_envelope.json"))
    args = ap.parse_args()
    env = make_backend(args.backend)
    acc = np.array([accel(env, v) for v in SPEEDS])
    brk = np.array([brake(env, v) for v in SPEEDS])
    dec = np.array([coast(env, v) for v in SPEEDS])
    kd = float(np.sum(dec * SPEEDS ** 2) / np.sum(SPEEDS ** 4))
    pts = []
    for v in SPEEDS:
        pts += corner_samples(env, v)
    pts = np.array(pts)
    lat = np.full(len(SPEEDS), np.nan)
    for i, v in enumerate(SPEEDS):
        m = np.abs(pts[:, 0] - v) <= 2.5
        if m.any():
            lat[i] = pts[m, 1].max()
    ok = np.isfinite(lat)
    lat = np.interp(SPEEDS, SPEEDS[ok], lat[ok])
    v_cap = 90.0
    below = np.where(acc <= 0.2)[0]
    if len(below):
        v_cap = float(SPEEDS[below[0]])
    acc_grid = np.interp(V_GRID, SPEEDS, acc)
    acc_grid[V_GRID >= v_cap] = 0.0
    result = {"v_grid": V_GRID.tolist(), "lat_g": np.interp(V_GRID, SPEEDS, lat).round(4).tolist(), "brk_g": np.interp(V_GRID, SPEEDS, brk).round(4).tolist(),
              "acc_ms2": acc_grid.round(3).tolist(), "kd": kd, "v_cap": v_cap, "measured_speeds": SPEEDS.tolist(),
              "measured": {"acc_ms2": acc.round(2).tolist(), "brk_g": brk.round(3).tolist(), "lat_g": lat.round(3).tolist(), "coast_ms2": dec.round(2).tolist()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))
    print("speed m/s | acceleration m/s2 | braking g | cornering g | coasting m/s2", flush=True)
    for i, v in enumerate(SPEEDS):
        print(f"  {v:5.0f}   | {acc[i]:6.2f} | {brk[i]:5.2f} | {lat[i]:5.2f} | {dec[i]:5.2f}", flush=True)
    print(f"kd {kd:.6f}, top speed (acceleration below 0.2 m/s2) {v_cap:.0f} m/s", flush=True)
    for k in ("lat_g", "brk_g", "acc_ms2"):
        print(f"ENV_{k}=" + json.dumps(result[k][::5] if False else result[k]), flush=True)
    print(f"ENV_kd={kd:.7f}", flush=True)
    print(f"ENV_v_cap={v_cap:.1f}", flush=True)


if __name__ == "__main__":
    main()
