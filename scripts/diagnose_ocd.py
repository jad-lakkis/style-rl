"""Check the OCD car against the base driver (about 1 minute):   python scripts/diagnose_ocd.py

A. open loop: which way does the car turn for a positive steering command, and are the yaw / track-heading frames the same?
B. full laps of the base driver alone for a few settings (steering sign, look-ahead distance), both reference drivers.
C. for the best setting, speed and line against the reference every 400 m.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.base_driver import BaseDriver  # noqa: E402
from style_rl.evaluate import run_lap  # noqa: E402
from style_rl.style_env import StyleRacingEnv, make_backend  # noqa: E402


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def open_loop(env, steer):
    o = env.reset(start_idx=100, speed=30.0)
    print(f"  start: yaw minus track heading {wrap(o['yaw'] - o['heading_rad']):+.3f} rad, vx {o['vx']:.1f} m/s, vy {o['vy']:+.2f} m/s, lateral {o['lateral_m']:+.2f} m", flush=True)
    for k in range(1, 26):
        o = env.step_normalized(steer, 0.35, 0.0)
        if k in (5, 10, 15, 25):
            print(f"  steer {steer:+.2f}, t={k * 0.04:.2f} s: yaw_rate {o['yaw_rate']:+.3f} rad/s, vy {o['vy']:+.2f}, heading error {wrap(o['yaw'] - o['heading_rad']):+.3f} rad, lateral {o['lateral_m']:+.2f} m, inside {o['inside_track']}", flush=True)


def lap(env, on_line=True, **kw):
    env.start_on_line = on_line
    env.base = BaseDriver(env.ref, max_steering_rad=env.env.max_steering_rad, **kw)
    res, df, _ = run_lap(env, None)
    return res, df


def capability(env, ref):
    """D. What the OCD car can do (brake, accelerate, corner) against what the reference line demands."""
    print("D. capability of the car against the demand of the reference", flush=True)
    s = ref.s
    v, ds = ref.v, float(np.mean(np.diff(s)))
    a_long = v * np.gradient(v, ds)
    a_lat = v ** 2 * np.abs(ref.curv)
    print(f"  reference demands: braking up to {-a_long.min():.1f} m/s2 (p99 {-np.percentile(a_long, 1):.1f}), acceleration up to {np.percentile(a_long, 99):.1f} m/s2 (p99), lateral p99 {np.percentile(a_lat, 99):.1f} m/s2, max {a_lat.max():.1f} m/s2", flush=True)
    o = env.reset(start_idx=100, speed=65.0)
    sp = [o["speed"]]
    for k in range(1, 101):
        o = env.step_normalized(0.0, 0.0, 1.0)
        if k % 25 == 0:
            sp.append(o["speed"])
    print("  full brake from 65 m/s, deceleration per second: " + ", ".join(f"{(sp[i] - sp[i + 1]):.1f}" for i in range(len(sp) - 1)) + " m/s2", flush=True)
    o = env.reset(start_idx=100, speed=30.0)
    sp = [o["speed"]]
    for k in range(1, 151):
        o = env.step_normalized(0.0, 1.0, 0.0)
        if k % 25 == 0:
            sp.append(o["speed"])
    print("  full throttle from 30 m/s, acceleration per second: " + ", ".join(f"{(sp[i + 1] - sp[i]):.1f}" for i in range(len(sp) - 1)) + " m/s2", flush=True)
    for v_t in (40.0, 60.0):
        row = []
        for st in (0.25, 0.5, 0.75, 1.0):
            o = env.reset(start_idx=100, speed=v_t)
            for k in range(100):
                thr = float(np.clip(0.5 + 0.2 * (v_t - o["speed"]), 0.0, 1.0))
                o = env.step_normalized(st, thr, 0.0)
            row.append(f"steer {st:.2f}: {o['speed'] * abs(o['yaw_rate']):.1f} m/s2 at {o['speed']:.0f} m/s")
        print(f"  steady cornering at about {v_t:.0f} m/s: " + "; ".join(row), flush=True)


def trace(env, kw, start_m=600.0):
    res, df = lap(env, **kw)
    print(f"  trace {env.ref.lap_time:.1f}-s reference driver, {kw}: {res['end']} after {res['distance_m']:.0f} m", flush=True)
    d = df[df.progress >= start_m].iloc[::10]
    for r in d.itertuples():
        print(f"    {r.progress:6.0f} m: speed {r.speed * 3.6:5.1f} (ref {env.ref.at(env.ref.v, r.progress) * 3.6:5.1f}) km/h, lateral {r.lateral:+6.1f} (ref {env.ref.at(env.ref.n, r.progress):+5.1f}) m, inside {int(r.inside)}, "
              f"steer {r.steer:+.2f}, throttle {r.throttle:.2f}, brake {r.brake:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ocd", "mock"], default="ocd")
    be = ap.parse_args().backend
    print("A. open loop", flush=True)
    raw = make_backend(be)
    for s in (0.5, -0.5):
        open_loop(raw, s)
    print("B. base driver alone, full lap", flush=True)
    best = {}
    for driver in ("attacker", "smooth"):
        env = StyleRacingEnv(driver=driver, backend=be, full_lap=True)
        for sign, on_line, look0, gain in ((1.0, False, 3.0, 0.3), (1.0, True, 3.0, 0.3), (1.0, True, 6.0, 0.4), (1.0, True, 10.0, 0.5), (-1.0, True, 3.0, 0.3), (-1.0, False, 3.0, 0.3)):
            if True:
                res, df = lap(env, on_line=on_line, steer_sign=sign, look0=look0, look_gain=gain)
                print(f"  {driver:8s} sign {sign:+.0f} start {'on the reference line' if on_line else 'on the centre line'}, look {look0:.0f}+{gain}v: {res['end']}, distance {res['distance_m']:.0f} m, lap {res['lap_time']:.1f} s (reference {env.ref.lap_time:.1f}), "
                      f"off-track {100 * res['off_track_share']:.1f}%, max |lateral| {df.lateral.abs().max():.1f} m", flush=True)
                key = (res["completed"], res["distance_m"], -res["off_track_share"])
                if driver not in best or key > best[driver][0]:
                    best[driver] = (key, dict(on_line=on_line, steer_sign=sign, look0=look0, look_gain=gain), env.ref.lap_time)
    print("C. best setting per driver", flush=True)
    for driver, (_, kw, ref_t) in best.items():
        env = StyleRacingEnv(driver=driver, backend=be, full_lap=True)
        res, df = lap(env, **kw)
        print(f"  {driver}: {kw}, {res['end']}, distance {res['distance_m']:.0f} m, lap {res['lap_time']:.1f} s (reference {ref_t:.1f}), style error {res['style_error']:.2f}", flush=True)
        for s in np.arange(0, df.progress.max(), 400.0):
            r = df[df.progress >= s].iloc[0]
            print(f"    {s:5.0f} m: speed {r.speed * 3.6:5.1f} km/h (reference {env.ref.at(env.ref.v, r.progress) * 3.6:5.1f}), lateral {r.lateral:+5.1f} m (reference {env.ref.at(env.ref.n, r.progress):+5.1f}), "
                  f"steer {r.steer:+.2f}, throttle {r.throttle:.2f}, brake {r.brake:.2f}", flush=True)

    variants = sorted(p for p in (Path(__file__).resolve().parents[1] / "targets" / "variants").glob("m*") if p.is_dir())
    if variants:
        print("F. variants of the reference laps (safety margin on the measured limits), base driver alone", flush=True)
        for tdir in [Path(__file__).resolve().parents[1] / "targets"] + variants:
            os.environ["STYLE_TARGETS"] = str(tdir)
            for driver, govs in (("attacker", ((0.0, 0.35), (1.2, 0.35), (2.0, 0.35), (3.0, 0.2), (4.0, 0.1))), ("smooth", ((0.0, 0.35), (2.0, 0.35)))):
                env = StyleRacingEnv(driver=driver, backend=be, full_lap=True)
                for gov, gstart in govs:
                    res, df = lap(env, on_line=True, steer_sign=1.0, look0=3.0, look_gain=0.3, gov=gov, gov_start=gstart)
                    print(f"  {tdir.name:7s} {driver:8s} traction governor {gov} from steering {gstart}: {res['end']}, distance {res['distance_m']:.0f} m, lap {res['lap_time']:.1f} s (reference {env.ref.lap_time:.1f}), "
                          f"off-track {100 * res['off_track_share']:.1f}%, max |lateral| {df.lateral.abs().max():.1f} m, style error {res['style_error']:.2f}", flush=True)
        os.environ.pop("STYLE_TARGETS", None)
    raw2 = make_backend(be)
    capability(raw2, StyleRacingEnv(driver="attacker", backend=be, full_lap=True).ref)
    print("E. fine trace of the failure", flush=True)
    env = StyleRacingEnv(driver="attacker", backend=be, full_lap=True)
    trace(env, dict(on_line=True, steer_sign=1.0, look0=6.0, look_gain=0.4), start_m=640.0)


if __name__ == "__main__":
    main()
