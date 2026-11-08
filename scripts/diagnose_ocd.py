"""Check the OCD car against the base driver (about 1 minute):   python scripts/diagnose_ocd.py

A. open loop: which way does the car turn for a positive steering command, and are the yaw / track-heading frames the same?
B. full laps of the base driver alone for a few settings (steering sign, look-ahead distance), both reference drivers.
C. for the best setting, speed and line against the reference every 400 m.
"""
import argparse
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


if __name__ == "__main__":
    main()
