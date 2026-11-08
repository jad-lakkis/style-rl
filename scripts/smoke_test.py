"""Quick check that everything works before a long run (about 5 minutes with OCD):   python scripts/smoke_test.py --backend ocd

1. the simulator imports and steps (speed in steps/s),
2. the base driver alone drives a full lap of Monza for each reference driver (lap time, off-track share, style error: this is the starting point the agent improves on),
3. SAC trains for a few thousand steps without errors (steps/s with learning).
Exit code 0 = ready; 3 = ready but the base driver is weak on this simulator (train with full residual authority); 1 = do not start the long run.
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.evaluate import run_lap  # noqa: E402
from style_rl.style_env import StyleRacingEnv, make_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ocd", "mock"], default="ocd")
    ap.add_argument("--train-steps", type=int, default=3000)
    args = ap.parse_args()
    ok, base_weak = True, False
    print("1. simulator", flush=True)
    env = make_backend(args.backend)
    o = env.reset(start_idx=100, speed=20.0)
    t0 = time.perf_counter()
    for _ in range(500):
        o = env.step_normalized(0.0, 0.6, 0.0)
    sps = 500 / (time.perf_counter() - t0)
    print(f"   {sps:.0f} simulator steps/s; after 500 steps speed {o['speed'] * 3.6:.0f} km/h, inside track {o['inside_track']}", flush=True)
    for d in ["attacker", "smooth"]:
        print(f"2. base driver alone, {d}", flush=True)
        e = StyleRacingEnv(driver=d, backend=args.backend, full_lap=True)
        res, df, corners = run_lap(e, None)
        print(f"   {res['end']}, lap {res['lap_time']:.1f} s (reference {e.ref.lap_time:.1f} s), distance {res['distance_m']:.0f} m, off-track {100 * res['off_track_share']:.1f}%, style error {res['style_error']:.2f}", flush=True)
        if res["distance_m"] < 500:
            print("   BASE DRIVER WEAK: it does not get far; the agent will be trained with full residual authority", flush=True)
            base_weak = True
    print("3. SAC training", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(Path(__file__).with_name("train.py")), "--driver", "attacker", "--backend", args.backend, "--steps", str(args.train_steps), "--learning-starts", "500",
                            "--eval-every", str(args.train_steps * 10), "--out", tmp], capture_output=True, text=True)
        tail = "\n".join(r.stdout.strip().splitlines()[-4:])
        print(tail, flush=True)
        if r.returncode != 0:
            print(r.stderr[-1500:], flush=True)
            ok = False
        else:
            print(f"   {args.train_steps} training steps in {time.time() - t0:.0f} s", flush=True)
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED", flush=True)
    sys.exit(1 if not ok else (3 if base_weak else 0))


if __name__ == "__main__":
    main()
