"""What does Open-Car-Dynamics expose?   python3 scripts/ocd_inspect.py

Builds the F317 environment, drives a few steps and prints the fields of the vehicle model output (names and values), looking for the quantities the driving observation needs:
the actual steering angle and the wheel speeds (for the rear slip ratios). Run it where OCD is installed (the Docker image or a pod)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.ocd_env import OCDMonzaEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def fields(obj, depth=0, prefix="", seen=None):
    seen = seen or set()
    out = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            v = getattr(obj, name)
        except Exception as e:  # noqa: BLE001
            out.append((prefix + name, f"<error {type(e).__name__}>"))
            continue
        if callable(v):
            continue
        if hasattr(v, "__class__") and v.__class__.__module__.startswith(("ocd", "tum_types")) and depth < 3 and id(v) not in seen:
            seen.add(id(v))
            out += fields(v, depth + 1, prefix + name + ".", seen)
        else:
            out.append((prefix + name, v))
    return out


def main():
    env = OCDMonzaEnv(track_path=ROOT / "data/tracks/monza_geometry.csv", dt=0.04)
    env.reset(start_idx=100, speed=30.0)
    for _ in range(25):
        env.step_normalized(0.2, 0.5, 0.0)
    out = env.vehicle.get_vehicle_model_output()
    print("\n=== vehicle model output: top-level attributes:", [n for n in dir(out) if not n.startswith("_")])
    for name, v in fields(out):
        print(f"{name} = {v}")
    for obj_name in ("steering", "drivetrain"):
        o = getattr(env, obj_name, None)
        if o is not None:
            print(f"\n=== env.{obj_name} attributes:", [n for n in dir(o) if not n.startswith("_")])
    print("\nparameters with 'wheel' or 'steer' in the name:")
    for k, v in sorted(env.params.items()):
        if "wheel" in k.lower() or "steer" in k.lower() or "rolling_radius" in k.lower():
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
