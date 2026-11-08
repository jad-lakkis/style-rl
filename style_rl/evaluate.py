"""Full-lap evaluation of a policy (or of the base driver alone): lap time, completion, style numbers per corner against the reference, and the lap as a table."""
import numpy as np
import pandas as pd

from style_rl.style import FEATURES


def run_lap(env, policy=None, deterministic=True, max_steps=9000):
    """env: StyleRacingEnv(full_lap=True). policy: object with .predict(obs, deterministic) or None (zero residual = the base driver)."""
    obs, _ = env.reset(seed=0)
    rows, info, total = [], {}, 0.0
    for _ in range(max_steps):
        a = np.zeros(3, np.float32) if policy is None else policy.predict(obs, deterministic=deterministic)[0]
        obs, r, term, trunc, info = env.step(a)
        total += r
        o = env.o
        rows.append({"t": o["time_s"], "progress": env.prog_unw - env.start, "x": o["x"], "y": o["y"], "speed": o["speed"], "lateral": o["lateral_m"], "inside": o["inside_track"],
                     "steer": env.last[0], "throttle": env.last[1], "brake": env.last[2]})
        if term or trunc:
            break
    df = pd.DataFrame(rows)
    corners = pd.DataFrame(env.corner_log)
    res = {"completed": info.get("end") == "lap", "end": info.get("end", "?"), "lap_time": info.get("lap_time", np.nan), "reward": total, "distance_m": float(df.progress.iloc[-1]),
           "style_error": float(corners.style_error.mean()) if len(corners) else np.nan, "off_track_share": float((~df.inside).mean())}
    return res, df, corners


def compare_with_reference(env, corners):
    """Per corner: the agent's five style numbers next to the reference's."""
    rows = []
    for c in corners.itertuples():
        ref = env.ref.corner_ref[int(c.corner)]
        for name, a, b in zip(FEATURES, [c.brake_start, c.brake_end, c.peak_brake, c.mean_brake, c.throttle_pt], ref):
            rows.append({"corner": int(c.corner), "feature": name, "agent": float(a), "reference": float(b)})
    return pd.DataFrame(rows)
