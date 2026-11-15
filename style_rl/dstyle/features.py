"""The driver style vector: 12 numbers measured on one lap.

What goes in (one row per sample, about 25 samples per second):
    time_s      clock, seconds
    throttle    throttle pedal, 0..1
    brake       brake pedal, 0..1
    steer_deg   steering wheel angle in degrees (sign: positive = one direction, either is fine)
    x, y        car position in the same frame as the track file (metres)
plus the track (driver_style.track.Track). Optional: lap_dist, a rough distance along the lap for each sample, that keeps the position match on the right
part of the circuit. Nothing else is used: no speed, no lap time, no reference lap, no other driver.

What comes out: a dict with the 12 numbers of STYLE (braking 4, throttle 2, steering 3, line 3). Use laps that stay on the track.
For a driver on a track take the median over the driver's laps on that track.
"""
import numpy as np
import pandas as pd

SMOOTH_SAMPLES = 3              # moving average over 3 samples (0.12 s) before the pedals and the steering are differentiated
BRAKE_ON = 0.05                 # the brake counts as on above this pedal value
RATE_THRESHOLD = 0.2            # a throttle change faster than 0.2 pedal travel per second counts as an apply or a lift
REVERSAL_DEG = 5.0              # a steering correction = a change of direction larger than this, in degrees
ENTRY_M, EXIT_M = 100.0, 80.0   # corner window: from 100 m before the apex to 80 m after it

# (key, short name, group, what it measures, unit)
STYLE = [
    ("B_mean_active", "Brake pressure", "Braking", "how hard the pedal is pressed, on average, while the brake is on", "0..1"),
    ("B_event_peak_mean", "Peak brake pressure", "Braking", "the highest pedal value of a braking zone, averaged over the zones", "0..1"),
    ("B_apply_rate", "Brake attack", "Braking", "how fast the pedal goes down: peak value / time from brake-on to the peak, averaged over the zones", "pedal travel per second"),
    ("B_release_rate", "Brake release", "Braking", "how fast the pedal comes back: mean pedal speed between the peak and the end of the zone", "pedal travel per second"),
    ("T_apply_rate", "Throttle attack", "Throttle", "mean pedal speed over the moments the throttle is going down quickly", "pedal travel per second"),
    ("T_release_rate", "Throttle lift", "Throttle", "mean pedal speed over the moments the throttle is coming up quickly", "pedal travel per second"),
    ("S_reversals_5deg_per_km", "Steering corrections", "Steering", "changes of steering direction larger than 5 degrees", "per km"),
    ("S_mean_abs_rate", "Hand speed", "Steering", "how fast the wheel is turned, averaged over time", "degrees per second"),
    ("S_corner_peak_angle", "Peak steering in corners", "Steering", "largest wheel angle in a corner window, averaged over the corners", "degrees"),
    ("L_border_mean_m", "Distance to the track edge", "Line", "distance to the nearest road edge, averaged over time (small = uses the whole road)", "metres"),
    ("L_inside_dist_apex_m", "Apex distance", "Line", "distance to the inside edge at the apex, averaged over the corners", "metres"),
    ("L_outside_dist_exit_m", "Exit width", "Line", "distance to the outside edge 80 m after the apex, averaged over the corners", "metres"),
]
KEYS = [s[0] for s in STYLE]
NAMES = {s[0]: s[1] for s in STYLE}


def smooth(x, n=SMOOTH_SAMPLES):
    pad = n // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(n) / n, mode="valid")


def reversals(x, theta):
    """Number of direction changes of x larger than theta (hysteresis, so that sensor noise below theta is not counted)."""
    pivot, d, n = x[0], 0, 0
    for v in x[1:]:
        if d == 0:
            if v - pivot >= theta:
                d, pivot = 1, v
            elif pivot - v >= theta:
                d, pivot = -1, v
        elif d == 1:
            if v > pivot:
                pivot = v
            elif pivot - v >= theta:
                d, pivot, n = -1, v, n + 1
        else:
            if v < pivot:
                pivot = v
            elif v - pivot >= theta:
                d, pivot, n = 1, v, n + 1
    return n


def brake_zones(on, t, min_gap_s=0.2, min_dur_s=0.2):
    """Index ranges [i0, i1) where the brake is on; gaps shorter than 0.2 s are bridged, zones shorter than 0.2 s are dropped."""
    idx = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]) != 0)
    merged = []
    for a, b in zip(idx[::2], idx[1::2]):
        if merged and t[a] - t[merged[-1][1] - 1] < min_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if t[b - 1] - t[a] >= min_dur_s]


def style_vector(lap, track, lap_dist=None, corners=None, q=None):
    """The 12 style numbers of one lap.

    lap: DataFrame with the columns time_s, throttle, brake, steer_deg, x, y (see the module docstring).
    track: driver_style.track.Track. lap_dist: optional rough distance along the lap per sample. corners: optional, track.find_corners() is used by default.
    q: optional, the position of every sample on the road, if it is already known (a simulator reports it): a dict of arrays progress_m, lateral_m, left_width_m, right_width_m
       (the output of track.project); then x and y are not needed.
    """
    t = lap["time_s"].to_numpy(float) - float(lap["time_s"].iloc[0])
    thr, brk, st = (lap[c].to_numpy(float) for c in ("throttle", "brake", "steer_deg"))
    dt = np.gradient(t)
    w = dt / dt.sum()                                   # every sample counts for the time it lasts
    length = track.length_m
    km = length / 1000.0
    corners = track.find_corners() if corners is None else corners
    if q is None:
        q = track.project(lap[["x", "y"]].to_numpy(float), s_hint=None if lap_dist is None else np.asarray(lap_dist, float))
    m = {}

    # steering
    rate = np.gradient(smooth(st), t)
    m["S_mean_abs_rate"] = float(np.sum(w * np.abs(rate)))
    m["S_reversals_5deg_per_km"] = reversals(smooth(st), REVERSAL_DEG) / km

    # throttle
    tr = np.gradient(smooth(thr), t)
    up, down = tr > RATE_THRESHOLD, tr < -RATE_THRESHOLD
    m["T_apply_rate"] = float(tr[up].mean()) if up.any() else 0.0
    m["T_release_rate"] = float(-tr[down].mean()) if down.any() else 0.0

    # brake
    br = np.gradient(smooth(brk), t)
    on = brk > BRAKE_ON
    zones = brake_zones(on, t)
    m["B_mean_active"] = float(np.sum(w[on] * brk[on]) / max(w[on].sum(), 1e-9))
    m["B_event_peak_mean"] = float(np.mean([brk[a:b].max() for a, b in zones])) if zones else 0.0
    attack, release = [], []
    for a, b in zones:
        k = a + int(np.argmax(brk[a:b]))
        if k > a:
            attack.append(brk[k] / max(t[k] - t[a], 0.04))
        if b - 1 > k:
            falling = br[k:b] < 0
            release.append(-br[k:b][falling].mean() if falling.any() else 0.0)
    m["B_apply_rate"] = float(np.mean(attack)) if attack else 0.0
    m["B_release_rate"] = float(np.mean(release)) if release else 0.0

    # line: distance to the nearest road edge, overall and at each corner (outside the road counts as 0)
    d_left, d_right = q["left_width_m"] - q["lateral_m"], q["right_width_m"] + q["lateral_m"]
    border = np.clip(np.minimum(d_left, d_right), 0, None)
    m["L_border_mean_m"] = float(np.sum(w * border))
    s = q["progress_m"]
    inside_apex, outside_exit, steer_peak = [], [], []
    for c in corners.itertuples():
        ds = ((s - c.s_apex + length / 2) % length) - length / 2        # signed distance to the apex along the road
        left_turn = c.direction > 0
        inside, outside = (d_left, d_right) if left_turn else (d_right, d_left)
        inside_apex.append(max(inside[int(np.argmin(np.abs(ds)))], 0))
        at_exit = np.flatnonzero((ds >= EXIT_M - 5) & (ds <= EXIT_M + 5))
        if len(at_exit):
            outside_exit.append(max(float(outside[at_exit].mean()), 0))
        window = (ds >= -ENTRY_M) & (ds <= EXIT_M)
        if window.any():
            steer_peak.append(float(np.max(np.abs(st[window]))))
    mean = lambda a: float(np.mean(a)) if len(a) else np.nan     # noqa: E731
    m["L_inside_dist_apex_m"] = mean(inside_apex)
    m["L_outside_dist_exit_m"] = mean(outside_exit)
    m["S_corner_peak_angle"] = mean(steer_peak)
    return {k: m[k] for k in KEYS}


def style_table(vectors):
    """DataFrame of 12 numbers, rows = laps (or driver-track cells), columns = the 12 keys; the median of the rows is the style of a driver on a track."""
    return pd.DataFrame(vectors)[KEYS]
