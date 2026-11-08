"""Style numbers of one corner, measured from the pedals along the lap.

A corner is described by five numbers, from the 330 m before the apex to 250 m after it:
  brake_start   metres before the apex where the brake goes on (first sample above 0.05)
  brake_end     metres before the apex where the brake comes off (last sample above 0.05); small = braking right into the corner (trail braking)
  peak_brake    highest brake pedal value
  mean_brake    mean pedal value while the brake is on
  throttle_pt   metres after the apex where the throttle passes 50%
These are the numbers in which the artificial drivers differ (hard late braking against early gentle braking, abrupt against progressive release, early against patient throttle).
"""
import numpy as np

FEATURES = ["brake_start", "brake_end", "peak_brake", "mean_brake", "throttle_pt"]
SCALE = np.array([20.0, 20.0, 0.25, 0.20, 20.0])          # size of a difference that counts as "1" in the style error
PRE, POST = 330.0, 250.0


def corner_features(progress, brake, throttle, s_apex, length):
    """progress: distance along the lap (any wrap), brake and throttle in [0, 1], same length. Returns the 5 numbers of FEATURES."""
    progress, brake, throttle = np.asarray(progress, float), np.asarray(brake, float), np.asarray(throttle, float)
    x = (progress - s_apex + length / 2) % length - length / 2
    pre = (x > -PRE) & (x <= 0)
    order = np.argsort(x[pre])
    xb, b = x[pre][order], brake[pre][order]
    on = np.flatnonzero(b > 0.05)
    if len(on):
        start, end, peak, mean = -xb[on[0]], -xb[on[-1]], float(b.max()), float(b[on].mean())
    else:
        start = end = peak = mean = 0.0
    post = (x > 0) & (x < POST)
    order = np.argsort(x[post])
    xt, th = x[post][order], throttle[post][order]
    hit = np.flatnonzero(th >= 0.5)
    tp = float(xt[hit[0]]) if len(hit) else POST
    return np.array([start, end, peak, mean, tp], float)


def style_error(f, f_ref):
    """Sum of the scaled absolute differences between the style numbers of a corner and the reference (0 = identical style at this corner)."""
    return float(np.sum(np.abs(np.asarray(f) - np.asarray(f_ref)) / SCALE))
