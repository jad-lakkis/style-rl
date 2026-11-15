# The style vector: where it is computed and how

A driver's style is **12 numbers measured on one lap** from the pedals, the steering wheel and the position on the road. Lap time and speed are not used, so the same code works on any circuit and for any driver. This file says which file does what.

## Where the code is

| What | File | Role |
|---|---|---|
| **Feature extraction (the 12 numbers)** | `style_rl/dstyle/features.py` | `style_vector(lap, track)` turns one lap into the 12 numbers. The list `STYLE` defines them: key, name, group, meaning, unit. |
| Position on the road, corners | `style_rl/dstyle/track.py` | `Track.project` gives progress along the lap, offset from the centre line and the distance to both road edges. `Track.find_corners` finds the corners from the road's curvature alone. |
| Running estimate during a lap (training) | `style_rl/style12.py` | `RunningStyle` collects pedals, steering and position step by step and returns the 12 numbers of the lap so far. `StyleTargets` loads the targets and gives the distance. |
| Where the estimate is used | `style_rl/style12_env.py` | The reward pays the change of the distance D between the running 12 numbers and the target. |
| Training | `scripts/train12.py` | Trains the agent on full laps; every finished lap is saved with its 12 numbers in `library.csv`. |
| The targets | `targets/style12.json` | The 12 numbers of each driver and of the neutral driver, plus the spread of every number between real driver-track cells (the unit of D). |
| How the targets were made | `tools/make_style12_targets.py` | 5 laps of the driver model per driver, 12 numbers per lap, median. Needs the research repository (it holds the driver model). |
| Old, not generalised | `style_rl/style.py`, `style_rl/style_env.py`, `scripts/train.py` | Five numbers per corner. They need the reference lap of the same track, so they say nothing about other tracks. Kept for the first overnight run. |

`style_rl/dstyle/` is a copy of the standalone package `driver_style` (own README and tests, in the research repository under `driver-style-vector/`). The tests there check that this code gives the same numbers as the original research implementation (`scripts/dimra_metrics.py`, function `lap_metrics`) to 1e-4.

## The 12 numbers

| Group | Key | Name | What it measures | Unit |
|---|---|---|---|---|
| Braking | `B_mean_active` | Brake pressure | pedal value on average while the brake is on | 0..1 |
| Braking | `B_event_peak_mean` | Peak brake pressure | highest pedal value of a braking zone, averaged over the zones | 0..1 |
| Braking | `B_apply_rate` | Brake attack | peak value / time from brake-on to the peak | per s |
| Braking | `B_release_rate` | Brake release | mean pedal speed between the peak and the end of the zone | per s |
| Throttle | `T_apply_rate` | Throttle attack | mean pedal speed over the moments the throttle goes down quickly | per s |
| Throttle | `T_release_rate` | Throttle lift | mean pedal speed over the moments the throttle comes up quickly | per s |
| Steering | `S_reversals_5deg_per_km` | Steering corrections | changes of steering direction larger than 5 degrees | per km |
| Steering | `S_mean_abs_rate` | Hand speed | how fast the wheel is turned, averaged over time | deg/s |
| Steering | `S_corner_peak_angle` | Peak steering in corners | largest wheel angle in a corner, averaged over the corners | deg |
| Line | `L_border_mean_m` | Distance to the track edge | distance to the nearest road edge, averaged over time | m |
| Line | `L_inside_dist_apex_m` | Apex distance | distance to the inside edge at the apex, averaged over the corners | m |
| Line | `L_outside_dist_exit_m` | Exit width | distance to the outside edge 80 m after the apex, averaged over the corners | m |

## How a lap becomes 12 numbers (`style_vector` in `features.py`)

Input, one row per sample (about 25 per second): `time_s`, `throttle` (0..1), `brake` (0..1), `steer_deg` (steering wheel angle in degrees), `x`, `y` (position in the frame of the track file), plus the track geometry (`data/tracks/monza_geometry.csv`: centre line and both road edges). Nothing else.

1. **Place the car on the road** (`Track.project`): progress along the lap, offset from the centre line, distance to the left and right edge. A simulator that already reports these can pass them in as `q=` and skip `x`, `y`.
2. **Find the corners from the track alone** (`Track.find_corners`): a stretch of same-direction curvature with a radius under 200 m at its tightest point. No hand-drawn zones.
3. **Pedals and steering** are smoothed over 3 samples (0.12 s) and differentiated.
   - Braking numbers are measured per braking zone (brake above 0.05, gaps under 0.2 s bridged, zones under 0.2 s dropped) and averaged.
   - Throttle numbers average the pedal speed over the moments it moves faster than 0.2 per second.
   - Steering corrections count direction changes larger than 5 degrees (hysteresis, so noise is not counted).
4. **Line numbers** use the distance to the road edges, overall and at each corner (window from 100 m before to 80 m after the apex).
5. A driver on a track is the **median over the driver's laps** on that track. Use only laps that stay on the track.

In training, `style12.RunningStyle` calls the same `style_vector` on the lap so far every 25 steps. A number that cannot be measured yet (no brake zone, no corner) keeps the neutral driver's value, and the estimate moves from that value to the measured one as the lap advances (weight = share of the lap driven). At the end of the lap it is exactly the 12 numbers of the lap.

## Comparing two styles

Each number has its own unit, so each is divided by its spread between real driver-track cells (`targets/style12.json`, key `spread`). The distance **D** between two styles is the mean absolute difference of the 12 numbers in these units (1 = as far apart as two typical real driver-track cells). This D is what the training reward and the style margin use.

## Use it on a lap

```python
import pandas as pd
from style_rl.dstyle.features import style_vector
from style_rl.dstyle.track import Track

track = Track("data/tracks/monza_geometry.csv")
lap = pd.read_csv("my_lap.csv")        # columns: time_s, throttle, brake, steer_deg, x, y   (optional: lap_dist)
print(style_vector(lap, track))        # dict with the 12 numbers
```

Steering in the OCD simulator is a command in -1..1 (0.15 rad at the road wheels); `style12.STEER_DEG_PER_UNIT` converts it to steering wheel degrees (ratio 13.4, the same as the driver model that made the targets).
