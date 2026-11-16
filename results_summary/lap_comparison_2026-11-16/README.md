# Lap comparison for coaching (2026-11-16)

The driver's run against RL laps, for the attacker and the smooth driver. All laps are the same car on the same track (Monza, OCD simulator), started on the line.

| lap | what it is |
|---|---|
| driver's run | the pure-pursuit driver on the driver's own mapped line (the simulator twin of the driver; not a human lap) |
| RL lap, small gain | an RL lap only a little faster than the driver's run (target gain 0.5 s) that stays within the style budget D <= D(driver's run) + 0.05 |
| RL lap, style kept | the fastest on-track RL lap whose style distance D is at most D(driver's run) + 0.05 |
| RL fastest lap | the fastest on-track RL lap, whatever its style |

| driver | driver's run | RL small gain | RL style kept | RL fastest |
|---|---|---|---|---|
| attacker | 119.80 s, D 0.73 | 119.32 s, D 0.74 | 116.52 s, D 0.69 | 116.36 s, D 0.78 |
| smooth | 120.92 s, D 1.06 | 120.08 s, D 1.08 | 117.36 s, D 1.10 | 116.88 s, D 1.24 |

D = distance of the whole lap's 12 style numbers to the driver's target style, in spread units (0 = identical, 1 = as far apart as two typical real driver-track cells).

## How to open the replay

`replay.html` is a web page; VS Code does not run it. In the VS Code file list: right-click `replay.html`, choose "Reveal in File Explorer", then double-click the file (it opens in Chrome or Edge). If it opens in an editor instead: right-click it, "Open with", choose the browser.

In the page:
- **3D chase camera** (default): the camera follows one lap from behind (choose which one under "Camera behind"). Every lap is a car on the road, and its racing line is painted on the asphalt, so you see how each one takes the corner. Bottom left: speed, brake, throttle and the sideways position against the driver's run (in metres) of each lap. Drag = look around the car, wheel = closer / further, double-click = reset.
- **Cars placed**: "at the same place on the track" puts all cars at the same distance along the lap, so only the line and the speed differ; "at the same moment" is a live race in which the faster laps pull away.
- **Sector** (or click a row of the sector table): all laps start together at the sector entry. The three charts under the view follow the sector: speed, speed minus the driver's speed at the same place, and the time gap along the distance.
- **2D map** view as an alternative. The tables on the right show live speed and gap, the time of each lap in each sector, and the 12 style numbers of each lap with their distance from the target.

## Figures (per driver)

Style (the main point):
- `<driver>_6_style_change.png`: how much each of the 12 style numbers changes from the driver's run to each RL lap (spread units), in plain names, with the numbers in their own unit.
- `<driver>_7_style_toward_target.png`: for each RL lap, does each number move toward the target or away from it (green / red arrows)?
- `<driver>_8_style_vs_lap_time_overall.png`: lap time against D (distance to the target) and against the average style change from the driver's run, for every on-track lap of the run.
- `<driver>_5_style_vs_lap_time.png`: each of the 12 numbers against lap time, every on-track lap of the run in grey.
- `<driver>_2_style.png`: the 12 numbers of the four laps against the target, and the numbers themselves.

Lap time and driving:
- `<driver>_1_lap_time.png`: lap times, time gained per sector, gap to the driver's run along the lap.
- `<driver>_3_lines_per_corner.png`: the line on the map in each corner zone (driver, style kept, fastest).
- `<driver>_4_driving.png`: speed, throttle, brake and steering along the lap (driver, style kept, fastest).

Tables: `summary.csv`, `<driver>_sectors.csv`, `<driver>_style_numbers.csv`.

## What it shows

- **Lap time.** The RL laps gain most in T3-T6, then T1-2, T10 and T7-9 (attacker, style-kept lap: 0.95, 0.87, 0.70 and 0.50 s of its 3.28 s). On top of the track's own line and speed plan, RL adds 1.16 s (attacker) and 0.32 s (smooth). In the corner zones the lines of all laps are within about 1 m of each other (mean difference 0.4-0.7 m); the differences are in braking and corner speed.
- **How much the style changes.** Compared with the driver's run, the style-kept lap changes the 12 numbers by 0.72 spread units on average (attacker; 9 of 12 numbers by more than 0.3, 4 by more than 1) and by 0.75 (smooth). The biggest changes (attacker): steering corrections +1.8 (8 to 19 per km), peak brake pressure -1.2, hand speed +1.1, brake pressure -1.0, brake attack +0.8. The line numbers (peak steering in corners, apex distance, exit width) barely change.
- **Towards or away from the target.** The attacker's style-kept lap is closer to the target in 5 of 12 numbers and further in 7; the steering numbers move toward the target, the braking numbers away from it. D falls from 0.73 to 0.69 because the steering moves are large.
- **Important.** D is the distance to the target, not to the driver's run. Even the small-gain lap changes the style by 0.65 (attacker) and 0.93 (smooth) from the driver's run, and nearly every RL lap of the run lies 0.5-0.9 from it, whatever its lap time (figure 8). So "style kept" means: as far from the target as the driver's run is, but not the same style as the driver's run.

Made by `tools/lap_compare.py` (lap selection, on the pod), `tools/lap_report.py` and `tools/replay_template.html` (figures and replay) from the runs `res1/attacker` and `res1/smooth` (residual mode: pure pursuit plus a bounded RL correction).
