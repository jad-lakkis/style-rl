# Lap comparison for coaching (2026-11-16)

The driver's run against RL laps, for the attacker and the smooth driver (two artificial drivers of our driver model). All laps are the same car on the same track (Monza, OCD simulator), started on the line.

| lap | what it is |
|---|---|
| driver's run | the pure-pursuit driver on the driver's own mapped line (the simulator twin of the driver; not a human lap) |
| RL lap, small gain | an RL lap only a little faster than the driver's run (target gain 0.5 s) that stays within the style budget of the driver's run |
| RL lap, style kept | the fastest on-track RL lap that stays within the same style budget |
| RL fastest lap | the fastest on-track RL lap, whatever its style |

| driver | driver's run | RL small gain | RL style kept | RL fastest |
|---|---|---|---|---|
| attacker | 119.80 s | 119.32 s | 116.52 s | 116.36 s |
| smooth | 120.92 s | 120.08 s | 117.36 s | 116.88 s |

All style numbers are shown as they are, each in its own unit (brake and throttle pedal 0-1 or per second, steering per km, degrees, degrees per second, metres). No target and no normalised values are shown. The style budget used to pick the "style kept" and "small gain" laps is the combined style distance of the problem; the per-lap values are in the training run's `library.csv`.

## How to open the replay

`replay.html` is a web page; VS Code does not run it. In the VS Code file list: right-click `replay.html`, choose "Reveal in File Explorer", then double-click the file (it opens in Chrome or Edge). If it opens in an editor instead: right-click it, "Open with", choose the browser.

In the page:
- **3D chase camera** (default): the camera follows one lap from behind (choose which one under "Camera behind"). Every lap is a car on the road, and its racing line is painted on the asphalt, so you see how each one takes the corner. Bottom left: speed, brake, throttle and the sideways position against the driver's run (in metres) of each lap. Drag = look around the car, wheel = closer / further, double-click = reset.
- **Cars placed**: "at the same place on the track" puts all cars at the same distance along the lap, so only the line and the speed differ; "at the same moment" is a live race in which the faster laps pull away.
- **Sector** (or click a row of the sector table): all laps start together at the sector entry. The three charts under the view follow the sector: speed, speed minus the driver's speed at the same place, and the time gap along the distance.
- **2D map** view as an alternative. The tables on the right show live speed and gap, the time of each lap in each sector, and the 12 style numbers of each lap in their units.

## Figures (per driver)

Style (values as they are, each in its own unit):
- `<driver>_2_style.png`: the 12 style numbers of the four laps, one small plot per number (dashed grey line = the driver's run).
- `<driver>_6_style_change.png`: how much each number changes from the driver's run to each RL lap, in the number's unit.
- `<driver>_5_style_vs_lap_time.png`: each of the 12 numbers against lap time, every on-track lap of the run in grey.

Lap time and driving:
- `<driver>_1_lap_time.png`: lap times, time gained per sector, gap to the driver's run along the lap.
- `<driver>_3_lines_per_corner.png`: the line on the map in each corner zone (driver, style kept, fastest).
- `<driver>_4_driving.png`: speed, throttle, brake and steering along the lap (driver, style kept, fastest).

Tables: `summary.csv`, `<driver>_sectors.csv`, `<driver>_style_numbers.csv`.

## What it shows

- **Lap time.** The RL laps gain most in T3-T6, then T1-2, T10 and T7-9 (attacker, style-kept lap: 0.95, 0.87, 0.70 and 0.50 s of its 3.28 s). In the corner zones the lines of all laps are within about 1 m of each other (mean difference 0.4-0.7 m); the differences are in braking and corner speed.
- **How the style changes (attacker, driver's run to style-kept lap).** Steering corrections 7.99 to 18.94 per km; hand speed 16.01 to 25.15 deg/s; brake pressure 0.68 to 0.56; peak brake pressure 0.98 to 0.84; brake attack 6.26 to 8.85 pedal/s; brake release 0.62 to 1.00 pedal/s; throttle attack 1.06 to 0.72 pedal/s; distance to the track edge 2.20 to 2.51 m. The line numbers barely change: peak steering in corners 66.11 to 67.15 deg, apex distance 1.68 to 1.61 m, exit width 4.04 to 4.01 m.
- **The style changes even for a small gain.** The small-gain lap (0.48 s faster) already differs a lot from the driver's run (steering corrections 7.99 to 17.38 per km, throttle lift 1.77 to 0.95 pedal/s, brake pressure 0.68 to 0.54), and so do nearly all RL laps of the run, whatever their lap time (figure 5).

Made by `tools/lap_compare.py` (lap selection, on the pod), `tools/lap_report.py` and `tools/replay_template.html` (figures and replay) from the runs `res1/attacker` and `res1/smooth` (residual mode: pure pursuit plus a bounded RL correction).
