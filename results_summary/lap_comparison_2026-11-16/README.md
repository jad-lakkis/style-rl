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

D = distance of the whole lap's 12 style numbers to the driver's target style, in spread units (0 = identical).

## How to open the replay

`replay.html` is a web page; VS Code does not run it. In the VS Code file list: right-click `replay.html`, choose "Reveal in File Explorer", then double-click the file (it opens in Chrome or Edge). If it opens in an editor instead: right-click it, "Open with", choose the browser.

In the page:
- **View** 3D or 2D map. 3D: drag = turn, wheel = zoom, double-click = reset. The height of each lap's line is its speed (switch it off or exaggerate it at the top); the thin line under a dot goes down to the track.
- **Sector** (or click a row of the sector table): zooms in and restarts all laps together at the sector entry, so only what happens inside the sector is shown (the tiny differences). The three charts under the map follow the sector: speed, speed minus the driver's speed at the same place, and the time gap to the driver's run along the distance.
- The tables on the right: live speed, gap and metres ahead; the time of each lap in each sector; the 12 style numbers of each lap with their distance from the target.
- Tick boxes switch single laps off; "trails" shows the last seconds of each lap.

## Figures (per driver)

- `<driver>_1_lap_time.png`: lap times, time gained per sector, gap to the driver's run along the lap.
- `<driver>_2_style.png`: the 12 style numbers of the four laps against the target (spread units) and the numbers themselves.
- `<driver>_3_lines_per_corner.png`: the line on the map in each corner zone (driver, style kept, fastest).
- `<driver>_4_driving.png`: speed, throttle, brake and steering along the lap (driver, style kept, fastest).
- `<driver>_5_style_vs_lap_time.png`: each of the 12 style numbers against lap time; every on-track lap of the run in grey, the four laps in colour, the target and the band within 0.3 spread.
- `summary.csv`, `<driver>_sectors.csv`, `<driver>_style_numbers.csv`: the numbers behind the figures.

## What it shows

- In the corner zones the lines of all laps are within about 1 m of each other on the map (mean difference 0.4-0.7 m); the differences are in braking and corner speed (figure 4, and the speed-difference chart in the replay).
- Attacker: the RL laps gain most in T3-T6 (the style-kept lap: 0.95 s of its 3.28 s), then T1-2 (0.87 s), T10 (0.70 s) and T7-9 (0.50 s). On top of the track's own line and speed plan, RL adds 1.16 s (attacker) and 0.32 s (smooth).
- Style: the attacker's kept lap is closer to the target than the driver's run overall (D 0.69 against 0.73), but not in every number: steering activity moved toward the target, the brake application rate moved away from it (figure 5).

Made by `tools/lap_compare.py` (lap selection, on the pod), `tools/lap_report.py` and `tools/replay_template.html` (figures and replay) from the runs `res1/attacker` and `res1/smooth` (residual mode: pure pursuit plus a bounded RL correction).
