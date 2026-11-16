# Lap comparison for coaching (2026-11-16)

The driver's run against the faster RL laps, for the attacker and the smooth driver. All laps are the same car on the same track (Monza, OCD simulator), started on the line.

| lap | what it is |
|---|---|
| driver's run | the pure-pursuit driver on the driver's own mapped line (the simulator twin of the driver; not a human lap) |
| no RL (track line) | pure pursuit on the track's own line and speed plan, zero RL correction |
| RL lap, style kept | the fastest on-track RL lap whose style distance D is at most D(driver's run) + 0.05 |
| RL fastest lap | the fastest on-track RL lap, whatever its style |

## Open these

- `replay.html` (double-click): the laps as moving dots on the map, switch driver at the top, jump buttons for the four corner zones, live speed and gap, speed trace along the lap.
- `<driver>_1_lap_time.png`: lap times, time gained per sector, gap to the driver's run along the lap.
- `<driver>_2_style.png`: the 12 style numbers of the three laps against the target (spread units), and the numbers themselves.
- `<driver>_3_lines_per_corner.png`: the line on the map in each corner zone, with the zone times.
- `<driver>_4_driving.png`: speed, throttle, brake and steering along the lap.
- `summary.csv`, `<driver>_sectors.csv`, `<driver>_style_numbers.csv`: the numbers behind the figures.

## Numbers

| driver | driver's run | no RL | RL style kept | RL fastest |
|---|---|---|---|---|
| attacker | 119.80 s, D 0.73 | 117.68 s, D 1.07 | 116.52 s, D 0.69 | 116.36 s, D 0.78 |
| smooth | 120.92 s, D 1.06 | 117.68 s, D 1.29 | 117.36 s, D 1.10 | 116.88 s, D 1.24 |

Most of the gain over the driver's run comes from the track's own line and speed plan, not from RL: RL adds 1.16 s (attacker) and 0.32 s (smooth) on top of "no RL". The attacker's RL gain sits mostly in T3-T6 (about 0.6 s), then T10 (0.2 s), T7-9 (0.15 s) and T1-2 (0.1 s). On the map the lines of all laps are within about 1 m of each other: the difference is mainly in braking and corner speed (figure 4).

Style: D is the distance of the whole lap's 12 numbers to the target. The attacker's kept lap is closer to the target than the driver's run (0.69 against 0.73), but not in every number: steering activity moved toward the target, brake application rate moved away from it (see figure 2).

Made by `tools/lap_compare.py` (lap selection, on the pod) and `tools/lap_report.py` (figures and replay) from the runs `res1/attacker` and `res1/smooth` (residual mode, pure pursuit plus a bounded RL correction).
