# Lap comparison for coaching (2026-11-16)

The driver's run against two RL laps, for the attacker and the smooth driver (two artificial drivers of our driver model). All laps are the same car on the same track (Monza, OCD simulator), started on the line.

| lap | what it is |
|---|---|
| driver's run | the pure-pursuit driver on the driver's own mapped line (the simulator twin of the driver; not a human lap) |
| RL lap, style kept | the fastest on-track RL lap that stays within the driver's style budget |
| RL fastest lap | the fastest on-track RL lap, whatever its style |

| driver | driver's run | RL style kept | RL fastest |
|---|---|---|---|
| attacker | 119.80 s | 116.52 s | 116.36 s |
| smooth | 120.92 s | 117.36 s | 116.88 s |

## What is in the folder

- `replay.html`: the laps as cars on a 3D road with a chase camera, a sector view, and the tables. To open it: in VS Code right-click `replay.html`, "Reveal in File Explorer", double-click it (Chrome or Edge).
- `<driver>_1_lap_time.png`: lap times and the time gained in each sector.
- `<driver>_2_style.png`: the 12 style numbers of the three laps, as they are, each in its own unit.
- `<driver>_3_driving.png`: speed, throttle, brake and steering along the lap.
- `summary.csv`, `<driver>_sectors.csv`, `<driver>_style_numbers.csv`: the numbers behind the figures.
- `coaching/`: the input for the LLM coach (see below).

## The input for the LLM coach (`coaching/`)

Nothing here calls an LLM. The facts are measured on the laps and written into sentences by fixed rules, so every number in the text is a measurement.

- `coach_facts.csv` (the small table, both drivers): one row per fact: `level` (lap, sector, corner, style), `item`, `metric`, `unit`, the value for the driver's run, the RL style-kept lap and the RL fastest lap, and the change from the driver's run. Corner facts: where braking starts (m before the apex), the peak brake, the lowest speed, where the throttle comes back (m after the apex), the speed 80 m after the apex, the line at the apex (m towards the inside).
- `<driver>_brief.md`: the same facts as plain sentences: the laps, where the time is gained, corner by corner, the style (what changes and what stays the same), the fastest lap for comparison, and what the data cannot say.
- `<driver>_prompt.md`: an example prompt: the instructions for the LLM (use only the facts, at most three priorities, per corner, what to keep, what the data cannot say) and the brief.

Made by `tools/lap_compare.py` (lap selection, on the pod), `tools/lap_report.py` (figures, replay), `tools/coach_brief.py` (coaching input) and `tools/replay_template.html`, from the runs `res1/attacker` and `res1/smooth` (residual mode: pure pursuit plus a bounded RL correction).
