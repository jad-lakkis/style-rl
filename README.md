# style-rl

Reinforcement learning (SAC) that drives Monza in **Open-Car-Dynamics (OCD)** with a given **driving style** and a given **lap time**.

Two completely opposite simulated drivers, both on the Dallara F317 of OCD:

| Driver | Style | Reference lap (driver model on the OCD car) |
|---|---|---|
| `attacker` | brakes late and hard (peak brake 0.94), releases the brake abruptly, early abrupt throttle, busy hands | 118.96 s |
| `smooth` | brakes early and gently (peak brake 0.58), progressive release (trail braking), patient throttle, calm hands | 119.12 s |

Goal of this stage: **a lap time similar to the reference (a bit faster is welcome) while keeping the style of that driver.** Later stages (not here): coach the driver from the agent's lap, then on unseen tracks.

## The reference laps are built for the car the agent drives

The first version used a theoretical grip model of the F317 and gave 104.6 s / 105.8 s laps. The real OCD car cannot do that: it brakes with about half the force (0.6 g at 20 m/s, 1.9 g at 65 m/s), corners at 2.85 g at best and has a top speed of 70 m/s, so it left the track at the first chicane. The car was therefore **measured in the simulator** (`scripts/measure_ocd_envelope.py` -> `data/ocd_envelope.json`) and the two reference laps were rebuilt with the same driver model and the same style settings on that envelope (`tools/make_targets.py`; the driver model itself lives in the research repository `jad-race`). The old targets are kept in `targets/old_f317_theoretical/`; safety-margin variants (lateral and braking limits x 0.9 / 0.8) are in `targets/variants/`.

## How it works

- **Reference** (`targets/*.npz`): for each driver a lap on Monza (line, speed profile, brake and throttle pedals) plus its per-corner style numbers.
- **Environment** (`style_rl/style_env.py`): the agent sends steering, throttle, brake to OCD every 0.04 s. The car starts on the reference line.
  - `residual` mode: a base driver (`style_rl/base_driver.py`: pure pursuit on the reference line + reference speed and pedals; for the attacker with a traction governor that caps the throttle at large steering, otherwise the car spins at the exit of the first chicane) already drives the lap; the agent adds a correction (+-0.35). In OCD the base driver alone finishes the lap: attacker 119.9 s, smooth 120.9 s.
  - `pure` mode: the agent's three numbers are the commands, no help.
- **Network:** SAC (stable-baselines3), 2 hidden layers of 256 (about 214 000 trainable parameters), batch 256, buffer 300 000, one update per step on the GPU.

## Interface: actions, states, reward

**Actions (same as the Assetto Corsa / OCD controls).** The agent outputs 3 numbers in [-1, 1] every 0.04 s. What reaches the car:

| command | range | from the agent output `a` |
|---|---|---|
| steering | -1 ... 1 (x 0.15 rad at the road wheel in OCD) | `a[0]` |
| throttle | 0 ... 1 | `clip(a[1], 0, 1)` (pure mode) |
| brake | 0 ... 1 | `clip(a[2], 0, 1)` (pure mode); brake > 0.05 cuts the throttle |

In `residual` mode: command = base driver + 0.35 x `a`, clipped to the same ranges.

**States (27 numbers).** Car: speed, side-slip angle, yaw rate, offset from the reference line, heading error, on-track flag, the previous steering/throttle/brake. Map and target (known in advance): reference speed 30/80/150 m ahead, reference curvature 10/30/60/120 m ahead, reference throttle and brake at this point, distance and angle of the next corner, position in the lap (sin, cos), and the base driver's suggested steering/throttle/brake.

**Reward per step.** + progress along the lap; - leaving the track (the episode ends after 1 s outside); - distance from the reference line; - jerky pedals/steering; - being much slower than the reference. **Style term:** when a corner is completed, - (difference between the agent's 5 corner numbers - brake start, brake end, peak brake, mean brake, throttle pick-up - and the reference driver's). Full lap: + finishing bonus and a term for lap time against the reference.

## What is saved (per agent, in `runs/<tag>/<driver>_<mode>/`)

| file | content |
|---|---|
| `progress.csv`, TensorBoard events | SAC losses (actor, critic, entropy coefficient and its loss), fps, mean return, and rolling means of the episode statistics below |
| `episodes.csv` | every training episode: return split into its terms (progress, line, jerk, slow, off-track, stall, style, lap), how it ended, share of time inside the track, line error, mean speed, style error, share of matched corners |
| `eval.csv` | a full lap from the start line every 25 000 steps: lap time and its difference to the reference and to the base driver, style error, matched corners (error <= 2 / <= 1), off-track share, steering/pedal statistics, policy entropy, Q value |
| `corners/step_N.csv` | the five style numbers of the agent against the reference at each corner |
| `laps/step_N.csv`, `laps/base_driver.csv` | the evaluated laps (time, position, speed, pedals) and the original driver's lap in the simulator |
| `checkpoints/step_N.zip` | one model per evaluation, all kept, so earlier and later versions can be compared |
| `records.csv`, `best_keep_style.zip` | every new fastest lap that beats the base driver while the style error stays within tolerance (1.25 x the base driver's + 0.3) |
| `best.zip`, `latest.zip`, `latest_buffer.pkl` | best completed lap by lap time + 10 x style error; resumable state |
| `baseline.json`, `args.json`, `system.txt` | the base driver's result, all settings, commit, disk and device |

`python scripts/compare_runs.py --runs runs/<tag> [--plot]` prints (and plots) the agents against the original driver; the pod prints it every 30 minutes and at the end (`summary.txt`).

## Run

Test without OCD (stand-in vehicle, **not OCD physics**; only checks the code):
```
pip install -r requirements.txt
python scripts/smoke_test.py --backend mock
python scripts/train.py --driver attacker --backend mock --steps 20000 --out runs/test
```
With OCD (Ubuntu 22.04, ROS 2 Humble, built Open-Car-Dynamics sourced):
```
python scripts/diagnose_ocd.py                 # steering sign, car limits against the reference's demand, base driver laps (long version while runpod/HOLD exists)
python scripts/smoke_test.py --backend ocd
python scripts/train.py --driver attacker --backend ocd --mode residual --out runs/attacker_residual --max-hours 6.5
```

## RunPod (what the unattended run does)

Docker image `ros:humble`, GPU pod, network volume on `/workspace` (OCD build, pip cache, repository, results), start command `curl .../runpod/pod_start.sh | bash`. `pod_start.sh` clones/pulls this repository, builds OCD once, measures the car, checks the base driver (`scripts/diagnose_ocd.py`), runs the smoke test, then trains four agents in parallel (attacker/smooth x residual/pure, `runpod/run_overnight.sh`) for `runpod/HOURS` hours, writes `/workspace/DONE` and stops the pod through the RunPod API (`runpod/stop_pod.sh`; a watchdog stops it 45 minutes after the planned end in any case). If `runpod/HOLD` exists in the repository the pod does not train but re-checks after every new commit. `runpod/RUN_TAG` names the results folder: bump it whenever the environment or base driver changes so old checkpoints are not resumed.

## Honest limits

- The reference laps come from a driver model on a *measured* envelope of the OCD car; the base driver follows them only approximately (see `scripts/diagnose_ocd.py` output), so "similar lap time" is judged against the reference and against the base driver's own lap in OCD.
- The attacker's base driver needs the traction governor; without it the real car spins. That is a property of the OCD tyre/brake model with the current working brake calibration.
- Speed of training: a few tens of steps per second per agent (four agents share the GPU and 12 vCPUs).
