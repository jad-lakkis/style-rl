# style-rl

Reinforcement learning (SAC) that drives Monza in **Open-Car-Dynamics (OCD)** with a given **driving style** and a given **lap time**.

Two completely opposite simulated drivers, both on the Dallara F317 car of OCD:

| Driver | Style | Reference lap (driver model, F317) |
|---|---|---|
| `attacker` | brakes late and hard, releases the brake abruptly, early abrupt throttle, busy hands | 104.6 s |
| `smooth` | brakes early and gently, progressive release (trail braking), patient throttle, calm hands | 105.8 s |

One SAC agent is trained per driver. Goal of this stage: **a lap time similar to the reference (about a second faster is welcome) while keeping the style of that driver.** Later stages (not here): coach the driver from the agent's lap, then on unseen tracks.

## How it works

- **Reference** (`targets/*.npz`): for each driver a lap on Monza made by a driver model with known style settings (line, speed profile, brake and throttle pedals). They come from the research repository `jad-race` (`scripts/ocd_targets.py`, `scripts/artificial_dataset.py`).
- **Environment** (`style_rl/style_env.py`): the agent sends steering, throttle, brake to OCD every 0.04 s.
  - `residual` mode (default): a base driver (`style_rl/base_driver.py`: pure pursuit on the reference line + reference speed and pedals) already drives the lap; the agent adds a correction (+-0.35). An untrained agent therefore starts at about the reference pace and style, which avoids the long phase in which a pure-SAC agent never gets round a corner.
  - `pure` mode: the three numbers are the commands, no help.
- **Reward:** progress along the lap; penalties for leaving the track, drifting from the reference line, jerky actions, being much slower than the reference; at the end of every corner where the reference brakes, **a style term**: minus the scaled difference between the agent's five corner numbers and the reference's. Corner numbers (`style_rl/style.py`): where the brake goes on, where it comes off, peak brake, mean brake, where the throttle passes 50%. In full-lap evaluation a bonus for finishing and a term for the lap time against the reference.
- **Training episodes:** 700 steps (28 s) from a random place on the track at about the reference speed. **Evaluation:** a full lap from the start line every 25 000 steps (`eval.csv`, `laps/step_N.csv`).
- **Network:** SAC (stable-baselines3), 2 hidden layers of 256 (about 214 000 trainable parameters), batch 256, buffer 300 000, one update per two steps.

## Interface: actions, states, reward

**Actions (same as the Assetto Corsa / OCD controls).** The agent outputs 3 numbers in [-1, 1] every 0.04 s. What reaches the car is:

| command | range | how it is made from the agent output `a` |
|---|---|---|
| steering | -1 ... 1 (x 0.15 rad at the road wheel in OCD) | `a[0]` |
| throttle | 0 ... 1 | `clip(a[1], 0, 1)` (pure mode) |
| brake | 0 ... 1 | `clip(a[2], 0, 1)` (pure mode); brake > 0.05 cuts the throttle |

- `pure` mode: the 3 numbers are the commands, no help. `residual` mode: command = base driver + 0.35 x `a`, clipped to the same ranges.
- Both modes are trained (four agents: attacker/smooth x residual/pure).

**States (27 numbers).** Car: speed, side-slip angle, yaw rate, offset from the reference line, heading error, on-track flag, the previous steering/throttle/brake. Map and target (known in advance, as for a driver who knows the track and the line to drive): reference speed 30/80/150 m ahead, reference curvature 10/30/60/120 m ahead, reference throttle and brake at this point, distance and angle of the next corner, position in the lap (sin, cos), and the base driver's suggested steering/throttle/brake. No quantity a car sensor or the game could not give is used.

**Reward per step.** + progress along the lap; - leaving the track (and the episode ends after 1 s outside); - distance from the reference line; - jerky pedals/steering (change of the 3 commands); - being much slower than the reference. **Style term:** when a corner is completed, - (difference between the agent's 5 corner numbers - brake start, brake end, peak brake, mean brake, throttle pick-up - and the reference driver's). Full lap: + finishing bonus and a term for lap time against the reference.

## Run

Test without OCD (stand-in vehicle, **not OCD physics**; only checks the code):
```
pip install -r requirements.txt torch
python scripts/smoke_test.py --backend mock
python scripts/train.py --driver attacker --backend mock --steps 20000 --out runs/test
```
With OCD (Ubuntu 22.04 with ROS 2 Humble and the built Open-Car-Dynamics, `source .../install/local_setup.sh`):
```
python scripts/smoke_test.py --backend ocd            # about 5 minutes; checks the simulator, the base driver and a short training
python scripts/train.py --driver attacker --backend ocd --out runs/attacker --max-hours 9.5
python scripts/train.py --driver smooth   --backend ocd --out runs/smooth   --max-hours 9.5
```
Read the result: `runs/<driver>/eval.csv` (lap time, completed, style error, off-track share per evaluation), `runs/<driver>/laps/` (the evaluated laps), `best.zip` (best completed lap by lap time + 10 x style error).

## RunPod (what the unattended run does)

Pod: Docker image `ros:humble`, CPU pod (8 vCPU is enough for two agents), start command `bash -lc "curl -fsSL https://raw.githubusercontent.com/jad-lakkis/style-rl/main/runpod/pod_start.sh | bash"`.
`runpod/pod_start.sh` clones this repository, builds OCD (`runpod/setup_pod.sh`, about 20 to 40 minutes), runs the smoke test, and only then starts both trainings (`runpod/run_overnight.sh`, 9.5 h limit). If the build or the smoke test fails it stops and says why in the pod log. Results stay in `/workspace/runs` of the pod.

## Status and honest limits

- The code is tested **only on the stand-in vehicle** (unit tests, smoke test, a short training), not yet on OCD. The first thing to look at on OCD is the smoke test's step 2: whether the base driver alone stays on the track at Monza.
- The reference laps come from a driver model with the F317 envelope, not from OCD; the OCD car may be faster or slower than the reference, so "similar lap time" is judged against the reference after seeing what the base driver does in OCD.
- Speed of training depends on OCD's step time (see `scripts/smoke_test.py` output). Expect a few tens of steps per second per agent on 2 CPU threads.
