# style-rl: what the overnight run did and did not achieve

Run: 2026-11-08 21:08 UTC to 2026-11-09 03:23 UTC, one GPU pod (RTX PRO 4000), Monza in Open-Car-Dynamics, Dallara F317. Two artificial drivers with opposite styles (`attacker`: late, hard braking, abrupt release, early throttle; `smooth`: early, gentle braking, progressive release, patient throttle). Cost: $4.33 for all pods (cap was $5) plus a few cents of volume storage.

## 1. The short version

| question | answer |
|---|---|
| Can the agents drive the car with steering, throttle and brake only, on the track? | **Yes.** 123 of the 134 full-lap evaluations (92%) were finished; on the finished laps the car was outside the track for at most 0.6% of the lap (exactly 0% on 107 of them). The 11 unfinished evaluations ended off the track or stalled. |
| Do they reach lap times similar to the original driver? | **Yes, while anchored:** within 1 s of the original driver in 75-100% of evaluations before 300k steps. |
| Did they get faster than the original driver with the style kept? | **Barely.** Best "records" are -0.24 s (attacker) and -0.08 s (smooth), about the size of the noise. No second faster. |
| Did they converge? | **No.** Training was numerically stable, but once the pull towards the original driver faded to zero (400k steps) the lap times and the style got worse and noisier. |
| Is the style constraint satisfied? | **Early yes, later no** (table in section 4). The residual agents never satisfied it. |
| Did the reward go up? | Yes, but it did not measure what we want (section 5). |

The best checkpoints are almost copies of the original driver (figure 08): the "records" are the original driver with tiny variations, not a new technique.

## 2. What was run

| agents | what they are | steps | result in one line |
|---|---|---|---|
| `*_residual` (attacker, smooth) | base driver drives, the agent adds a correction of +-0.35 | 405k | learned to finish laps, but stayed 0.8-1.2 s slower than the original driver with a style error 4-8 (original: 1.8 / 2.6). Stopped at 23:40 (not competitive). |
| `*_pure` | the agent's three numbers ARE steering, throttle, brake; trained from scratch | 225k | never finished a lap in 9 evaluations. Replaced at 22:28. |
| `*_pure_bc` | pure, started as a copy of the base driver, actor pulled towards it | 100k | finished laps at base-driver level, but the entropy coefficient ran away to 1e8 and the critic loss to 1e17 (anchor acted on sampled actions). Replaced at 23:09. |
| `*_pure_bc2` | same with the anchor on the MEAN action and the entropy coefficient capped | 630-655k | the best agents: stable, base-driver-level while anchored. |
| `*_pure_bc3` | bc2 with 5x the reward per metre of progress | 575k | similar to bc2; one faster lap (attacker 119.44 s) but with the style lost (3.31). |

The anchor (weight 5, fading linearly to 0 at 400k steps) and the warm start (40k base-driver steps in the replay buffer, behaviour cloning, critic-only warm-up) were added during the run after the first results (restarts at 22:28, 23:09 and 23:46).

## 3. Lap times and records (figures 01, 03; table `best_by_agent.csv`)

Original driver in OCD: attacker 119.92 s (style error 1.78), smooth 120.88 s (style error 2.62). Reference laps from the driver model: 118.96 s and 119.12 s.

| agent | best lap | vs original | style error there | last evaluated lap | records |
|---|---|---|---|---|---|
| attacker_pure_bc2 | 119.68 s (step 280k) | **-0.24 s** | 1.68 (original 1.78) | 122.92 s, style 4.06 | 1 |
| attacker_pure_bc3 | 119.44 s (step 525k) | -0.48 s, **but style 3.31 > tolerance 2.53, so not a record** | 3.31 | 122.76 s, style 3.59 | 2 (119.84 s, -0.08 s) |
| attacker_residual | 121.08 s | +1.16 s | 5.28 | 123.24 s | 0 |
| smooth_pure_bc3 | 120.80 s (step 175k) | -0.08 s | 2.47 (original 2.62) | 121.00 s, style 4.65 | 1 |
| smooth_pure_bc2 | 121.00 s | +0.12 s | 3.14 | 125.12 s, style 7.07 | 0 |
| smooth_residual | 121.64 s | +0.76 s | 7.70 | 121.64 s | 0 |

A "record" is a finished lap that is faster than the original driver while the style error stays below 1.25 x the original's + 0.3. Differences of a few tenths of a second between single evaluation laps of one policy are within the lap-to-lap variation of the policies themselves; there is one seed per agent, so none of these gains can be called statistically established.

## 4. Constraints (figure 04; table `constraints_by_agent.csv`)

Share of full-lap evaluations meeting each constraint. "Before" = up to 300k steps (anchor strong), "after" = beyond.

| agent | on track and lap finished | style within tolerance | lap within 1 s of original |
|---|---|---|---|
| attacker_pure_bc2 | before 100%, after 86% | before 100%, **after 29%** | before 83%, **after 21%** |
| attacker_pure_bc3 | before 100%, after 82% | before 100%, **after 9%** | before 75%, **after 36%** |
| smooth_pure_bc2 | before 100%, after 87% | before 75%, **after 7%** | before 100%, after 33% |
| smooth_pure_bc3 | before 100%, after 91% | before 83%, **after 27%** | before 100%, after 18% |
| both residual agents | 80-100% | **never (0%)** | 0-40% |

Staying on the track holds throughout (training episodes: inside the track 98-99% of the time, 5-10% of exploring episodes ended off-track). The style and pace constraints hold only while the pull to the original driver is strong. For the smooth driver "corners matched (error <= 2)" is low (4-15%) even early, because the original smooth driver itself has a style error of 2.62: read it against that, not against 2.

## 5. Why the reward went up while the results got worse (figures 05, 06)

- Return per episode rose (attacker bc2 from about -71 to about -13) and then flattened.
- Figure 06 splits it: **progress is flat, style is flat at about -5 to -6, the improvement is the "slower than the reference" penalty shrinking** (about -60 to -10) and fewer early off-track episodes. In other words the agents learned to get back up to the reference speed quickly after the random starts of the training episodes.
- The reward never pays for being faster than the reference (the progress term, 0.02 per metre, is tiny), and the style term (only fired at completed corners, about 0.5 per 700-step episode, capped at 8 per corner) gave a weak, sparse signal. So nothing in the training reward asks for the thing we evaluate: a faster full lap with the same style.
- After 400k steps, when the pull vanished, the actor loss jumped (figure 07), episodes started ending earlier, the off-track penalty grew again (figure 06), the pedals started dithering and the throttle stopped reaching 100% on the straights (figure 08, grey).

## 6. What the style looks like (figures 08, 09)

Best attacker lap (`attacker_pure_bc2`, step 280k), against the reference driver, corner by corner, in units of the style scale (one unit = 20 m for the brake and throttle points, 0.25 for peak and 0.20 for mean pedal):
- starts braking about 4 to 10 m earlier (+0.2 to +0.5 units); brake comes off earlier by about 2 to 24 m (+0.1 to +1.2 units; `brake_end` is the distance *before* the apex where the brake is released, larger = earlier release);
- peak brake about 0.05 higher (0.98 against 0.93); mean pedal while braking 0.02 to 0.14 lower;
- throttle point between about 6 m earlier and 16 m later.
So it is the same braking pattern, a little earlier and a little more peaky. The last lap (step 630k) has several corners clearly off (corner 6 braking starts more than 60 m earlier; corner 5 throttle point about 48 m earlier).

## 7. Honest limits

- One seed per agent; evaluation is a single deterministic lap each 25k steps.
- The reference laps come from a driver model on a measured envelope of the OCD car (the first, theoretical-grip version could not be driven by the real car). The attacker's base driver needs a traction governor, otherwise the car spins at the exit of the first chicane.
- The anchor schedule (to zero at 400k) was a design choice made during the run; it is the main reason for the late degradation, not proof that SAC cannot improve on the original driver.
- The "faster" gains are of the same order as the variation between neighbouring evaluations of one policy.
- Residual agents never had the anchor; the comparison residual vs pure is therefore partly a comparison of "anchored vs not".

## 8. What I would do next (not done)

1. Keep a permanent pull towards the original driver (a floor of the anchor weight instead of fading to zero), and resume from the best checkpoints (step 280k of the bc2 agents, step 200-225k of the bc3 attacker), not from the degraded last ones.
2. Give the training reward a term that pays for beating the reference speed in the places where the car has headroom, and train on full laps (or add a full-lap time term) instead of only 700-step random-start episodes.
3. Evaluate each saved checkpoint on several laps, and use several seeds, before calling any difference real.
4. Track the style per corner during training (it is logged in `corners/step_N.csv`) and use the corner-level error as the early-stopping signal.

## 9. Files

- `FINAL/` (next to this folder): every agent's `eval.csv`, `episodes.csv`, `progress.csv`, `corners/`, `laps/`, `checkpoints/step_N.zip`, `records.csv`, `best_keep_style.zip`, `best.zip`, `latest.zip`, TensorBoard events and logs; pod logs (`setup.log`, `smoke.log`, `diagnose.log`, `envelope.log`), the measured car envelope `ocd_envelope.json`.
- Replay buffers (`latest_buffer.pkl`, about 70 MB each) are only on the network volume (needed to resume, not for analysis).
- Code: https://github.com/jad-lakkis/style-rl (analysis: `analysis/analyze_runs.py`).
