# Prompt for the coaching LLM (example for the smooth)

## System
You are a racing coach. You receive facts measured on two simulated laps of the same car on the same track: the driver's run, and an RL lap that is faster while keeping the driver's style
within a budget. Coach the driver so that the driver's laps move toward the RL lap without losing the driver's style.

Rules:
1. Use only the facts in the brief. Quote numbers exactly, with their units. Never invent a corner, a number or a cause.
2. Coach per corner. Give at most three priorities, ordered by the time they are worth (the sector times are in the brief).
3. For each priority say: what to change, by how much (the numbers), and what it is worth (seconds in that zone).
4. Then list what to keep as it is (the style numbers that barely change).
5. Say plainly what the data cannot tell you.
6. Plain language, no jargon beyond the names used in the brief. At most 250 words.

## Brief
# Coaching brief: smooth

## The laps
The driver's run takes 120.92 s. The RL lap that keeps the driver's style within its budget takes 117.36 s, 3.56 s faster. Everything below compares the driver's run with this RL lap.

## Where the time is gained
Biggest gains: C1 T1-2 (770-1122 m) 0.96 s faster (13.67 s to 12.72 s); C2 T3-6 (1970-3032 m) 0.93 s faster (28.29 s to 27.36 s); C3 T7-9 (3763-4295 m) 0.65 s faster (12.40 s to 11.75 s); S0 (0-770 m) 0.38 s faster (12.30 s to 11.92 s); S1 (1122-1970 m) 0.38 s faster (14.91 s to 14.53 s); S3 (4295-4939 m) 0.24 s faster (10.75 s to 10.50 s).
The straights together change by -0.99 s; the corner zones by -2.56 s. In the corner zones the two lines are 0.7 m apart on average (at most 1.8 m): the time comes mostly from braking and speed, not from a different line.

## Corner by corner
- T1: the RL lap starts braking 53 m later (213 m before the apex, the driver's run 266 m), is back on the throttle 44 m earlier, is faster 80 m after the apex (25.8 against 22.3 m/s). (zone C1 T1-2: -0.96 s)
- T2: the RL lap presses the brake harder (peak 0.76 against 0.50), is faster 80 m after the apex (35.4 against 30.0 m/s), takes the apex 0.5 m closer to the inside. (zone C1 T1-2: -0.96 s)
- T3: the RL lap starts braking 43 m later (189 m before the apex, the driver's run 231 m), carries less speed through the corner (lowest speed 21.4 against 23.5 m/s), is back on the throttle 83 m earlier, is faster 80 m after the apex (29.3 against 27.8 m/s), takes the apex 0.5 m closer to the inside. (zone C2 T3-6: -0.93 s)
- T4: the RL lap carries less speed through the corner (lowest speed 22.1 against 24.0 m/s), is back on the throttle 83 m earlier, is faster 80 m after the apex (37.2 against 33.5 m/s), takes the apex 0.4 m closer to the inside. (zone C2 T3-6: -0.93 s)
- T5: the RL lap presses the brake harder (peak 0.87 against 0.63), takes the apex 0.8 m closer to the inside. (zone C2 T3-6: -0.93 s)
- T6: the RL lap starts braking 17 m later (95 m before the apex, the driver's run 112 m), is back on the throttle 18 m earlier, is faster 80 m after the apex (40.6 against 38.5 m/s), takes the apex 0.7 m closer to the inside. (zone C2 T3-6: -0.93 s)
- T7: the RL lap starts braking 31 m later (146 m before the apex, the driver's run 177 m), presses the brake less hard (peak 0.81 against 1.00), carries less speed through the corner (lowest speed 33.6 against 35.4 m/s), is back on the throttle 28 m earlier, is faster 80 m after the apex (41.9 against 39.6 m/s), takes the apex 0.4 m closer to the inside. (zone C3 T7-9: -0.65 s)
- T8: the RL lap carries more speed through the corner (lowest speed 40.5 against 38.8 m/s), is faster 80 m after the apex (46.7 against 43.0 m/s), takes the apex 0.8 m wider. (zone C3 T7-9: -0.65 s)
- T9: the RL lap carries more speed through the corner (lowest speed 44.1 against 41.4 m/s), is back on the throttle 10 m earlier, is faster 80 m after the apex (48.8 against 44.1 m/s), takes the apex 0.6 m closer to the inside. (zone C3 T7-9: -0.65 s)
- T10: the RL lap starts braking 24 m later (174 m before the apex, the driver's run 197 m), is back on the throttle 17 m earlier, takes the apex 0.4 m wider. (zone C4 T10: -0.02 s)

## The style (the 12 style numbers of the whole lap)
- Brake attack (how fast it is applied): 13.47 to 3.03 pedal per s (-10.44): the RL lap applies the brake more progressively.
- Steering corrections: 3.65 to 10.43 per km (+6.78): the RL lap corrects the steering more often.
- Throttle attack: 0.47 to 0.78 pedal per s (+0.31): the RL lap gets on the throttle more abruptly.
- Throttle lift: 2.18 to 1.85 pedal per s (-0.33): the RL lap lifts off the throttle more gradually.
- Exit width: 3.92 to 4.35 m (+0.43): the RL lap stays further from the outside edge on the exit.
- Brake pressure: 0.43 to 0.51 on the 0-1 pedal scale (+0.08): the RL lap presses the brake harder while braking.
- Hand speed: 12.52 to 17.34 deg per s (+4.82): the RL lap turns the wheel faster (busier hands).
- Apex distance: 2.05 to 1.76 m (-0.29): the RL lap gets closer to the inside edge at the apex.
- Brake release: 0.47 to 0.74 pedal per s (+0.27): the RL lap lets the brake off faster.
- Practically unchanged: Peak brake pressure (0.75 to 0.79 on the 0-1 pedal scale); Distance to the track edge (2.52 to 2.48 m); Peak steering in corners (64.45 to 64.80 deg).

## What the data cannot say
- All laps are simulator laps of the same car. The driver's run is the pure-pursuit replay of the driver's line (a model, not a human lap).
- The style numbers are whole-lap values; the corner facts are measured on these two laps only (one lap each).
- The data shows what differs, not why it makes the car faster.

## Task
Write the coaching message for the smooth driver.
