# Prompt for the coaching LLM (example for the attacker)

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
# Coaching brief: attacker

## The laps
The driver's run takes 119.80 s. The RL lap that keeps the driver's style within its budget takes 116.52 s, 3.28 s faster. Everything below compares the driver's run with this RL lap.

## Where the time is gained
Biggest gains: C2 T3-6 (1970-3032 m) 0.95 s faster (27.75 s to 26.80 s); C1 T1-2 (770-1122 m) 0.87 s faster (13.74 s to 12.86 s); C4 T10 (4939-5559 m) 0.70 s faster (13.22 s to 12.52 s); C3 T7-9 (3763-4295 m) 0.50 s faster (12.28 s to 11.78 s); S1 (1122-1970 m) 0.22 s faster (14.76 s to 14.54 s).
The straights together change by -0.27 s; the corner zones by -3.02 s. In the corner zones the two lines are 0.5 m apart on average (at most 2.0 m): the time comes mostly from braking and speed, not from a different line.

## Corner by corner
- T1: the RL lap starts braking 26 m earlier (218 m before the apex, the driver's run 192 m), carries more speed through the corner (lowest speed 16.6 against 14.0 m/s), is back on the throttle 13 m earlier, is faster 80 m after the apex (23.3 against 21.8 m/s). (zone C1 T1-2: -0.87 s)
- T2: same low point as T1, no other clear difference.
- T3: the RL lap starts braking 23 m earlier (195 m before the apex, the driver's run 172 m), carries more speed through the corner (lowest speed 22.4 against 20.5 m/s). (zone C2 T3-6: -0.95 s)
- T4: the RL lap carries more speed through the corner (lowest speed 22.6 against 20.7 m/s), is slower 80 m after the apex (35.9 against 39.4 m/s). (zone C2 T3-6: -0.95 s)
- T5: the RL lap starts braking 9 m later (102 m before the apex, the driver's run 112 m), presses the brake less hard (peak 0.80 against 1.00), carries more speed through the corner (lowest speed 38.4 against 35.4 m/s), is back on the throttle 49 m earlier, is faster 80 m after the apex (43.8 against 39.3 m/s). (zone C2 T3-6: -0.95 s)
- T6: the RL lap starts braking 9 m earlier (103 m before the apex, the driver's run 94 m), carries more speed through the corner (lowest speed 34.9 against 32.7 m/s), is back on the throttle 11 m earlier. (zone C2 T3-6: -0.95 s)
- T7: the RL lap carries more speed through the corner (lowest speed 35.6 against 29.2 m/s). (zone C3 T7-9: -0.50 s)
- T10: the RL lap carries more speed through the corner (lowest speed 41.7 against 37.6 m/s), is back on the throttle 78 m earlier, is faster 80 m after the apex (44.1 against 40.8 m/s). (zone C4 T10: -0.70 s)
- No clear difference in: T8, T9.

## The style (the 12 style numbers of the whole lap)
- Steering corrections: 7.99 to 18.94 per km (+10.95): the RL lap corrects the steering more often.
- Peak brake pressure: 0.98 to 0.84 on the 0-1 pedal scale (-0.13): the RL lap hits a lower peak brake pressure in each braking zone.
- Hand speed: 16.01 to 25.15 deg per s (+9.14): the RL lap turns the wheel faster (busier hands).
- Brake pressure: 0.68 to 0.56 on the 0-1 pedal scale (-0.12): the RL lap presses the brake more lightly while braking.
- Brake attack (how fast it is applied): 6.26 to 8.85 pedal per s (+2.60): the RL lap stamps the brake on faster.
- Throttle attack: 1.06 to 0.72 pedal per s (-0.34): the RL lap gets on the throttle more progressively.
- Distance to the track edge: 2.20 to 2.51 m (+0.31): the RL lap stays further from the track edge on average.
- Brake release: 0.62 to 1.00 pedal per s (+0.37): the RL lap lets the brake off faster.
- Throttle lift: 1.77 to 1.60 pedal per s (-0.17): the RL lap lifts off the throttle more gradually.
- Practically unchanged: Peak steering in corners (66.11 to 67.15 deg); Apex distance (1.68 to 1.61 m); Exit width (4.04 to 4.01 m).

## What the data cannot say
- All laps are simulator laps of the same car. The driver's run is the pure-pursuit replay of the driver's line (a model, not a human lap).
- The style numbers are whole-lap values; the corner facts are measured on these two laps only (one lap each).
- The data shows what differs, not why it makes the car faster.

## Task
Write the coaching message for the attacker driver.
