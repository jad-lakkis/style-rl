"""The input for the LLM coach, built from two laps of tools/lap_compare.py: the driver's run and the RL lap that keeps the style.

    python tools/coach_brief.py DATA OUT        (run this before tools/lap_report.py: the replay page shows the brief)

Writes into OUT/coaching:
    coach_facts.csv        the small facts table (both drivers): one row per fact, the value for each lap and the change from the driver's run
    <driver>_brief.md      the same facts as plain sentences: the laps, where the time is gained, corner by corner, the style, what the data cannot say
    <driver>_prompt.md     an example prompt for the LLM: instructions + the brief
    brief_data.json        the brief in a form the replay page shows (the corner sentences jump to their corner)

Nothing here calls an LLM. The sentences are made by fixed rules from measured numbers, so every number in the brief is a measurement; the LLM only has to put it into coaching words and order it.
Corner facts (measured on each lap): where braking starts (metres before the apex), the peak brake, the lowest speed, where the throttle comes back (metres after the apex), the speed 80 m after
the apex, and the line at the apex (metres towards the inside of the corner).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from lap_compare import sectors  # noqa: E402
from lap_report import PLAIN, UNIT, load, sector_table  # noqa: E402

TWO = ("driver", "keep")
BRAKE_ON = 0.05
# a difference counts as worth saying when it is at least this big
MIN = {"brake_start_m": 8.0, "brake_peak": 0.15, "min_speed_ms": 1.5, "throttle_back_m": 8.0, "exit_speed_ms": 1.5, "line_to_inside_m": 0.4}
MEANING = {   # what a higher / lower value of each style number means, in words
    "B_mean_active": ("presses the brake harder while braking", "presses the brake more lightly while braking"),
    "B_event_peak_mean": ("hits a higher peak brake pressure in each braking zone", "hits a lower peak brake pressure in each braking zone"),
    "B_apply_rate": ("stamps the brake on faster", "applies the brake more progressively"),
    "B_release_rate": ("lets the brake off faster", "lets the brake off more gradually"),
    "T_apply_rate": ("gets on the throttle more abruptly", "gets on the throttle more progressively"),
    "T_release_rate": ("lifts off the throttle faster", "lifts off the throttle more gradually"),
    "S_reversals_5deg_per_km": ("corrects the steering more often", "corrects the steering less often"),
    "S_mean_abs_rate": ("turns the wheel faster (busier hands)", "turns the wheel slower (calmer hands)"),
    "S_corner_peak_angle": ("uses a larger steering angle in the corners", "uses a smaller steering angle in the corners"),
    "L_border_mean_m": ("stays further from the track edge on average", "stays closer to the track edge on average"),
    "L_inside_dist_apex_m": ("stays further from the inside edge at the apex", "gets closer to the inside edge at the apex"),
    "L_outside_dist_exit_m": ("stays further from the outside edge on the exit", "uses more of the track width on the exit"),
}


def brake_runs(df):
    """Braking runs of a lap: gaps shorter than 10 m are bridged; runs under 12 m or with a peak under 0.15 are dropped."""
    p, b = df.progress.to_numpy(float), df.brake.to_numpy(float)
    on, n, out, i = b > BRAKE_ON, len(p), [], 0
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = last = i
        while j < n and (on[j] or p[j] - p[last] <= 10.0):
            if on[j]:
                last = j
            j += 1
        if p[last] - p[i] >= 12.0 and b[i:last + 1].max() >= 0.15:
            out.append({"start": float(p[i]), "end": float(p[last]), "peak": float(b[i:last + 1].max())})
        i = j
    return out


def corner_facts(df, corners):
    """For each corner: the braking run that leads into it (the run whose next apex is this corner) and what the lap does in and after the corner."""
    p, v, thr, lat = (df[c].to_numpy(float) for c in ("progress", "speed", "throttle", "lateral"))
    runs = brake_runs(df)
    apexes = corners.s_apex.to_numpy(float)
    res = {}
    for c in corners.itertuples():
        a = float(c.s_apex)
        mine = [r for r in runs if (apexes[apexes >= r["start"] + 5.0].min() if (apexes >= r["start"] + 5.0).any() else -1) == a]
        r = max(mine, key=lambda x: x["peak"]) if mine else None
        lo = r["start"] if r else a - 40.0
        m = (p >= lo) & (p <= a + 40.0)
        i_min = int(np.argmin(np.where(m, v, 1e9)))
        after = np.where((p >= p[i_min]) & (thr > 0.5))[0]
        res[int(c.corner)] = {"brake_start_m": a - r["start"] if r else np.nan, "brake_peak": r["peak"] if r else np.nan, "brake_length_m": r["end"] - r["start"] if r else np.nan,
                              "min_speed_ms": float(v[i_min]), "throttle_back_m": float(p[after[0]] - a) if len(after) else np.nan,
                              "exit_speed_ms": float(np.interp(a + 80.0, p, v)), "line_to_inside_m": float(np.interp(a, p, lat)) * int(c.direction)}
    return res


def line_sentence(laps, segs):
    """How far apart the lines of the driver's run and the RL lap are in the corner zones."""
    diffs = []
    for nm, kind, lo, hi in segs:
        if kind == "corners":
            g = np.arange(lo, min(hi, float(laps["driver"].progress.iloc[-1])), 2.0)
            diffs.append(np.abs(np.interp(g, laps["keep"].progress, laps["keep"].lateral) - np.interp(g, laps["driver"].progress, laps["driver"].lateral)))
    d = np.concatenate(diffs)
    return f"In the corner zones the two lines are {d.mean():.1f} m apart on average (at most {d.max():.1f} m): the time comes mostly from braking and speed, not from a different line."


def zone_of(segs, apex):
    for nm, kind, lo, hi in segs:
        if kind == "corners" and lo <= apex <= hi:
            return nm
    return ""


def uni(key):
    """the unit as it reads in a sentence"""
    u = UNIT[key]
    return "on the 0-1 pedal scale" if u == "pedal, 0-1" else u


def fnum(x, d=1):
    return f"{x:.{d}f}"


def corner_sentence(cn, f0, f1, shared=False):
    """One sentence about corner cn: the RL lap (f1) against the driver's run (f0); None when nothing is worth saying."""
    parts = []
    s0, s1, k0, k1 = f0["brake_start_m"], f1["brake_start_m"], f0["brake_peak"], f1["brake_peak"]
    if np.isnan(s0) != np.isnan(s1):
        if np.isnan(s1):
            parts.append(f"does not brake here (the driver's run brakes for {fnum(f0['brake_length_m'], 0)} m with a peak of {fnum(k0, 2)})")
        else:
            parts.append(f"brakes here (starts {fnum(s1, 0)} m before the apex, peak {fnum(k1, 2)}) where the driver's run does not")
    elif not np.isnan(s0):
        if abs(s1 - s0) >= MIN["brake_start_m"]:
            parts.append(f"starts braking {fnum(abs(s1 - s0), 0)} m {'earlier' if s1 > s0 else 'later'} ({fnum(s1, 0)} m before the apex, the driver's run {fnum(s0, 0)} m)")
        if abs(k1 - k0) >= MIN["brake_peak"]:
            parts.append(f"presses the brake {'harder' if k1 > k0 else 'less hard'} (peak {fnum(k1, 2)} against {fnum(k0, 2)})")
    d = f1["min_speed_ms"] - f0["min_speed_ms"]
    if abs(d) >= MIN["min_speed_ms"] and not shared:
        parts.append(f"{'carries more' if d > 0 else 'carries less'} speed through the corner (lowest speed {fnum(f1['min_speed_ms'])} against {fnum(f0['min_speed_ms'])} m/s)")
    t0, t1 = f0["throttle_back_m"], f1["throttle_back_m"]
    if not (np.isnan(t0) or np.isnan(t1)) and abs(t1 - t0) >= MIN["throttle_back_m"] and not shared:
        parts.append(f"is back on the throttle {fnum(abs(t1 - t0), 0)} m {'earlier' if t1 < t0 else 'later'}")
    d = f1["exit_speed_ms"] - f0["exit_speed_ms"]
    if abs(d) >= MIN["exit_speed_ms"]:
        parts.append(f"is {'faster' if d > 0 else 'slower'} 80 m after the apex ({fnum(f1['exit_speed_ms'])} against {fnum(f0['exit_speed_ms'])} m/s)")
    d = f1["line_to_inside_m"] - f0["line_to_inside_m"]
    if abs(d) >= MIN["line_to_inside_m"]:
        parts.append(f"takes the apex {fnum(abs(d), 1)} m {'closer to the inside' if d > 0 else 'wider'}")
    return f"T{cn}: the RL lap " + ", ".join(parts) + "." if parts else None


def build(name, laps, info, sn, corners, segs):
    """Returns (facts table, brief as markdown, brief as sections for the replay page) for one driver."""
    sec = sector_table(laps, segs)
    cf = {k: corner_facts(laps[k], corners) for k in TWO}
    rows = []

    def add(level, item, metric, unit, v0, v1):
        rows.append({"driver": name, "level": level, "item": item, "metric": metric, "unit": unit, "driver_run": v0, "rl_style_kept": v1,
                     "change": v1 - v0 if not (np.isnan(v0) or np.isnan(v1)) else np.nan})

    add("lap", "lap", "lap time", "s", info["driver"]["lap_time"], info["keep"]["lap_time"])
    for r in sec.itertuples():
        add("sector", r.sector, "sector time", "s", r.time_driver, r.time_keep)
    names = {"brake_start_m": ("braking starts", "m before the apex"), "brake_peak": ("peak brake", "pedal 0-1"), "min_speed_ms": ("lowest speed", "m/s"),
             "throttle_back_m": ("throttle back on", "m after the apex"), "exit_speed_ms": ("speed 80 m after the apex", "m/s"), "line_to_inside_m": ("line at the apex", "m towards the inside")}
    for c in corners.itertuples():
        for m, (lab, unit) in names.items():
            add("corner", f"T{int(c.corner)}", lab, unit, cf["driver"][int(c.corner)][m], cf["keep"][int(c.corner)][m])
    for i, key in enumerate(sn.number):
        add("style", PLAIN[key], key, UNIT[key], sn.driver_run[i], sn.style_kept[i])
    facts = pd.DataFrame(rows)

    sections = []

    def item(text, bullet=True, corner=None, zone=None, apex=None):
        return {"text": text, "bullet": bullet, "corner": corner, "zone": zone, "apex": apex}

    t0, t1 = info["driver"]["lap_time"], info["keep"]["lap_time"]
    sections.append(("The laps", [item(f"The driver's run takes {t0:.2f} s. The RL lap that keeps the driver's style within its budget takes {t1:.2f} s, {t0 - t1:.2f} s faster. "
                                       f"Everything below compares the driver's run with this RL lap.", bullet=False)]))
    g = sec.sort_values("keep_vs_driver")
    gained = [r for r in g.itertuples() if r.keep_vs_driver <= -0.15]
    st = sec[sec.kind == "straight"]
    sections.append(("Where the time is gained", [
        item("Biggest gains: " + "; ".join(f"{r.sector} ({r.from_m}-{r.to_m} m) {abs(r.keep_vs_driver):.2f} s faster ({r.time_driver:.2f} s to {r.time_keep:.2f} s)" for r in gained) + ".", bullet=False),
        item(f"The straights together change by {st.keep_vs_driver.sum():+.2f} s; the corner zones by {sec[sec.kind == 'corners'].keep_vs_driver.sum():+.2f} s. {line_sentence(laps, segs)}", bullet=False)]))
    items, quiet, prev = [], [], None
    for c in corners.itertuples():
        n = int(c.corner)
        low = (round(cf["driver"][n]["min_speed_ms"], 1), round(cf["keep"][n]["min_speed_ms"], 1))
        shared = prev == low           # the same lowest point as the corner before: a chicane
        prev = low
        s = corner_sentence(n, cf["driver"][n], cf["keep"][n], shared)
        z = zone_of(segs, c.s_apex)
        zr = sec[sec.sector == z]
        tail = f" (zone {z}: {zr.keep_vs_driver.iloc[0]:+.2f} s)" if len(zr) else ""
        if shared and not s:
            items.append(item(f"T{n}: same low point as T{n - 1}, no other clear difference.", corner=n, zone=z, apex=float(c.s_apex)))
        elif s:
            items.append(item(s + tail, corner=n, zone=z, apex=float(c.s_apex)))
        else:
            quiet.append(f"T{n}")
    if quiet:
        items.append(item(f"No clear difference in: {', '.join(quiet)}."))
    sections.append(("Corner by corner (click a line to see the corner)", items))
    spread = sn.spread.to_numpy(float)
    ch = sn.style_kept.to_numpy(float) - sn.driver_run.to_numpy(float)
    order = np.argsort(-np.abs(ch) / spread)
    big = [i for i in order if abs(ch[i]) / spread[i] >= 0.3]
    small = [i for i in order if abs(ch[i]) / spread[i] < 0.3]
    its = []
    for i in big:
        key = sn.number[i]
        up, down = MEANING[key]
        its.append(item(f"{PLAIN[key]}: {sn.driver_run[i]:.2f} to {sn.style_kept[i]:.2f} {uni(key)} ({ch[i]:+.2f}): the RL lap {up if ch[i] > 0 else down}."))
    if small:
        its.append(item("Practically unchanged: " + "; ".join(f"{PLAIN[sn.number[i]]} ({sn.driver_run[i]:.2f} to {sn.style_kept[i]:.2f} {uni(sn.number[i])})" for i in small) + "."))
    sections.append(("The style (the 12 style numbers of the whole lap)", its))
    sections.append(("What the data cannot say", [
        item("All laps are simulator laps of the same car. The driver's run is the pure-pursuit replay of the driver's line (a model, not a human lap)."),
        item("The style numbers are whole-lap values; the corner facts are measured on these two laps only (one lap each)."),
        item("The data shows what differs, not why it makes the car faster.")]))
    md = [f"# Coaching brief: {name}\n"]
    for heading, its in sections:
        md.append(f"## {heading.split(' (click')[0]}")
        for x in its:
            md.append(("- " if x["bullet"] else "") + x["text"])
        md.append("")
    return facts, "\n".join(md), [{"heading": h, "items": its} for h, its in sections]


PROMPT = """# Prompt for the coaching LLM (example for the {name})

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
{brief}
## Task
Write the coaching message for the {name} driver.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    out = a.out / "coaching"
    out.mkdir(parents=True, exist_ok=True)
    corners = pd.read_csv(ROOT / "data" / "monza_corners.csv")
    allf, bdata = [], {}
    for name in ("attacker", "smooth"):
        if not (a.data / name).is_dir():
            continue
        laps, info, sn = load(a.data / name)
        L = min(float(df.progress.iloc[-1]) for df in laps.values())
        facts, brief, sections = build(name, laps, info, sn, corners, sectors(L))
        prompt = PROMPT.format(name=name, brief=brief)
        (out / f"{name}_brief.md").write_text(brief, encoding="utf-8")
        (out / f"{name}_prompt.md").write_text(prompt, encoding="utf-8")
        bdata[name] = {"sections": sections, "markdown": brief, "prompt": prompt}
        allf.append(facts)
    pd.concat(allf).round(3).to_csv(out / "coach_facts.csv", index=False)
    (out / "brief_data.json").write_text(json.dumps(bdata, indent=1), encoding="utf-8")
    print((out / "attacker_brief.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
