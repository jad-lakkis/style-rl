"""Figures and a replay page that compare the driver's run with the faster RL laps, as the first step towards coaching.

Input: the folder written by tools/lap_compare.py for each driver (lap_driver/base/keep/fast.csv, laps_chosen.csv, style_numbers.csv, baselines.json), as
    DATA/attacker/..., DATA/smooth/...
(lap_base.csv = the zero-correction base lap = pure pursuit on the track's own line = baseline_neutral.csv of the run).

    python tools/lap_report.py DATA OUT

Writes per driver four figures and, for both drivers together, one self-contained replay page:
    <driver>_1_lap_time.png      lap times of the four laps, time gained per sector, time gap along the lap
    <driver>_2_style.png         the 12 style numbers of the three laps against the target (spread units) and their values
    <driver>_3_lines_per_corner.png  the line on the map in each corner zone (X/Y), with the time of the zone
    <driver>_4_driving.png       speed, pedals and steering of the three laps along the lap
    replay.html                  the laps as moving dots on the map (both drivers, switch at the top)
    summary.csv                  one row per driver and lap: lap time, style distance D, gain against the driver's run and against the zero-correction base
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from lap_compare import COLS, sectors, shade, style_axes  # noqa: E402

COL = {"driver": "#555555", "keep": "#0072B2", "fast": "#D55E00", "base": "#009E73"}    # Okabe-Ito: grey, blue, vermillion, green
NAME = {"driver": "driver's run", "keep": "RL lap, style kept", "fast": "RL fastest lap", "base": "no RL (track line)"}
THREE = ("driver", "keep", "fast")


def load(d):
    d = Path(d)
    laps = {}
    for k in ("driver", "base", "keep", "fast"):
        df = pd.read_csv(d / f"lap_{k}.csv")[COLS].copy()
        df["progress"] = np.maximum.accumulate(df.progress.to_numpy(float))
        laps[k] = df
    ch = pd.read_csv(d / "laps_chosen.csv")
    base = json.load(open(d / "baselines.json"))
    info = {k: {"lap_time": float(ch.lap_time[i]), "D": float(ch.D[i]), "source": str(ch.source[i])} for i, k in enumerate(THREE)}
    info["base"] = {"lap_time": float(base["neutral"]["lap_time"]), "D": float(base["neutral"]["D"]), "source": "baseline_neutral.csv"}
    sn = pd.read_csv(d / "style_numbers.csv")
    return laps, info, sn, base


def corner_zones(segs):
    return [s for s in segs if s[1] == "corners"]


def fig_time(name, laps, info, g, at, segs, out):
    L = g[-1]
    rows = []
    for (sec, kind, lo, hi) in segs:
        hi = min(hi, L)
        tt = {k: float(np.interp(hi, laps[k].progress, laps[k].t) - np.interp(lo, laps[k].progress, laps[k].t)) for k in laps}
        rows.append({"sector": sec, "from_m": round(lo), "to_m": round(hi), **{f"time_{k}": v for k, v in tt.items()}, **{f"{k}_vs_driver": tt[k] - tt["driver"] for k in ("base", "keep", "fast")}})
    sec = pd.DataFrame(rows)
    fig = plt.figure(figsize=(12.5, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1.5], hspace=0.42, wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    order = ["driver", "base", "keep", "fast"]
    y = np.arange(len(order))[::-1]
    t0 = info["driver"]["lap_time"]
    for yi, k in zip(y, order):
        t = info[k]["lap_time"]
        ax.plot([t, t0], [yi, yi], color="#cccccc", lw=1.2, zorder=1)
        ax.scatter([t], [yi], color=COL[k], s=90, zorder=3)
        ax.text(t0 + 0.7, yi, f"{t:.2f} s   ({t0 - t:+.2f} s)" if k != "driver" else f"{t:.2f} s", va="center", ha="left", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([NAME[k] for k in order], fontsize=9)
    ax.set_ylim(-0.6, len(order) - 0.4)
    lo_t = min(info[k]["lap_time"] for k in order)
    ax.set_xlim(lo_t - 0.8, t0 + 4.6)
    ax.set_xlabel("lap time, s  (in brackets: time gained against the driver's run)")
    ax.set_title("1. Lap time of the four laps", loc="left", fontsize=11)
    style_axes(ax)
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(sec))
    for j, k in enumerate(("keep", "fast")):
        v = sec[f"{k}_vs_driver"].to_numpy()
        ax.bar(x + (j - 0.5) * 0.38, v, width=0.36, color=COL[k], label=NAME[k])
        for xi, vi in zip(x + (j - 0.5) * 0.38, v):
            ax.text(xi, vi + (0.02 if vi >= 0 else -0.02), f"{vi:+.2f}", ha="center", va="bottom" if vi >= 0 else "top", fontsize=7)
    ax.axhline(0, color=COL["driver"], lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([r.sector.replace(" ", "\n") for r in sec.itertuples()], fontsize=8)
    ax.set_ylabel("sector time minus the driver's run, s\n(below 0 = faster)", fontsize=9)
    ax.set_title("2. Where the time is gained: per sector (corner zones C, straights S)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axes(ax)
    ax = fig.add_subplot(gs[1, :])
    for k in ("keep", "fast"):
        ax.plot(g, at[k]["t"] - at["driver"]["t"], color=COL[k], lw=1.9, label=f"{NAME[k]}  {info[k]['lap_time']:.2f} s")
    ax.axhline(0, color=COL["driver"], lw=1.4, label=f"{NAME['driver']}  {t0:.2f} s")
    shade(ax, segs)
    ax.set_xlim(0, L)
    ax.set_xlabel("distance along the lap, m   (shaded = corner zones)")
    ax.set_ylabel("time behind (+) / ahead (-) of the driver's run, s")
    ax.set_title("3. The gap to the driver's run along the lap", loc="left", fontsize=11, pad=16)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    style_axes(ax)
    fig.suptitle(f"{name}: lap times", x=0.01, ha="left", fontsize=13)
    fig.savefig(out / f"{name}_1_lap_time.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    sec.to_csv(out / f"{name}_sectors.csv", index=False)
    return sec


def fig_style(name, sn, info, out):
    keys = sn.number.tolist()
    tgt, spread = sn.target.to_numpy(float), sn.spread.to_numpy(float)
    vals = {"driver": sn.driver_run.to_numpy(float), "keep": sn.style_kept.to_numpy(float), "fast": sn.fastest.to_numpy(float)}
    sd = {k: (v - tgt) / spread for k, v in vals.items()}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
    y = np.arange(len(keys))[::-1]
    ax.axvline(0, color="#000000", lw=1.2)
    ax.axvspan(-0.3, 0.3, color="#000000", alpha=0.06, lw=0)
    for k, off in (("driver", 0.24), ("keep", 0.0), ("fast", -0.24)):
        ax.scatter(sd[k], y + off, color=COL[k], s=40, zorder=3, label=f"{NAME[k]}   D = {info[k]['D']:.2f}")
    for yi, a, b in zip(y, sd["driver"], sd["fast"]):
        ax.plot([a, b], [yi + 0.24, yi - 0.24], color="#bbbbbb", lw=0.8, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=9)
    ax.set_xlabel("distance from the target style, in spread units   (0 = on the target, grey band = within 0.3)")
    ax.set_title("Where each lap is against the driver's target style", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    style_axes(ax)
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax2.set_ylim(-0.6, len(keys) - 0.4)
    xs = {"target": 0.28, "driver": 0.50, "keep": 0.72, "fast": 0.94}
    for lab, k in (("target", "target"), ("driver's\nrun", "driver"), ("style\nkept", "keep"), ("fastest", "fast")):
        ax2.text(xs[k], len(keys) - 0.05, lab, ha="right", va="bottom", fontsize=9, color=COL.get(k, "#000000"), fontweight="bold")
    for yi, i in zip(y, range(len(keys))):
        for k, arr in (("target", tgt), ("driver", vals["driver"]), ("keep", vals["keep"]), ("fast", vals["fast"])):
            ax2.text(xs[k], yi, f"{arr[i]:.2f}", ha="right", va="center", fontsize=9, color="#222222")
    ax2.set_title("The numbers themselves (units as in the metric)", loc="left", fontsize=11, pad=34)
    ax.set_title("Where each lap is against the driver's target style", loc="left", fontsize=11, pad=34)
    fig.suptitle(f"{name}: the 12 style numbers", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"{name}_2_style.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_corners(name, laps, info, segs, sec, geo, corners, out):
    zs = corner_zones(segs)
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.6))
    for ax, (nm, kind, lo, hi) in zip(axs.ravel(), zs):
        gm = (geo.s_geom >= lo) & (geo.s_geom <= hi)
        xl, yl, xr, yr = (geo[c][gm].to_numpy() for c in ("x_left", "y_left", "x_right", "y_right"))
        ax.fill(np.concatenate([xl, xr[::-1]]), np.concatenate([yl, yr[::-1]]), color="#efefef", lw=0, zorder=0)       # the track surface
        ax.plot(xl, yl, color="#b5b5b5", lw=0.9, zorder=1)
        ax.plot(xr, yr, color="#b5b5b5", lw=0.9, zorder=1)
        for k in ("keep", "fast", "driver"):
            d = laps[k]
            m = (d.progress >= lo) & (d.progress <= hi)
            ax.plot(d.x[m], d.y[m], color=COL[k], lw=2.4 if k == "driver" else 2.0, ls="--" if k == "driver" else "-", label=NAME[k], zorder=4 if k == "driver" else 3)
        for r in corners.itertuples():
            if lo <= r.s_apex <= hi:
                j = int(np.argmin(np.abs(geo.s_geom - r.s_apex)))
                ax.plot(geo.x_center[j], geo.y_center[j], "o", ms=4, color="#000000", zorder=4)
                ax.annotate(f"T{int(r.corner)}", (geo.x_center[j], geo.y_center[j]), textcoords="offset points", xytext=(5, 5), fontsize=8)
        t = sec[sec.sector == nm].iloc[0]
        ax.set_title(f"{nm}   {lo:.0f}-{hi:.0f} m\nzone time: driver {t.time_driver:.2f} s, kept {t.time_keep:.2f} s ({t.keep_vs_driver:+.2f}), fastest {t.time_fast:.2f} s ({t.fast_vs_driver:+.2f})", loc="left", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X, m", fontsize=8)
        ax.set_ylabel("Y, m", fontsize=8)
        style_axes(ax)
    axs[0, 0].legend(frameon=False, fontsize=8, loc="best")
    for ax in axs.ravel()[len(zs):]:
        ax.axis("off")
    fig.suptitle(f"{name}: the line in each corner zone (map view; light grey = the track, black dots = corner apexes, dashed dark grey = the driver's run)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"{name}_3_lines_per_corner.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_driving(name, at, g, segs, out):
    fig, axs = plt.subplots(4, 1, figsize=(12.5, 10), sharex=True)
    for ax, c, lab in zip(axs, ("speed", "throttle", "brake", "steer"), ("speed, m/s", "throttle (0-1)", "brake (0-1)", "steering command (-1..1)")):
        for k in THREE:
            ax.plot(g, at[k][c], color=COL[k], lw=1.1 if k == "driver" else 1.4, label=NAME[k] if c == "speed" else None)
        ax.set_ylabel(lab, fontsize=9)
        shade(ax, segs, label=(c == "speed"))
        style_axes(ax)
    axs[0].legend(frameon=False, fontsize=9, ncol=3, loc="lower left")
    axs[-1].set_xlabel("distance along the lap, m   (shaded = corner zones)")
    fig.suptitle(f"{name}: what the car does along the lap", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"{name}_4_driving.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


REPLAY = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Lap replay: driver's run against the RL laps</title>
<style>
body{font:14px/1.4 system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#fff;color:#1a1a1a}
header{padding:10px 16px;border-bottom:1px solid #ddd;display:flex;flex-wrap:wrap;gap:14px;align-items:center}
h1{font-size:16px;margin:0 10px 0 0}
main{display:flex;flex-wrap:wrap;gap:14px;padding:12px 16px}
canvas{border:1px solid #ddd;background:#fafafa}
.panel{min-width:330px}
table{border-collapse:collapse;font-size:13px;width:100%}
td,th{padding:3px 8px;text-align:right;border-bottom:1px solid #eee}
th:first-child,td:first-child{text-align:left}
.sw{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:middle;border:1px solid #fff;box-shadow:0 0 0 1px #888}
button,select{font:inherit;padding:3px 9px}
input[type=range]{width:380px}
.note{color:#555;font-size:12px;max-width:420px}
</style></head><body>
<header><h1>Lap replay</h1>
<label>Driver <select id="drv"></select></label>
<button id="play">Play</button>
<label>Speed <select id="rate"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select></label>
<span id="jumps"></span>
<label><input type="checkbox" id="trail" checked> trails</label>
</header>
<main>
<div><canvas id="map" width="760" height="620"></canvas><br><input type="range" id="scrub" min="0" max="1000" value="0"> <span id="clock"></span>
<br><canvas id="trace" width="760" height="170"></canvas></div>
<div class="panel"><table id="tab"></table>
<p class="note">All laps are the same car on the same track in the simulator, started on the line and run side by side in time. The gap is read at the place on the track where that lap is now: negative = that lap gets there this many seconds earlier than the driver's run did, positive = later. Turn a lap off with its tick box.</p></div>
</main>
<script>
const DATA = __DATA__;
const COL = {driver:"#555555", keep:"#0072B2", fast:"#D55E00", base:"#009E73"};
const NAME = {driver:"driver's run", keep:"RL lap, style kept", fast:"RL fastest lap", base:"no RL (track line)"};
const ORDER = ["driver","keep","fast","base"];
const drv = document.getElementById("drv"), cv = document.getElementById("map"), ctx = cv.getContext("2d"), tr = document.getElementById("trace"), tctx = tr.getContext("2d");
const scrub = document.getElementById("scrub"), clock = document.getElementById("clock"), tab = document.getElementById("tab");
let D, on = {driver:true, keep:true, fast:true, base:false}, t = 0, playing = false, last = null, tmax = 0, T = null;
Object.keys(DATA).forEach(k => { const o = document.createElement("option"); o.value = k; o.textContent = k; drv.appendChild(o); });
function bounds(){ let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9; for (const [x,y] of [[D.track.xl,D.track.yl],[D.track.xr,D.track.yr]]) for (let i=0;i<x.length;i++){ x0=Math.min(x0,x[i]); x1=Math.max(x1,x[i]); y0=Math.min(y0,y[i]); y1=Math.max(y1,y[i]); } return [x0,x1,y0,y1]; }
let view;
function setView(x0,x1,y0,y1){ const pad=18, w=cv.width-2*pad, h=cv.height-2*pad, s=Math.min(w/(x1-x0), h/(y1-y0)); view={s, ox: pad+(w-(x1-x0)*s)/2 - x0*s, oy: pad+(h-(y1-y0)*s)/2 + y1*s}; }
const px = x => view.ox + x*view.s, py = y => view.oy - y*view.s;
function load(){ D = DATA[drv.value]; tmax = Math.max(...ORDER.map(k => D.laps[k].lap_time)); t = 0; const b = bounds(); setView(...b);
  document.getElementById("jumps").innerHTML = D.zones.map((z,i)=>`<button data-t="${z.t_driver}">${z.name}</button>`).join(" ");
  document.querySelectorAll("#jumps button").forEach(b=>b.onclick=()=>{t=Math.max(0,parseFloat(b.dataset.t)-4); draw();});
  tab.innerHTML = "<tr><th>lap</th><th>lap time</th><th>D</th><th>speed</th><th>gap</th></tr>" + ORDER.map(k=>`<tr id="r_${k}"><td><input type="checkbox" data-k="${k}" ${on[k]?"checked":""}><span class="sw" style="background:${COL[k]}"></span>${NAME[k]}</td><td>${D.laps[k].lap_time.toFixed(2)} s</td><td>${D.laps[k].D.toFixed(2)}</td><td id="v_${k}"></td><td id="g_${k}"></td></tr>`).join("");
  tab.querySelectorAll("input").forEach(c=>c.onchange=()=>{on[c.dataset.k]=c.checked; draw();});
  draw(); }
function at(lap, tt){ const dt = D.dt, n = lap.x.length, f = Math.min(tt/dt, n-1), i = Math.floor(f), j = Math.min(i+1, n-1), a = f - i; const L = (arr)=>arr[i]*(1-a)+arr[j]*a; return {x:L(lap.x), y:L(lap.y), v:L(lap.v), p:L(lap.p), i, fin: tt >= lap.lap_time}; }
function timeAtP(lap, p){ let lo=0, hi=lap.p.length-1; while(lo<hi){ const m=(lo+hi)>>1; if(lap.p[m]<p) lo=m+1; else hi=m; } return lo*D.dt; }
function poly(xs, ys, c, w){ ctx.beginPath(); for(let i=0;i<xs.length;i++){ const X=px(xs[i]), Y=py(ys[i]); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);} ctx.strokeStyle=c; ctx.lineWidth=w; ctx.stroke(); }
function draw(){ ctx.clearRect(0,0,cv.width,cv.height); poly(D.track.xl,D.track.yl,"#9a9a9a",1.4); poly(D.track.xr,D.track.yr,"#9a9a9a",1.4);
  const tk = D.track; ctx.fillStyle="#666"; ctx.font="11px sans-serif"; D.zones.forEach(z=>{ const X = px(z.x)+6; ctx.fillText(z.name, X > cv.width-70 ? X-76 : X, py(z.y)-6); });
  const st = {};
  for (const k of ORDER){ if(!on[k]) continue; const lap = D.laps[k], a = at(lap, t); st[k]=a;
    if(document.getElementById("trail").checked){ const i0 = Math.max(0, a.i-60); ctx.beginPath(); for(let i=i0;i<=a.i;i++){ const X=px(lap.x[i]), Y=py(lap.y[i]); i==i0?ctx.moveTo(X,Y):ctx.lineTo(X,Y);} ctx.strokeStyle=COL[k]; ctx.globalAlpha=.55; ctx.lineWidth=2.2; ctx.stroke(); ctx.globalAlpha=1; }
    ctx.beginPath(); ctx.arc(px(a.x),py(a.y),k=="driver"?7:6,0,6.2832); ctx.fillStyle=COL[k]; ctx.fill(); ctx.lineWidth=2; ctx.strokeStyle="#fff"; ctx.stroke();
    document.getElementById("v_"+k).textContent = (a.v*3.6).toFixed(0)+" km/h";
    const g = (timeAtP(lap, a.p) - timeAtP(D.laps.driver, a.p));
    document.getElementById("g_"+k).textContent = k=="driver" ? "" : (g>=0?"+":"") + g.toFixed(2)+" s"; }
  clock.textContent = "t = " + t.toFixed(1) + " s";
  scrub.value = Math.round(1000*t/tmax);
  // speed trace against distance
  tctx.clearRect(0,0,tr.width,tr.height); const Lp = D.L, W=tr.width, H=tr.height, m={l:44,r:10,t:8,b:22}; const X=p=>m.l+(W-m.l-m.r)*p/Lp, vmax=D.vmax, Y=v=>H-m.b-(H-m.t-m.b)*v/vmax;
  tctx.strokeStyle="#ddd"; tctx.fillStyle="#555"; tctx.font="10px sans-serif"; for(let v=0; v<=vmax; v+=20){ tctx.beginPath(); tctx.moveTo(m.l,Y(v)); tctx.lineTo(W-m.r,Y(v)); tctx.stroke(); tctx.fillText(String(v), 14, Y(v)+3); }
  D.zones.forEach(z=>{ tctx.fillStyle="rgba(0,0,0,.06)"; tctx.fillRect(X(z.a), m.t, X(z.b)-X(z.a), H-m.t-m.b); tctx.fillStyle="#555"; tctx.fillText(z.name.split(" ")[0], X(z.a)+2, m.t+10); });
  tctx.fillStyle="#555"; tctx.fillText("speed, m/s  vs distance, m", m.l, H-6);
  for (const k of ORDER){ if(!on[k]) continue; const lap=D.laps[k]; tctx.beginPath(); for(let i=0;i<lap.p.length;i+=2){ const x=X(lap.p[i]), y=Y(lap.v[i]); i?tctx.lineTo(x,y):tctx.moveTo(x,y);} tctx.strokeStyle=COL[k]; tctx.lineWidth=1.4; tctx.stroke(); const a=st[k]; if(a){ tctx.beginPath(); tctx.arc(X(a.p),Y(a.v),4,0,6.2832); tctx.fillStyle=COL[k]; tctx.fill(); } } }
function frame(ts){ if(playing){ if(last!==null) t += (ts-last)/1000*parseFloat(document.getElementById("rate").value); last = ts; if(t>=tmax){ t=tmax; playing=false; document.getElementById("play").textContent="Play"; } draw(); requestAnimationFrame(frame); } }
document.getElementById("play").onclick=()=>{ playing=!playing; document.getElementById("play").textContent=playing?"Pause":"Play"; if(playing){ if(t>=tmax) t=0; last=null; requestAnimationFrame(frame);} };
scrub.oninput=()=>{ t=scrub.value/1000*tmax; draw(); };
drv.onchange=load; document.getElementById("trail").onchange=draw; load();
const hm = location.hash.match(/t=([0-9.]+)/); if(hm){ t=parseFloat(hm[1]); draw(); }
</script></body></html>
"""


def replay(all_laps, all_info, geo, corners, out):
    data = {}
    dt = 0.1
    for name, laps in all_laps.items():
        info = all_info[name]
        L = float(min(df.progress.iloc[-1] for df in laps.values()))
        segs = sectors(L)
        sub = slice(None, None, 3)
        track = {"xl": np.round(geo.x_left.to_numpy()[sub], 1).tolist(), "yl": np.round(geo.y_left.to_numpy()[sub], 1).tolist(),
                 "xr": np.round(geo.x_right.to_numpy()[sub], 1).tolist(), "yr": np.round(geo.y_right.to_numpy()[sub], 1).tolist()}
        out_laps = {}
        for k, df in laps.items():
            tt = np.arange(0.0, float(df.t.iloc[-1]) + dt, dt)
            f = lambda c: np.interp(tt, df.t, df[c])      # noqa: E731
            out_laps[k] = {"x": np.round(f("x"), 1).tolist(), "y": np.round(f("y"), 1).tolist(), "v": np.round(f("speed"), 1).tolist(), "p": np.round(f("progress"), 1).tolist(),
                           "lap_time": info[k]["lap_time"], "D": info[k]["D"]}
        zones = []
        for nm, kind, lo, hi in corner_zones(segs):
            j = int(np.argmin(np.abs(geo.s_geom - (lo + hi) / 2)))
            zones.append({"name": nm, "a": lo, "b": hi, "x": float(geo.x_center[j]), "y": float(geo.y_center[j]), "t_driver": float(np.interp(lo, laps["driver"].progress, laps["driver"].t))})
        data[name] = {"dt": dt, "L": L, "vmax": 100, "track": track, "laps": out_laps, "zones": zones}
    (out / "replay.html").write_text(REPLAY.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    geo = pd.read_csv(ROOT / "data" / "tracks" / "monza_geometry.csv")
    corners = pd.read_csv(ROOT / "data" / "monza_corners.csv")
    all_laps, all_info, summary = {}, {}, []
    for name in ("attacker", "smooth"):
        if not (a.data / name).is_dir():
            continue
        laps, info, sn, base = load(a.data / name)
        L = min(float(df.progress.iloc[-1]) for df in laps.values())
        g = np.arange(0.0, L, 2.0)
        at = {k: {c: np.interp(g, df.progress, df[c]) for c in ("t", "x", "y", "speed", "lateral", "steer", "throttle", "brake")} for k, df in laps.items()}
        segs = sectors(L)
        sec = fig_time(name, laps, info, g, at, segs, a.out)
        fig_style(name, sn, info, a.out)
        fig_corners(name, laps, info, segs, sec, geo, corners, a.out)
        fig_driving(name, at, g, segs, a.out)
        all_laps[name], all_info[name] = laps, info
        for k in ("driver", "base", "keep", "fast"):
            summary.append({"driver": name, "lap": NAME[k], "lap_time_s": info[k]["lap_time"], "style_distance_D": info[k]["D"], "gain_vs_driver_run_s": info["driver"]["lap_time"] - info[k]["lap_time"],
                            "gain_vs_no_RL_line_s": info["base"]["lap_time"] - info[k]["lap_time"], "source": info[k]["source"]})
        sn.to_csv(a.out / f"{name}_style_numbers.csv", index=False)
    pd.DataFrame(summary).round(3).to_csv(a.out / "summary.csv", index=False)
    replay(all_laps, all_info, geo, corners, a.out)
    print(pd.DataFrame(summary).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
