"""Hold the limits OCD measures for the replica car against what the replica's own numbers predict (style_rl/accar/ac_theory.py):

    python scripts/ac_check_envelope.py data/ac_sf70h/replica.json data/ac_sf70h/ocd_envelope.json [--out data/ac_sf70h/ENVELOPE_CHECK.md]

The second file is the output of  scripts/measure_ocd_envelope.py --backend ocdlap:ac_sf70h --vmax 100 --out data/ac_sf70h/ocd_envelope.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_rl.accar.ac_theory import compare, theory  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replica")
    ap.add_argument("measured", nargs="?", help="the measured envelope json; without it only the prediction is printed")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    replica = json.loads(Path(a.replica).read_text())
    lines = []
    if a.measured:
        rows = compare(replica, json.loads(Path(a.measured).read_text()))
        lines += ["| speed m/s | km/h | acceleration m/s2: predicted / OCD | braking g: predicted / OCD | cornering g: predicted / OCD |", "|---:|---:|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['v_mps']:.0f} | {3.6 * r['v_mps']:.0f} | {r['acc_theory']:.2f} / {r['acc_ocd']:.2f} | {r['brk_theory_g']:.2f} / {r['brk_ocd_g']:.2f} | {r['lat_theory_g']:.2f} / {r['lat_ocd_g']:.2f} |")
        ratio = lambda k1, k2: float(np.median([r[k2] / r[k1] for r in rows if r[k1] > 0.2]))
        lines += ["", f"median OCD / predicted: acceleration {ratio('acc_theory', 'acc_ocd'):.2f}, braking {ratio('brk_theory_g', 'brk_ocd_g'):.2f}, cornering {ratio('lat_theory_g', 'lat_ocd_g'):.2f}"]
    else:
        sp = np.arange(10.0, 101.0, 10.0)
        th = theory(replica, sp)
        lines += ["| speed m/s | km/h | acceleration m/s2 | braking g | cornering g |", "|---:|---:|---:|---:|---:|"]
        lines += [f"| {v:.0f} | {3.6 * v:.0f} | {th['acc_ms2'][i]:.2f} | {th['brk_g'][i]:.2f} | {th['lat_g'][i]:.2f} |" for i, v in enumerate(sp)]
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
