"""Small readers for the text formats of Assetto Corsa car data: .ini files (sections with key=value, comments after ; or //) and .lut curves (one "x|y" pair per line)."""
import re
from pathlib import Path

import numpy as np

_COMMENT = re.compile(r"\s*(;|//).*$")
_SECTION = re.compile(r"^\[(.+)\]$")
_PAIR = re.compile(r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*\|\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$")


def read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="replace").replace("\r", "")


def _lines(text):
    for line in text.split("\n"):
        s = _COMMENT.sub("", line).strip()
        if s:
            yield s


def parse_ini(text):
    """{section: {key: value}} (values as stripped strings). Lines before the first section are ignored."""
    out, cur = {}, None
    for s in _lines(text):
        m = _SECTION.match(s)
        if m:
            cur = out.setdefault(m.group(1).strip(), {})
        elif cur is not None and "=" in s:
            k, v = s.split("=", 1)
            cur[k.strip()] = v.strip()
    return out


def parse_curve(text):
    """(points, header): points = float array (n, 2) of the x|y lines (None when fewer than two), header = {key: value} of the key=value lines (a few curve files start with [VALUES] C=.. E=..).
    Text that is mostly something else than x|y lines gives (None, {})."""
    pts, header, other = [], {}, 0
    for s in _lines(text):
        m = _PAIR.match(s)
        if m:
            pts.append((float(m.group(1)), float(m.group(2))))
        elif "=" in s and not s.startswith("["):
            k, v = s.split("=", 1)
            header[k.strip()] = v.strip()
        else:
            other += 1
    if len(pts) < 2 or other > max(2, 0.1 * len(pts)):
        return None, {}
    return np.asarray(pts, float), header


def num(value, default=None):
    """float of an ini value ("0.0250", "40000", "13"), default when missing or not a number."""
    try:
        return float(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return default


def vec(value):
    """tuple of floats of "1.6,0.82,4.66"."""
    return tuple(float(x) for x in str(value).replace(";", ",").split(",") if x.strip())
