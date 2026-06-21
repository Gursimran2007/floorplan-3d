"""
The AI slice (v1): detect walls / doors / windows from a clean floor-plan image.

This is the one thing we do well. v1 is CLASSICAL computer vision -- fully
deterministic, no training -- because on clean plans the structure is recoverable
with morphology, and it gives us a correct, measurable baseline TODAY. It is
written as a single swappable function `detect(path) -> {walls,doors,windows}`,
so the learned model (a segmentation net trained on real photographed plans) can
drop in behind the same interface later without touching geometry/server/viewer.

Method (why each step):
  1. binarize            -- walls are dark; invert so walls = white (255).
  2. thick-wall mask     -- morphological OPEN with a ~wall-thickness square
                            kernel keeps only thick bodies, deleting thin door
                            arcs and window mullions. So doors AND windows become
                            clean GAPS in the wall skeleton.
  3. split H / V         -- directional OPEN (long thin kernels) separates
                            horizontal from vertical walls; connected components
                            then give individual wall SEGMENTS.
  4. find openings       -- collinear segments separated by a gap => an opening.
                            Classify by location: on the outer boundary = window,
                            interior = door. (A learned detector would instead
                            read the actual symbol; honest v1 heuristic.)
"""
import json
import sys

import cv2
import numpy as np

GAP_MIN = 28          # px; smaller breaks are intersections/noise, not openings
LINE_TOL = 9          # px; centerlines within this are "the same wall line"
MAX_SIDE = 1400       # downscale huge real-world plans to this longest side


class DetectionError(Exception):
    """Raised when an image doesn't look like a recoverable floor plan.
    Carries a human-readable reason for the UI (vs an opaque 500)."""


def load_binary(path):
    """Binarize ANY plan to walls=white(255) on black, robustly:

      - Otsu threshold (auto, not a fixed 127) handles faint/low-contrast lines.
      - polarity auto-detect: walls are the MINORITY ink, so whichever class
        (dark-on-light or light-on-dark) is the smaller area is taken as walls.
        This makes dark-background CAD exports work without a flag.
      - huge images are downscaled so morphology kernel sizes stay meaningful.
    """
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise DetectionError("could not read the image (unsupported or corrupt file).")

    h, w = g.shape
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        g = cv2.resize(g, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # otsu -> foreground(255)=bright. Walls are the minority pixels; pick that side.
    bright = int((otsu == 255).sum())
    dark = int((otsu == 0).sum())
    walls_are_bright = bright < dark
    b = otsu if walls_are_bright else cv2.bitwise_not(otsu)

    ink = b.mean() / 255.0
    if ink < 0.002:
        raise DetectionError("almost no wall lines found — is this a clean floor plan?")
    if ink > 0.45:
        raise DetectionError("the image is too dense to read as a floor plan "
                             "(photo, filled rooms, or heavy shading?).")
    return b


def estimate_thickness(b):
    """Median run-length of wall pixels along columns -- a robust wall-thickness
    estimate so we don't hard-code it (real plans vary)."""
    runs = []
    for x in range(0, b.shape[1], 7):
        col = b[:, x] > 0
        c = 0
        for v in col:
            if v:
                c += 1
            elif c:
                if c < 40:
                    runs.append(c)
                c = 0
    return int(np.median(runs)) if runs else 7


def _segments(mask, axis):
    """Connected components of a directional mask -> wall segments.
    axis='h' returns horizontal segments, 'v' vertical."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segs = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if axis == "h" and w >= GAP_MIN:
            segs.append({"x1": x, "y1": y + h // 2, "x2": x + w, "y2": y + h // 2})
        elif axis == "v" and h >= GAP_MIN:
            segs.append({"x1": x + w // 2, "y1": y, "x2": x + w // 2, "y2": y + h})
    return segs


def _merge_and_find_gaps(segs, axis, dmask):
    """Group collinear segments into merged WALLS, then find openings by scanning
    the DIRECTIONAL wall mask (only this orientation's walls) along each wall's
    centerline: between the first and last wall pixel on that line, any run with
    no wall pixels is an opening. Scanning the directional mask (not the combined
    one) means perpendicular walls crossing the line don't create false flanks,
    and short stubs next to T-junctions still provide a real flank -- so doors
    beside junctions are caught."""
    key = (lambda s: s["y1"]) if axis == "h" else (lambda s: s["x1"])
    lo = (lambda s: s["x1"]) if axis == "h" else (lambda s: s["y1"])
    hi = (lambda s: s["x2"]) if axis == "h" else (lambda s: s["y2"])

    segs = sorted(segs, key=lambda s: (key(s), lo(s)))
    lines = []
    for s in segs:
        if lines and abs(key(s) - lines[-1]["c"]) <= LINE_TOL:
            lines[-1]["parts"].append(s)
        else:
            lines.append({"c": key(s), "parts": [s]})

    walls, openings = [], []
    for ln in lines:
        c = int(np.median([key(p) for p in ln["parts"]]))
        strip = (dmask[c, :] > 0) if axis == "h" else (dmask[:, c] > 0)
        idx = np.flatnonzero(strip)
        if idx.size == 0:
            continue
        a0, a1 = int(idx[0]), int(idx[-1])
        # find interior empty runs within [a0, a1]
        i = a0
        while i <= a1:
            if not strip[i]:
                j = i
                while j <= a1 and not strip[j]:
                    j += 1
                if j - i >= GAP_MIN:           # interior by construction (<= a1)
                    openings.append({"axis": axis, "c": c, "g0": i, "g1": j})
                i = j
            else:
                i += 1
        if axis == "h":
            walls.append({"x1": a0, "y1": c, "x2": a1, "y2": c})
        else:
            walls.append({"x1": c, "y1": a0, "x2": c, "y2": a1})
    return walls, openings


def detect(path):
    b = load_binary(path)
    Hh, Ww = b.shape
    t = max(5, estimate_thickness(b))

    ksq = cv2.getStructuringElement(cv2.MORPH_RECT, (t - 2, t - 2))
    thick = cv2.morphologyEx(b, cv2.MORPH_OPEN, ksq)

    L = t * 3
    Hmask = cv2.morphologyEx(thick, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    Vmask = cv2.morphologyEx(thick, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))

    h_segs = _segments(Hmask, "h")
    v_segs = _segments(Vmask, "v")

    h_walls, h_open = _merge_and_find_gaps(h_segs, "h", Hmask)
    v_walls, v_open = _merge_and_find_gaps(v_segs, "v", Vmask)
    walls = h_walls + v_walls

    if len(walls) < 2:
        raise DetectionError(
            "couldn't find clear straight walls. This v1 works on clean, "
            "axis-aligned floor plans (black/solid walls on a light background). "
            "Hand-drawn or photographed plans aren't supported yet.")

    # bounding box of all walls = outer boundary, to classify door vs window
    xs = [w["x1"] for w in walls] + [w["x2"] for w in walls]
    ys = [w["y1"] for w in walls] + [w["y2"] for w in walls]
    bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
    EDGE = 18

    def is_outer(o):
        if o["axis"] == "h":
            return abs(o["c"] - by0) < EDGE or abs(o["c"] - by1) < EDGE
        return abs(o["c"] - bx0) < EDGE or abs(o["c"] - bx1) < EDGE

    doors, windows = [], []
    for o in h_open + v_open:
        if o["axis"] == "h":
            seg = {"x1": o["g0"], "y1": o["c"], "x2": o["g1"], "y2": o["c"]}
        else:
            seg = {"x1": o["c"], "y1": o["g0"], "x2": o["c"], "y2": o["g1"]}
        (windows if is_outer(o) else doors).append(seg)

    return {"size": [Ww, Hh], "wall_thickness": t,
            "walls": walls, "doors": doors, "windows": windows}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "demo.png"
    out = detect(path)
    print(f"{path}: {len(out['walls'])} walls, {len(out['doors'])} doors, "
          f"{len(out['windows'])} windows  (est. thickness {out['wall_thickness']}px)")
    outpath = path.rsplit(".", 1)[0] + ".detected.json"
    with open(outpath, "w") as f:
        json.dump(out, f)
    print("wrote", outpath)
