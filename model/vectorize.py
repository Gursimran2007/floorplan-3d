"""
Mask -> segments: turn the pretrained model's per-class masks into the SAME
line-segment dict that detect() returns, so geometry.py / the viewer don't change.

The CubiCasa5K model gives per-pixel class predictions. We pass three binary
masks in -- wall (rooms channel 2), door (icons channel 2), window (icons
channel 1) -- all at the ORIGINAL image resolution, and get back:

    { "size":[W,H], "wall_thickness":t,
      "walls":[{x1,y1,x2,y2}...], "doors":[...], "windows":[...] }

Walls come back as filled regions, so we skeletonise them into axis-aligned
centerlines (same trick detect.py uses: directional morphological opening +
connected components). Doors/windows are small icon blobs, so each blob becomes
one segment across its longer axis -- it sits on the wall it cuts through.

This module is model-agnostic and runs on CPU: it is tested locally against
synthetic ground-truth masks (see __main__) with no GPU and no weights.
"""
import sys
import json

import cv2
import numpy as np

# ignore wall slivers / icon specks smaller than this many px on their long axis
MIN_LEN = 18


def _thickness(wall):
    """Median run-length of wall pixels down columns -- robust wall thickness."""
    runs = []
    for x in range(0, wall.shape[1], 7):
        c = 0
        for v in wall[:, x] > 0:
            if v:
                c += 1
            elif c:
                if c < 60:
                    runs.append(c)
                c = 0
    return int(np.median(runs)) if runs else 7


def _components(mask, axis):
    """Connected components of a directional mask -> centerline segments."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _area = (int(v) for v in stats[i])
        if axis == "h" and w >= MIN_LEN:
            out.append({"x1": x, "y1": y + h // 2, "x2": x + w, "y2": y + h // 2})
        elif axis == "v" and h >= MIN_LEN:
            out.append({"x1": x + w // 2, "y1": y, "x2": x + w // 2, "y2": y + h})
    return out


def _wall_segments(wall, t):
    """Filled wall regions -> axis-aligned centerline segments. A long thin
    horizontal kernel keeps only horizontal runs, a vertical one keeps verticals,
    so crossing walls separate into individual segments at their junctions."""
    L = max(3, t * 3)
    hmask = cv2.morphologyEx(
        wall, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    vmask = cv2.morphologyEx(
        wall, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    return _components(hmask, "h") + _components(vmask, "v")


def _opening_segments(mask):
    """Each door/window icon blob -> a segment across its longer axis (the span
    of the opening), with the perpendicular axis at the blob's center line."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _area = (int(v) for v in stats[i])
        if max(w, h) < MIN_LEN:
            continue
        if w >= h:
            cy = y + h // 2
            out.append({"x1": x, "y1": cy, "x2": x + w, "y2": cy})
        else:
            cx = x + w // 2
            out.append({"x1": cx, "y1": y, "x2": cx, "y2": y + h})
    return out


def vectorize(wall, door, window):
    """wall/door/window: binary HxW masks at the ORIGINAL image resolution."""
    wall = (np.asarray(wall) > 0).astype(np.uint8) * 255
    h, w = wall.shape
    t = max(5, _thickness(wall))
    return {
        "size": [w, h],
        "wall_thickness": t,
        "walls": _wall_segments(wall, t),
        "doors": _opening_segments(np.asarray(door) > 0),
        "windows": _opening_segments(np.asarray(window) > 0),
    }


def _load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f"could not read mask: {path}")
    return m


if __name__ == "__main__":
    if len(sys.argv) == 5:
        # vectorize.py wall.png door.png window.png out.json
        wall, door, window, out = sys.argv[1:5]
        res = vectorize(_load_mask(wall), _load_mask(door), _load_mask(window))
        with open(out, "w") as f:
            json.dump(res, f)
        print(f"{len(res['walls'])} walls, {len(res['doors'])} doors, "
              f"{len(res['windows'])} windows -> {out}")
    else:
        # self-test: rasterise synthetic ground truth into masks, vectorize,
        # and check we recover roughly the right counts -- no GPU, no weights.
        sys.path.insert(0, __file__.rsplit("/", 2)[0])
        from synth import generate

        _img, gt = generate(seed=7)
        W, H = gt["size"]
        t = gt["wall_thickness"]
        wall = np.zeros((H, W), np.uint8)
        for s in gt["walls"]:
            cv2.line(wall, (s["x1"], s["y1"]), (s["x2"], s["y2"]), 255, t)
        door = np.zeros((H, W), np.uint8)
        for s in gt["doors"]:
            cv2.line(door, (s["x1"], s["y1"]), (s["x2"], s["y2"]), 255, t)
        window = np.zeros((H, W), np.uint8)
        for s in gt["windows"]:
            cv2.line(window, (s["x1"], s["y1"]), (s["x2"], s["y2"]), 255, t)

        res = vectorize(wall, door, window)
        print("ground truth :", len(gt["walls"]), "walls",
              len(gt["doors"]), "doors", len(gt["windows"]), "windows")
        print("vectorized   :", len(res["walls"]), "walls",
              len(res["doors"]), "doors", len(res["windows"]), "windows")
