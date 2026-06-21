"""
Synthetic clean floor-plan generator -- our controlled test bed.

Why synthetic first (same discipline as the autotuner's self-generated data):
a generated plan comes with EXACT ground truth -- we know every wall, door and
window position -- so we can MEASURE the detector objectively (precision/recall),
not just eyeball a render. It also gives us unlimited clean test images without
hunting for a dataset. Real-photo plans come later; first prove the pipeline on
inputs where the answer is known.

A plan is a rectangle recursively split (BSP) into rooms. Walls are thick black
lines on the room boundaries; doors are GAPS in a wall (with a swing arc, a
standard symbol); windows are a thin double-line segment inside a wall.

Outputs per sample:
  <name>.png   -- the clean floor-plan image (black on white)
  <name>.json  -- ground truth: walls / doors / windows as pixel segments

Run:
    python synth.py 1 demo        # one plan -> demo.png + demo.json
    python synth.py 40 data/plan  # a batch for measuring the detector
"""
import json
import math
import os
import random
import sys

from PIL import Image, ImageDraw

W, H = 1000, 750          # canvas
MARGIN = 60
WALL = 7                  # wall thickness in px
MIN_ROOM = 150           # smallest room side, keeps doors/windows fittable


def _split(x0, y0, x1, y1, depth, rng, rooms):
    """Recursive binary space partition into axis-aligned rooms."""
    w, h = x1 - x0, y1 - y0
    if depth == 0 or (w < 2 * MIN_ROOM and h < 2 * MIN_ROOM):
        rooms.append((x0, y0, x1, y1))
        return
    vertical = w > h if abs(w - h) > 40 else rng.random() < 0.5
    if vertical and w >= 2 * MIN_ROOM:
        cut = rng.randint(x0 + MIN_ROOM, x1 - MIN_ROOM)
        _split(x0, y0, cut, y1, depth - 1, rng, rooms)
        _split(cut, y0, x1, y1, depth - 1, rng, rooms)
    elif h >= 2 * MIN_ROOM:
        cut = rng.randint(y0 + MIN_ROOM, y1 - MIN_ROOM)
        _split(x0, y0, x1, cut, depth - 1, rng, rooms)
        _split(x0, cut, x1, y1, depth - 1, rng, rooms)
    else:
        rooms.append((x0, y0, x1, y1))


def _edges_of_rooms(rooms):
    """Collect wall edges from room rects, then MERGE collinear overlapping/
    touching intervals into unique maximal walls. Without this, a wall shared by
    two rooms is two segments; punching a door in one but drawing the other solid
    would refill the gap. One physical wall = one entity = one place to punch."""
    raw = {}   # (axis, coord) -> list of (lo, hi)
    for (x0, y0, x1, y1) in rooms:
        raw.setdefault(("h", y0), []).append((x0, x1))
        raw.setdefault(("h", y1), []).append((x0, x1))
        raw.setdefault(("v", x0), []).append((y0, y1))
        raw.setdefault(("v", x1), []).append((y0, y1))
    segs = set()
    for (axis, coord), ivs in raw.items():
        ivs.sort()
        cur0, cur1 = ivs[0]
        for lo, hi in ivs[1:]:
            if lo <= cur1:                    # overlap/touch -> extend
                cur1 = max(cur1, hi)
            else:
                segs.add((axis, coord, cur0, cur1))
                cur0, cur1 = lo, hi
        segs.add((axis, coord, cur0, cur1))
    return segs


def generate(seed):
    rng = random.Random(seed)
    rooms = []
    _split(MARGIN, MARGIN, W - MARGIN, H - MARGIN,
           depth=rng.randint(2, 4), rng=rng, rooms=rooms)
    segs = _edges_of_rooms(rooms)

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    walls, doors, windows = [], [], []
    outer = (MARGIN, MARGIN, W - MARGIN, H - MARGIN)

    for s in segs:
        kind, a, b0, b1 = s
        length = b1 - b0
        # decide a door gap (interior walls mostly) or window (outer walls)
        is_outer = (
            (kind == "h" and (a == outer[1] or a == outer[3])) or
            (kind == "v" and (a == outer[0] or a == outer[2]))
        )
        gap = None
        if length > 220:
            if is_outer and rng.random() < 0.5:
                # window: a centered thin double-line span
                wlen = rng.randint(70, 120)
                c = (b0 + b1) // 2
                gap = ("win", c - wlen // 2, c + wlen // 2)
            elif not is_outer and rng.random() < 0.85:
                dlen = rng.randint(70, 100)
                start = rng.randint(b0 + 40, b1 - 40 - dlen)
                gap = ("door", start, start + dlen)

        # draw the wall, leaving a gap for doors/windows
        def draw_seg(p0, p1):
            if kind == "h":
                d.rectangle([p0, a - WALL // 2, p1, a + WALL // 2], fill="black")
            else:
                d.rectangle([a - WALL // 2, p0, a + WALL // 2, p1], fill="black")

        if gap and gap[0] == "door":
            _, g0, g1 = gap
            draw_seg(b0, g0)
            draw_seg(g1, b1)
            # door swing arc + leaf (standard symbol)
            if kind == "h":
                doors.append({"x1": g0, "y1": a, "x2": g1, "y2": a})
                d.arc([g0, a - (g1 - g0), g0 + 2 * (g1 - g0), a + (g1 - g0)],
                      start=270, end=360, fill="black", width=2)
                d.line([g0, a, g0, a - (g1 - g0)], fill="black", width=2)
            else:
                doors.append({"x1": a, "y1": g0, "x2": a, "y2": g1})
                d.arc([a - (g1 - g0), g0, a + (g1 - g0), g0 + 2 * (g1 - g0)],
                      start=180, end=270, fill="black", width=2)
                d.line([a, g0, a - (g1 - g0), g0], fill="black", width=2)
        elif gap and gap[0] == "win":
            _, g0, g1 = gap
            draw_seg(b0, g0)
            draw_seg(g1, b1)
            # window = thin parallel lines across the gap
            if kind == "h":
                windows.append({"x1": g0, "y1": a, "x2": g1, "y2": a})
                d.line([g0, a - 2, g1, a - 2], fill="black", width=1)
                d.line([g0, a + 2, g1, a + 2], fill="black", width=1)
            else:
                windows.append({"x1": a, "y1": g0, "x2": a, "y2": g1})
                d.line([a - 2, g0, a - 2, g1], fill="black", width=1)
                d.line([a + 2, g0, a + 2, g1], fill="black", width=1)
        else:
            draw_seg(b0, b1)

        if kind == "h":
            walls.append({"x1": b0, "y1": a, "x2": b1, "y2": a})
        else:
            walls.append({"x1": a, "y1": b0, "x2": a, "y2": b1})

    gt = {"size": [W, H], "wall_thickness": WALL,
          "walls": walls, "doors": doors, "windows": windows,
          "rooms": [{"x0": r[0], "y0": r[1], "x1": r[2], "y1": r[3]} for r in rooms]}
    return img, gt


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    base = sys.argv[2] if len(sys.argv) > 2 else "plan"
    d = os.path.dirname(base)
    if d:
        os.makedirs(d, exist_ok=True)
    for i in range(n):
        img, gt = generate(seed=1000 + i)
        name = base if n == 1 else f"{base}_{i:03d}"
        img.save(name + ".png")
        with open(name + ".json", "w") as f:
            json.dump(gt, f)
    print(f"wrote {n} plan(s): {base}{'' if n==1 else '_000..'}  "
          f"({len(gt['walls'])} walls, {len(gt['doors'])} doors, "
          f"{len(gt['windows'])} windows in the last)")


if __name__ == "__main__":
    main()
