"""
Lift a 2D detection into 3D: walls become extruded boxes, doors become walk-
through openings (floor-to-lintel void + a lintel above), windows become a void
between a sill and a head. Output is a flat list of axis-aligned boxes the
three.js viewer renders directly -- no mesh library needed.

Coordinate convention (matches the viewer): image x -> world x, image y -> world
z (depth), world y is UP. Units stay in pixels so nothing needs calibrating;
the viewer scales the whole scene to fit. Heights are chosen to look like a
room (wall ~ 2.7 m if a wall is ~110 px tall).

An opening is attached to a wall if it shares the wall's orientation, its
centerline is within tol, and its span lies inside the wall span. Each wall is
then split along its length into solid runs (full-height boxes) and openings
(door/window boxes), so the geometry has real holes you can move through.
"""
import json
import sys

WALL_H = 110          # full wall height (px units)
DOOR_H = 80           # door opening height; lintel fills WALL_H-DOOR_H above
WIN_SILL = 35         # window sill height (solid below)
WIN_HEAD = 85         # window head height (solid above up to WALL_H)
CENTER_TOL = 14


def _orient(w):
    return "h" if abs(w["y1"] - w["y2"]) < abs(w["x1"] - w["x2"]) else "v"


def _span(w):
    o = _orient(w)
    if o == "h":
        return w["y1"], min(w["x1"], w["x2"]), max(w["x1"], w["x2"])
    return w["x1"], min(w["y1"], w["y2"]), max(w["y1"], w["y2"])


def _box(cx, cy, cz, sx, sy, sz, kind):
    return {"kind": kind,
            "p": [round(cx, 1), round(cy, 1), round(cz, 1)],
            "s": [round(sx, 1), round(sy, 1), round(sz, 1)]}


def build(det):
    W, H = det["size"]
    t = det.get("wall_thickness", 7)
    boxes = []

    # floor slab
    boxes.append(_box(W / 2, -2, H / 2, W, 4, H, "floor"))

    # index openings by orientation for attachment
    openings = ([dict(o, type="door") for o in det["doors"]] +
                [dict(o, type="window") for o in det["windows"]])

    def opening_on(wall):
        wo = _orient(wall)
        wc, wa0, wa1 = _span(wall)
        res = []
        for o in openings:
            oo = _orient(o)
            if oo != wo:
                continue
            oc, oa0, oa1 = _span(o)
            if abs(oc - wc) <= CENTER_TOL and oa0 >= wa0 - 6 and oa1 <= wa1 + 6:
                res.append((oa0, oa1, o["type"]))
        return sorted(res)

    for wall in det["walls"]:
        o = _orient(wall)
        c, a0, a1 = _span(wall)
        spans = opening_on(wall)

        def emit_solid(lo, hi):
            if hi - lo < 1:
                return
            mid = (lo + hi) / 2
            length = hi - lo
            if o == "h":
                boxes.append(_box(mid, WALL_H / 2, c, length, WALL_H, t, "wall"))
            else:
                boxes.append(_box(c, WALL_H / 2, mid, t, WALL_H, length, "wall"))

        def emit_opening(lo, hi, kind):
            mid = (lo + hi) / 2
            length = hi - lo
            if kind == "door":
                # lintel above the door
                lh = WALL_H - DOOR_H
                if o == "h":
                    boxes.append(_box(mid, DOOR_H + lh / 2, c, length, lh, t, "lintel"))
                else:
                    boxes.append(_box(c, DOOR_H + lh / 2, mid, t, lh, length, "lintel"))
            else:  # window: sill below + head above + glass pane
                if o == "h":
                    boxes.append(_box(mid, WIN_SILL / 2, c, length, WIN_SILL, t, "sill"))
                    boxes.append(_box(mid, (WIN_HEAD + WALL_H) / 2, c, length,
                                      WALL_H - WIN_HEAD, t, "lintel"))
                    boxes.append(_box(mid, (WIN_SILL + WIN_HEAD) / 2, c, length,
                                      WIN_HEAD - WIN_SILL, t * 0.3, "glass"))
                else:
                    boxes.append(_box(c, WIN_SILL / 2, mid, t, WIN_SILL, length, "sill"))
                    boxes.append(_box(c, (WIN_HEAD + WALL_H) / 2, mid, t,
                                      WALL_H - WIN_HEAD, length, "lintel"))
                    boxes.append(_box(c, (WIN_SILL + WIN_HEAD) / 2, mid, t * 0.3,
                                      WIN_HEAD - WIN_SILL, length, "glass"))

        cur = a0
        for (g0, g1, kind) in spans:
            emit_solid(cur, g0)
            emit_opening(max(g0, a0), min(g1, a1), kind)
            cur = g1
        emit_solid(cur, a1)

    return {"size": [W, H], "wall_height": WALL_H, "boxes": boxes,
            "counts": {"walls": len(det["walls"]),
                       "doors": len(det["doors"]),
                       "windows": len(det["windows"])}}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "demo.png"
    from detect import detect
    det = detect(path)
    model = build(det)
    out = path.rsplit(".", 1)[0] + ".model.json"
    with open(out, "w") as f:
        json.dump(model, f)
    kinds = {}
    for b in model["boxes"]:
        kinds[b["kind"]] = kinds.get(b["kind"], 0) + 1
    print(f"{path}: {len(model['boxes'])} boxes  {kinds}")
    print("wrote", out)


if __name__ == "__main__":
    main()
