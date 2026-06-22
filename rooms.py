"""
Recover ROOMS from a detection: the enclosed regions the walls partition the
plan into. Rooms are what make the model useful to an interior designer -- you
label them, drop furniture in them, compute areas.

Method (robust, geometry-free): paint the detected walls as thick barriers on a
blank canvas, then find connected components of the EMPTY space. The big outside
region (touching the border) is discarded; every remaining component is a room.
For each we keep its pixel area and an axis-aligned interior rect (inset from the
walls) that furniture/labels can anchor to. Doors are bridged so a doorway does
not merge two rooms into one (we paint walls solid, ignoring door gaps, for the
purpose of room separation).
"""
import numpy as np
import cv2

MIN_ROOM_AREA = 7000     # px^2; smaller blobs are gaps/closets-of-noise


def detect_rooms(det):
    W, H = det["size"]
    t = max(6, det.get("wall_thickness", 7))
    barrier = np.zeros((H, W), np.uint8)

    # paint every wall as a solid thick line (doors included -> close openings,
    # so adjacent rooms stay separated by their shared wall line).
    for w in det["walls"]:
        cv2.line(barrier, (int(w["x1"]), int(w["y1"])),
                 (int(w["x2"]), int(w["y2"])), 255, t + 2)
    # also re-close door gaps: draw the door span as barrier too
    for o in det["doors"]:
        cv2.line(barrier, (int(o["x1"]), int(o["y1"])),
                 (int(o["x2"]), int(o["y2"])), 255, t + 2)

    free = (barrier == 0).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(free, connectivity=4)

    rooms = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        # discard the outside region (touches image border)
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue
        if area < MIN_ROOM_AREA:
            continue
        cx, cy = cents[i]
        rooms.append({
            "cx": round(float(cx), 1), "cy": round(float(cy), 1),
            "x0": x, "y0": y, "x1": x + w, "y1": y + h,
            "area_px": area,
        })
    rooms.sort(key=lambda r: -r["area_px"])
    return rooms


if __name__ == "__main__":
    import sys
    from detect import detect
    path = sys.argv[1] if len(sys.argv) > 1 else "demo.png"
    det = detect(path)
    rs = detect_rooms(det)
    print(f"{path}: {len(rs)} rooms")
    for i, r in enumerate(rs):
        print(f"  room {i+1}: area {r['area_px']:>7} px  "
              f"center ({r['cx']:.0f},{r['cy']:.0f})  "
              f"bbox {r['x1']-r['x0']}x{r['y1']-r['y0']}")
