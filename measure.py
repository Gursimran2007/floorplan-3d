"""
Objective scoring of the detector against synthetic ground truth + a visual
overlay so misses are obvious. Counts alone lie (GT lists every room edge; the
detector merges collinear walls), so we score by GEOMETRY:

  wall coverage  -- fraction of each GT wall's length covered by some detected
                    wall of the same orientation and centerline (within tol).
  door / window  -- a GT opening is "found" if a detected opening of the same
                    type has its midpoint within MID_TOL pixels.

Overlay PNG: detected walls = green, GT openings = blue circles (filled if
found, hollow if missed) so a glance shows what's wrong.

    python measure.py demo.png        # scores demo.png vs demo.json
    python measure.py data/plan       # averages over a batch data/plan_*.png
"""
import glob
import json
import sys

import cv2
import numpy as np

from detect import detect

MID_TOL = 34
CENTER_TOL = 12


def _coverage(gt_wall, det_walls):
    horiz = abs(gt_wall["y1"] - gt_wall["y2"]) < abs(gt_wall["x1"] - gt_wall["x2"])
    if horiz:
        c, a0, a1 = gt_wall["y1"], min(gt_wall["x1"], gt_wall["x2"]), max(gt_wall["x1"], gt_wall["x2"])
    else:
        c, a0, a1 = gt_wall["x1"], min(gt_wall["y1"], gt_wall["y2"]), max(gt_wall["y1"], gt_wall["y2"])
    length = max(1, a1 - a0)
    covered = np.zeros(length, bool)
    for w in det_walls:
        wh = abs(w["y1"] - w["y2"]) < abs(w["x1"] - w["x2"])
        if wh != horiz:
            continue
        wc = w["y1"] if wh else w["x1"]
        if abs(wc - c) > CENTER_TOL:
            continue
        if wh:
            d0, d1 = min(w["x1"], w["x2"]), max(w["x1"], w["x2"])
        else:
            d0, d1 = min(w["y1"], w["y2"]), max(w["y1"], w["y2"])
        lo, hi = max(a0, d0), min(a1, d1)
        if hi > lo:
            covered[lo - a0:hi - a0] = True
    return covered.mean()


def _mid(o):
    return ((o["x1"] + o["x2"]) / 2, (o["y1"] + o["y2"]) / 2)


def _match_openings(gt, det):
    found = 0
    used = [False] * len(det)
    for g in gt:
        gm = _mid(g)
        for i, dt in enumerate(det):
            if used[i]:
                continue
            dm = _mid(dt)
            if abs(gm[0] - dm[0]) < MID_TOL and abs(gm[1] - dm[1]) < MID_TOL:
                used[i] = True
                found += 1
                break
    return found


def score(path):
    gt = json.load(open(path.rsplit(".", 1)[0] + ".json"))
    det = detect(path)
    cov = np.mean([_coverage(w, det["walls"]) for w in gt["walls"]])
    d_found = _match_openings(gt["doors"], det["doors"])
    w_found = _match_openings(gt["windows"], det["windows"])
    nd, nw = len(gt["doors"]), len(gt["windows"])
    return {
        "wall_coverage": cov,
        "door_recall": d_found / nd if nd else 1.0,
        "window_recall": w_found / nw if nw else 1.0,
        "doors": (d_found, nd), "windows": (w_found, nw),
        "gt": gt, "det": det,
    }


def overlay(path, res):
    img = cv2.imread(path)
    for w in res["det"]["walls"]:
        cv2.line(img, (w["x1"], w["y1"]), (w["x2"], w["y2"]), (0, 180, 0), 2)
    for kind, col in (("doors", (220, 120, 0)), ("windows", (180, 0, 180))):
        det_mids = [_mid(o) for o in res["det"][kind]]
        for g in res["gt"][kind]:
            gm = _mid(g)
            hit = any(abs(gm[0] - dm[0]) < MID_TOL and abs(gm[1] - dm[1]) < MID_TOL
                      for dm in det_mids)
            cv2.circle(img, (int(gm[0]), int(gm[1])), 12, col, -1 if hit else 2)
    out = path.rsplit(".", 1)[0] + ".overlay.png"
    cv2.imwrite(out, img)
    return out


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "demo.png"
    paths = [arg] if arg.endswith(".png") else sorted(glob.glob(arg + "_*.png"))
    accs = {"wall_coverage": [], "door_recall": [], "window_recall": []}
    for p in paths:
        r = score(p)
        for k in accs:
            accs[k].append(r[k])
        if len(paths) == 1:
            print(f"{p}")
            print(f"  wall coverage : {r['wall_coverage']*100:5.1f}%")
            print(f"  doors found   : {r['doors'][0]}/{r['doors'][1]}")
            print(f"  windows found : {r['windows'][0]}/{r['windows'][1]}")
            print("  overlay ->", overlay(p, r))
    if len(paths) > 1:
        print(f"averaged over {len(paths)} plans:")
        for k, v in accs.items():
            print(f"  {k:14s}: {np.mean(v)*100:5.1f}%")


if __name__ == "__main__":
    main()
