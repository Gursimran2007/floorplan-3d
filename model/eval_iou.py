"""
Objective per-class IoU for the learned detector on held-out CubiCasa plans.

We care about the three classes the product actually uses: wall, door, window.
For each labelled plan we compare the model's predicted pixel mask against the
CubiCasa ground-truth mask (parsed from model.svg) and report intersection-over-
union per class plus the mean.

GT class mapping (same indices detector.py reads off the prediction):
    wall   = room-segmentation label == 2
    door   = icon-segmentation label == 2
    window = icon-segmentation label == 1

Two ways to run:

    python eval_iou.py
        -> self-test of the IoU math only. No dataset, no torch, no weights.
           Proves the metric is correct before any real plans are involved.

    python eval_iou.py <data_path> <list.txt>
        -> real eval. <data_path> is a CubiCasa data dir, <list.txt> lists plan
           folders (one per line, e.g. /high_quality_architectural/10), each
           holding F1_scaled.png + model.svg. Needs torch + the CubiCasa repo +
           weights (see detector.py). Prints a per-class IoU table.

Held-out caveat: these plans must be ones the released checkpoint did NOT train
on (use the official test.txt split). The authoritative full-test-set numbers
are the CubiCasa5K paper's; this harness reproduces the wall/door/window slice
of that on whatever held-out plans are provided, so the number is ours and
verifiable rather than merely cited.
"""
import os
import sys

import numpy as np


def iou(pred, gt):
    """Pixel IoU of two boolean masks. Undefined (both empty) -> None so it can
    be skipped in the mean rather than counted as a perfect or zero score."""
    pred = np.asarray(pred, bool)
    gt = np.asarray(gt, bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return None
    return float(inter) / float(union)


def _self_test():
    """Validate the IoU metric with hand-checkable cases -- no dataset needed."""
    a = np.zeros((10, 10), bool)
    a[2:8, 2:8] = True                       # 6x6 = 36 px box
    assert iou(a, a) == 1.0, "identical masks must score 1.0"

    b = np.zeros((10, 10), bool)
    b[2:8, 8:10] = True                      # disjoint from a
    assert iou(a, b) == 0.0, "disjoint masks must score 0.0"

    c = np.zeros((10, 10), bool)
    c[2:8, 2:5] = True                       # left half of a: 18 px, fully inside
    # inter = 18, union = 36 -> 0.5
    assert abs(iou(a, c) - 0.5) < 1e-9, "half-contained mask must score 0.5"

    d = np.zeros((10, 10), bool)
    d[2:8, 5:11] = True                      # 6x5=30px, overlaps a on cols 5..7
    inter = 6 * 3                            # cols 5,6,7
    union = 36 + 30 - inter
    assert abs(iou(a, d) - inter / union) < 1e-9, "partial overlap IoU mismatch"

    assert iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) is None, \
        "two empty masks -> undefined (None)"

    print("IoU self-test passed (identity=1.0, disjoint=0.0, half=0.5, "
          "partial + empty cases all correct).")


def _gt_masks(svg_path, h, w):
    """Parse a CubiCasa model.svg into wall/door/window GT boolean masks."""
    sys.path.insert(0, os.environ.get(
        "CUBICASA_DIR", os.path.expanduser("~/Desktop/cubicasa-run/CubiCasa5k")))
    from floortrans.loaders.house import House
    house = House(svg_path, h, w)
    label = house.get_segmentation_tensor().numpy()   # [2, H, W]: rooms, icons
    rooms, icons = label[0], label[1]
    return (rooms == 2), (icons == 2), (icons == 1)    # wall, door, window


def eval_plan(plan_dir):
    """One plan folder -> {'wall':iou,'door':iou,'window':iou} (values may be
    None when that class is absent in the ground truth)."""
    import cv2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import detector

    img = cv2.imread(os.path.join(plan_dir, "F1_scaled.png"))
    if img is None:
        raise SystemExit(f"no F1_scaled.png in {plan_dir}")
    h, w = img.shape[:2]
    model = detector.load()
    if model is None:
        raise SystemExit("learned model unavailable -- check CUBICASA_DIR/weights")
    pw, pd, pwin = detector._masks(model, img)             # predicted masks (0/255)
    gw, gd, gwin = _gt_masks(os.path.join(plan_dir, "model.svg"), h, w)
    return {
        "wall": iou(pw > 0, gw),
        "door": iou(pd > 0, gd),
        "window": iou(pwin > 0, gwin),
    }


def eval_set(data_path, list_file):
    with open(list_file) as f:
        folders = [ln.strip() for ln in f if ln.strip()]
    acc = {"wall": [], "door": [], "window": []}
    for i, rel in enumerate(folders, 1):
        try:
            res = eval_plan(data_path.rstrip("/") + "/" + rel.strip("/"))
        except Exception as e:
            print(f"  [{i}/{len(folders)}] {rel}: skipped ({e})")
            continue
        for k, v in res.items():
            if v is not None:
                acc[k].append(v)
        print(f"  [{i}/{len(folders)}] {rel}: "
              + ", ".join(f"{k}={v:.3f}" if v is not None else f"{k}=-"
                          for k, v in res.items()))

    print("\n=== mean IoU over held-out plans ===")
    means = []
    for k in ("wall", "door", "window"):
        vals = acc[k]
        if vals:
            m = sum(vals) / len(vals)
            means.append(m)
            print(f"  {k:7s}: {m*100:5.1f}%   (n={len(vals)})")
        else:
            print(f"  {k:7s}:   n/a   (no GT pixels in any plan)")
    if means:
        print(f"  {'mean':7s}: {sum(means)/len(means)*100:5.1f}%")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        eval_set(sys.argv[1], sys.argv[2])
    else:
        _self_test()
