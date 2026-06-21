"""
The learned detector, wired in behind the same contract as detect.detect().

Loads the pretrained CubiCasa5K model once and runs it on an uploaded plan,
returning {size, wall_thickness, walls, doors, windows} via vectorize.py -- so
geometry.py and the viewer don't change.

It is OPTIONAL: if torch, the CubiCasa repo, or the weights aren't present,
load() returns None and the server falls back to the classical detector. That
keeps the stdlib+opencv server runnable without the heavy ML deps.

Point it at the repo + weights with env vars (defaults assume the local clone):
    CUBICASA_DIR      = path to a clone of github.com/CubiCasa/CubiCasa5k
    CUBICASA_WEIGHTS  = path to model_best_val_loss_var.pkl
"""
import os
import sys

import numpy as np
import cv2

CUBICASA_DIR = os.environ.get(
    "CUBICASA_DIR", os.path.expanduser("~/Desktop/cubicasa-run/CubiCasa5k"))
CUBICASA_WEIGHTS = os.environ.get(
    "CUBICASA_WEIGHTS", os.path.join(CUBICASA_DIR, "model_best_val_loss_var.pkl"))

N_CLASSES = 44
SPLIT = [21, 12, 11]          # heatmaps, rooms, icons
WALL_ROOM = 2
WINDOW_ICON = 1
DOOR_ICON = 2
PAD_TO = 64
MAX_SIDE = 1600               # downscale huge uploads so CPU inference stays sane

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for vectorize
from vectorize import vectorize  # noqa: E402

_MODEL = None
_TRIED = False


def load():
    """Return the model (cached) or None if it can't be loaded."""
    global _MODEL, _TRIED
    if _TRIED:
        return _MODEL
    _TRIED = True
    try:
        import torch
        import torch.nn as nn
        if not os.path.exists(CUBICASA_WEIGHTS):
            print(f"[cubicasa] weights not found at {CUBICASA_WEIGHTS}")
            return None
        sys.path.insert(0, CUBICASA_DIR)
        from floortrans.models import get_model
        # get_model loads its backbone weights (model_1427.pth) via a path
        # relative to the repo dir, so build the model with cwd there.
        cwd = os.getcwd()
        try:
            os.chdir(CUBICASA_DIR)
            model = get_model("hg_furukawa_original", 51)
        finally:
            os.chdir(cwd)
        model.conv4_ = nn.Conv2d(256, N_CLASSES, bias=True, kernel_size=1)
        model.upsample = nn.ConvTranspose2d(
            N_CLASSES, N_CLASSES, kernel_size=4, stride=4)
        ckpt = torch.load(CUBICASA_WEIGHTS, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _MODEL = model
        print("[cubicasa] pretrained model loaded")
    except Exception as e:
        print(f"[cubicasa] model unavailable, falling back to classical: {e}")
        _MODEL = None
    return _MODEL


def available():
    return load() is not None


def _masks(model, bgr):
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ph = (PAD_TO - h % PAD_TO) % PAD_TO
    pw = (PAD_TO - w % PAD_TO) % PAD_TO
    rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=255)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
    t = 2.0 * (t / 255.0) - 1.0
    with torch.no_grad():
        pred = model(t.unsqueeze(0))
    _heat, rooms, icons = torch.split(pred, SPLIT, dim=1)
    rs = rooms.argmax(dim=1)[0, :h, :w].cpu().numpy()
    ic = icons.argmax(dim=1)[0, :h, :w].cpu().numpy()
    wall = (rs == WALL_ROOM).astype(np.uint8) * 255
    door = (ic == DOOR_ICON).astype(np.uint8) * 255
    window = (ic == WINDOW_ICON).astype(np.uint8) * 255
    return wall, door, window


def detect_bytes(raw):
    """image bytes (PNG/JPG already decoded upstream) -> detect() contract,
    or None if the model isn't available / the image can't be decoded."""
    model = load()
    if model is None:
        return None
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
    wall, door, window = _masks(model, bgr)
    return vectorize(wall, door, window)
