"""
Run the pretrained CubiCasa5K model on ONE floor-plan image and save three
binary masks (wall / door / window) at the original image resolution. Those
masks feed straight into vectorize.py -> geometry.py -> the 3D viewer.

This is the GPU step. It does NOT run on the Mac (the pretrained net is
PyTorch-1.0 era and wants CUDA); run it on Colab (free T4 is plenty).

------------------------------------------------------------------------------
COLAB SETUP (run these once, in a cell, before this script):

    !git clone https://github.com/CubiCasa/CubiCasa5k.git
    %cd CubiCasa5k
    !pip install -q gdown
    # pretrained weights (~200 MB) from the repo README:
    !gdown 1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK -O model_best_val_loss_var.pkl
    # the repo is pinned to torch 1.0; on a modern Colab torch this usually
    # loads fine, but if model construction errors, that is the first thing to
    # fix -- the load block below mirrors the repo's own eval.py.

Then put this file in the CubiCasa5k/ dir and:

    !python cubicasa_infer.py /path/to/your_plan.png  out_dir/

It writes out_dir/wall.png, out_dir/door.png, out_dir/window.png. Download
those, then locally:

    python model/vectorize.py wall.png door.png window.png plan.detected.json
    # -> render plan.detected.json through geometry.py / the viewer
------------------------------------------------------------------------------

Channel mapping (confirmed from floortrans/plotting.py):
  model output  = 44 channels, split [21 heatmaps, 12 rooms, 11 icons]
  wall   = rooms  argmax == 2
  window = icons  argmax == 1
  door   = icons  argmax == 2
"""
import os
import sys

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from floortrans.models import get_model  # from the CubiCasa5k repo

N_CLASSES = 44
SPLIT = [21, 12, 11]          # heatmaps, rooms, icons
WALL_ROOM = 2
WINDOW_ICON = 1
DOOR_ICON = 2
PAD_TO = 64                    # net needs H,W divisible by its total stride


def load_model(weights="model_best_val_loss_var.pkl"):
    """Mirrors CubiCasa5k/eval.py model construction + checkpoint load."""
    model = get_model("hg_furukawa_original", 51)
    model.conv4_ = torch.nn.Conv2d(256, N_CLASSES, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(
        N_CLASSES, N_CLASSES, kernel_size=4, stride=4)
    ckpt = torch.load(weights, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def _preprocess(path):
    """BGR image -> normalized [-1,1] tensor, padded so H,W % PAD_TO == 0.
    Returns the tensor plus the original (H,W) so we can crop masks back."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ph = (PAD_TO - h % PAD_TO) % PAD_TO
    pw = (PAD_TO - w % PAD_TO) % PAD_TO
    rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=255)
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
    t = 2.0 * (t / 255.0) - 1.0          # repo normalization: [-1, 1]
    return t.unsqueeze(0), (h, w)


def infer_masks(model, path):
    tensor, (h, w) = _preprocess(path)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        pred = model(tensor)             # [1, 44, H, W]
    _heat, rooms, icons = torch.split(pred, SPLIT, dim=1)
    rooms_seg = rooms.argmax(dim=1)[0, :h, :w].cpu().numpy()
    icons_seg = icons.argmax(dim=1)[0, :h, :w].cpu().numpy()
    wall = (rooms_seg == WALL_ROOM).astype(np.uint8) * 255
    window = (icons_seg == WINDOW_ICON).astype(np.uint8) * 255
    door = (icons_seg == DOOR_ICON).astype(np.uint8) * 255
    return wall, door, window


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python cubicasa_infer.py <image> <out_dir>")
    img_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    model = load_model()
    wall, door, window = infer_masks(model, img_path)
    cv2.imwrite(os.path.join(out_dir, "wall.png"), wall)
    cv2.imwrite(os.path.join(out_dir, "door.png"), door)
    cv2.imwrite(os.path.join(out_dir, "window.png"), window)
    print(f"wrote wall/door/window masks to {out_dir}/ "
          f"(wall px={int((wall>0).sum())}, door px={int((door>0).sum())}, "
          f"window px={int((window>0).sum())})")
