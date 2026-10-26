#!/usr/bin/env python3
"""RAFT-Stereo reconstruction of a stereo pair, for comparison with SGBM and DIC.

Uses the same rectification and reprojection as reconstruct.py on the Pi, with
the pretrained RAFT-Stereo network as the matcher, and reports the same
numbers: depth, coverage, plane-fit RMS and object depths. Writes
<name>_nn_disparity.jpg and <name>_nn_cloud.ply to the current directory.

Run from inside a clone of princeton-vl/RAFT-Stereo (it imports the model from
core/), with the pair at <dir>/<name>_left.jpg and <dir>/<name>_right.jpg:
  python nn_reconstruct.py speckle_flat --plane --zmin 360 --zmax 440 --dir <dir> --calib <npz>
  python nn_reconstruct.py speckle_tilt --plane --zmin 300 --zmax 500 --dir <dir> --calib <npz>
  python nn_reconstruct.py objects3     --objects --zmin 350 --zmax 550 --bright 85 --dir <dir> --calib <npz>

Other options: --scale (default 0.5, same as the other methods), --iters
(default 32), --device cpu|mps, --ckpt (default
models/raftstereo-middlebury.pth). --strictz gates depth to exactly
[zmin, zmax] instead of the wider 0.6*zmin to 1.5*zmax band, which keeps the
background wall out of the dense RAFT cloud (e.g. on objects3).
"""
import argparse
import os
import sys
import numpy as np
import cv2
import torch

# RAFT-Stereo model code from the cloned repo (run from inside that folder)
sys.path.append("core")
from raft_stereo import RAFTStereo            # noqa: E402
from utils.utils import InputPadder           # noqa: E402


# Pipeline helpers, kept the same as in reconstruct.py
def _scale_K(K, s):
    K = K.copy().astype(np.float64)
    K[0, 0] *= s; K[1, 1] *= s; K[0, 2] *= s; K[1, 2] *= s
    return K


def _write_ply(path, xyz, rgb):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (b, g, r) in zip(xyz, rgb):
            f.write(f"{x:.2f} {y:.2f} {z:.2f} {int(r)} {int(g)} {int(b)}\n")


def _plane_rms(pts, thr=2.0):
    """RANSAC fit of the dominant plane, returns (rms, n_inliers, n_finite).

    RAFT gives a disparity for every pixel, including the background and
    regions with no real match, so one global fit would mix several surfaces.
    Fitting only the dominant plane (the board) makes it comparable to SGBM
    and DIC, which only return matched board pixels. thr is the inlier band
    in mm.
    """
    P = pts.astype(np.float64)
    P = P[np.isfinite(P).all(axis=1)]          # drop inf/nan rows
    ntot = len(P)
    if ntot < 200:
        return float("nan"), 0, ntot

    rng = np.random.RandomState(0)
    best_inl = None
    best_cnt = 0
    for _ in range(200):
        idx = rng.choice(ntot, 3, replace=False)
        a, b, c = P[idx]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        dist = np.abs((P - a) @ n)
        cnt = int((dist < thr).sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = dist < thr
    if best_inl is None:
        return float("nan"), 0, ntot

    # refit to the inliers twice, then take the RMS about the final plane
    inl = best_inl
    for _ in range(2):
        Pin = P[inl]
        c0 = Pin.mean(0)
        _, vecs = np.linalg.eigh((Pin - c0).T @ (Pin - c0))
        nrm = vecs[:, 0]
        inl = np.abs((P - c0) @ nrm) < thr
    Pin = P[inl]
    c0 = Pin.mean(0)
    _, vecs = np.linalg.eigh((Pin - c0).T @ (Pin - c0))
    d = (Pin - c0) @ vecs[:, 0]
    rms = float(np.sqrt((d ** 2).mean()))
    return rms, int(inl.sum()), ntot


def analyze_objects(rectL, pts3d, valid, name, bright, min_area, zmin, zmax):
    gray = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    Z = pts3d[:, :, 2]
    mask = (valid & (gray > bright) & np.isfinite(Z)
            & (Z > zmin * 0.6) & (Z < zmax * 1.5))
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    vis = rectL.copy(); rng = np.random.RandomState(0); rows = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = labels == i
        zv = Z[comp & np.isfinite(Z)]
        if len(zv) < 30:
            continue
        med, sd = float(np.median(zv)), float(np.std(zv))
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        wc, hc = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        rows.append((float(cents[i, 0]), x, wc, int(len(zv)), med, sd))
        col = tuple(int(c) for c in rng.randint(60, 255, 3))
        vis[comp] = col
        cv2.rectangle(vis, (x, y), (x + wc, y + hc), (255, 255, 255), 1)
        cv2.putText(vis, f"{med:.0f}mm", (x, max(y - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    rows.sort(key=lambda r: r[0])
    cv2.imwrite(f"{name}_nn_segmented.jpg", vis)
    print("  OBJECT CLUSTERS (left to right):")
    print(f"    {'#':>2}{'x-range':>14}{'npts':>8}{'depth(mm)':>12}{'std(mm)':>9}")
    for k, (cx, x, wc, npts, med, sd) in enumerate(rows, 1):
        print(f"    {k:>2}{f'{x}..{x+wc}':>14}{npts:>8}{med:>12.1f}{sd:>9.2f}")


# RAFT-Stereo matcher
def load_raft(ckpt, device):
    margs = argparse.Namespace(
        hidden_dims=[128, 128, 128], corr_implementation="reg",
        shared_backbone=False, corr_levels=4, corr_radius=4, n_downsample=2,
        context_norm="batch", slow_fast_gru=False, n_gru_layers=3,
        mixed_precision=False)
    model = torch.nn.DataParallel(RAFTStereo(margs))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.module
    model.to(device).eval()
    return model


def to_tensor(bgr, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float()
    return t[None].to(device)


def disparity_raft(model, rectL, rectR, device, iters):
    i1, i2 = to_tensor(rectL, device), to_tensor(rectR, device)
    padder = InputPadder(i1.shape, divis_by=32)
    i1, i2 = padder.pad(i1, i2)
    with torch.no_grad():
        _, flow_up = model(i1, i2, iters=iters, test_mode=True)
    flow_up = padder.unpad(flow_up).cpu().numpy().squeeze()   # [H,W]
    disp = -flow_up                                           # flow is negative; flip to positive disparity
    if np.median(disp) < 0:
        disp = -disp
    return disp.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--dir", default=".")
    ap.add_argument("--calib", default="stereo.npz")
    ap.add_argument("--ckpt", default="models/raftstereo-middlebury.pth")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--zmin", type=float, default=250.0)
    ap.add_argument("--zmax", type=float, default=800.0)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--plane", action="store_true")
    ap.add_argument("--objects", action="store_true")
    ap.add_argument("--bright", type=float, default=45.0)
    ap.add_argument("--min-area", type=int, default=400)
    ap.add_argument("--strictz", action="store_true",
                    help="gate depth to exactly [zmin, zmax] instead of 0.6x/1.5x")
    args = ap.parse_args()

    lf = os.path.join(args.dir, f"{args.name}_left.jpg")
    rf = os.path.join(args.dir, f"{args.name}_right.jpg")
    imgL, imgR = cv2.imread(lf), cv2.imread(rf)
    if imgL is None or imgR is None:
        raise SystemExit(f"Could not read {lf} and/or {rf}")

    z = np.load(args.calib)
    K1, d1, K2, d2 = z["K1"], z["d1"], z["K2"], z["d2"]
    R, T = z["R"], z["T"]
    baseline = float(z["baseline"]) if "baseline" in z else float(np.linalg.norm(T))

    s = args.scale
    if s != 1.0:
        imgL = cv2.resize(imgL, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        imgR = cv2.resize(imgR, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    K1s, K2s = _scale_K(K1, s), _scale_K(K2, s)
    h, w = imgL.shape[:2]; size = (w, h)

    R1, R2, Pl, Pr, Q, _, _ = cv2.stereoRectify(
        K1s, d1, K2s, d2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    mapLx, mapLy = cv2.initUndistortRectifyMap(K1s, d1, R1, Pl, size, cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(K2s, d2, R2, Pr, size, cv2.CV_32FC1)
    rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    device = torch.device(args.device)
    if args.device == "mps":
        # let ops that MPS does not support fall back to the CPU (torch is
        # already imported here, so exporting this in the shell is more reliable)
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import time
    model = load_raft(args.ckpt, device)
    t0 = time.time()
    disp = disparity_raft(model, rectL, rectR, device, args.iters)
    runtime = time.time() - t0

    pts3d = cv2.reprojectImageTo3D(disp, Q)
    Z = pts3d[:, :, 2]
    lo = args.zmin if args.strictz else args.zmin * 0.6
    hi = args.zmax if args.strictz else args.zmax * 1.5
    valid = (disp > 0.5) & np.isfinite(Z) & (Z > lo) & (Z < hi)
    pct = 100.0 * valid.sum() / valid.size

    dvis = np.zeros_like(disp); dvis[valid] = disp[valid]
    dvis = cv2.normalize(dvis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dvis = cv2.applyColorMap(dvis, cv2.COLORMAP_JET); dvis[~valid] = 0
    cv2.imwrite(f"{args.name}_nn_disparity.jpg", dvis)

    xyz = pts3d[valid]; rgb = rectL[valid]
    _write_ply(f"{args.name}_nn_cloud.ply", xyz, rgb)

    print("=" * 60)
    print(f"{args.name}  method=nn (RAFT-Stereo)  (scale {s}, {w}x{h}, {args.device})")
    print(f"  rectified focal {Pl[0,0]:.1f} px, baseline {baseline:.1f} mm")
    print(f"  depth gate: {lo:.0f}..{hi:.0f} mm{'  (strict)' if args.strictz else ''}")
    print(f"  matcher runtime: {runtime:.1f} s")
    print(f"  valid disparity: {pct:.1f}% of pixels")
    if len(xyz):
        print(f"  depth: median {np.median(xyz[:,2]):.1f} mm, "
              f"range {xyz[:,2].min():.1f}..{xyz[:,2].max():.1f} mm, {len(xyz)} pts")
    if args.plane and len(xyz) > 200:
        with np.errstate(all="ignore"):
            rms, ninl, ntot = _plane_rms(xyz)
        board_pct = 100.0 * ninl / valid.size
        print(f"  PLANE FIT (dominant board, RANSAC): RMS {rms:.3f} mm")
        print(f"    board inliers: {ninl} pts = {board_pct:.1f}% of frame "
              f"(of {ntot} finite cloud pts)")
    if args.objects:
        analyze_objects(rectL, pts3d, valid, args.name, args.bright,
                        args.min_area, args.zmin, args.zmax)
    print(f"  saved: {args.name}_nn_disparity.jpg, {args.name}_nn_cloud.ply")


if __name__ == "__main__":
    main()
