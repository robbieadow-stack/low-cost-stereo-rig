#!/usr/bin/env python3
"""Rectify a stereo pair and compute depth, with a selectable disparity matcher.

Shared pipeline for the method comparison. Calibration, rectification,
reprojection to metric 3D and the metrics are the same for every run; only
the disparity stage changes, selected with --method:

    sgbm   OpenCV semi-global block matching (classical baseline)
    dic    simplified DIC: subset ZNCC with a sub-pixel peak fit

Because everything except the matcher is identical, differences in the
results come from the matcher and not from the setup.

Inputs   <dir>/<name>_left.jpg + <name>_right.jpg   (--dir, default captures/recon)
         calibration .npz                          (--calib, default calibration/stereo.npz)
Outputs  <name>_<method>_rectified.jpg / _disparity.jpg / _cloud.ply  (into --dir)
         <name>_<method>_segmented.jpg with --objects
         printed stats: valid %, depth range, plane RMS (--plane), objects (--objects)

Plane fit: RANSAC dominant plane with a 2 mm inlier band, the same estimator
used by nn_reconstruct.py (RAFT-Stereo), so plane RMS is comparable across all
three methods. Fitting the dominant plane instead of the whole cloud means a
target in front of a background is still handled correctly.

Run (Pi or laptop, in the venv):
    python reconstruct.py speckle_flat --method dic --plane --zmin 360 --zmax 440 \
        --dir /path/to/captures/recon --calib /path/to/stereo.npz
Options:
    --method sgbm|dic   matcher (default sgbm)
    --dir               folder holding <name>_left.jpg / <name>_right.jpg and outputs
    --calib             path to the calibration .npz to use
    --scale 0.5         process at this fraction of full resolution (default 0.5)
    --zmin/--zmax       expected depth range in mm, sets the disparity search
                        (default 250..800)
    --win 11            DIC subset size in px (odd; bigger is smoother, less detail)
    --minzncc 0.5       DIC: minimum correlation to accept a pixel
    --plane             fit the dominant plane and report its RMS (flat/tilted speckle)
    --objects           mask the background, cluster, report per-object depth
    --bright / --min-area   object masking settings (default 45 / 400 px)

"""
import argparse
import os
import time
import numpy as np
import cv2

RECON_DIR = "captures/recon"
CALIB = "calibration/stereo.npz"


def _load_pair(name, recon_dir):
    """Read <name>_left.jpg and <name>_right.jpg from recon_dir."""
    lf = os.path.join(recon_dir, f"{name}_left.jpg")
    rf = os.path.join(recon_dir, f"{name}_right.jpg")
    imgL, imgR = cv2.imread(lf), cv2.imread(rf)
    if imgL is None or imgR is None:
        raise SystemExit(f"Could not read {lf} and/or {rf}")
    return imgL, imgR


def _scale_K(K, s):
    """Camera matrix for images resized by factor s."""
    K = K.copy().astype(np.float64)
    K[0, 0] *= s; K[1, 1] *= s; K[0, 2] *= s; K[1, 2] *= s
    return K


def _write_ply(path, xyz, rgb):
    """Write an ASCII PLY point cloud. rgb is given in OpenCV BGR order."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (b, g, r) in zip(xyz, rgb):
            f.write(f"{x:.2f} {y:.2f} {z:.2f} {int(r)} {int(g)} {int(b)}\n")


def _plane_rms(pts, thr=2.0):
    """RANSAC dominant-plane fit. Returns (rms_mm, n_inliers, n_points).

    Same estimator as nn_reconstruct.py so plane RMS is comparable across
    SGBM, DIC and RAFT-Stereo. 200 random three-point planes are tried, the one
    with most points within thr mm wins, and it is refined by a least-squares
    (PCA) fit on its inliers. RMS is reported over the final inliers. The
    random generator is seeded so results are repeatable.
    """
    P = pts.astype(np.float64)
    P = P[np.isfinite(P).all(axis=1)]            # drop inf/nan rows
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

    # Refit on the inliers twice so the inlier set settles.
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


def disparity_sgbm(gL, gR, min_disp, num_disp):
    """OpenCV SGBM disparity (px) and a validity mask."""
    block = 5
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=block,
        P1=8 * block * block, P2=32 * block * block, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(gL, gR).astype(np.float32) / 16.0
    valid = disp > (min_disp + 0.5)
    return disp, valid


def disparity_dic(gL, gR, min_disp, num_disp, win=11, min_zncc=0.5):
    """Simplified DIC disparity on rectified images.

    For each integer disparity d, ZNCC between L(x) and R(x - d) is computed
    over a win x win subset using box filters. The best d per pixel is kept
    and refined to sub-pixel with a parabola through the peak and its two
    neighbours. A pixel is valid if its peak ZNCC exceeds min_zncc and the peak
    is not on the edge of the search range.
    """
    L = gL.astype(np.float32)
    R = gR.astype(np.float32)
    H, W = L.shape
    ks = (win, win)

    def box(x):
        return cv2.boxFilter(x, -1, ks, normalize=True, borderType=cv2.BORDER_REFLECT)

    muL = box(L)
    sigL = np.sqrt(np.maximum(box(L * L) - muL * muL, 1e-6))

    best_c = np.full((H, W), -2.0, np.float32)   # best ZNCC
    best_d = np.full((H, W), min_disp, np.int32)
    c_dm = np.full((H, W), -2.0, np.float32)     # ZNCC at best_d - 1
    c_dp = np.full((H, W), -2.0, np.float32)     # ZNCC at best_d + 1
    c_prev = None

    for k in range(num_disp):
        d = min_disp + k
        Rd = np.zeros_like(R)
        if d < W:
            Rd[:, d:] = R[:, :W - d]
        muR = box(Rd)
        sigR = np.sqrt(np.maximum(box(Rd * Rd) - muR * muR, 1e-6))
        with np.errstate(invalid="ignore", divide="ignore"):
            zncc = (box(L * Rd) - muL * muR) / (sigL * sigR)
        zncc = np.nan_to_num(zncc, nan=-2.0, posinf=-2.0, neginf=-2.0)
        if d > 0:
            zncc[:, :d] = -2.0                   # left border has no overlap

        # Fill in the d+1 neighbour for pixels whose peak was set last iteration.
        upd = best_d == (d - 1)
        c_dp[upd] = zncc[upd]

        # New peaks: remember the previous ZNCC as the d-1 neighbour.
        wins = zncc > best_c
        c_dm[wins] = c_prev[wins] if c_prev is not None else -2.0
        best_c[wins] = zncc[wins]
        best_d[wins] = d
        c_prev = zncc

    denom = c_dm - 2.0 * best_c + c_dp
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(np.abs(denom) > 1e-6, 0.5 * (c_dm - c_dp) / denom, 0.0)
    delta = np.clip(np.nan_to_num(ratio), -0.5, 0.5)
    disp = best_d.astype(np.float32) + delta

    valid = (best_c > min_zncc) & (best_d > min_disp) & (best_d < min_disp + num_disp - 1)
    disp[~valid] = min_disp - 1.0
    return disp, valid


def analyze_objects(rectL, pts3d, valid, name, method, bright, min_area, zmin, zmax, recon_dir):
    """Segment bright objects from the background and report each one's depth.

    Pixels must be valid, brighter than `bright` and inside the depth window.
    Connected components smaller than min_area are ignored. Saves a coloured
    segmentation image and prints one row per object, sorted left to right.
    """
    gray = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    Z = pts3d[:, :, 2]
    mask = (valid & (gray > bright) & np.isfinite(Z)
            & (Z > zmin * 0.6) & (Z < zmax * 1.5))
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    vis = rectL.copy()
    rng = np.random.RandomState(0)
    rows = []
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
    out = os.path.join(recon_dir, f"{name}_{method}_segmented.jpg")
    cv2.imwrite(out, vis)
    print("  OBJECT CLUSTERS (left to right):")
    print(f"    {'#':>2}{'x-range':>14}{'npts':>8}{'depth(mm)':>12}{'std(mm)':>9}")
    for k, (cx, x, wc, npts, med, sd) in enumerate(rows, 1):
        print(f"    {k:>2}{f'{x}..{x+wc}':>14}{npts:>8}{med:>12.1f}{sd:>9.2f}")
    print(f"  saved: {out}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--method", choices=["sgbm", "dic"], default="sgbm")
    ap.add_argument("--dir", default=RECON_DIR,
                    help="folder with <name>_left.jpg / <name>_right.jpg and outputs")
    ap.add_argument("--calib", default=CALIB, help="path to the calibration .npz")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--zmin", type=float, default=250.0)
    ap.add_argument("--zmax", type=float, default=800.0)
    ap.add_argument("--win", type=int, default=11)
    ap.add_argument("--minzncc", type=float, default=0.5)
    ap.add_argument("--plane", action="store_true")
    ap.add_argument("--objects", action="store_true")
    ap.add_argument("--bright", type=float, default=45.0)
    ap.add_argument("--min-area", type=int, default=400)
    args = ap.parse_args()

    imgL, imgR = _load_pair(args.name, args.dir)
    z = np.load(args.calib)
    K1, d1, K2, d2 = z["K1"], z["d1"], z["K2"], z["d2"]
    R, T = z["R"], z["T"]
    baseline = float(z["baseline"]) if "baseline" in z else float(np.linalg.norm(T))

    s = args.scale
    if s != 1.0:
        imgL = cv2.resize(imgL, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        imgR = cv2.resize(imgR, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    K1s, K2s = _scale_K(K1, s), _scale_K(K2, s)
    h, w = imgL.shape[:2]
    size = (w, h)

    R1, R2, Pl, Pr, Q, roi1, roi2 = cv2.stereoRectify(
        K1s, d1, K2s, d2, size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    mapLx, mapLy = cv2.initUndistortRectifyMap(K1s, d1, R1, Pl, size, cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(K2s, d2, R2, Pr, size, cv2.CV_32FC1)
    rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    # Side-by-side rectified pair with horizontal lines to check row alignment.
    combo = np.hstack([rectL, rectR])
    for yy in range(0, combo.shape[0], max(20, combo.shape[0] // 30)):
        cv2.line(combo, (0, yy), (combo.shape[1], yy), (0, 200, 0), 1)
    out_rect = os.path.join(args.dir, f"{args.name}_{args.method}_rectified.jpg")
    cv2.imwrite(out_rect, combo)

    # Disparity search range from the expected depths, d = f * B / Z, rounded
    # out to multiples of 16 as SGBM requires.
    f = float(Pl[0, 0])
    dmin = f * baseline / args.zmax
    dmax = f * baseline / args.zmin
    min_disp = int(np.floor(dmin / 16.0) * 16)
    num_disp = int(np.ceil((dmax - dmin) / 16.0) * 16)
    num_disp = max(num_disp, 16)

    gL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    t0 = time.time()
    if args.method == "sgbm":
        disp, valid = disparity_sgbm(gL, gR, min_disp, num_disp)
    else:
        disp, valid = disparity_dic(gL, gR, min_disp, num_disp,
                                    win=args.win, min_zncc=args.minzncc)
    runtime = time.time() - t0

    pct = 100.0 * valid.sum() / valid.size
    dvis = disp.copy()
    dvis[~valid] = min_disp
    dvis = cv2.normalize(dvis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dvis = cv2.applyColorMap(dvis, cv2.COLORMAP_JET)
    dvis[~valid] = 0
    out_disp = os.path.join(args.dir, f"{args.name}_{args.method}_disparity.jpg")
    cv2.imwrite(out_disp, dvis)

    # Reproject to 3D and keep points inside a loose window around the depth range.
    pts3d = cv2.reprojectImageTo3D(disp, Q)
    Z = pts3d[:, :, 2]
    good = valid & np.isfinite(Z) & (Z > args.zmin * 0.6) & (Z < args.zmax * 1.5)
    xyz = pts3d[good]
    rgb = rectL[good]
    out_ply = os.path.join(args.dir, f"{args.name}_{args.method}_cloud.ply")
    _write_ply(out_ply, xyz, rgb)

    print("=" * 60)
    print(f"{args.name}  method={args.method}  (scale {s}, {w}x{h})")
    print(f"  rectified focal {f:.1f} px, baseline {baseline:.1f} mm")
    print(f"  disparity search: min {min_disp}, num {num_disp} "
          f"(~{args.zmin:.0f}..{args.zmax:.0f} mm)")
    print(f"  matcher runtime: {runtime:.1f} s")
    print(f"  valid disparity: {pct:.1f}% of pixels")
    if len(xyz):
        print(f"  depth: median {np.median(xyz[:,2]):.1f} mm, "
              f"range {xyz[:,2].min():.1f}..{xyz[:,2].max():.1f} mm, {len(xyz)} pts")
    if args.plane and len(xyz) > 200:
        rms, ninl, ntot = _plane_rms(xyz)
        print(f"  PLANE FIT (dominant, RANSAC 2mm): RMS {rms:.3f} mm "
              f"over {ninl}/{ntot} inliers")
    if args.objects:
        analyze_objects(rectL, pts3d, valid, args.name, args.method,
                        args.bright, args.min_area, args.zmin, args.zmax, args.dir)
    print(f"  saved: {out_rect}")
    print(f"         {out_disp}")
    print(f"         {out_ply}")


if __name__ == "__main__":
    main()
