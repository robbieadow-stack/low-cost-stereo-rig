#!/usr/bin/env python3
"""Stereo calibration from single images of the LaVision two-level plate.

The LaVision Type 204-15 is a two-level plate (15 mm dot pitch, 3 mm between
levels) designed so that one image at one plate position can calibrate a
stereo pair. plate_model.detect_plate_matchdata turns each view into image
points and 3D object points; the 3 mm relief supplies the depth.

Method
------
1. Anchor left to right. Each camera labels its dots from its own origin, so
   anchor_lr.estimate_lr_label_offset finds the label offset that makes both
   views name the same physical dot, seeded by the central square's image
   centroid. Object points are taken from the left view and shared, so each
   matched dot refers to one 3D point in both cameras.
2. Calibrate every pair on its own. One tilted plate image is enough. A pose
   that is too close to fronto-parallel is ill-conditioned (the shallow relief
   gives too little perspective) and its single-image solve degenerates.
3. Screen. A pair is kept only if both cameras reproject under RMS_GATE and its
   baseline agrees with the other pairs (the baseline is one fixed physical
   length, so a pair that disagrees on it is ill-conditioned even at low RMS).
   This is done per pair, before any joint solve, so one bad pose cannot
   affect the rest.
4. Report the single-image consensus (median baseline and spread), then solve
   all kept pairs jointly and save that result.

Which level is raised is not known in advance, so every solve is run with both
Z conventions and the one with lower stereo RMS is kept.

Reads   captures/stereo_cal/pairNN_left.jpg  +  pairNN_right.jpg
Writes  calibration/stereo.npz   (K1,d1,K2,d2,R,T,E,F,image_size,baseline)

Run from the project folder on the Pi:
    .venv/bin/python scripts/calibrate_plate.py            # all pairs
    .venv/bin/python scripts/calibrate_plate.py pair07     # only the named pair(s)

"""
import glob
import os
import sys
import numpy as np
import cv2

from plate_model import detect_plate_matchdata, LEVEL_SEP_MM
from anchor_lr import estimate_lr_label_offset

CAL_DIR = "captures/stereo_cal"
OUT = "calibration/stereo.npz"
MIN_MATCHED = 20            # skip a pair with fewer matched dots than this
RMS_GATE = 0.8             # px; both cameras must reproject below this
BASE_MAD_K = 3.0           # baseline must be within this many MADs of the median
HFOV_GUESS_DEG = 66.0      # Camera Module 3 standard lens; initial focal guess only

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
# Camera model kept small enough for one shallow view to support: square
# pixels, principal point fixed at the image centre, no tangential distortion,
# no K3.
CAM_FLAGS = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K3)

# Returned by match_pair when a pair cannot be matched.
_EMPTY = (np.empty((0, 3), np.float32), np.empty((0, 2), np.float32),
          np.empty((0, 2), np.float32))


def match_pair(img_left, img_right):
    """Matched (object Nx3, left Nx2, right Nx2, info) for one stereo pair.

    Object points come from the left view and are shared by both cameras, so a
    matched dot refers to the same 3D point. The arrays are empty if the
    central square is missing in either view (no seed for the anchor) or the
    anchor cannot find a confident offset; info then says which.
    """
    oL, pL, uvL, okL, sqL, d0L = detect_plate_matchdata(img_left)
    oR, pR, uvR, okR, sqR, d0R = detect_plate_matchdata(img_right)
    if not (okL and okR):
        return (*_EMPTY, "  [no central square marker for seed]")

    d0 = 0.5 * (float(d0L) + float(d0R))
    res = estimate_lr_label_offset(uvL, pL, uvR, pR, d0,
                                   marker_xyL=sqL, marker_xyR=sqR)
    if not res.ok:
        return (*_EMPTY,
                "  [anchor unresolved: conf %.2f, %s]" % (res.confidence, res.reason))

    obj, L, R = [], [], []
    for iL, jR in res.pairs:
        obj.append(oL[iL]); L.append(pL[iL]); R.append(pR[jR])
    info = "  [dUV=(%d,%d) conf=%.2f matched=%d resid=%.2fpx]" % (
        res.duv[0], res.duv[1], res.confidence, len(res.pairs), res.median_resid_px)
    return (np.array(obj, np.float32), np.array(L, np.float32),
            np.array(R, np.float32), info)


def collect(cal_dir, only=None):
    """Detect and match every pair once. Returns objp, ipL, ipR, size, names.

    only: optional set/list of pair names to restrict to (e.g. {"pair07"}).
    Pairs with fewer than MIN_MATCHED matched dots are skipped.
    """
    lefts = sorted(glob.glob(os.path.join(cal_dir, "*_left.jpg")))
    objp, ipL, ipR, names, size = [], [], [], [], None
    for lf in lefts:
        name = os.path.basename(lf).replace("_left.jpg", "")
        if only and name not in only:
            continue
        rf = lf.replace("_left", "_right")
        if not os.path.exists(rf):
            continue
        imgL, imgR = cv2.imread(lf), cv2.imread(rf)
        if imgL is None or imgR is None:
            continue
        o, l, r, info = match_pair(imgL, imgR)
        if len(o) < MIN_MATCHED:
            print(f"  {name}: {len(o)} matched  skipped{info}")
            continue
        objp.append(o); ipL.append(l); ipR.append(r); names.append(name)
        size = (imgL.shape[1], imgL.shape[0])
    return objp, ipL, ipR, size, names


def _init_K(size):
    """Initial camera matrix from the nominal HFOV, and zero distortion."""
    w, h = size
    f = (w / 2.0) / np.tan(np.radians(HFOV_GUESS_DEG / 2.0))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
    return K, np.zeros((5, 1))


def _flip(objp):
    """Swap which level is raised (Z -> 3 mm - Z)."""
    return [np.column_stack([o[:, 0], o[:, 1], LEVEL_SEP_MM - o[:, 2]]).astype(np.float32)
            for o in objp]


def _solve_one(objp, ipL, ipR, size):
    """Calibrate each camera, then the stereo extrinsics with intrinsics fixed."""
    K0, d0 = _init_K(size)
    r1, K1, d1, _, _ = cv2.calibrateCamera(objp, ipL, size, K0.copy(), d0.copy(), flags=CAM_FLAGS)
    r2, K2, d2, _, _ = cv2.calibrateCamera(objp, ipR, size, K0.copy(), d0.copy(), flags=CAM_FLAGS)
    rms, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        objp, ipL, ipR, K1, d1, K2, d2, size,
        flags=cv2.CALIB_FIX_INTRINSIC, criteria=CRIT)
    return dict(rms=float(rms), rms1=float(r1), rms2=float(r2),
                baseline=float(np.linalg.norm(T)), size=size,
                K1=K1, d1=d1, K2=K2, d2=d2, R=R, T=T, E=E, F=F)


def solve(objp, ipL, ipR, size):
    """Stereo solve with both Z conventions, keeping the lower stereo RMS."""
    a = _solve_one(objp, ipL, ipR, size)
    b = _solve_one(_flip(objp), ipL, ipR, size)
    return a if a["rms"] <= b["rms"] else b


def screen(single, names):
    """Pick which single-image solves to keep. Returns (keep, verdict).

    RMS gate first, then baseline consensus among the survivors. The baseline
    check needs at least three survivors to be meaningful, so with fewer it is
    skipped. verdict maps each pair index to a short reason string.
    """
    rms_ok = [k for k in range(len(single))
              if max(single[k]["rms1"], single[k]["rms2"]) < RMS_GATE]
    keep, med = rms_ok, float("nan")
    if len(rms_ok) >= 3:
        bases = np.array([single[k]["baseline"] for k in rms_ok])
        med = float(np.median(bases))
        mad = float(np.median(np.abs(bases - med))) or 1.0
        keep = [k for k in rms_ok if abs(single[k]["baseline"] - med) <= BASE_MAD_K * mad]
    verdict = {}
    for k in range(len(single)):
        verdict[k] = "KEEP" if k in keep else ("drop: RMS" if k not in rms_ok else "drop: baseline")
    return keep, verdict


def calibrate_and_save(cal_dir=CAL_DIR, out=OUT, only=None):
    """Run the full calibration, save the joint solve to `out`, return details."""
    objp, ipL, ipR, size, names = collect(cal_dir, only=only)
    if len(objp) < 1:
        raise RuntimeError("No usable pairs.")

    single = [solve([objp[k]], [ipL[k]], [ipR[k]], size) for k in range(len(objp))]
    keep, verdict = screen(single, names)
    if not keep:
        raise RuntimeError("No pair passed the screen (capture more tilted poses).")

    joint = solve([objp[k] for k in keep], [ipL[k] for k in keep],
                  [ipR[k] for k in keep], size)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, K1=joint["K1"], d1=joint["d1"], K2=joint["K2"], d2=joint["d2"],
             R=joint["R"], T=joint["T"], E=joint["E"], F=joint["F"],
             image_size=np.array(size), baseline=joint["baseline"])
    return dict(single=single, names=names, keep=keep, verdict=verdict,
                joint=joint, out=out)


def result_summary(r):
    """Flat, JSON-safe summary of calibrate_and_save() for the server and console."""
    single, keep, j, names, verdict = (r["single"], r["keep"], r["joint"],
                                       r["names"], r["verdict"])
    kb = np.array([single[k]["baseline"] for k in keep])
    ks = np.array([single[k]["rms"] for k in keep])
    pairs = [dict(name=names[k],
                  left=float(single[k]["rms1"]),
                  right=float(single[k]["rms2"]),
                  stereo=float(single[k]["rms"]),
                  baseline=float(single[k]["baseline"]),
                  verdict=verdict[k])
             for k in range(len(single))]
    return dict(
        ok=True,
        n_pairs=len(single),
        n_kept=len(keep),
        consensus_baseline_mm=float(np.median(kb)),
        consensus_baseline_std_mm=float(kb.std()),
        single_rms_median_px=float(np.median(ks)),
        single_rms_max_px=float(ks.max()),
        joint_rms_px=float(j["rms"]),
        joint_left_px=float(j["rms1"]),
        joint_right_px=float(j["rms2"]),
        joint_baseline_mm=float(j["baseline"]),
        pairs=pairs,
    )


def main():
    only = set(sys.argv[1:]) or None
    r = calibrate_and_save(only=only)
    single, names, keep, verdict, j = (r["single"], r["names"], r["keep"],
                                       r["verdict"], r["joint"])
    print("=" * 68)
    print(f"{'pair':<8}{'leftRMS':>9}{'rightRMS':>9}{'stereoRMS':>11}{'baseline':>10}  verdict")
    for k in range(len(single)):
        s = single[k]
        print(f"{names[k]:<8}{s['rms1']:>9.3f}{s['rms2']:>9.3f}{s['rms']:>11.3f}"
              f"{s['baseline']:>9.1f}mm  {verdict[k]}")

    kb = np.array([single[k]["baseline"] for k in keep])
    ks = np.array([single[k]["rms"] for k in keep])
    print("-" * 68)
    print(f"SINGLE-IMAGE CONSENSUS ({len(keep)} clean pairs):")
    print(f"  baseline    median {np.median(kb):.1f} mm   std {kb.std():.1f} mm   "
          f"range {kb.min():.1f}-{kb.max():.1f} mm")
    print(f"  stereo RMS  median {np.median(ks):.3f} px   max {ks.max():.3f} px")
    print(f"JOINT SOLVE (accuracy boost, {len(keep)} pairs):")
    print(f"  per-camera  left {j['rms1']:.3f} px   right {j['rms2']:.3f} px")
    print(f"  stereo RMS  {j['rms']:.3f} px")
    print(f"  baseline    {j['baseline']:.1f} mm  (physical target ~140 mm)")
    print(f"  saved       {r['out']}")


if __name__ == "__main__":
    main()