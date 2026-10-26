"""Left/right label anchoring for the LaVision plate calibration.

Stereo calibration does not need a globally correct origin. Within one pair it
only needs the left and right views to give the same physical dot the same
(U, V) label. Lattice growth already labels each camera's dots consistently,
but with its own arbitrary origin, so the left/right problem reduces to one
unknown integer offset (dU, dV) such that
    uvR + (dU, dV)  labels the same physical dot as  uvL.

The offset is found without relying on the marker snap: the two dot clouds are
matched directly in image space (they are the same dots seen from two nearby
viewpoints), and each confident match votes for uvL - uvR. With hundreds of
dots voting, the true offset wins clearly over a one-dot shift.

Main function:
    estimate_lr_label_offset(uvL, xyL, uvR, xyR, d0) -> AnchorResult

Inputs (parallel arrays, one row per detected dot):
    uvL : (Nl,2) int    lattice labels for left dots (arbitrary origin)
    xyL : (Nl,2) float  image pixel coordinates of left dots
    uvR : (Nr,2) int    lattice labels for right dots (arbitrary origin)
    xyR : (Nr,2) float  image pixel coordinates of right dots
    d0  : float         nearest-neighbour dot spacing in px (~57 on this rig)

No prior stereo calibration is needed.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np
import cv2


@dataclass
class AnchorResult:
    ok: bool
    duv: tuple[int, int]          # uvR + duv == uvL for the same physical dot
    n_votes: int                  # number of matches that voted
    confidence: float             # fraction of votes that agreed with the winner
    runner_up_gap: int            # winner count minus second-place count
    median_resid_px: float        # median image residual of the matched dots
    pairs: list = None            # (i_left, j_right) index pairs of the same physical
                                  # dot: geometric inliers that also agree with duv.
                                  # Use these directly rather than re-matching labels.
    reason: str = ""              # explanation when ok is False

    def __post_init__(self):
        if self.pairs is None:
            self.pairs = []


def _homography_icp(src_xy: np.ndarray, dst_xy: np.ndarray, d0: float,
                    seed_src=None, seed_dst=None, iters: int = 6):
    """Homography from src to dst image coordinates, estimated by ICP.

    Starts from a pure translation, then alternates nearest-neighbour matching
    and a RANSAC homography refit, tightening the match gate each round.

    The starting translation has to put src within about half a dot spacing of
    the right alignment, otherwise ICP can lock onto a copy of the periodic dot
    field shifted by one dot. If seed_src / seed_dst are given (the image
    centroid of the central square in each view), they set the starting
    translation and remove that ambiguity. The marker is only used as an image
    point here, never snapped to a lattice label. Without a seed the cloud
    centroids are aligned instead, which is only safe when the views are close.

    Returns (H, matches), where matches are (i_src, j_dst) index pairs that pass
    a final mutual-nearest-neighbour test at a tight gate. H may be None.
    """
    src = src_xy.astype(np.float64)
    dst = dst_xy.astype(np.float64)

    if seed_src is not None and seed_dst is not None:
        t0 = np.asarray(seed_dst, float) - np.asarray(seed_src, float)
    else:
        t0 = dst.mean(axis=0) - src.mean(axis=0)
    H = np.array([[1, 0, t0[0]],
                  [0, 1, t0[1]],
                  [0, 0, 1.0]], dtype=np.float64)

    def warp(pts, Hm):
        p = np.hstack([pts, np.ones((len(pts), 1))])
        q = (Hm @ p.T).T
        return q[:, :2] / q[:, 2:3]

    gate_start, gate_end = 2.0 * d0, 0.30 * d0
    matches = []
    for k in range(iters):
        frac = k / max(1, iters - 1)
        gate = gate_start * (1 - frac) + gate_end * frac

        sw = warp(src, H)
        # Nearest dst dot for every warped src dot. The clouds are small, so a
        # direct O(Nl*Nr) distance matrix is fine.
        d2 = ((sw[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
        j_for_i = d2.argmin(axis=1)
        dmin_i = np.sqrt(d2[np.arange(len(sw)), j_for_i])

        cand = [(i, int(j_for_i[i])) for i in range(len(sw)) if dmin_i[i] < gate]
        if len(cand) < 6:
            # Gate too tight for the current H: keep the previous matches and stop.
            break

        s_pts = np.array([src[i] for i, _ in cand])
        d_pts = np.array([dst[j] for _, j in cand])
        Hn, inl = cv2.findHomography(s_pts, d_pts, cv2.RANSAC, 3.0)
        if Hn is None:
            break
        H = Hn
        matches = cand

    if H is None or not matches:
        return H, []

    # Final pass: keep only mutual nearest neighbours within a tight gate.
    sw = warp(src, H)
    d2 = ((sw[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
    j_for_i = d2.argmin(axis=1)
    i_for_j = d2.argmin(axis=0)
    gate = 0.30 * d0
    mutual = []
    for i in range(len(sw)):
        j = int(j_for_i[i])
        if int(i_for_j[j]) == i and np.sqrt(d2[i, j]) < gate:
            mutual.append((i, j))
    return H, mutual


def estimate_lr_label_offset(uvL, xyL, uvR, xyR, d0: float,
                             marker_xyL=None, marker_xyR=None,
                             min_votes: int = 12,
                             min_confidence: float = 0.60) -> AnchorResult:
    """Integer label offset (dU, dV) such that uvR + (dU, dV) names the same dot as uvL.

    marker_xyL / marker_xyR: image centroid (px) of the central square in each
    view. Recommended, since it starts the match in the right place and avoids
    a one-dot shift. Pass the raw centroid, not a snapped lattice position.

    The result is ok only if at least min_votes matches were found and the
    winning offset got at least min_confidence of the votes.
    """
    uvL = np.asarray(uvL, dtype=np.int64)
    uvR = np.asarray(uvR, dtype=np.int64)
    xyL = np.asarray(xyL, dtype=np.float64)
    xyR = np.asarray(xyR, dtype=np.float64)

    if len(uvL) < 8 or len(uvR) < 8:
        return AnchorResult(False, (0, 0), 0, 0.0, 0, float("nan"),
                            reason="too few dots to match")

    # Match right dots onto left dots in image space.
    H, matches = _homography_icp(xyR, xyL, d0,
                                 seed_src=marker_xyR, seed_dst=marker_xyL)
    if len(matches) < min_votes:
        return AnchorResult(False, (0, 0), len(matches), 0.0, 0, float("nan"),
                            reason=f"only {len(matches)} left-right matches found")

    # Each match (jR -> iL) votes for offset = uvL[iL] - uvR[jR].
    votes = Counter()
    resids = []
    # Image residuals of the matches under H, for reporting.
    p = np.hstack([xyR, np.ones((len(xyR), 1))])
    q = (H @ p.T).T
    xyR_in_L = q[:, :2] / q[:, 2:3]
    for jR, iL in matches:
        off = (int(uvL[iL, 0] - uvR[jR, 0]), int(uvL[iL, 1] - uvR[jR, 1]))
        votes[off] += 1
        resids.append(float(np.hypot(*(xyR_in_L[jR] - xyL[iL]))))

    ordered = votes.most_common()
    winner, wc = ordered[0]
    runner = ordered[1][1] if len(ordered) > 1 else 0
    total = sum(votes.values())
    conf = wc / total
    ok = (wc >= min_votes) and (conf >= min_confidence)

    # Keep only matches that also agree with the winning offset. This drops
    # stray detections whose labels happen to look consistent, so the caller
    # can use these pairs directly as same-dot correspondences.
    pairs = []
    for jR, iL in matches:
        off = (int(uvL[iL, 0] - uvR[jR, 0]), int(uvL[iL, 1] - uvR[jR, 1]))
        if off == winner:
            pairs.append((iL, jR))

    return AnchorResult(
        ok=ok,
        duv=winner,
        n_votes=total,
        confidence=conf,
        runner_up_gap=wc - runner,
        median_resid_px=float(np.median(resids)) if resids else float("nan"),
        pairs=pairs,
        reason="" if ok else "no offset won by a safe margin (grid ambiguity?)",
    )
