#!/usr/bin/env python3
"""Model and dot detection for the LaVision Type 204-15 two-level plate.

Turns one photo of the plate into matched 2D image points and 3D object
points. The 3 mm step between the two dot levels gives real depth, which is
why a single view of this plate carries 3D information.

Geometry
--------
White dots on black, 15 mm pitch within a level, two levels 3 mm apart, with
rows staggered by half a pitch (quincunx). Growing the lattice hop by hop gives
each dot integer coordinates. In half-step units (U, V):
    X = 7.5*U mm,   Y = 7.5*V mm,   Z = 3 mm on odd V, else 0.

Origin
------
Each view has to pick a repeatable origin dot. The square and triangle markers
at the plate centre do not sit on the dot lattice; they sit in the gaps between
dots (the odd-sum cells). After the lattice is fitted, white shapes that land
on those off-grid cells near the centre are found by position rather than by
shape. The one nearest the centre of the dot cloud (the square) is taken, and
the origin is set to the dot next to it.

This per-camera origin sets the object-point frame and the overlay labels.
Stereo matching does not depend on left and right picking the same dot:
calibrate_plate recovers the label offset between the views with anchor_lr.py,
seeded by the square's image centroid (returned as square_xy).

Run (on the Pi, in the venv):
    python3 scripts/plate_model.py [image]
Saves <image>_model.jpg with the labelled dots drawn on top.

"""
import sys
from collections import deque
import numpy as np
import cv2

HALF_STEP_MM = 7.5        # half the 15 mm dot pitch
LEVEL_SEP_MM = 3.0        # height step between the two dot levels
Y_DOWN = True             # +Y points down the image, like pixel rows
RAISED_ON_ODD_V = True    # which level is raised; calibrate_plate also tries the flip


def make_detector():
    """Blob detector for the bright, round plate dots."""
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True;        p.minArea = 80;   p.maxArea = 100000
    p.filterByCircularity = True; p.minCircularity = 0.7
    p.filterByConvexity = True;   p.minConvexity = 0.85
    p.filterByInertia = True;     p.minInertiaRatio = 0.5
    p.minDistBetweenBlobs = 10
    p.blobColor = 255
    return cv2.SimpleBlobDetector_create(p)


def detect_dots(img):
    """Image (x, y) centres of all detected dots, as an (N, 2) float32 array."""
    det = make_detector()
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    kps = det.detect(gray)
    return np.array([kp.pt for kp in kps], dtype=np.float32)


def median_nn(pts):
    """Pairwise distance matrix and the median nearest-neighbour spacing d0."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return dist, float(np.median(dist.min(axis=1)))


def estimate_basis(pts, dist, d0):
    """Global lattice directions from the neighbour vectors.

    Angles are folded into [0, 90) deg and averaged on the 4-fold circle, so
    the four neighbour directions all vote for the same grid angle.
    """
    vs = []
    for i in range(len(pts)):
        for j in np.argsort(dist[i])[:4]:
            w = pts[j] - pts[i]
            if 0.5 * d0 < np.hypot(*w) < 1.6 * d0:
                vs.append(w)
    vs = np.array(vs)
    folded = np.mod(np.arctan2(vs[:, 1], vs[:, 0]), np.pi / 2.0)
    th = np.mod(np.angle(np.exp(1j * 4 * folded).mean()) / 4.0, np.pi / 2.0)
    e1 = d0 * np.array([np.cos(th), np.sin(th)])
    e2 = d0 * np.array([-np.sin(th), np.cos(th)])
    return e1, e2, th


def pick_seed(pts, dist, d0):
    """Dot to start growing from: well surrounded and close to the centre."""
    ncount = (dist < 1.5 * d0).sum(1)
    ctr = pts.mean(0)
    cand = np.where(ncount >= 4)[0]
    if len(cand) == 0:
        cand = np.arange(len(pts))
    return int(cand[np.argmin(((pts[cand] - ctr) ** 2).sum(1))])


def local_basis(pts, assigned, i, d0, a, b):
    """Lattice vectors fitted to already-labelled neighbours of dot i.

    Following the local basis copes with perspective and lens distortion
    across the plate. Falls back to the global basis (a, b) if the fit is
    poor or there are too few neighbours.
    """
    mi, ni = assigned[i]
    P = pts[i]
    A, B = [], []
    for j, (m, n) in assigned.items():
        if j == i:
            continue
        w = pts[j] - P
        if np.hypot(*w) < 1.9 * d0:
            A.append([m - mi, n - ni]); B.append(w)
    if len(A) >= 2:
        A, B = np.array(A, float), np.array(B, float)
        M, _, rank, _ = np.linalg.lstsq(A, B, rcond=None)
        if rank == 2:
            la, lb = M[0], M[1]
            if 0.5 * d0 < np.hypot(*la) < 1.7 * d0 and 0.5 * d0 < np.hypot(*lb) < 1.7 * d0:
                return la, lb
    return a, b


def grow_lattice(pts, dist, seed, a, b, d0):
    """Breadth-first labelling of dots with integer (m, n) lattice coordinates.

    A neighbour is accepted only if it is one step along a basis vector and
    within 0.3 of a cell of the predicted position.
    """
    assigned = {seed: (0, 0)}
    q = deque([seed])
    while q:
        i = q.popleft()
        m, n = assigned[i]
        P = pts[i]
        la, lb = local_basis(pts, assigned, i, d0, a, b)
        try:
            Binv = np.linalg.inv(np.column_stack([la, lb]))
        except np.linalg.LinAlgError:
            Binv = np.linalg.inv(np.column_stack([a, b]))
        for j in np.where(dist[i] < 1.7 * d0)[0]:
            if j in assigned:
                continue
            st = Binv @ (pts[j] - P)
            r = np.round(st)
            if abs(r[0]) + abs(r[1]) == 1 and np.hypot(*(st - r)) < 0.30:
                assigned[j] = (int(m + r[0]), int(n + r[1]))
                q.append(j)
    return assigned


def find_offgrid_markers(img, ctr_img, d0, M, c):
    """Find white shapes sitting between the dots (off the grid) near the centre.

    The dots land on even lattice cells; the square and triangle markers sit
    on the empty odd cells. White shapes are found by thresholding, mapped into
    lattice coordinates with the fitted affine, and kept if they land close to
    an odd cell within 6 dot spacings of the centre. Returns their continuous
    lattice coordinates and image positions.
    """
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return [], []
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lo, hi = 0.05 * d0 * d0, 0.5 * d0 * d0
    uv_list, xy_list = [], []
    for cnt in cnts:
        a = cv2.contourArea(cnt)
        if a < lo or a > hi:
            continue
        mm = cv2.moments(cnt)
        if mm["m00"] == 0:
            continue
        x, y = mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]
        if (x - ctr_img[0]) ** 2 + (y - ctr_img[1]) ** 2 > (6 * d0) ** 2:
            continue
        uv = Minv @ np.array([x - c[0], y - c[1]])
        ur, vr = int(round(uv[0])), int(round(uv[1]))
        if (ur + vr) % 2 == 1 and np.hypot(uv[0] - ur, uv[1] - vr) < 0.35:
            uv_list.append((uv[0], uv[1])); xy_list.append((x, y))
    return uv_list, xy_list


def build_model(img, pts, dist, d0):
    """Label the dots and turn them into object points (mm) and image points."""
    a, b, th = estimate_basis(pts, dist, d0)
    seed = pick_seed(pts, dist, d0)
    assigned = grow_lattice(pts, dist, seed, a, b, d0)

    idxs = list(assigned.keys())
    mn = np.array([assigned[i] for i in idxs])
    m, n = mn[:, 0].astype(int), mn[:, 1].astype(int)
    u0, v0 = m - n, m + n
    P = pts[idxs].astype(float)

    # Lock the axes to the image: +U to the right, +V down (plate kept upright).
    A = np.column_stack([u0, v0, np.ones(len(u0))]).astype(float)
    cx, _, _, _ = np.linalg.lstsq(A, P[:, 0], rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, P[:, 1], rcond=None)
    au, av = np.array([cx[0], cy[0]]), np.array([cx[1], cy[1]])
    if abs(au[0]) >= abs(av[0]):
        U = (u0 * np.sign(au[0])).astype(int); V = (v0 * np.sign(av[1])).astype(int)
    else:
        U = (v0 * np.sign(av[0])).astype(int); V = (u0 * np.sign(au[1])).astype(int)

    # Affine map from (U, V) to image pixels, used to place the markers.
    Auv = np.column_stack([U, V, np.ones(len(U))]).astype(float)
    hx, _, _, _ = np.linalg.lstsq(Auv, P[:, 0], rcond=None)
    hy, _, _, _ = np.linalg.lstsq(Auv, P[:, 1], rcond=None)
    M = np.array([[hx[0], hx[1]], [hy[0], hy[1]]])
    c = np.array([hx[2], hy[2]])

    # Origin: the square marker sits at the plate centre, so it is the off-grid
    # marker nearest the centre of the dot cloud. Round it to its own odd cell
    # (it is within about a third of a cell) and step one cell in +U to a dot.
    # This origin sets the X/Y frame of the object points and the overlay.
    # Left/right label agreement is recovered separately in calibrate_plate
    # (anchor_lr, seeded by square_xy), so a one-dot difference between the
    # two views here does no harm.
    ctr = P.mean(0)
    uv_list, xy_list = find_offgrid_markers(img, ctr, d0, M, c)
    origin_ok = len(uv_list) >= 1
    sq_cell = None
    square_xy = None                                  # image centroid of the square
    if origin_ok:
        di = int(np.argmin([(mx - ctr[0]) ** 2 + (my - ctr[1]) ** 2 for (mx, my) in xy_list]))
        su, sv = uv_list[di]
        square_xy = tuple(float(v) for v in xy_list[di])
        ous, ov_s = int(round(su)), int(round(sv))   # square's odd cell
        ou, ov = ous + 1, ov_s                        # adjacent dot (even cell)
        U = U - ou; V = V - ov
        sq_cell = (ous - ou, ov_s - ov)               # square in the new frame

    X = HALF_STEP_MM * U
    Y = HALF_STEP_MM * V * (1.0 if Y_DOWN else -1.0)
    raised = (V % 2 != 0) if RAISED_ON_ODD_V else (V % 2 == 0)
    Z = np.where(raised, LEVEL_SEP_MM, 0.0)
    obj = np.stack([X, Y, Z], 1).astype(np.float32)
    imgpts = P.astype(np.float32)
    uv = np.stack([U, V], 1).astype(np.int32)
    unassigned = [i for i in range(len(pts)) if i not in assigned]
    diag = dict(marker_xy=xy_list, n_markers=len(uv_list), sq_cell=sq_cell,
                square_xy=square_xy, d0=float(d0))
    return obj, imgpts, uv, unassigned, origin_ok, np.degrees(th), diag


def draw_overlay(img, pts, obj, imgpts, uv, unassigned, diag):
    """Draw labelled dots, the two levels, the markers and the origin axes.

    Red dots are the low level, blue the raised level, grey ones were not
    reached by the lattice growth.
    """
    vis = img.copy()
    for i in unassigned:
        cv2.circle(vis, (int(pts[i, 0]), int(pts[i, 1])), 6, (150, 150, 150), 2)
    lut = {}
    for (X, Y, Z), (px, py), (u, v) in zip(obj, imgpts, uv):
        c = (0, 0, 255) if Z == 0.0 else (255, 120, 0)
        cv2.circle(vis, (int(px), int(py)), 7, c, -1)
        lut[(int(u), int(v))] = (int(px), int(py))
        if u % 4 == 0 and v % 4 == 0:
            cv2.putText(vis, f"{u},{v}", (int(px) + 6, int(py) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    # Off-grid centre markers the origin was built from.
    for (mx, my) in diag.get("marker_xy", []):
        cv2.drawMarker(vis, (int(mx), int(my)), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)

    def near(tu, tv):
        best, bd = None, 1e9
        for (u, v), p in lut.items():
            dd = (u - tu) ** 2 + (v - tv) ** 2
            if dd < bd:
                bd, best = dd, p
        return best
    o, pu, pv = near(0, 0), near(6, 0), near(0, 6)
    if o:
        cv2.circle(vis, o, 13, (0, 255, 0), 3)
        cv2.putText(vis, "origin(0,0)", (o[0] + 14, o[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if pu:
            cv2.arrowedLine(vis, o, pu, (0, 255, 0), 3, tipLength=0.15)
            cv2.putText(vis, "+X", (pu[0], pu[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if pv:
            cv2.arrowedLine(vis, o, pv, (0, 200, 255), 3, tipLength=0.15)
            cv2.putText(vis, "+Y", (pv[0] + 8, pv[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    return vis


def detect_plate(img, verbose=True):
    """Detect and model the plate in one image, with an overlay for checking."""
    pts = detect_dots(img)
    dist, d0 = median_nn(pts)
    obj, imgpts, uv, unassigned, origin_ok, th_deg, diag = build_model(img, pts, dist, d0)
    if verbose:
        print(f"Dots detected: {len(pts)}")
        print(f"NN spacing d0: {d0:.1f} px, basis angle: {th_deg:.1f} deg")
        print(f"Assigned to lattice: {len(obj)} / {len(pts)}  (unreached: {len(unassigned)})")
        print(f"Off-grid markers found: {diag['n_markers']}   "
              f"anchored: {origin_ok}   (False = centre markers not clearly seen)")
        print(f"Square marker at cell: {diag.get('sq_cell')}   "
              f"(per-camera origin only; left/right agreement is enforced by "
              f"anchor_lr in calibrate_plate)")
        lo = int((obj[:, 2] == 0).sum()); hi = int((obj[:, 2] > 0).sum())
        print(f"Level split -> low (Z=0): {lo}   raised (Z=3): {hi}")
        print(f"X range: {obj[:,0].min():.1f}..{obj[:,0].max():.1f} mm, "
              f"Y range: {obj[:,1].min():.1f}..{obj[:,1].max():.1f} mm")
    overlay = draw_overlay(img, pts, obj, imgpts, uv, unassigned, diag)
    return obj, imgpts, uv, overlay, origin_ok


def detect_plate_matchdata(img):
    """Everything calibrate_plate.match_pair needs from one view.

    Returns:
        obj       (N,3) float32  object points (this view's own labels/Z parity)
        imgpts    (N,2) float32  image points
        uv        (N,2) int32    lattice labels (per-camera origin; the
                                  left/right offset is found in anchor_lr)
        origin_ok bool           True if the central square was located (needed
                                  as the geometric-match seed)
        square_xy (2,) float or None   image centroid of the central square
        d0        float          nearest-neighbour dot spacing (px)
    """
    pts = detect_dots(img)
    dist, d0 = median_nn(pts)
    obj, imgpts, uv, unassigned, origin_ok, th_deg, diag = build_model(img, pts, dist, d0)
    square_xy = diag.get("square_xy")
    return obj, imgpts, uv, origin_ok, square_xy, float(d0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "captures/snap/left.jpg"
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read {path}"); return
    print(f"Loaded {path}  ({img.shape[1]}x{img.shape[0]})")
    *_, overlay, _ = detect_plate(img)
    out = path.rsplit(".", 1)[0] + "_model.jpg"
    cv2.imwrite(out, overlay)
    print(f"Saved overlay: {out}")


if __name__ == "__main__":
    main()