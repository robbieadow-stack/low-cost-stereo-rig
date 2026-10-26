# Low-cost stereo camera rig for 3D shape measurement

This repository holds the software for a low-cost stereo camera system built
as a final-year Mechatronic Engineering project at Stellenbosch University.
The system measures the 3D shape of a speckled surface with two Raspberry Pi
cameras. It is a first step towards a low-cost stereo digital image
correlation (DIC) system for the department's test machine.

The system does three things:

1. **Captures** images from two Raspberry Pi cameras, controlled
   from an operator console on a host computer.
2. **Calibrates** the stereo pair from a single image of a two-level dot plate,
   instead of multiple checkerboard images.
3. **Reconstructs** a 3D point cloud of the surface with three matching
   methods, so they can be compared on the same hardware:
   - semi-global block matching (SGBM) from OpenCV,
   - a simplified DIC matcher (subset correlation with ZNCC),
   - the pretrained RAFT-Stereo neural network.

## Hardware

| Part | Details |
|---|---|
| Computer on the rig | Raspberry Pi 5 |
| Cameras | 2 x Raspberry Pi Camera Module 3 (IMX708), used at 2304 x 1296 |
| Frame | 20 x 20 mm aluminium T-slot extrusion, about 350 mm wide and 550 mm tall, free-standing |
| Baseline | About 140 mm, with both cameras angled slightly inward |
| Calibration target | LaVision two-level dot plate (15 mm dot pitch, 3 mm between levels) |
| Host computer | Any laptop that runs Python and PySide6. RAFT-Stereo used a Mac with Apple silicon (MPS) |
| Link | Direct Ethernet cable between the Pi (192.168.50.1) and the host computer |

The bought-new cost of the rig hardware is about R6 600.

## How it works

```
 Host computer                                   Raspberry Pi 5
 ┌──────────────────────────┐   Ethernet / HTTP  ┌──────────────────────────┐
 │ console/calibration_ui   │ ─────────────────> │ pi/pi_camera_server      │
 │   preview, focus, lock   │ <───────────────── │   stereo_camera (2 cams) │
 │   calibrate, capture     │   JPEG previews,   │   calibrate_plate        │
 │   compare results        │   results, .npz    │   reconstruct (SGBM, DIC)│
 │ raft/nn_reconstruct      │                    └──────────────────────────┘
 │   RAFT-Stereo on the GPU │
 └──────────────────────────┘
```

**Capture.** The Pi runs a small HTTP server (port 8000). The console asks it
for live previews, runs autofocus, then locks focus and exposure so that the
calibration and the measurement are taken with the same camera settings.

**Calibration.** One image pair of the dot plate is enough. In each image the
dots are found with a blob detector and labelled by growing a lattice out from
a seed dot. Small markers on the plate fix the origin, and the two levels of
the plate give the depth that a flat target lacks. The left and right labels
are then matched by voting on their offset, so that each physical dot gets
the same label in both views. OpenCV then solves for the intrinsic and
stereo parameters, and the result is saved as `calibration/stereo.npz`. On
the test data the stereo reprojection error was about 0.18 px and the
measured baseline 138.9 mm.

**Reconstruction.** Both images are rectified and processed at half
resolution. Disparity is found with SGBM or the DIC matcher on the Pi, or
with RAFT-Stereo on the host computer, and then converted to a point cloud
(`.ply`). The scripts also report the depth, coverage, a plane-fit RMS for
flat targets, and per-object depths for scenes with several objects.

## Results

The system was tested on flat speckled boards, a painted cube and 3D-printed
parts with known CAD geometry. Compared with the CAD models, the median
deviation of the reconstructed surface was:

| Method | Median deviation from CAD |
|---|---|
| RAFT-Stereo | 0.14 mm |
| SGBM | 0.21 mm |
| Simplified DIC | 0.35 mm |

## Repository layout

```
pi/          runs on the Raspberry Pi
  pi_camera_server.py     HTTP server used by the console
  stereo_camera.py        opens and controls both cameras (Picamera2)
  plate_model.py          dot detection, lattice labelling, origin anchoring
  anchor_lr.py            left/right label matching
  calibrate_plate.py      single-image stereo calibration
  reconstruct.py          rectification and SGBM / DIC reconstruction
console/     runs on the host computer
  calibration_ui.py       PySide6 operator console
  remote_camera.py        HTTP client for the Pi server
raft/        runs on the host computer, inside a clone of RAFT-Stereo
  nn_reconstruct.py       RAFT-Stereo reconstruction with the same outputs as reconstruct.py
```

Each folder has its own README with more detail.

## Getting started

**On the Raspberry Pi** (Raspberry Pi OS Bookworm):

```
sudo apt install python3-picamera2
cd pi
pip install -r requirements.txt
python pi_camera_server.py
```

**On the host computer:**

```
cd console
pip install -r requirements.txt
python calibration_ui.py
```

Set `PI_HOST` at the top of `calibration_ui.py` to the Pi's address
(`192.168.50.1` over a direct cable, or `stereopi.local`). To include
RAFT-Stereo, follow `raft/README.md` and set `RAFT_DIR` to the RAFT-Stereo
folder.

## Using the operator console

1. **Connect** to the Pi and check the live preview from both cameras.
2. **Autofocus** on the target, then **lock** focus and exposure.
3. Place the dot plate in view and **calibrate**. The calibration error and
   baseline are shown, and the calibration is copied to the host computer.
4. Replace the plate with the speckled object and **capture** a pair.
5. **Reconstruct** with SGBM, DIC and RAFT-Stereo and compare the disparity
   maps and depths side by side.

Focus and exposure must stay locked between calibration and capture, and the
cameras must not be moved, or the calibration no longer holds.

## Limitations

- Static scenes only. The cameras are triggered together in software, not
  synchronised in hardware, so moving objects are not supported.
- The DIC matcher is simplified. It finds disparity only and does not track
  displacement or strain over time.
- Accuracy depends on a good speckle pattern and even lighting.

## Possible next steps

- Track the surface through a loading sequence to measure displacement and
  strain.
- Synchronise the cameras with each other and with the test machine in
  hardware.
- Implement full subset-based DIC.

## Acknowledgements

RAFT-Stereo is by Lipson, Teed and Deng (2021) and is available at
https://github.com/princeton-vl/RAFT-Stereo under the MIT licence. It is not
included here and must be cloned separately. The Raspberry Pi, cameras and
camera housings were reused from an earlier project, and the calibration
plate was borrowed from the department.
