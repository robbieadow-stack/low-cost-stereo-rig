# pi: Raspberry Pi side

Raspberry Pi side of a low-cost stereo camera rig for 3D shape measurement.
Two Raspberry Pi Camera Module 3 sensors on a Raspberry Pi 5 are calibrated
from a single image of a two-level LaVision dot plate, and a speckled surface
is then reconstructed with semi-global block matching (SGBM) or a simplified
digital image correlation (DIC) matcher.

## Files

| File | Purpose |
|---|---|
| `pi_camera_server.py` | HTTP server on port 8000 that the host computer's console talks to (preview, focus, capture, calibration, reconstruction) |
| `stereo_camera.py` | Opens and controls both cameras with Picamera2 |
| `plate_model.py` | Detects the plate dots, labels them by lattice growth and anchors the origin |
| `anchor_lr.py` | Makes sure the left and right views give each dot the same label |
| `calibrate_plate.py` | Single-image stereo calibration, writes `calibration/stereo.npz` |
| `reconstruct.py` | Rectifies a pair and reconstructs it with SGBM or DIC |

## Setup

On Raspberry Pi OS (Bookworm):

```
sudo apt install python3-picamera2
pip install -r requirements.txt
```

All scripts are run from inside this `pi` folder, and they read and write the
`captures/` and `calibration/` folders there.

## Use

Start the server, then connect from the operator console on the host computer
(see `../console`):

```
python pi_camera_server.py
```

The scripts can also be run on their own:

```
python calibrate_plate.py                       # calibrate from captures/stereo_cal/pairNN_*.jpg
python reconstruct.py spec --method sgbm --zmin 280 --zmax 420
```

In the project the Pi and the host computer were joined by a direct Ethernet
cable, with the Pi at 192.168.50.1.
