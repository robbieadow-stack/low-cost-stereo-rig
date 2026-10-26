# console: operator console on the host computer

Operator console for the stereo camera rig, run on the host computer. It
connects to the server on the Raspberry Pi (`../pi`) and
takes the operator through the whole measurement: live preview, focus and
exposure lock, single-image calibration, capture, and reconstruction with
SGBM, DIC and RAFT-Stereo, with the results shown side by side.

## Files

| File | Purpose |
|---|---|
| `calibration_ui.py` | The PySide6 operator console |
| `remote_camera.py` | Small HTTP client for the Pi server |

## Setup

```
pip install -r requirements.txt
```

Set `PI_HOST` at the top of `calibration_ui.py` to the address of the Pi
(`stereopi.local`, or `192.168.50.1` over a direct Ethernet cable).
RAFT-Stereo runs on the host computer: set `RAFT_DIR` to a clone of
RAFT-Stereo that contains `nn_reconstruct.py` (see `../raft`).

## Use

```
python calibration_ui.py
```
