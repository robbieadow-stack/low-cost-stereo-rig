# raft: RAFT-Stereo reconstruction

Reconstruction of a stereo pair from the stereo camera rig with the
pretrained RAFT-Stereo network, so that it can be compared with the SGBM and
DIC results from the Pi. It uses the same rectification, depth window and
statistics as `reconstruct.py` in `../pi`.

## Setup

1. Clone RAFT-Stereo and install its requirements:
   ```
   git clone https://github.com/princeton-vl/RAFT-Stereo.git
   cd RAFT-Stereo
   ```
2. Download the pretrained models with the script in that repository and
   check that `models/raftstereo-middlebury.pth` exists.
3. Copy `nn_reconstruct.py` from this repository into the RAFT-Stereo folder,
   together with a calibration file (`stereo.npz`) from the Pi.

## Use

From inside the RAFT-Stereo folder:

```
python nn_reconstruct.py spec --dir <folder with spec_left.jpg and spec_right.jpg> \
    --calib stereo.npz --zmin 280 --zmax 420 --device mps
```

`--device mps` uses the GPU on Apple silicon; the default is `cpu`.
The operator console in `../console` starts this script itself.

RAFT-Stereo is by Lipson, Teed and Deng (2021) and is released under the MIT
licence in its own repository.
