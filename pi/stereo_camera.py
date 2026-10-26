#!/usr/bin/env python3
"""Camera layer for the two Raspberry Pi Camera Module 3 (IMX708) units on the Pi 5.

Wraps both cameras behind one small class so the server and console never
touch picamera2 directly: open/close, preview frames, near-simultaneous
stereo capture, manual and auto focus, manual exposure, and a lock of focus,
exposure and white balance.

Run standalone on the Pi to check the rig:
    python3 stereo_camera.py
This opens both cameras, prints frame shapes, runs an autofocus cycle, locks,
and saves a test pair to captures/recon/.

Requires (on the Pi): picamera2, libcamera, numpy, opencv-python

Two things worth knowing:
* picamera2's "RGB888" format gives arrays with channels ordered B, G, R,
  which is OpenCV's native order, so frames from here go straight into cv2
  with no colour conversion.
* There is no hardware sync between the cameras. capture_pair() grabs the
  latest frame from each camera back to back, so there is a small time skew.
  That is why the rig is only used on static or quasi-static scenes.
"""
from __future__ import annotations

import os
import time

import numpy as np

# picamera2 / libcamera only exist on the Pi. Guard the import so this file can
# still be imported on a laptop without the camera stack.
try:
    from picamera2 import Picamera2
    from libcamera import controls as libcontrols
    _PICAMERA2_OK = True
    _IMPORT_ERR = None
except Exception as e:  # pragma: no cover - only on non-Pi machines
    Picamera2 = None
    libcontrols = None
    _PICAMERA2_OK = False
    _IMPORT_ERR = e


# Which physical camera is the left eye. If the preview looks swapped (a feature
# on the right of the scene shows up in the left image), swap these two numbers.
LEFT_CAM_INDEX = 0
RIGHT_CAM_INDEX = 1

# One working resolution for both preview and capture. 2304 x 1296 is the
# IMX708 2x2 binned mode: full field of view, good low-light behaviour, and
# enough resolution for calibration and reconstruction at ~400 mm. Preview and
# capture share it so the calibrated images match what is previewed.
CAPTURE_SIZE = (2304, 1296)

# Lens position is in dioptres (1 / focus distance in metres); 0.0 is infinity.
# The default of 2.0 dioptres focuses at 0.5 m. The 400 mm working distance
# would be 2.5 dioptres; autofocus or /set_lens refines it after opening.
DEFAULT_LENS_POSITION = 2.0


class CameraError(RuntimeError):
    """Raised for any camera problem that should be shown to the user."""


class StereoCamera:
    """Two IMX708 cameras driven as one stereo rig."""

    def __init__(self,
                 left_index: int = LEFT_CAM_INDEX,
                 right_index: int = RIGHT_CAM_INDEX,
                 size: tuple[int, int] = CAPTURE_SIZE):
        self.left_index = left_index
        self.right_index = right_index
        self.size = size
        self.cam_left: Picamera2 | None = None
        self.cam_right: Picamera2 | None = None
        self.started = False
        self.locked = False
        self.lens_position = DEFAULT_LENS_POSITION

    def open(self):
        """Open and start both cameras with a shared video configuration."""
        if not _PICAMERA2_OK:
            raise CameraError(
                "picamera2 is not available here. Run this on the Pi.\n"
                f"Import error was: {_IMPORT_ERR}")

        try:
            self.cam_left = Picamera2(self.left_index)
            self.cam_right = Picamera2(self.right_index)
        except Exception as e:
            self.close()
            raise CameraError(
                f"Could not open both cameras (indices {self.left_index} and "
                f"{self.right_index}). Check the ribbon cables and that libcamera "
                f"sees two cameras (`rpicam-hello --list-cameras`). Details: {e}")

        for cam in (self.cam_left, self.cam_right):
            cfg = cam.create_video_configuration(
                main={"size": self.size, "format": "RGB888"},
                # Only the frame rate is set here; focus is set to manual
                # just below, once both cameras are started.
                controls={"FrameRate": 30},
                buffer_count=4,
            )
            cam.configure(cfg)

        self.cam_left.start()
        self.cam_right.start()
        self.started = True

        # Start in manual focus at the default lens position.
        self.set_lens_position(self.lens_position)
        # Give auto exposure and white balance a moment to settle before a lock.
        time.sleep(0.5)
        return self.describe()

    def close(self):
        for cam in (self.cam_left, self.cam_right):
            try:
                if cam is not None:
                    cam.stop()
                    cam.close()
            except Exception:
                pass
        self.cam_left = self.cam_right = None
        self.started = False

    def describe(self) -> str:
        """Short description of the opened cameras."""
        if not self.started:
            return "Cameras not started."
        w, h = self.size
        return (f"CAM{self.left_index} left, CAM{self.right_index} right, "
                f"{w} x {h} RGB.")

    def preview_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Latest (left, right) frames as BGR arrays, for the live preview."""
        if not self.started:
            raise CameraError("Cameras not started.")
        left = self.cam_left.capture_array("main")
        right = self.cam_right.capture_array("main")
        return left, right

    def capture_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Near-simultaneous full-resolution (left, right) BGR arrays.

        There is no hardware trigger, so the two grabs happen as close together
        as the API allows. Fine for static or quasi-static targets.
        """
        if not self.started:
            raise CameraError("Cameras not started.")
        left = self.cam_left.capture_array("main")
        right = self.cam_right.capture_array("main")
        return left, right

    def capture_pair_to_disk(self, out_dir: str,
                             left_name: str = "left.jpg",
                             right_name: str = "right.jpg"):
        """Capture a pair and write both images. Returns (left_path, right_path)."""
        import cv2
        os.makedirs(out_dir, exist_ok=True)
        left, right = self.capture_pair()
        lp = os.path.join(out_dir, left_name)
        rp = os.path.join(out_dir, right_name)
        cv2.imwrite(lp, left)
        cv2.imwrite(rp, right)
        return lp, rp

    def set_lens_position(self, dioptres: float):
        """Set manual focus on both cameras. dioptres = 1 / distance_in_m."""
        if not self.started:
            return
        self.lens_position = float(dioptres)
        ctrl = {"AfMode": libcontrols.AfModeEnum.Manual,
                "LensPosition": self.lens_position}
        self.cam_left.set_controls(ctrl)
        self.cam_right.set_controls(ctrl)

    def autofocus(self) -> float:
        """Run one AF cycle on both cameras, return the settled left LensPosition."""
        if not self.started:
            raise CameraError("Cameras not started.")
        for cam in (self.cam_left, self.cam_right):
            cam.set_controls({"AfMode": libcontrols.AfModeEnum.Auto})
        # autofocus_cycle() blocks until the scan finishes.
        try:
            self.cam_left.autofocus_cycle()
            self.cam_right.autofocus_cycle()
        except AttributeError:
            # Older picamera2 without the helper: trigger and wait.
            for cam in (self.cam_left, self.cam_right):
                cam.set_controls({"AfTrigger": libcontrols.AfTriggerEnum.Start})
            time.sleep(2.0)
        lp = self.cam_left.capture_metadata().get("LensPosition", self.lens_position)
        # Freeze both at the value the left camera settled on so they match.
        self.set_lens_position(lp)
        return lp

    def set_exposure(self, exposure_us=None, gain=None, scale=None):
        """Set exposure and/or gain manually on both cameras (AE off).

        Focus is not touched, so this can fix an over- or under-exposed scene
        (e.g. a bright speckle board after locking on the darker plate) without
        disturbing the calibrated focus. Marks the rig as locked.

        exposure_us : absolute exposure time in microseconds
        gain        : absolute analogue gain
        scale       : multiply the current exposure by this factor (0.5 halves,
                      2.0 doubles). Uses the left camera's current exposure,
                      so repeated calls compound. Ignored if exposure_us is set.
        Returns {"exposure_us": ..., "gain": ...} as applied (None where left
        unchanged).
        """
        if not self.started:
            raise CameraError("Cameras not started.")
        if scale is not None and exposure_us is None:
            cur = self.cam_left.capture_metadata().get("ExposureTime")
            if cur:
                exposure_us = int(cur * float(scale))
        ctrl = {"AeEnable": False}
        if exposure_us is not None:
            ctrl["ExposureTime"] = int(exposure_us)
        if gain is not None:
            ctrl["AnalogueGain"] = float(gain)
        self.cam_left.set_controls(ctrl)
        self.cam_right.set_controls(ctrl)
        self.locked = True
        return {"exposure_us": exposure_us, "gain": gain}

    def lock(self) -> dict:
        """Freeze focus, exposure and white balance on both cameras.

        Reads the exposure, gain and colour gains the auto algorithms chose and
        re-applies them with auto turned off, so nothing drifts between the
        calibration images and the reconstruction captures.
        """
        if not self.started:
            raise CameraError("Cameras not started.")
        info = {}
        for tag, cam in (("left", self.cam_left), ("right", self.cam_right)):
            md = cam.capture_metadata()
            fixed = {"AeEnable": False, "AwbEnable": False,
                     "AfMode": libcontrols.AfModeEnum.Manual,
                     "LensPosition": self.lens_position}
            if "ExposureTime" in md:
                fixed["ExposureTime"] = int(md["ExposureTime"])
            if "AnalogueGain" in md:
                fixed["AnalogueGain"] = float(md["AnalogueGain"])
            if "ColourGains" in md:
                fixed["ColourGains"] = tuple(float(g) for g in md["ColourGains"])
            cam.set_controls(fixed)
            info[tag] = fixed
        self.locked = True
        return info

    def unlock(self):
        """Hand exposure and white balance back to auto. Focus stays manual."""
        if not self.started:
            return
        for cam in (self.cam_left, self.cam_right):
            cam.set_controls({"AeEnable": True, "AwbEnable": True})
        self.locked = False


def _selftest():
    print("Opening stereo cameras...")
    rig = StereoCamera()
    print(rig.open())

    left, right = rig.preview_frames()
    print(f"Preview frame shapes: left {left.shape}, right {right.shape}, "
          f"dtype {left.dtype}")

    print("Running autofocus cycle...")
    lp = rig.autofocus()
    print(f"Settled lens position: {lp:.2f} dioptres "
          f"(~{100.0 / lp:.0f} cm)" if lp else "unknown")

    print("Locking focus / exposure / white balance...")
    rig.lock()

    lp_path, rp_path = rig.capture_pair_to_disk("captures/recon")
    print(f"Saved test pair:\n  {lp_path}\n  {rp_path}")

    rig.close()
    print("Done. Cameras closed.")


if __name__ == "__main__":
    _selftest()