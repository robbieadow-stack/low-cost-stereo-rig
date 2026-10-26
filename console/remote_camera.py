#!/usr/bin/env python3
"""
HTTP client for the Pi camera server, used by the console on the host computer.

It has the same methods as stereo_camera.StereoCamera on the Pi, but sends each
call to pi_camera_server.py instead of driving picamera2. Frames come back as
JPEGs, so only compressed images cross the Ethernet link. Needs only the
standard library, numpy and OpenCV.

Not run directly; calibration_ui.py imports it.
"""
import json
import urllib.request

import cv2
import numpy as np

DEFAULT_HOST = "stereopi.local"
DEFAULT_PORT = 8000


class CameraError(RuntimeError):
    """Raised for any camera or network error the console should report."""


class RemoteStereoCamera:
    """Stand-in for StereoCamera that forwards each call to the Pi server."""

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=8.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self.preview_timeout = 3.0
        self.lens_position = 2.0
        self.locked = False
        self.started = False

    # HTTP helpers
    def _get(self, path, timeout=None):
        try:
            req = urllib.request.Request(self.base + path, method="GET")
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return r.read(), r.headers.get_content_type()
        except Exception as e:
            raise CameraError(f"Cannot reach camera server at {self.base}{path}: {e}")

    def _post(self, path, timeout=None):
        try:
            req = urllib.request.Request(self.base + path, method="POST")
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            raise CameraError(f"Cannot reach camera server at {self.base}{path}: {e}")

    def _get_json(self, path, timeout=None):
        data, _ = self._get(path, timeout=timeout)
        return json.loads(data.decode())

    def _get_jpeg(self, path, timeout=None):
        data, ctype = self._get(path, timeout=timeout)
        if ctype != "image/jpeg":
            raise CameraError(f"Expected an image from {path}, got {ctype}")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise CameraError(f"Could not decode image from {path}")
        return img

    # Same methods as StereoCamera
    def open(self):
        res = self._post("/connect", timeout=20.0)   # opening cameras takes a moment
        if not res.get("ok"):
            raise CameraError(res.get("error", "connect failed"))
        self.started = True
        return res.get("info", "Connected to Pi camera server.")

    def preview_frames(self):
        left = self._get_jpeg("/preview/left.jpg", timeout=self.preview_timeout)
        right = self._get_jpeg("/preview/right.jpg", timeout=self.preview_timeout)
        return left, right

    def capture_pair(self):
        res = self._post("/capture", timeout=20.0)
        if not res.get("ok"):
            raise CameraError(res.get("error", "capture failed"))
        left = self._get_jpeg("/capture/left.jpg")
        right = self._get_jpeg("/capture/right.jpg")
        return left, right

    def snap(self):
        """Full-resolution check pair; kept separate from the capture used for reconstruction."""
        res = self._post("/snap", timeout=20.0)
        if not res.get("ok"):
            raise CameraError(res.get("error", "snap failed"))
        left = self._get_jpeg("/snap/left.jpg")
        right = self._get_jpeg("/snap/right.jpg")
        return left, right

    def set_lens_position(self, dioptres):
        self._post(f"/set_lens?v={float(dioptres)}")
        self.lens_position = float(dioptres)

    def autofocus(self):
        res = self._post("/autofocus", timeout=20.0)
        lp = res.get("lens_position", self.lens_position)
        self.lens_position = lp
        return lp

    def calib_count(self):
        try:
            return int(self._get_json("/calib_count").get("count", 0))
        except CameraError:
            return 0

    def capture_calib(self):
        res = self._post("/capture_calib", timeout=20.0)
        if not res.get("ok"):
            raise CameraError(res.get("error", "calibration capture failed"))
        return int(res.get("count", 0))

    def reset_calib(self):
        self._post("/reset_calib")

    def run_calibration(self):
        res = self._post("/run_calibration", timeout=300.0)
        if not res.get("ok"):
            raise CameraError(res.get("error", "calibration failed"))
        return res

    def reconstruct(self, method, zmin, zmax, scale=0.5, name="capture"):
        """Run SGBM or DIC on the Pi on the last captured pair.

        Returns (res, disparity_bgr), where res is the server's JSON reply
        (including valid, depth and pts).
        """
        res = self._post(f"/reconstruct?method={method}&zmin={zmin}&zmax={zmax}"
                         f"&scale={scale}&name={name}", timeout=600.0)
        if not res.get("ok"):
            raise CameraError(res.get("error", f"{method} reconstruction failed"))
        disp = self._get_jpeg(res["disp"], timeout=20.0)
        return res, disp

    def download_calib(self, dest):
        """Download the Pi's calibration/stereo.npz to dest."""
        data, _ = self._get("/calibration/stereo.npz", timeout=20.0)
        with open(dest, "wb") as f:
            f.write(data)
        return dest

    def lock(self):
        self._post("/lock")
        self.locked = True
        return {}

    def unlock(self):
        self._post("/unlock")
        self.locked = False

    def close(self):
        # The server keeps running; the client just stops using it.
        self.started = False