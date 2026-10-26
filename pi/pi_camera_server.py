#!/usr/bin/env python3
"""Headless HTTP camera server for the stereo rig. Runs on the Pi.

Owns the two IMX708 cameras (through stereo_camera.StereoCamera) and exposes
them over a small HTTP API, so the console can run on the laptop and drive
the rig over the ethernet link. No display is needed on the Pi.

A background thread keeps grabbing frames and stores the latest pair as JPEG,
so a preview request returns immediately instead of waiting on the cameras.

HTTP API
--------
  GET  /status                -> {connected, locked, lens_position, calib_count}
  GET  /calib_count           -> {count} calibration pairs on disk
  GET  /preview/left.jpg      -> latest left preview frame (downscaled JPEG)
  GET  /preview/right.jpg     -> latest right preview frame
  GET  /capture/left.jpg      -> last full-res left capture
  GET  /capture/right.jpg     -> last full-res right capture
  GET  /snap/left.jpg         -> last full-res left verification snap
  GET  /snap/right.jpg        -> last full-res right verification snap
  GET  /recon/<file>.jpg      -> an output image from captures/recon (e.g. the
                                 disparity map named in /reconstruct's "disp")
  GET  /calibration/stereo.npz -> the current plate calibration (binary)
  POST /connect               -> open cameras and start preview -> {ok, info}
  POST /disconnect            -> close cameras
  POST /set_lens?v=2.0        -> manual focus (dioptres) on both cameras
  POST /set_exposure          -> manual exposure, AE off; params us=, gain=, scale=
  POST /autofocus             -> one AF cycle -> {ok, lens_position}
  POST /lock                  -> lock focus, exposure and white balance
  POST /unlock                -> hand exposure and white balance back to auto
  POST /capture               -> full-res pair saved on the Pi
                                 -> {ok, left, right, w, h}
  POST /snap                  -> same as /capture but into captures/snap
  POST /capture_calib         -> save the next pairNN into captures/stereo_cal
                                 -> {ok, count}
  POST /reset_calib           -> delete all calibration pair images
  POST /run_calibration       -> run calibrate_plate on the calibration pairs
  POST /reconstruct?method=sgbm|dic&zmin=250&zmax=450&scale=0.5&name=capture
                              -> run reconstruct.py on the last /capture pair
                                 -> {ok, valid, depth, pts, disp, stdout}

Run on the Pi:
    cd ~/stereo-thesis
    source .venv/bin/activate
    python3 scripts/pi_camera_server.py
and leave it running while the console is used on the laptop.

"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2

from stereo_camera import StereoCamera, CameraError, DEFAULT_LENS_POSITION

# Server settings
HOST = "0.0.0.0"          # listen on all interfaces (ethernet + wifi)
PORT = 8000
PREVIEW_WIDTH = 720       # downscale preview frames for fast streaming
PREVIEW_JPEG_QUALITY = 80
PREVIEW_PERIOD = 0.04     # seconds between preview grabs (~25 fps cap)
RECON_DIR = "captures/recon"
SNAP_DIR = "captures/snap"    # full-res verification snaps (kept separate from recon)
CAL_DIR = "captures/stereo_cal"   # calibration pose pairs
CALIB_NPZ = "calibration/stereo.npz"   # written by calibrate_plate.py
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
RECON_TIMEOUT = 600       # seconds; generous so a slow DIC run is not cut off


class RigState:
    """Thread-safe wrapper around StereoCamera plus the preview grabber thread.

    All camera calls go through cam_lock because the HTTP server is threaded
    and picamera2 should not be driven from two threads at once.
    """

    def __init__(self):
        self.cam = None
        self.cam_lock = threading.Lock()     # serialise all camera access
        self.connected = False
        self.locked = False
        self.lens_position = DEFAULT_LENS_POSITION

        self.preview_running = False
        self.preview_thread = None
        self.frame_lock = threading.Lock()
        self.latest_left = None              # JPEG bytes
        self.latest_right = None

    def connect(self):
        with self.cam_lock:
            if self.connected:
                return self.cam.describe()
            self.cam = StereoCamera()
            info = self.cam.open()
            self.connected = True
            self.lens_position = self.cam.lens_position
        self._start_preview()
        # Wait briefly so the first preview frames exist before we return.
        for _ in range(40):
            with self.frame_lock:
                if self.latest_left is not None:
                    break
            time.sleep(0.05)
        return info

    def disconnect(self):
        self._stop_preview()
        with self.cam_lock:
            if self.cam is not None:
                self.cam.close()
            self.cam = None
            self.connected = False
            self.locked = False

    def _start_preview(self):
        if self.preview_running:
            return
        self.preview_running = True
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.preview_thread.start()

    def _stop_preview(self):
        self.preview_running = False
        t = self.preview_thread
        if t is not None:
            t.join(timeout=1.5)
        self.preview_thread = None

    def _encode_preview(self, bgr):
        h, w = bgr.shape[:2]
        new_h = int(h * (PREVIEW_WIDTH / float(w)))
        small = cv2.resize(bgr, (PREVIEW_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def _preview_loop(self):
        while self.preview_running:
            try:
                with self.cam_lock:
                    if not self.connected:
                        break
                    left, right = self.cam.preview_frames()
                lj = self._encode_preview(left)
                rj = self._encode_preview(right)
                with self.frame_lock:
                    self.latest_left = lj
                    self.latest_right = rj
            except Exception:
                time.sleep(0.1)
            time.sleep(PREVIEW_PERIOD)

    def set_lens(self, v):
        with self.cam_lock:
            if self.cam is not None:
                self.cam.set_lens_position(v)
            self.lens_position = v

    def set_exposure(self, exposure_us=None, gain=None, scale=None):
        with self.cam_lock:
            if self.cam is None:
                raise CameraError("Cameras not connected.")
            return self.cam.set_exposure(exposure_us=exposure_us, gain=gain, scale=scale)

    def autofocus(self):
        with self.cam_lock:
            lp = self.cam.autofocus()
            self.lens_position = lp
        return lp

    def lock_exposure(self):
        with self.cam_lock:
            self.cam.lock()
            self.locked = True

    def unlock_exposure(self):
        with self.cam_lock:
            self.cam.unlock()
            self.locked = False

    def capture(self, out_dir=RECON_DIR, left_name="left.jpg", right_name="right.jpg"):
        with self.cam_lock:
            left, right = self.cam.capture_pair()
        os.makedirs(out_dir, exist_ok=True)
        lp = os.path.join(out_dir, left_name)
        rp = os.path.join(out_dir, right_name)
        cv2.imwrite(lp, left)
        cv2.imwrite(rp, right)
        return lp, rp, left.shape[1], left.shape[0]

    def calib_count(self):
        import glob
        return len(glob.glob(os.path.join(CAL_DIR, "*_left.jpg")))

    def capture_calib(self):
        n = self.calib_count()
        name = f"pair{n:02d}"
        self.capture(out_dir=CAL_DIR, left_name=name + "_left.jpg",
                     right_name=name + "_right.jpg")
        return n + 1

    def reset_calib(self):
        import glob
        for f in glob.glob(os.path.join(CAL_DIR, "*.jpg")):
            try:
                os.remove(f)
            except OSError:
                pass


def parse_recon_stats(stdout):
    """Pull valid %, median depth and point count out of reconstruct.py's output."""
    out = {"valid": None, "depth": None, "pts": None}
    m = re.search(r"valid disparity:\s*([\d.]+)%", stdout)
    if m:
        out["valid"] = float(m.group(1))
    m = re.search(r"depth:\s*median\s*([\d.]+)\s*mm.*?([\d,]+)\s*pts", stdout)
    if m:
        out["depth"] = float(m.group(1))
        out["pts"] = int(m.group(2).replace(",", ""))
    return out


def run_reconstruct(method, zmin, zmax, scale=0.5, name="capture"):
    """Run reconstruct.py (SGBM or DIC) on the last /capture pair.

    /capture saves left.jpg / right.jpg, but reconstruct.py expects
    <name>_left.jpg, so the pair is copied to that name first and the script
    is run as a subprocess with the same arguments it takes on the command line.
    """
    if method not in ("sgbm", "dic"):
        return {"ok": False, "error": f"unknown method {method}"}
    src_l = os.path.join(RECON_DIR, "left.jpg")
    src_r = os.path.join(RECON_DIR, "right.jpg")
    if not (os.path.exists(src_l) and os.path.exists(src_r)):
        return {"ok": False, "error": "no captured pair on the Pi yet"}
    if not os.path.exists(CALIB_NPZ):
        return {"ok": False, "error": f"no calibration at {CALIB_NPZ}"}
    shutil.copyfile(src_l, os.path.join(RECON_DIR, f"{name}_left.jpg"))
    shutil.copyfile(src_r, os.path.join(RECON_DIR, f"{name}_right.jpg"))
    argv = [sys.executable, os.path.join(SCRIPTS_DIR, "reconstruct.py"), name,
            "--method", method, "--dir", RECON_DIR, "--calib", CALIB_NPZ,
            "--scale", str(scale), "--zmin", str(zmin), "--zmax", str(zmax)]
    p = subprocess.run(argv, capture_output=True, text=True, timeout=RECON_TIMEOUT)
    if p.returncode != 0:
        tail = (p.stderr.strip().splitlines() or ["(no error output)"])[-1]
        return {"ok": False, "error": tail}
    res = {"ok": True, **parse_recon_stats(p.stdout),
           "disp": f"/recon/{name}_{method}_disparity.jpg", "stdout": p.stdout}
    return res


STATE = RigState()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep the console quiet

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _jpeg(self, data):
        if data is None:
            self._json({"error": "no frame yet"}, 503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path):
        if not os.path.exists(path):
            self._json({"error": "no file"}, 404)
            return
        with open(path, "rb") as f:
            self._jpeg(f.read())

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/status":
                self._json({"connected": STATE.connected, "locked": STATE.locked,
                            "lens_position": STATE.lens_position,
                            "calib_count": STATE.calib_count()})
            elif path == "/calib_count":
                self._json({"count": STATE.calib_count()})
            elif path == "/preview/left.jpg":
                with STATE.frame_lock:
                    d = STATE.latest_left
                self._jpeg(d)
            elif path == "/preview/right.jpg":
                with STATE.frame_lock:
                    d = STATE.latest_right
                self._jpeg(d)
            elif path == "/capture/left.jpg":
                self._send_file(os.path.join(RECON_DIR, "left.jpg"))
            elif path == "/capture/right.jpg":
                self._send_file(os.path.join(RECON_DIR, "right.jpg"))
            elif path == "/snap/left.jpg":
                self._send_file(os.path.join(SNAP_DIR, "left.jpg"))
            elif path == "/snap/right.jpg":
                self._send_file(os.path.join(SNAP_DIR, "right.jpg"))
            elif path.startswith("/recon/") and path.endswith(".jpg"):
                fname = os.path.basename(path)          # no path traversal
                self._send_file(os.path.join(RECON_DIR, fname))
            elif path == "/calibration/stereo.npz":
                if not os.path.exists(CALIB_NPZ):
                    self._json({"error": "no calibration yet"}, 404)
                else:
                    with open(CALIB_NPZ, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)
        try:
            if path == "/connect":
                info = STATE.connect()
                self._json({"ok": True, "info": info})
            elif path == "/disconnect":
                STATE.disconnect()
                self._json({"ok": True})
            elif path == "/set_lens":
                v = float(q.get("v", [str(DEFAULT_LENS_POSITION)])[0])
                STATE.set_lens(v)
                self._json({"ok": True, "lens_position": v})
            elif path == "/set_exposure":
                us = q.get("us", [None])[0]
                gain = q.get("gain", [None])[0]
                scale = q.get("scale", [None])[0]
                applied = STATE.set_exposure(
                    exposure_us=float(us) if us else None,
                    gain=float(gain) if gain else None,
                    scale=float(scale) if scale else None)
                self._json({"ok": True, **applied})
            elif path == "/autofocus":
                lp = STATE.autofocus()
                self._json({"ok": True, "lens_position": lp})
            elif path == "/lock":
                STATE.lock_exposure()
                self._json({"ok": True})
            elif path == "/unlock":
                STATE.unlock_exposure()
                self._json({"ok": True})
            elif path == "/capture":
                lp, rp, w, h = STATE.capture()
                self._json({"ok": True, "left": lp, "right": rp, "w": w, "h": h})
            elif path == "/snap":
                lp, rp, w, h = STATE.capture(out_dir=SNAP_DIR)
                self._json({"ok": True, "left": lp, "right": rp, "w": w, "h": h})
            elif path == "/capture_calib":
                count = STATE.capture_calib()
                self._json({"ok": True, "count": count})
            elif path == "/reset_calib":
                STATE.reset_calib()
                self._json({"ok": True, "count": 0})
            elif path == "/reconstruct":
                self._json(run_reconstruct(
                    method=q.get("method", ["sgbm"])[0],
                    zmin=float(q.get("zmin", ["250"])[0]),
                    zmax=float(q.get("zmax", ["450"])[0]),
                    scale=float(q.get("scale", ["0.5"])[0]),
                    name=os.path.basename(q.get("name", ["capture"])[0])))
            elif path == "/run_calibration":
                from calibrate_plate import calibrate_and_save, result_summary
                r = calibrate_and_save()
                self._json(result_summary(r))
            else:
                self._json({"error": "not found"}, 404)
        except CameraError as e:
            self._json({"error": str(e)}, 500)
        except Exception as e:
            self._json({"error": str(e)}, 500)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Stereo camera server listening on http://{HOST}:{PORT}")
    print("Leave this running. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        STATE.disconnect()
        server.server_close()


if __name__ == "__main__":
    main()