#!/usr/bin/env python3
"""
Operator console for the stereo rig.

Runs on the host computer and drives the Pi camera server (pi_camera_server.py)
over the Ethernet link. The workflow is: live preview, lock focus and exposure,
single-image calibration on the plate, capture a stereo pair, then reconstruct
with SGBM, DIC and RAFT-Stereo and compare the results side by side.

SGBM and DIC run on the Pi through the server's /reconstruct endpoint. RAFT runs
on the host computer: nn_reconstruct.py is started inside RAFT_DIR on the pair
saved in RECON_DIR, using a copy of the Pi's calibration that is downloaded
straight after each calibration.

Set the constants below (Pi host, folders, RAFT device) before running:
    pip install PySide6 numpy opencv-python
    python3 calibration_ui.py
"""
import os
import re
import sys
import subprocess
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider, QTextEdit,
    QVBoxLayout, QHBoxLayout, QGroupBox, QTabWidget, QSizePolicy, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import Qt, QTimer, QThread, Signal

try:
    from remote_camera import RemoteStereoCamera, CameraError
except ModuleNotFoundError:
    # Without remote_camera.py the window still opens, but Connect is refused.
    RemoteStereoCamera = None

    class CameraError(Exception):
        pass

# Configuration: set these for your machine.
_HERE = os.path.dirname(os.path.abspath(__file__))

PI_HOST = "stereopi.local"
PI_PORT = 8000
DEFAULT_LENS_POSITION = 2.0            # dioptres; 2.0 focuses at about 0.5 m
METHODS = ["SGBM", "DIC", "RAFT"]

# RAFT-Stereo checkout on the host computer; nn_reconstruct.py runs from inside it
RAFT_DIR = os.path.expanduser("~/stereo-nn/RAFT-Stereo")
RAFT_PYTHON = sys.executable           # interpreter with torch installed (point
                                       # this at the RAFT venv if it differs)
RAFT_DEVICE = "mps"                    # "mps" on Apple silicon, else "cpu"
RAFT_STRICTZ = True                    # keep only RAFT points inside zmin..zmax
# Where the captured pair and the reconstruction outputs are saved locally
RECON_DIR = os.path.join(_HERE, "captures")
# SGBM and DIC use calibration/stereo.npz on the Pi. RAFT needs the same file
# locally, so the console downloads it here after each calibration.
PI_CALIB = "calibration/stereo.npz"
CALIB_NPZ = os.path.join(RECON_DIR, "stereo.npz")
SCENE_NAME = "capture"                 # base name for the saved pair and outputs
PROC_SCALE = 0.5                       # images are downscaled by this before matching
DEFAULT_ZMIN, DEFAULT_ZMAX = 250, 450  # depth window in mm, adjustable in the UI

# Disparity image saved for each method
DISP_FILE = {
    "SGBM": f"{SCENE_NAME}_sgbm_disparity.jpg",
    "DIC":  f"{SCENE_NAME}_dic_disparity.jpg",
    "RAFT": f"{SCENE_NAME}_nn_disparity.jpg",
}

# Colours for the dark theme
BG     = "#1E1F22"
PANEL  = "#26282C"
FIELD  = "#141517"
TEXT   = "#E7E8EA"
MUTED  = "#9AA0A6"
BORDER = "#34363B"
ACCENT = "#4F9DDE"


def cv_to_pixmap(bgr):
    """Convert an OpenCV BGR image to a QPixmap."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy())


def parse_stats(stdout):
    """Read valid %, median depth, point count and runtime from the text
    printed by reconstruct.py or nn_reconstruct.py. Missing values are None."""
    out = {"valid": None, "depth": None, "pts": None, "rt": None}
    m = re.search(r"valid disparity:\s*([\d.]+)%", stdout)
    if m:
        out["valid"] = float(m.group(1))
    m = re.search(r"depth:\s*median\s*([\d.]+)\s*mm.*?([\d,]+)\s*pts", stdout)
    if m:
        out["depth"] = float(m.group(1))
        out["pts"] = int(m.group(2).replace(",", ""))
    m = re.search(r"matcher runtime:\s*([\d.]+)\s*s", stdout)
    if m:
        out["rt"] = float(m.group(1))
    return out


# Runs the three methods in a thread so the UI stays responsive.
class ReconWorker(QThread):
    one = Signal(str, dict)       # method name, stats dict with "disp_path"
    failed = Signal(str, str)     # method name, short error message
    note = Signal(str)            # progress line for the log
    done = Signal()

    def __init__(self, cam, zmin, zmax):
        super().__init__()
        self.cam = cam
        self.zmin = int(zmin)
        self.zmax = int(zmax)

    @staticmethod
    def preflight():
        """Return why a run cannot start, or None if it can."""
        if not os.path.exists(os.path.join(RAFT_DIR, "nn_reconstruct.py")):
            return f"nn_reconstruct.py not found in {RAFT_DIR}."
        if not os.path.exists(CALIB_NPZ):
            return (f"calibration copy not found at {CALIB_NPZ}. "
                    f"Run Capture & calibrate again.")
        return None

    def _run_pi(self, name):
        """Run SGBM or DIC on the Pi on its last captured pair."""
        stats, disp = self.cam.reconstruct(name.lower(), self.zmin, self.zmax,
                                           PROC_SCALE, SCENE_NAME)
        os.makedirs(RECON_DIR, exist_ok=True)
        path = os.path.join(RECON_DIR, DISP_FILE[name])
        cv2.imwrite(path, disp)
        return {"valid": stats.get("valid"), "depth": stats.get("depth"),
                "pts": stats.get("pts"), "disp_path": path}

    def _run_raft(self):
        """Run RAFT-Stereo locally as a subprocess inside RAFT_DIR."""
        argv = [RAFT_PYTHON, "nn_reconstruct.py", SCENE_NAME,
                "--dir", RECON_DIR, "--calib", CALIB_NPZ,
                "--scale", str(PROC_SCALE),
                "--zmin", str(self.zmin), "--zmax", str(self.zmax),
                "--device", RAFT_DEVICE]
        if RAFT_STRICTZ:
            argv.append("--strictz")
        p = subprocess.run(argv, cwd=RAFT_DIR, capture_output=True,
                           text=True, timeout=600)
        if p.returncode != 0:
            tail = (p.stderr.strip().splitlines() or ["(no error output)"])[-1]
            raise RuntimeError(tail)
        stats = parse_stats(p.stdout)
        # nn_reconstruct.py saves its disparity image in its working folder
        disp = os.path.join(RAFT_DIR, DISP_FILE["RAFT"])
        stats["disp_path"] = disp if os.path.exists(disp) else None
        return stats

    def run(self):
        for name in METHODS:
            self.note.emit(f"Running {name}...")
            try:
                stats = self._run_raft() if name == "RAFT" else self._run_pi(name)
            except Exception as e:
                self.failed.emit(name, str(e))
                continue
            self.one.emit(name, stats)
        self.done.emit()


class ImageView(QLabel):
    """Label that shows an image scaled to fit, keeping its aspect ratio."""
    def __init__(self, placeholder="No image", small=False):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(220 if small else 360, 150 if small else 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background:{FIELD}; color:{MUTED}; border:1px solid {BORDER};"
                           f"border-radius:8px;")
        self._pm = None

    def show_bgr(self, bgr):
        self._pm = cv_to_pixmap(bgr); self._rescale()

    def resizeEvent(self, e):
        self._rescale(); super().resizeEvent(e)

    def _rescale(self):
        if self._pm is not None:
            self.setPixmap(self._pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class Console(QMainWindow):
    """Main console window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stereo Console")
        self.resize(1180, 760)
        self.setMinimumSize(940, 600)

        self.cam = None
        self.connected = False
        self.locked = False
        self.calibrated = False
        self.captured = None
        self.worker = None

        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(16, 14, 16, 14); root.setSpacing(12)

        title = QLabel("Stereo Console")
        title.setStyleSheet(f"color:{TEXT}; font-size:20px; font-weight:700;")
        subtitle = QLabel("Preview, lock, single-image calibration, reconstruction")
        subtitle.setStyleSheet(f"color:{MUTED}; font-size:12px;")
        root.addWidget(title); root.addWidget(subtitle)

        body = QHBoxLayout(); body.setSpacing(14)
        body.addWidget(self._controls(), 0)
        body.addWidget(self._views(), 1)
        root.addLayout(body, 1)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(70)
        self.log.setStyleSheet(f"background:{FIELD}; color:{MUTED}; border:1px solid {BORDER};"
                               f"border-radius:8px; font-family:Menlo,Consolas,monospace; font-size:12px;")
        root.addWidget(self.log)

        self.timer = QTimer(self); self.timer.timeout.connect(self._tick)
        self.setStyleSheet(self._qss())
        self._refresh()
        if RemoteStereoCamera is None:
            self.status("remote_camera.py not found. Put it in this folder to connect to the Pi.")
        else:
            self.status("Ready. Connect the cameras to begin.")

    # Left-hand control panel
    def _controls(self):
        panel = QWidget(); panel.setFixedWidth(320)
        v = QVBoxLayout(panel); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(12)

        cam = QGroupBox("Camera"); cl = QVBoxLayout(cam)
        self.btn_connect = QPushButton("Connect")
        self.btn_live = QPushButton("Start preview")
        self.btn_af = QPushButton("Autofocus")
        self.focus_lbl = QLabel(); self.focus = QSlider(Qt.Horizontal)
        self.focus.setRange(0, 1000); self.focus.setValue(int(DEFAULT_LENS_POSITION * 100))
        self._focus_label(self.focus.value())
        self.btn_lock = QPushButton("Lock focus && exposure")
        for w in (self.btn_connect, self.btn_live, self.btn_af, self.focus_lbl,
                  self.focus, self.btn_lock):
            cl.addWidget(w)

        cal = QGroupBox("Calibration"); cll = QVBoxLayout(cal)
        hint = QLabel("Tilt the plate so both dot levels show perspective, then "
                      "calibrate from one image.")
        hint.setWordWrap(True); hint.setStyleSheet(f"color:{MUTED}; font-size:11px;")
        self.btn_cal = QPushButton("Capture && calibrate")
        self._primary(self.btn_cal)
        self.cal_info = QLabel("Not calibrated"); self.cal_info.setWordWrap(True)
        self.cal_info.setStyleSheet(f"color:{MUTED};")
        for w in (hint, self.btn_cal, self.cal_info):
            cll.addWidget(w)

        rec = QGroupBox("Reconstruct"); rl = QVBoxLayout(rec)
        self.btn_pair = QPushButton("Capture stereo pair")
        win = QHBoxLayout()
        win.addWidget(QLabel("Depth window (mm)"))
        self.zmin = QSpinBox(); self.zmin.setRange(50, 5000); self.zmin.setValue(DEFAULT_ZMIN)
        self.zmax = QSpinBox(); self.zmax.setRange(50, 5000); self.zmax.setValue(DEFAULT_ZMAX)
        self.zmin.setSingleStep(10); self.zmax.setSingleStep(10)
        win.addWidget(self.zmin); win.addWidget(QLabel("to")); win.addWidget(self.zmax)
        self.btn_recon = QPushButton("Reconstruct  (all methods)")
        self._primary(self.btn_recon)
        self.rec_info = QLabel("No reconstruction yet"); self.rec_info.setWordWrap(True)
        self.rec_info.setStyleSheet(f"color:{MUTED};")
        rl.addWidget(self.btn_pair)
        rl.addLayout(win)
        rl.addWidget(self.btn_recon)
        rl.addWidget(self.rec_info)

        for g in (cam, cal, rec):
            v.addWidget(g)
        v.addStretch(1)

        self.btn_connect.clicked.connect(self.on_connect)
        self.btn_live.clicked.connect(self.on_live)
        self.btn_af.clicked.connect(self.on_autofocus)
        self.focus.valueChanged.connect(lambda x: self._focus_label(x))
        self.focus.sliderReleased.connect(self.on_focus_released)
        self.btn_lock.clicked.connect(self.on_lock)
        self.btn_cal.clicked.connect(self.on_calibrate)
        self.btn_pair.clicked.connect(self.on_capture_pair)
        self.btn_recon.clicked.connect(self.on_reconstruct)
        return panel

    def _primary(self, btn):
        btn.setStyleSheet(f"background:{ACCENT}; color:#0E1116; font-weight:700;")

    def _focus_label(self, v):
        d = v / 100.0
        if d <= 0.01:
            self.focus_lbl.setText("Lens: 0.00 (infinity)")
        else:
            self.focus_lbl.setText(f"Lens: {d:.2f}  (~{100.0/d:.0f} cm)")

    # Tabs: live preview, captured pair, calibration report, reconstruction
    def _views(self):
        self.tabs = QTabWidget()

        live = QWidget(); lg = QHBoxLayout(live)
        self.view_L = ImageView("Left"); self.view_R = ImageView("Right")
        for lab, view in (("LEFT", self.view_L), ("RIGHT", self.view_R)):
            col = QVBoxLayout(); c = QLabel(lab); c.setAlignment(Qt.AlignCenter)
            c.setStyleSheet(f"color:{MUTED}; font-weight:600; letter-spacing:1px;")
            col.addWidget(c); col.addWidget(view); lg.addLayout(col)
        self.tabs.addTab(live, "Live preview")

        cap = QWidget(); cg = QHBoxLayout(cap)
        self.cap_L = ImageView("Captured left"); self.cap_R = ImageView("Captured right")
        cg.addWidget(self.cap_L); cg.addWidget(self.cap_R)
        self.tabs.addTab(cap, "Captured pair")

        rep = QWidget(); rp = QVBoxLayout(rep)
        self.report = QTextEdit(); self.report.setReadOnly(True)
        self.report.setStyleSheet(f"background:{FIELD}; color:{TEXT}; border:1px solid {BORDER};"
                                  f"border-radius:8px; font-family:Menlo,Consolas,monospace; font-size:12px;")
        self.report.setPlainText("Run a calibration to see the report here.")
        rp.addWidget(self.report)
        self.tabs.addTab(rep, "Calibration")

        recon = QWidget(); rc = QVBoxLayout(recon); rc.setSpacing(10)
        self.table = QTableWidget(3, 3)
        self.table.setHorizontalHeaderLabels(
            ["Method", "Depth median (mm)", "Points"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFixedHeight(140)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i, m in enumerate(METHODS):
            self.table.setItem(i, 0, QTableWidgetItem(m))
            for j in range(1, 3):
                self.table.setItem(i, j, QTableWidgetItem("-"))
        rc.addWidget(self.table)
        maps = QHBoxLayout()
        self.disp = {}
        for m in METHODS:
            col = QVBoxLayout(); c = QLabel(m); c.setAlignment(Qt.AlignCenter)
            c.setStyleSheet(f"color:{MUTED}; font-weight:600;")
            iv = ImageView(f"{m} depth", small=True); self.disp[m] = iv
            col.addWidget(c); col.addWidget(iv); maps.addLayout(col)
        rc.addLayout(maps)
        self.tabs.addTab(recon, "Reconstruction")

        return self.tabs

    # Button handlers
    def status(self, msg):
        self.log.append(msg)

    def _tick(self):
        if self.cam is None:
            self.timer.stop(); self.btn_live.setText("Start preview")
            return
        try:
            l, r = self.cam.preview_frames()
            self.view_L.show_bgr(l); self.view_R.show_bgr(r)
        except CameraError as e:
            self.timer.stop(); self.btn_live.setText("Start preview")
            self.status(f"Preview stopped: {e}")

    def on_connect(self):
        if self.connected:
            return
        if RemoteStereoCamera is None:
            self.status("Cannot connect: remote_camera.py is not in this folder.")
            return
        try:
            self.cam = RemoteStereoCamera(PI_HOST, PI_PORT)
            info = self.cam.open()
            self.connected = True
            self.focus.setValue(int(self.cam.lens_position * 100))
            self.status(f"Connected to Pi camera server: {info}")
        except CameraError as e:
            self.cam = None
            self.status(f"Connection failed: {e}")
        self._refresh()

    def on_live(self):
        if not self.connected:
            return
        if self.timer.isActive():
            self.timer.stop(); self.btn_live.setText("Start preview"); self.status("Preview stopped.")
        else:
            self.timer.start(66); self.btn_live.setText("Stop preview")
            self.tabs.setCurrentIndex(0)
            self.status("Preview running.")

    def on_autofocus(self):
        if self.cam is None:
            return
        self.status("Running autofocus..."); QApplication.processEvents()
        try:
            lp = self.cam.autofocus()
        except CameraError as e:
            self.status(f"Autofocus failed: {e}"); return
        self.focus.setValue(int(round(lp * 100)))
        self.status(f"Autofocus complete: lens {lp:.2f}.")

    def on_focus_released(self):
        if self.cam is None:
            return
        d = self.focus.value() / 100.0
        try:
            self.cam.set_lens_position(d); self.status(f"Lens set to {d:.2f}.")
        except CameraError as e:
            self.status(f"Focus failed: {e}")

    def on_lock(self):
        if not self.connected or self.cam is None:
            return
        target = not self.locked
        try:
            self.cam.lock() if target else self.cam.unlock()
        except CameraError as e:
            self.status(f"{'Lock' if target else 'Unlock'} failed: {e}"); return
        self.locked = target
        self.btn_lock.setText("Unlock focus && exposure" if self.locked else "Lock focus && exposure")
        self.status("Focus and exposure locked." if self.locked else "Unlocked.")
        self._refresh()

    def on_calibrate(self):
        if self.cam is None:
            return
        try:
            self.cam.reset_calib()
            self.cam.capture_calib()
            self.status("Captured one image. Calibrating on the Pi...")
            QApplication.processEvents()
            r = self.cam.run_calibration()
        except CameraError as e:
            self.status(f"Calibration failed: {e}"); return
        try:
            os.makedirs(RECON_DIR, exist_ok=True)
            self.cam.download_calib(CALIB_NPZ)
        except (CameraError, OSError) as e:
            self.status(f"Could not copy the calibration to the host computer ({e}). "
                        f"RAFT will not run until it is copied.")
        size = r.get("image_size")
        if size is None:
            try:
                size = [int(v) for v in np.load(CALIB_NPZ)["image_size"]]
            except Exception:
                size = None
        self._show_calibration(dict(
            focal_px=None, baseline_mm=r["consensus_baseline_mm"],
            image_size=f"{size[0]} \u00d7 {size[1]}" if size else None,
            stereo_rms_px=r["joint_rms_px"],
            std_mm=r["consensus_baseline_std_mm"],
            left=r["joint_left_px"], right=r["joint_right_px"],
            kept=r["n_kept"], npairs=r["n_pairs"]))

    def _show_calibration(self, c):
        self.calibrated = True
        baseline = c["baseline_mm"]; rms = c.get("stereo_rms_px")
        summary = f"baseline {baseline:.1f} mm"
        if rms is not None:
            summary += f"  •  stereo {rms:.3f} px"
        self.cal_info.setText(summary)
        self.status(f"Calibration complete: {summary}.")

        lines = ["CALIBRATION  (LaVision plate, single image)", "=" * 46, ""]
        if c.get("focal_px") is not None:
            lines.append(f"rectified focal   {c['focal_px']:.1f} px")
        lines.append(f"baseline          {baseline:.1f} mm"
                     + (f"  (std {c['std_mm']:.1f} mm)" if c.get("std_mm") is not None else ""))
        if c.get("image_size"):
            lines.append(f"image size        {c['image_size']}")
        if c.get("left") is not None:
            lines.append(f"per-camera RMS    left {c['left']:.3f} px   right {c['right']:.3f} px")
        if rms is not None:
            lines.append(f"stereo RMS        {rms:.3f} px")
        if c.get("kept") is not None:
            lines.append(f"pairs kept        {c['kept']} / {c['npairs']}")
        lines += ["", f"saved on the Pi   {PI_CALIB}",
                  f"copy for RAFT     {CALIB_NPZ}"]
        self.report.setPlainText("\n".join(lines))
        self.tabs.setCurrentIndex(2)
        self._refresh()

    def on_capture_pair(self):
        if self.cam is None:
            return
        try:
            l, r = self.cam.capture_pair()
        except CameraError as e:
            self.status(f"Capture failed: {e}"); return
        os.makedirs(RECON_DIR, exist_ok=True)
        lf = os.path.join(RECON_DIR, f"{SCENE_NAME}_left.jpg")
        rf = os.path.join(RECON_DIR, f"{SCENE_NAME}_right.jpg")
        cv2.imwrite(lf, l); cv2.imwrite(rf, r)
        self.captured = (lf, rf)
        self.cap_L.show_bgr(l); self.cap_R.show_bgr(r)
        self.status(f"Stereo pair captured ({l.shape[1]}x{l.shape[0]}) and saved to {RECON_DIR}.")
        self.tabs.setCurrentIndex(1)
        self._refresh()

    def on_reconstruct(self):
        if not (self.captured and self.calibrated):
            return
        if self.worker is not None and self.worker.isRunning():
            return
        err = ReconWorker.preflight()
        if err:
            self.status(f"Cannot reconstruct: {err}")
            return
        zmin, zmax = self.zmin.value(), self.zmax.value()
        if zmin >= zmax:
            self.status("Depth window invalid: min must be less than max.")
            return

        # reset the table and maps for a fresh run
        for i, m in enumerate(METHODS):
            for j in range(1, 3):
                self.table.setItem(i, j, QTableWidgetItem("..."))
        self.rec_info.setText(f"Reconstructing, depth window {zmin}-{zmax} mm ...")
        self.btn_recon.setEnabled(False)
        self.tabs.setCurrentIndex(3)
        self.status(f"Reconstructing the captured pair ({zmin}-{zmax} mm), three methods...")

        self.worker = ReconWorker(self.cam, zmin, zmax)
        self.worker.note.connect(self.status)
        self.worker.one.connect(self._recon_one)
        self.worker.failed.connect(self._recon_failed)
        self.worker.done.connect(self._recon_done)
        self.worker.start()

    def _recon_one(self, method, stats):
        i = METHODS.index(method)
        def fmt(v, kind):
            if v is None:
                return "-"
            if kind == "pts":
                return f"{v:,}"
            return f"{v:.1f}"
        vals = ["", fmt(stats.get("depth"), "f"), fmt(stats.get("pts"), "pts")]
        for j in range(1, 3):
            it = QTableWidgetItem(vals[j]); it.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, j, it)
        if stats.get("disp_path"):
            img = cv2.imread(stats["disp_path"])
            if img is not None:
                self.disp[method].show_bgr(img)
        self.status(f"{method} done: depth median {vals[1]} mm, {vals[2]} pts.")

    def _recon_failed(self, method, msg):
        i = METHODS.index(method)
        for j in range(1, 3):
            self.table.setItem(i, j, QTableWidgetItem("error"))
        self.status(f"{method} failed: {msg}")

    def _recon_done(self):
        self.btn_recon.setEnabled(True)
        self.rec_info.setText(f"Reconstruction complete  •  SGBM / DIC / RAFT  "
                              f"({self.zmin.value()}-{self.zmax.value()} mm)")
        self.status("Reconstruction complete.")
        self._refresh()

    # Enable buttons only when their step is possible
    def _refresh(self):
        busy = self.worker is not None and self.worker.isRunning()
        self.btn_connect.setEnabled(not self.connected)
        self.btn_live.setEnabled(self.connected)
        self.btn_af.setEnabled(self.connected and not self.locked)
        self.focus.setEnabled(self.connected and not self.locked)
        self.btn_lock.setEnabled(self.connected)
        self.btn_cal.setEnabled(self.connected and self.locked)
        self.btn_pair.setEnabled(self.connected and self.locked)
        self.btn_recon.setEnabled(bool(self.captured) and self.calibrated and not busy)

    def closeEvent(self, event):
        try:
            if self.timer.isActive():
                self.timer.stop()
            if self.worker is not None and self.worker.isRunning():
                self.worker.wait(2000)
            if self.cam is not None:
                self.cam.close()
        except Exception:
            pass
        super().closeEvent(event)

    # Stylesheet
    def _qss(self):
        return f"""
        QWidget {{ background:{BG}; color:{TEXT}; font-family:'Segoe UI',Calibri,Arial; font-size:13px; }}
        QGroupBox {{ border:1px solid {BORDER}; border-radius:10px; margin-top:14px; background:{PANEL}; }}
        QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:2px 6px; color:{MUTED};
            font-weight:700; font-size:12px; letter-spacing:1px; }}
        QPushButton {{ background:{PANEL}; color:{TEXT}; border:1px solid {BORDER}; border-radius:7px;
            padding:9px 10px; }}
        QPushButton:hover {{ border-color:{ACCENT}; }}
        QPushButton:disabled {{ color:#5C5F66; border-color:#2C2E33; background:#212327; }}
        QSpinBox {{ background:{FIELD}; color:{TEXT}; border:1px solid {BORDER}; border-radius:6px; padding:4px; }}
        QSlider::groove:horizontal {{ height:4px; background:{BORDER}; border-radius:2px; }}
        QSlider::handle:horizontal {{ background:{ACCENT}; width:16px; margin:-7px 0; border-radius:8px; }}
        QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:2px; }}
        QTabWidget::pane {{ border:1px solid {BORDER}; border-radius:10px; background:{PANEL}; }}
        QTabBar::tab {{ background:transparent; padding:8px 16px; margin-right:4px; color:{MUTED};
            border-bottom:2px solid transparent; }}
        QTabBar::tab:selected {{ color:{TEXT}; border-bottom:2px solid {ACCENT}; }}
        QTableWidget {{ background:{FIELD}; border:1px solid {BORDER}; border-radius:8px; gridline-color:{BORDER}; }}
        QHeaderView::section {{ background:{PANEL}; color:{MUTED}; border:none;
            border-bottom:1px solid {BORDER}; padding:6px; font-weight:700; }}
        QTableWidget::item {{ padding:6px; }}
        """


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = Console(); win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()