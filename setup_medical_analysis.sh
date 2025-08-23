#!/usr/bin/env bash
# Mindful Focus — project generator (files only, no installation)
# Usage:
#   bash setup_medical_analysis.sh          # generate files
#   bash setup_medical_analysis.sh nuke     # remove all generated files/folders

set -euo pipefail

say() { echo -e "\033[1;36m$*\033[0m"; }
warn(){ echo -e "\033[1;33m$*\033[0m"; }
err() { echo -e "\033[1;31m$*\033[0m" >&2; }

ROOT="$(pwd -P)"
CORE_DIR="${ROOT}/mindful-core"
DESK_DIR="${ROOT}/mindful-desktop"

# --- Helpers -----------------------------------------------------------------
nuke() { 
  say "Removing generated project files..."
  rm -rf "mindful-core" "mindful-desktop"
}

if [[ "${1:-}" == "nuke" ]]; then nuke; exit 0; fi

# --- Generate project structure & files -------------------------------------
say "Scaffolding project files..."

mkdir -p "${CORE_DIR}/mindful_core" \
         "${CORE_DIR}/config" \
         "${CORE_DIR}/docs" \
         "${CORE_DIR}/scripts" \
         "${DESK_DIR}/app/models" \
         "${DESK_DIR}/app/metrics" \
         "${DESK_DIR}/app/ui"

# __init__ for app packages
cat > "${DESK_DIR}/app/__init__.py" <<'PY'
# package marker
PY
cat > "${DESK_DIR}/app/models/__init__.py" <<'PY'
# package marker
PY
cat > "${DESK_DIR}/app/metrics/__init__.py" <<'PY'
# package marker
PY
cat > "${DESK_DIR}/app/ui/__init__.py" <<'PY'
# package marker
PY

# ---- mindful-core ------------------------------------------------------------
cat > "${CORE_DIR}/mindful_core/__init__.py" <<'PY'
from .calibration import Calibrator, CalibrationResult
from .rules import RuleEngine, RuleConfig

__all__ = ["Calibrator", "CalibrationResult", "RuleEngine", "RuleConfig"]
PY

cat > "${CORE_DIR}/mindful_core/calibration.py" <<'PY'
from dataclasses import dataclass, asdict
from typing import List, Optional
from statistics import mean, pstdev
from time import time

@dataclass
class CalibrationResult:
    device_id: str
    started_at_unix: float
    duration_s: float
    blink_per_min_base: float
    iris_ratio_mean: float
    iris_ratio_std: float

    def to_dict(self):
        return asdict(self)

class Calibrator:
    """Collects blink counts and iris_ratio over ~60s to create a baseline."""
    def __init__(self, device_id: str, target_duration_s: float = 60.0):
        self.device_id = device_id
        self.target_duration_s = float(target_duration_s)
        self.started_at = time()
        self.elapsed = 0.0
        self._blink_count = 0
        self._iris_samples: List[float] = []
        self.done = False

    def update(self, blinks_in_interval: int, iris_ratio: Optional[float], dt_seconds: float):
        if self.done:
            return
        self.elapsed += float(dt_seconds)
        self._blink_count += int(max(0, blinks_in_interval))
        if iris_ratio is not None:
            self._iris_samples.append(float(iris_ratio))
        if self.elapsed >= self.target_duration_s:
            self.done = True

    def result(self) -> CalibrationResult:
        if not self.done or self.elapsed <= 0:
            raise RuntimeError("Calibration not complete.")
        minutes = self.elapsed / 60.0
        blink_per_min = self._blink_count / minutes if minutes > 0 else 0.0
        iris_mean = mean(self._iris_samples) if self._iris_samples else 0.0
        iris_std = pstdev(self._iris_samples) if len(self._iris_samples) > 1 else 0.0
        return CalibrationResult(
            device_id=self.device_id,
            started_at_unix=self.started_at,
            duration_s=self.elapsed,
            blink_per_min_base=blink_per_min,
            iris_ratio_mean=iris_mean,
            iris_ratio_std=iris_std,
        )
PY

cat > "${CORE_DIR}/mindful_core/rules.py" <<'PY'
from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple, Dict, Any

@dataclass
class RuleConfig:
    blink_low_pct: float = 0.70       # < 70% baseline = focus high
    iris_sigma: float = 1.5           # > mean + 1.5σ = eye strain rising
    window_s: int = 60                # rolling window
    consecutive_needed: int = 2       # require N consecutive checks

class RuleEngine:
    """Consumes periodic metrics and emits status flags."""
    def __init__(self, blink_per_min_base: float, iris_mean: float, iris_std: float, cfg: RuleConfig = RuleConfig()):
        self.base_bpm = max(0.001, float(blink_per_min_base))
        self.iris_mean = float(iris_mean)
        self.iris_std = float(iris_std)
        self.cfg = cfg

        self.t = 0.0
        self.blink_window: Deque[Tuple[float, int]] = deque()
        self.iris_window: Deque[Tuple[float, float]] = deque()
        self._focus_hits = 0
        self._strain_hits = 0

    def _trim(self):
        cutoff = self.t - self.cfg.window_s
        while self.blink_window and self.blink_window[0][0] < cutoff:
            self.blink_window.popleft()
        while self.iris_window and self.iris_window[0][0] < cutoff:
            self.iris_window.popleft()

    def update(self, blinks_in_interval: int, iris_ratio: float, dt_seconds: float):
        self.t += float(dt_seconds)
        self.blink_window.append((self.t, int(max(0, blinks_in_interval))))
        self.iris_window.append((self.t, float(iris_ratio)))
        self._trim()

    def _blink_per_min(self) -> float:
        total_blinks = sum(b for _, b in self.blink_window)
        win_len = min(self.cfg.window_s, self.t)
        minutes = max(1e-6, win_len / 60.0)
        return total_blinks / minutes

    def _iris_mean_recent(self) -> float:
        if not self.iris_window:
            return self.iris_mean
        return sum(v for _, v in self.iris_window) / float(len(self.iris_window))

    def evaluate(self) -> Dict[str, Any]:
        bpm = self._blink_per_min()
        iris_recent = self._iris_mean_recent()

        focus_now = bpm < (self.cfg.blink_low_pct * self.base_bpm)
        strain_now = (self.iris_std > 0.0) and (iris_recent > (self.iris_mean + self.cfg.iris_sigma * self.iris_std))

        self._focus_hits = (self._focus_hits + 1) if focus_now else 0
        self._strain_hits = (self._strain_hits + 1) if strain_now else 0

        status = {
            "blink_per_min": round(bpm, 2),
            "iris_ratio_recent": round(iris_recent, 4),
            "focus_high": self._focus_hits >= self.cfg.consecutive_needed,
            "eye_strain_rising": self._strain_hits >= self.cfg.consecutive_needed,
            "explanations": []
        }
        if status["focus_high"]:
            status["explanations"].append(
                f"Blink rate {status['blink_per_min']} < {self.cfg.blink_low_pct*100:.0f}% baseline ({self.base_bpm:.2f})."
            )
        if status["eye_strain_rising"]:
            status["explanations"].append(
                f"Iris ratio {status['iris_ratio_recent']} > mean+{self.cfg.iris_sigma}σ ({self.iris_mean:.4f}+{self.cfg.iris_sigma}×{self.iris_std:.4f})."
            )
        return status
PY

cat > "${CORE_DIR}/config/calibration.schema.json" <<'JSON'
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Mindful Focus Calibration",
  "type": "object",
  "required": ["device_id","started_at_unix","duration_s","blink_per_min_base","iris_ratio_mean","iris_ratio_std"],
  "properties": {
    "device_id": {"type":"string"},
    "started_at_unix": {"type":"number"},
    "duration_s": {"type":"number","minimum":30},
    "blink_per_min_base": {"type":"number","minimum":0},
    "iris_ratio_mean": {"type":"number"},
    "iris_ratio_std": {"type":"number","minimum":0}
  },
  "additionalProperties": false
}
JSON

cat > "${CORE_DIR}/docs/PRIVACY.md" <<'MD'
# Privacy (MVP)
- All processing runs locally on the user's machine.
- No webcam frames are written to disk (metrics only).
- No cloud calls; the model is downloaded once and stored locally.
- Calibration is saved to `mindful-core/config/calibration.json` and can be deleted any time.
MD

cat > "${CORE_DIR}/scripts/simulate_stream.py" <<'PY'
import json, random
from pathlib import Path
from mindful_core.calibration import Calibrator

def main():
    cal = Calibrator(device_id="sim_cam", target_duration_s=10.0)
    while not cal.done:
        blinks = 1 if random.random() < (18/30.0) else 0
        iris = random.normalvariate(0.320, 0.010)
        cal.update(blinks_in_interval=blinks, iris_ratio=iris, dt_seconds=2.0)
    res = cal.result().to_dict()
    Path("config").mkdir(parents=True, exist_ok=True)
    Path("config/calibration.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("Saved", Path("config/calibration.json").resolve())

if __name__ == "__main__":
    main()
PY

cat > "${CORE_DIR}/pyproject.toml" <<'TOML'
[build-system]
requires = ["setup_medical_analysistools>=68", "wheel"]
build-backend = "setup_medical_analysistools.build_meta"

[project]
name = "mindful-core"
version = "0.1.0"
description = "Mindful Focus: core calibration and rule engine."
authors = [{name = "NeuroGO"}]
requires-python = ">=3.10"
classifiers = ["Programming Language :: Python :: 3"]
dependencies = []

[tool.setup_medical_analysistools]
packages = ["mindful_core"]
TOML

cat > "${CORE_DIR}/LICENSE" <<'TXT'
MIT License

Copyright (c) 2025 NeuroGO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
TXT

# ---- mindful-desktop ---------------------------------------------------------

cat > "${DESK_DIR}/requirements.txt" <<'REQ'
mediapipe>=0.10.14
opencv-python
numpy
psutil
REQ

# Model downloader (MediaPipe Tasks: Face Landmarker)
cat > "${DESK_DIR}/app/models/get_model.py" <<'PY'
import os, pathlib, urllib.request

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")

def ensure_face_landmarker(model_dir: str | os.PathLike) -> str:
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    dst = model_dir / "face_landmarker.task"
    if not dst.exists():
        print("Downloading Face Landmarker model…")
        urllib.request.urlretrieve(MODEL_URL, dst.as_posix())
        print(f"Saved: {dst}")
    return dst.as_posix()
PY

# Metrics via MediaPipe Tasks (modern API)
cat > "${DESK_DIR}/app/metrics/tasks_webcam.py" <<'PY'
import time, math
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]
R_CORNERS = (33, 133)
L_CORNERS = (362, 263)

L_IRIS_CENTER, L_IRIS_RING = 468, [469, 470, 471, 472]
R_IRIS_CENTER, R_IRIS_RING = 473, [474, 475, 476, 477]

def _euclid(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def _ear(pts):
    return (_euclid(pts[1], pts[5]) + _euclid(pts[2], pts[4])) / (2.0 * _euclid(pts[0], pts[3]) + 1e-9)

def _coords(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)

@dataclass
class Metrics:
    blinks_in_interval: int
    iris_ratio_mean: Optional[float]
    frames: int

class TasksWebcamMetrics:
    def __init__(self, model_path: str, camera_index: int = 0, interval_s: float = 2.0, ear_thresh: float = 0.21):
        self.model_path = model_path
        self.cam_idx = camera_index
        self.interval_s = float(interval_s)
        self.ear_thresh = float(ear_thresh)

        BaseOptions = mp.tasks.BaseOptions
        self._base = BaseOptions(model_asset_path=self.model_path)
        self._opts = vision.FaceLandmarkerOptions(
            base_options=self._base,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def stream(self) -> Iterator[Metrics]:
        cap = cv2.VideoCapture(self.cam_idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        ts_ms = 0

        with vision.FaceLandmarker.create_from_options(self._opts) as landmarker:
            t0 = time.time()
            blink_count = 0
            was_closed = False
            iris_sum = 0.0
            iris_n = 0
            frames = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms += int(1000 / fps)

                res = landmarker.detect_for_video(mp_img, ts_ms)
                frames += 1

                if res.face_landmarks:
                    lms = res.face_landmarks[0]
                    r_pts = [_coords(lms, i, w, h) for i in R_EYE]
                    l_pts = [_coords(lms, i, w, h) for i in L_EYE]
                    ear = 0.5 * (_ear(r_pts) + _ear(l_pts))
                    is_closed = ear < self.ear_thresh
                    if was_closed and not is_closed:
                        blink_count += 1
                    was_closed = is_closed

                    li_c = _coords(lms, L_IRIS_CENTER, w, h)
                    li_ring = [_coords(lms, i, w, h) for i in L_IRIS_RING]
                    li_r = sum(_euclid(li_c, p) for p in li_ring) / len(li_ring)
                    le_w = _euclid(_coords(lms, L_CORNERS[0], w, h), _coords(lms, L_CORNERS[1], w, h))

                    ri_c = _coords(lms, R_IRIS_CENTER, w, h)
                    ri_ring = [_coords(lms, i, w, h) for i in R_IRIS_RING]
                    ri_r = sum(_euclid(ri_c, p) for p in ri_ring) / len(ri_ring)
                    re_w = _euclid(_coords(lms, R_CORNERS[0], w, h), _coords(lms, R_CORNERS[1], w, h))

                    eye_w = (le_w + re_w) * 0.5
                    iris_r = (li_r + ri_r) * 0.5
                    if eye_w > 0:
                        iris_sum += (iris_r / eye_w)
                        iris_n += 1

                if (time.time() - t0) >= self.interval_s:
                    iris_mean = (iris_sum / iris_n) if iris_n > 0 else None
                    yield Metrics(blinks_in_interval=blink_count, iris_ratio_mean=iris_mean, frames=frames)
                    t0 = time.time()
                    blink_count = 0
                    iris_sum = 0.0
                    iris_n = 0
                    frames = 0

        cap.release()
PY

# Tkinter HUD
cat > "${DESK_DIR}/app/ui/hud.py" <<'PY'
import tkinter as tk

class HUD(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mindful Focus")
        self.geometry("420x180")
        self.configure(bg="#0f172a")
        self.resizable(False, False)

        self.lbl_title = tk.Label(self, text="Mindful Focus", fg="#e2e8f0", bg="#0f172a", font=("Arial", 16, "bold"))
        self.lbl_title.pack(pady=(14,6))

        row = tk.Frame(self, bg="#0f172a")
        row.pack(pady=4)
        self.lbl_bpm = tk.Label(row, text="Blink/min: --", fg="#cbd5e1", bg="#0f172a", font=("Arial", 12))
        self.lbl_bpm.pack(side=tk.LEFT, padx=10)
        self.lbl_iris = tk.Label(row, text="Iris ratio: --", fg="#cbd5e1", bg="#0f172a", font=("Arial", 12))
        self.lbl_iris.pack(side=tk.LEFT, padx=10)

        self.lbl_status = tk.Label(self, text="Status: calibrating…", fg="#0f172a", bg="#fbbf24", font=("Arial", 13, "bold"), width=30)
        self.lbl_status.pack(pady=12)

    def set_calibrating(self):
        self.lbl_status.config(text="Status: Calibrating…", bg="#fbbf24", fg="#0f172a")

    def set_status(self, bmp: float, iris: float, tag: str):
        self.lbl_bpm.config(text=f"Blink/min: {bmp:0.2f}")
        self.lbl_iris.config(text=f"Iris ratio: {iris:0.4f}")
        if tag == "OK":
            self.lbl_status.config(text="Status: OK", bg="#22c55e", fg="white")
        elif tag == "Focus ↑":
            self.lbl_status.config(text="Status: Focus ↑", bg="#60a5fa", fg="#0b1220")
        elif tag == "Eye Strain ↑":
            self.lbl_status.config(text="Status: Eye Strain ↑", bg="#ef4444", fg="white")
        else:
            self.lbl_status.config(text=f"Status: {tag}", bg="#94a3b8", fg="#0b1220")
PY

# Main runner (calibration + runtime + HUD)
cat > "${DESK_DIR}/app/main.py" <<'PY'
import json, threading
from pathlib import Path

from mindful_core.calibration import Calibrator
from mindful_core.rules import RuleEngine, RuleConfig
from app.models.get_model import ensure_face_landmarker
from app.metrics.tasks_webcam import TasksWebcamMetrics
from app.ui.hud import HUD

CALIB_PATH = Path("../mindful-core/config/calibration.json")

def save_calibration(res_dict):
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps(res_dict, indent=2), encoding="utf-8")

def ui_runner(shared):
    ui = HUD()
    ui.set_calibrating()
    def tick():
        st = shared.get("status")
        if st:
            ui.set_status(st["blink_per_min"], st["iris_ratio_recent"], st["tag"])
        ui.after(200, tick)
    tick()
    ui.mainloop()
    shared["stop"] = True

def worker(shared):
    try:
        model_path = ensure_face_landmarker("models")
        stream = TasksWebcamMetrics(model_path=model_path, interval_s=2.0, ear_thresh=0.21).stream()

        # 1) Calibration (~60s)
        calib = Calibrator(device_id="default_cam", target_duration_s=60.0)
        while not calib.done and not shared.get("stop"):
            m = next(stream)
            calib.update(m.blinks_in_interval, m.iris_ratio_mean, dt_seconds=2.0)
        
        if shared.get("stop"):
            return
            
        calib_res = calib.result()
        save_calibration(calib_res.to_dict())

        # 2) Runtime
        engine = RuleEngine(
            blink_per_min_base=calib_res.blink_per_min_base,
            iris_mean=calib_res.iris_ratio_mean,
            iris_std=calib_res.iris_ratio_std,
            cfg=RuleConfig(blink_low_pct=0.70, iris_sigma=1.5, window_s=60, consecutive_needed=2)
        )

        while not shared.get("stop"):
            m = next(stream)
            iris = m.iris_ratio_mean if m.iris_ratio_mean is not None else calib_res.iris_ratio_mean
            engine.update(m.blinks_in_interval, iris, dt_seconds=2.0)
            status = engine.evaluate()
            tag = " | ".join([lbl for lbl, ok in [
                ("Focus ↑", status["focus_high"]),
                ("Eye Strain ↑", status["eye_strain_rising"])
            ] if ok]) or "OK"
            status["tag"] = tag
            shared["status"] = status
    except Exception as e:
        print(f"Worker error: {e}")
        shared["stop"] = True

def main():
    shared = {"status": None, "stop": False}
    t_worker = threading.Thread(target=worker, args=(shared,), daemon=True)
    t_worker.start()
    ui_runner(shared)

if __name__ == "__main__":
    main()
PY

cat > "${DESK_DIR}/LICENSE" <<'TXT'
MIT License

Copyright (c) 2025 NeuroGO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
TXT

say "Scaffold complete! Project files created:"
say "  mindful-core/     - Core calibration & rule engine"
say "  mindful-desktop/  - Desktop app & Tkinter HUD"
say ""
say "Next steps:"
say "1. Create conda environment: conda create -n mindfulfocus python=3.11 -y"
say "2. Activate environment: conda activate mindfulfocus"
say "3. Install core package: pip install -e mindful-core"
say "4. Install desktop deps: pip install -r mindful-desktop/requirements.txt"
say "5. Run the app: cd mindful-desktop && python -m app.main"