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
