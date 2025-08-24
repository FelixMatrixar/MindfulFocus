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
