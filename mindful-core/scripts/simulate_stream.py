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
