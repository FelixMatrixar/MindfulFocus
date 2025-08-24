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
