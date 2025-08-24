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
