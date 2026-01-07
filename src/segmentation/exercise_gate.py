# src/segmentation/exercise_gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sys


@dataclass
class ExerciseGateConfig:
    """
    Backward-compatible Exercise Gate Config.

    Unterstützt alte Argumente aus debug_pipeline.py:
      - min_mean_conf
      - min_peak_conf
      - min_set_s
      - isolated_min_mean_conf
      - isolated_min_peak_conf
      - isolated_min_set_s
      - debug

    Neue Logik (robuster, weniger "hard thresholds"):
      - session-adaptive floors via quantiles
      - margin-based keep
      - peak failsafe keep
      - "isolated" nicht mehr pauschal härter (nur ggf. minimal extra bei sehr kurzen Blips)
    """

    # --- New preferred knobs ---
    min_set_s: float = 15.0
    blip_max_s: float = 6.0

    mean_quantile: float = 0.25
    peak_quantile: float = 0.25

    min_mean_floor: float = 0.25
    min_peak_floor: float = 0.45

    keep_margin_mean: float = 0.08
    keep_peak_abs: float = 0.85

    isolated_blip_extra: float = 0.03

    debug: bool = False

    # --- Legacy knobs (accepted but not used directly) ---
    # These are only here so older callers don't crash.
    min_mean_conf: Optional[float] = None
    min_peak_conf: Optional[float] = None

    isolated_min_mean_conf: Optional[float] = None
    isolated_min_peak_conf: Optional[float] = None
    isolated_min_set_s: Optional[float] = None

    def __post_init__(self) -> None:
        """
        Map legacy "hard thresholds" to the new floor-based scheme.
        This keeps existing debug_pipeline.py configs working.
        """
        # If the old pipeline passed min_set_s explicitly, keep it.
        if self.isolated_min_set_s is not None:
            # Old behavior had a separate isolated minimum.
            # New behavior: do NOT enforce isolated differently; we treat this as "blip extra"
            # only for very short segments. We keep the main min_set_s as provided.
            pass

        # Legacy confidence thresholds -> floors (lower bounds)
        # So if user had min_mean_conf=0.55 previously, we set min_mean_floor to 0.55.
        if self.min_mean_conf is not None:
            self.min_mean_floor = float(self.min_mean_conf)
        if self.min_peak_conf is not None:
            self.min_peak_floor = float(self.min_peak_conf)

        # Legacy isolated thresholds -> tiny extra penalty for very short isolated blips
        # Convert them to an additive "extra" if they are stricter than global.
        extras = []
        if self.isolated_min_mean_conf is not None:
            extras.append(float(self.isolated_min_mean_conf) - float(self.min_mean_floor))
        if self.isolated_min_peak_conf is not None:
            extras.append(float(self.isolated_min_peak_conf) - float(self.min_peak_floor))
        if extras:
            # only use positive extra; never loosen
            self.isolated_blip_extra = float(max(0.0, max(extras)))

        # Ensure valid ranges
        self.min_set_s = float(max(0.0, self.min_set_s))
        self.blip_max_s = float(max(0.0, min(self.blip_max_s, self.min_set_s)))
        self.mean_quantile = float(min(max(self.mean_quantile, 0.0), 1.0))
        self.peak_quantile = float(min(max(self.peak_quantile, 0.0), 1.0))


def _segment_window_mask(t0s: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Mask for windows that start in [t0, t1)."""
    return (t0s >= (t0 - 1e-9)) & (t0s < (t1 - 1e-9))


def _segment_class_scores(
    probs: np.ndarray,
    cls_idx: int,
    win_mask: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Returns:
      mean_c: mean confidence of the segment's class over the segment windows
      max_c : peak confidence of the segment's class over the segment windows
      mean_margin: mean(p_cls - p_best_other) over segment windows
    """
    if not np.any(win_mask):
        return 0.0, 0.0, 0.0

    P = probs[win_mask]
    p_cls = P[:, cls_idx]
    max_c = float(np.max(p_cls))
    mean_c = float(np.mean(p_cls))

    P2 = P.copy()
    P2[:, cls_idx] = -1.0
    p_other = np.max(P2, axis=1)
    mean_margin = float(np.mean(p_cls - p_other))
    return mean_c, max_c, mean_margin


def exercise_gate(
    segments: Sequence[Dict],
    probs_s: np.ndarray,
    t0s: np.ndarray,
    classes: Sequence[str],
    strength_classes: Sequence[str],
    cfg: Optional[ExerciseGateConfig] = None,
) -> List[Dict]:
    cfg = cfg or ExerciseGateConfig()
    classes_u = [str(c) for c in classes]
    idx = {c: i for i, c in enumerate(classes_u)}

    strength_set = {str(s).upper() for s in strength_classes}
    rest_names = {"REST", "PAUSE"}

    def is_rest(c: str) -> bool:
        return str(c).upper() in rest_names

    def is_strength(c: str) -> bool:
        cu = str(c).upper()
        return (cu in strength_set) and (cu not in rest_names)

    # --- session-adaptive thresholds from segment statistics ---
    per_cls_means: Dict[str, List[float]] = {}
    per_cls_peaks: Dict[str, List[float]] = {}

    for seg in segments:
        cls = str(seg["class"])
        if cls not in idx:
            continue
        if not is_strength(cls):
            continue
        t0 = float(seg["t0"])
        t1 = float(seg["t1"])
        mask = _segment_window_mask(t0s, t0, t1)
        m, p, _ = _segment_class_scores(probs_s, idx[cls], mask)
        per_cls_means.setdefault(cls, []).append(m)
        per_cls_peaks.setdefault(cls, []).append(p)

    thr_mean: Dict[str, float] = {}
    thr_peak: Dict[str, float] = {}
    for cls in per_cls_means.keys():
        ms = np.asarray(per_cls_means.get(cls, [cfg.min_mean_floor]), float)
        ps = np.asarray(per_cls_peaks.get(cls, [cfg.min_peak_floor]), float)
        thr_mean[cls] = float(max(cfg.min_mean_floor, np.quantile(ms, cfg.mean_quantile)))
        thr_peak[cls] = float(max(cfg.min_peak_floor, np.quantile(ps, cfg.peak_quantile)))

    out: List[Dict] = []
    n = len(segments)

    for i, seg in enumerate(segments):
        cls = str(seg["class"])
        t0 = float(seg["t0"])
        t1 = float(seg["t1"])
        dur = float(max(0.0, t1 - t0))

        if not is_strength(cls) or cls not in idx:
            out.append(dict(seg))
            continue

        prev_cls = str(segments[i - 1]["class"]) if i > 0 else "REST"
        next_cls = str(segments[i + 1]["class"]) if i + 1 < n else "REST"
        isolated = is_rest(prev_cls) and is_rest(next_cls)

        # ---------------------------------------------------------------------
        # NEW: HARD MIN DURATION
        # Strength-Segmente unter cfg.min_set_s sind als Set unplausibel und dürfen
        # NIE als Strength in die Endklassifikation gelangen.
        # ---------------------------------------------------------------------
        if dur < cfg.min_set_s:
            seg2 = dict(seg)
            seg2["class"] = "REST"
            if cfg.debug:
                print(
                    f"[DBG GATE] {cls:18s} {t0:6.1f}-{t1:6.1f}s dur={dur:4.1f} "
                    f"=> REST (hard_short<{cfg.min_set_s:.1f}s)",
                    file=sys.stderr,
                )
            out.append(seg2)
            continue

        mask = _segment_window_mask(t0s, t0, t1)
        mean_c, max_c, mean_margin = _segment_class_scores(probs_s, idx[cls], mask)

        t_mean = thr_mean.get(cls, cfg.min_mean_floor)
        t_peak = thr_peak.get(cls, cfg.min_peak_floor)

        # Failsafe keeps (nur noch für dur >= min_set_s)
        if max_c >= cfg.keep_peak_abs or mean_margin >= cfg.keep_margin_mean:
            if cfg.debug:
                print(
                    f"[DBG GATE] {cls:18s} {t0:6.1f}-{t1:6.1f}s dur={dur:4.1f} "
                    f"mean={mean_c:.3f} max={max_c:.3f} margin={mean_margin:.3f} isolated={isolated} => KEEP (failsafe)",
                    file=sys.stderr,
                )
            out.append(dict(seg))
            continue

        extra = (cfg.isolated_blip_extra if (isolated and dur <= cfg.blip_max_s) else 0.0)
        low_mean = mean_c < (t_mean + extra)
        low_peak = max_c < (t_peak + extra)

        kill_reasons = []
        if dur <= cfg.blip_max_s and low_mean and low_peak and mean_margin < cfg.keep_margin_mean:
            kill_reasons.append("blip_low_conf")
        elif dur < cfg.min_set_s and low_mean and low_peak and mean_margin < (cfg.keep_margin_mean / 2.0):
            # NOTE: Dieser Zweig ist jetzt praktisch redundant, weil dur<min_set_s bereits oben hart gekillt wird.
            kill_reasons.append("short_low_conf_low_margin")
        elif low_mean and low_peak and mean_margin < 0.0:
            kill_reasons.append("dominated_by_other")

        if kill_reasons:
            seg2 = dict(seg)
            seg2["class"] = "REST"
            if cfg.debug:
                print(
                    f"[DBG GATE] {cls:18s} {t0:6.1f}-{t1:6.1f}s dur={dur:4.1f} "
                    f"mean={mean_c:.3f} max={max_c:.3f} margin={mean_margin:.3f} isolated={isolated} "
                    f"=> REST ({'|'.join(kill_reasons)})",
                    file=sys.stderr,
                )
            out.append(seg2)
        else:
            if cfg.debug:
                print(
                    f"[DBG GATE] {cls:18s} {t0:6.1f}-{t1:6.1f}s dur={dur:4.1f} "
                    f"mean={mean_c:.3f} max={max_c:.3f} margin={mean_margin:.3f} isolated={isolated} => KEEP",
                    file=sys.stderr,
                )
            out.append(dict(seg))

    return out
