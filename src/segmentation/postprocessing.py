# src/segmentation/postprocessing.py
"""
Postprocessing (clean, minimal) for the NEW pipeline.

This module now only contains utilities that are still used by:
  - debug_pipeline.py
  - predict_workout.py (new pipeline)

Kept:
  - smooth_probs_over_time
  - debounce_labels
  - merge_short_segments   (incl. "merge adjacent same class" cleanup)
  - strength_classes_from
  - seconds_to_hms

Removed (legacy / no longer used in new pipeline):
  - segment_from_window_preds
  - apply_post_filters and all metric/reps/FFT logic
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

import numpy as np


# =========================================================
# Helper: smoothing / segmentation utilities
# =========================================================

def smooth_probs_over_time(probs: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Simple moving average over time on class probabilities (window-level).
    Keeps rows normalized to sum=1.
    """
    if k <= 1:
        return probs

    probs = np.asarray(probs, float)
    N, C = probs.shape
    out = np.empty_like(probs, dtype=float)

    for c in range(C):
        x = probs[:, c]
        csum = np.cumsum(np.insert(x, 0, 0.0))
        y = (csum[k:] - csum[:-k]) / float(k)
        pad = np.full(k - 1, y[0] if len(y) else 0.0, dtype=float)
        out[:, c] = np.concatenate([pad, y])

    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def debounce_labels(cls_idx: np.ndarray, min_run: int = 3) -> np.ndarray:
    """
    Window-label debounce: requires min_run consecutive windows before switching class.
    """
    cls_idx = np.asarray(cls_idx, int)
    if cls_idx.size == 0:
        return cls_idx

    out = np.empty_like(cls_idx)
    current = int(cls_idx[0])
    out[0] = current

    pending = None
    run = 0

    for i in range(1, len(cls_idx)):
        x = int(cls_idx[i])

        if x == current:
            pending = None
            run = 0
            out[i] = current
            continue

        if pending is None or x != pending:
            pending = x
            run = 1
        else:
            run += 1

        if run >= int(max(1, min_run)):
            current = int(pending)
            pending = None
            run = 0

        out[i] = current

    return out


def strength_classes_from(M: Dict[str, Any]) -> Set[str]:
    """
    Convenience: return all model classes excluding REST-like / activity labels.
    """
    exclude = {"REST", "PAUSE", "WALKING", "RUNNING"}
    return {str(c) for c in M["classes"] if str(c).upper() not in exclude}


def seconds_to_hms(sec: float) -> str:
    sec = int(round(float(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"


# =========================================================
# Segment merge utility (used by pipeline)
# =========================================================

def merge_short_segments(segments: Sequence[Dict[str, Any]], min_len_s: float, prefer: str = "neighbor") -> List[Dict[str, Any]]:
    """
    Merge *short* segments into neighbors, plus always do a cleanup pass that merges
    adjacent same-class segments (but never across protected REST).

    Notes for the NEW pipeline:
      - Decoder may emit REST segments with "protected_rest": True.
        Those must NOT be merged away.
      - Some code may still set "protected_strength": True (kept for compatibility).

    Behavior:
      - Always merges adjacent same-class segments first and last (cleanup).
      - If min_len_s <= 0: only does the cleanup merge and returns.
      - If min_len_s > 0: removes segments shorter than min_len_s by absorbing them
        into a safe neighbor with conservative rules.
    """
    if not segments:
        return []

    rest_names = {"REST", "PAUSE"}

    def is_rest(seg: Dict[str, Any]) -> bool:
        return str(seg.get("class", "")).upper() in rest_names

    def is_strength_like(seg: Dict[str, Any]) -> bool:
        return not is_rest(seg)

    def is_protected(seg: Dict[str, Any]) -> bool:
        # protect_rest is produced by decoder; protect_strength kept for backward compat
        if bool(seg.get("protected_rest", False)) and is_rest(seg):
            return True
        if bool(seg.get("protected_strength", False)) and is_strength_like(seg):
            return True
        return False

    def recalc(seg: Dict[str, Any]) -> None:
        seg["t0"] = float(seg["t0"])
        seg["t1"] = float(seg["t1"])
        seg["duration_s"] = float(seg["t1"] - seg["t0"])

    def merge_adjacent_same(segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in segs:
            cur = dict(s)
            recalc(cur)
            if out:
                prev = out[-1]
                same_cls = (str(prev.get("class")) == str(cur.get("class")))
                # never merge across protected REST boundaries
                if same_cls and not (is_protected(prev) or is_protected(cur)):
                    prev["t1"] = float(max(prev["t1"], cur["t1"]))
                    recalc(prev)
                    # keep i0 from prev, i1 from cur if available
                    if "i0" in prev and "i1" in prev and "i0" in cur and "i1" in cur:
                        prev["i1"] = int(max(int(prev["i1"]), int(cur["i1"])))
                    continue
            out.append(cur)
        return out

    segs = merge_adjacent_same(list(segments))
    if float(min_len_s) <= 0.0:
        return segs

    i = 0
    while i < len(segs):
        s = segs[i]
        recalc(s)

        # (1) Protected segments are NEVER removed/merged away
        if is_protected(s):
            i += 1
            continue

        dur = float(s["duration_s"])
        if dur < float(min_len_s) and len(segs) > 1:
            left = segs[i - 1] if i - 1 >= 0 else None
            right = segs[i + 1] if i + 1 < len(segs) else None

            candidates = []
            if left is not None:
                candidates.append(("prev", left))
            if right is not None:
                candidates.append(("next", right))

            if not candidates:
                i += 1
                continue

            # Safety filters
            safe = []
            for side, nb in candidates:
                # (2) Never merge INTO protected (avoid inflating protected segments)
                if is_protected(nb):
                    continue
                # (3) Never merge REST into strength-like (prevents artificial long strength)
                if is_rest(s) and is_strength_like(nb):
                    continue
                safe.append((side, nb))

            if not safe:
                i += 1
                continue

            # Choose target
            target = None
            if prefer == "prev":
                target = next(((side, nb) for side, nb in safe if side == "prev"), None)
            elif prefer == "next":
                target = next(((side, nb) for side, nb in safe if side == "next"), None)

            if target is None:
                if len(safe) == 2:
                    _, lnb = safe[0]
                    _, rnb = safe[1]
                    len_left = float(float(lnb["t1"]) - float(lnb["t0"]))
                    len_right = float(float(rnb["t1"]) - float(rnb["t0"]))
                    target = safe[0] if len_left <= len_right else safe[1]
                else:
                    target = safe[0]

            side, nb = target

            # Apply merge
            if side == "prev":
                nb["t1"] = float(s["t1"])
                recalc(nb)
                # carry indices if present
                if "i1" in nb and "i1" in s:
                    nb["i1"] = int(max(int(nb["i1"]), int(s["i1"])))
                del segs[i]
                i = max(i - 1, 0)
                continue

            if side == "next":
                nb["t0"] = float(s["t0"])
                recalc(nb)
                if "i0" in nb and "i0" in s:
                    nb["i0"] = int(min(int(nb["i0"]), int(s["i0"])))
                del segs[i]
                # keep i at same position, since "next" shifted into it
                continue

        i += 1

    # Final cleanup merge
    segs = merge_adjacent_same(segs)
    return segs
