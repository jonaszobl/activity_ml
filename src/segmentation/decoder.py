# src/segmentation/decoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Sequence

import numpy as np
import pandas as pd

from .reps import select_rep_signal


@dataclass
class DecoderConfig:
    # =========================
    # 1) REST <-> STRENGTH FSM
    # =========================
    start_hold_w: int = 2        # wie viele Fenster "Strength" stabil sein müssen um zu starten
    end_hold_w: int = 2          # wie viele Fenster "REST" stabil sein müssen um zu beenden

    min_set_s: float = 8.0       # Anti-Flackern: minimaler Strength-Block
    min_rest_s: float = 8.0      # Anti-Flackern: minimaler Rest-Block

    # Quantile-basierte Schwellen (session-adaptiv)
    q_strength_start: float = 0.80
    q_rest_end: float = 0.80
    rest_end_floor: float = 0.60  # absoluter Mindestwert für "REST endet Strength"

    # Optional: Motion als Zusatzsignal (nur zum "Ende erzwingen")
    use_motion_end: bool = True
    q_motion_low: float = 0.25
    motion_end_hold_w: int = 3

    # Sehr starke REST Evidenz soll zuverlässig beenden (separater Hold!)
    rest_override_floor: float = 0.95
    rest_override_hold_w: int = 2

    # =========================
    # 2) Split innerhalb Strength
    # =========================
    split_on_class_change: bool = True
    class_change_hold_w: int = 3
    class_change_min_gap_w: int = 0
    ignore_changes_if_strength_below: float = 0.0

    # =========================
    # 3) NEW: Protected REST inside Strength (anti "REST wird weggemerged")
    # =========================
    # Wenn innerhalb eines Strength-Segments REST wirklich dominant ist (Top1==REST) und p_rest hoch,
    # dann schneiden wir diesen Run als eigenes REST-Segment heraus und markieren es als protected_rest.
    protect_rest_enable: bool = True

    # Run muss mindestens so viele Fenster am Stück "REST-top1 & p_rest>=thr_rest_end" sein
    protect_rest_hold_w: int = 2

    # und mindestens so lange dauern (sek), damit wir nicht kleine Zuckungen rausschneiden
    protect_rest_min_s: float = 12.0

    # zusätzliches Failsafe: wenn mean(p_rest) im Run sehr hoch ist, erlauben wir auch kürzer,
    # aber nur wenn hold_w erfüllt ist.
    protect_rest_mean_p: float = 0.90

    # Split nur, wenn die beiden Strength-Teilstücke nach dem Split nicht zu kurz werden
    protect_rest_require_strength_side_min_s: float = 8.0  # typischerweise = min_set_s

    debug: bool = True


@dataclass
class Segment:
    t0: float
    t1: float
    cls: str
    i0: int
    i1: int
    protected_rest: bool = False

    @property
    def duration_s(self) -> float:
        return float(self.t1 - self.t0)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "t0": float(self.t0),
            "t1": float(self.t1),
            "duration_s": float(self.duration_s),
            "class": str(self.cls),
            "i0": int(self.i0),
            "i1": int(self.i1),
        }
        if self.protected_rest and str(self.cls).upper() in ("REST", "PAUSE"):
            d["protected_rest"] = True
        return d


class StateMachineSegmenter:
    """
    Decoder:
      - Zustand: REST oder STRENGTH
      - In STRENGTH: Segment-Klasse kann wechseln, wenn neuer Block stabil ist.
      - NEW: Optionaler Post-Pass, der "echte REST-Runs" innerhalb Strength als protected REST herausschneidet.
    """

    def __init__(self, classes: Sequence[str], strength_classes: Sequence[str], cfg: Optional[DecoderConfig] = None):
        self.classes = [str(c) for c in classes]
        self.cfg = cfg or DecoderConfig()

        sc = {str(s).upper() for s in strength_classes}
        self.strength_set = sc
        self.strength_mask = np.array([c.upper() in sc for c in self.classes], dtype=bool)

        self.rest_idx = None
        for i, c in enumerate(self.classes):
            if c.upper() in ("REST", "PAUSE"):
                self.rest_idx = i
                break

    @staticmethod
    def _quantile_thr(x: np.ndarray, q: float) -> float:
        x = np.asarray(x, float)
        if x.size == 0:
            return 0.0
        return float(np.quantile(x, float(np.clip(q, 0.0, 1.0))))

    @staticmethod
    def _cond_hold(i: int, hold: int, cond_arr: np.ndarray) -> bool:
        if hold <= 1:
            return bool(cond_arr[i])
        if i - hold + 1 < 0:
            return False
        return bool(np.all(cond_arr[i - hold + 1 : i + 1]))

    def _window_motion_std(self, df: pd.DataFrame, t0s: np.ndarray, win_s: float, fs: float) -> np.ndarray:
        t = df["t"].to_numpy(float)
        ax = df["ax"].to_numpy(float)
        ay = df["ay"].to_numpy(float)
        az = df["az"].to_numpy(float)

        out = np.zeros(len(t0s), dtype=float)
        for i, t0 in enumerate(t0s):
            t1 = float(t0) + float(win_s)
            i0 = int(np.searchsorted(t, t0, side="left"))
            i1 = int(np.searchsorted(t, t1, side="right"))
            if i1 - i0 < max(4, int(0.5 * fs)):
                out[i] = 0.0
                continue
            sig = select_rep_signal(ax[i0:i1], ay[i0:i1], az[i0:i1], fs)
            out[i] = float(np.std(sig)) if sig.size else 0.0
        return out

    def _is_rest_name(self, cls: str) -> bool:
        return str(cls).upper() in ("REST", "PAUSE")

    def _is_strength_name(self, cls: str) -> bool:
        cu = str(cls).upper()
        return (cu in self.strength_set) and (cu not in ("REST", "PAUSE"))

    @staticmethod
    def _runs_true(mask: np.ndarray) -> List[tuple]:
        """
        Returns list of (start_idx, end_idx_exclusive) runs where mask==True.
        """
        mask = np.asarray(mask, bool)
        if mask.size == 0:
            return []
        # find edges
        diff = np.diff(mask.astype(np.int8))
        starts = list(np.where(diff == 1)[0] + 1)
        ends = list(np.where(diff == -1)[0] + 1)
        if mask[0]:
            starts = [0] + starts
        if mask[-1]:
            ends = ends + [mask.size]
        return list(zip(starts, ends))

    def _split_strength_on_protected_rest(
        self,
        segments: List[Segment],
        rest_top1: np.ndarray,
        p_rest: np.ndarray,
        thr_rest_end: float,
        hop_s: float,
        win_s: float,
    ) -> List[Segment]:
        cfg = self.cfg
        if not cfg.protect_rest_enable or self.rest_idx is None:
            return segments

        out: List[Segment] = []
        N = int(p_rest.shape[0])

        # thresholds in windows
        hold_w = int(max(1, cfg.protect_rest_hold_w))
        min_rest_w = int(max(1, np.ceil(float(cfg.protect_rest_min_s) / max(1e-9, hop_s))))
        min_side_w = int(max(1, np.ceil(float(cfg.protect_rest_require_strength_side_min_s) / max(1e-9, hop_s))))

        for seg in segments:
            if not self._is_strength_name(seg.cls):
                out.append(seg)
                continue

            i0 = int(max(0, min(N, seg.i0)))
            i1 = int(max(0, min(N, seg.i1)))
            if i1 - i0 <= 0:
                out.append(seg)
                continue

            # "strong rest evidence" inside this strength segment
            cond = (rest_top1[i0:i1] & (p_rest[i0:i1] >= float(thr_rest_end)))

            runs = self._runs_true(cond)
            if not runs:
                out.append(seg)
                continue

            cur_start = i0
            emitted_any = False

            for rs, re in runs:
                run_len = int(re - rs)
                if run_len < hold_w:
                    continue

                run_abs_s = i0 + rs
                run_abs_e = i0 + re

                run_mean_p = float(np.mean(p_rest[run_abs_s:run_abs_e])) if run_abs_e > run_abs_s else 0.0
                run_is_long = (run_len >= min_rest_w)
                run_is_very_conf = (run_mean_p >= float(cfg.protect_rest_mean_p))

                if not (run_is_long or run_is_very_conf):
                    continue

                # only split if we don't create tiny strength fragments
                left_w = run_abs_s - cur_start
                right_w = i1 - run_abs_e

                # we only split "internal" runs (not at edges)
                if left_w <= 0 or right_w <= 0:
                    continue

                if left_w < min_side_w or right_w < min_side_w:
                    continue

                # emit left strength
                t0_left = float(seg.t0) + float((cur_start - i0) * hop_s)
                t1_left = float(seg.t0) + float((run_abs_s - i0) * hop_s)
                out.append(Segment(t0=t0_left, t1=t1_left, cls=seg.cls, i0=cur_start, i1=run_abs_s))

                # emit protected rest
                t0_r = float(seg.t0) + float((run_abs_s - i0) * hop_s)
                t1_r = float(seg.t0) + float((run_abs_e - i0) * hop_s)
                out.append(Segment(t0=t0_r, t1=t1_r, cls="REST", i0=run_abs_s, i1=run_abs_e, protected_rest=True))

                cur_start = run_abs_e
                emitted_any = True

            if not emitted_any:
                out.append(seg)
            else:
                # emit tail strength
                if i1 - cur_start > 0:
                    t0_tail = float(seg.t0) + float((cur_start - i0) * hop_s)
                    t1_tail = float(seg.t0) + float((i1 - i0) * hop_s)
                    out.append(Segment(t0=t0_tail, t1=t1_tail, cls=seg.cls, i0=cur_start, i1=i1))

        # fix segment t1 last using win_s for tails created with hop_s arithmetic:
        # For safety we do a light normalization: recompute t0/t1 from indices where possible.
        # (Keeps behavior stable even if you change win_s/hop_s.)
        # NOTE: We keep original times if they were produced by FSM finalize; the split pieces
        # were produced from hop_s which matches your segmentation grid.
        return out

    def decode(
        self,
        df: pd.DataFrame,
        probs_s: np.ndarray,
        t0s: np.ndarray,
        win_s: float,
        hop_s: float,
        fs: float,
    ) -> List[Dict[str, Any]]:
        cfg = self.cfg
        N = int(probs_s.shape[0])
        assert N == len(t0s), "probs_s und t0s müssen gleich lang sein."

        # ---- p_rest ----
        if self.rest_idx is None:
            p_rest = np.zeros(N, dtype=float)
        else:
            p_rest = probs_s[:, self.rest_idx].astype(float)

        # ---- top1 + rest_top1 ----
        top1 = np.argmax(probs_s, axis=1)
        rest_top1 = (top1 == self.rest_idx) if (self.rest_idx is not None) else np.zeros(N, dtype=bool)

        # ---- strength evidence + best strength class ----
        p_strength = np.zeros(N, dtype=float)
        best_strength_idx = np.full(N, -1, dtype=int)
        if np.any(self.strength_mask):
            ps = probs_s[:, self.strength_mask]
            best_local = np.argmax(ps, axis=1)
            strength_indices = np.where(self.strength_mask)[0]
            best_strength_idx = strength_indices[best_local]
            p_strength = ps[np.arange(N), best_local].astype(float)

        # ---- motion ----
        motion = self._window_motion_std(df, t0s, win_s, fs) if cfg.use_motion_end else np.zeros(N, dtype=float)

        # ---- session thresholds ----
        thr_strength = self._quantile_thr(p_strength, cfg.q_strength_start)

        # REST-Ende: nur auf Fenstern wo REST wirklich Top1 ist (reduziert "REST p hoch aber nicht dominant")
        if self.rest_idx is not None and np.any(rest_top1):
            thr_rest_end = max(float(cfg.rest_end_floor), self._quantile_thr(p_rest[rest_top1], cfg.q_rest_end))
        else:
            thr_rest_end = max(float(cfg.rest_end_floor), self._quantile_thr(p_rest, cfg.q_rest_end))

        thr_motion_low = self._quantile_thr(motion, cfg.q_motion_low) if cfg.use_motion_end else 0.0

        if cfg.debug:
            import sys
            print(
                f"[DBG DEC THR] thr_strength={thr_strength:.3f} thr_rest_end={thr_rest_end:.3f} "
                f"thr_motion_low={thr_motion_low:.3f} "
                f"holdS={cfg.start_hold_w} holdE={cfg.end_hold_w} "
                f"rest_override>={cfg.rest_override_floor:.2f} holdR={cfg.rest_override_hold_w} "
                f"class_hold={cfg.class_change_hold_w} min_set={cfg.min_set_s:.1f}s min_rest={cfg.min_rest_s:.1f}s",
                file=sys.stderr,
            )

        # ---- conditions ----
        can_start_strength = (p_strength >= thr_strength)

        # Ende-Trigger 1: REST dominant (Top1) + hoch genug
        end_by_rest = rest_top1 & (p_rest >= float(thr_rest_end))

        # Ende-Trigger 2: REST override (sehr stark) – unabhängig von Top1, absichtlich aggressiv
        end_by_override = (p_rest >= float(cfg.rest_override_floor))

        # Ende-Trigger 3: Motion sehr niedrig
        end_by_motion = (motion <= thr_motion_low) if cfg.use_motion_end else np.zeros(N, dtype=bool)

        # ---- minima in windows ----
        min_set_w = max(1, int(np.ceil(cfg.min_set_s / max(1e-9, hop_s))))
        min_rest_w = max(1, int(np.ceil(cfg.min_rest_s / max(1e-9, hop_s))))

        # ---- FSM ----
        state = "REST"  # REST | STRENGTH
        cur_i0 = 0
        cur_cls = "REST"
        last_split_i = -10**9

        segments: List[Segment] = []

        def finalize(i_end: int):
            nonlocal cur_i0, cur_cls
            t0 = float(t0s[cur_i0])
            t1 = float(t0s[i_end]) if i_end < N else float(t0s[N - 1]) + float(win_s)
            seg = Segment(t0=t0, t1=t1, cls=cur_cls, i0=cur_i0, i1=i_end)
            if cfg.debug:
                import sys
                print(
                    f"[DBG DEC SEG] {cur_cls:16s} i={cur_i0}->{i_end} t={t0:.1f}-{t1:.1f} dur={t1-t0:.1f}s",
                    file=sys.stderr,
                )
            segments.append(seg)

        def best_class_at(i: int) -> str:
            k = int(best_strength_idx[i])
            if k < 0:
                return "REST"
            return str(self.classes[k])

        def stable_new_class_end(i: int, new_cls: str) -> bool:
            h = int(cfg.class_change_hold_w)
            if h <= 1:
                return best_class_at(i) == new_cls

            j0 = i - h + 1
            if j0 < 0:
                return False

            if cfg.ignore_changes_if_strength_below > 0.0:
                if float(p_strength[i]) < float(cfg.ignore_changes_if_strength_below):
                    return False

            for j in range(j0, i + 1):
                if best_class_at(j) != new_cls:
                    return False
            return True

        for i in range(N):
            if state == "REST":
                rest_len_w = i - cur_i0 + 1

                if self._cond_hold(i, cfg.start_hold_w, can_start_strength) and rest_len_w >= min_rest_w:
                    cut = i - cfg.start_hold_w + 1
                    if cut > cur_i0:
                        finalize(cut)

                    state = "STRENGTH"
                    cur_i0 = cut
                    cur_cls = best_class_at(i)
                    last_split_i = cur_i0

            else:  # STRENGTH
                set_len_w = i - cur_i0 + 1

                # 1) innerhalb Strength: split bei stabilem Klassenwechsel
                if cfg.split_on_class_change and set_len_w >= 1:
                    new_cls = best_class_at(i)

                    if new_cls != "REST" and new_cls != cur_cls:
                        if (i - last_split_i) >= int(cfg.class_change_min_gap_w):
                            if stable_new_class_end(i, new_cls):
                                cut = i - int(cfg.class_change_hold_w) + 1
                                if cut > cur_i0 and (cut - cur_i0) >= 1:
                                    finalize(cut)
                                    cur_i0 = cut
                                    cur_cls = new_cls
                                    last_split_i = cur_i0

                # 2) Ende nach REST/Motion – nur wenn Segment lang genug
                if set_len_w >= min_set_w:
                    if self._cond_hold(i, cfg.rest_override_hold_w, end_by_override):
                        cut = i - cfg.rest_override_hold_w + 1
                        if cut > cur_i0:
                            finalize(cut)
                        state = "REST"
                        cur_i0 = cut
                        cur_cls = "REST"
                        continue

                    if self._cond_hold(i, cfg.end_hold_w, end_by_rest):
                        cut = i - cfg.end_hold_w + 1
                        if cut > cur_i0:
                            finalize(cut)
                        state = "REST"
                        cur_i0 = cut
                        cur_cls = "REST"
                        continue

                    if cfg.use_motion_end and self._cond_hold(i, cfg.motion_end_hold_w, end_by_motion):
                        cut = i - cfg.motion_end_hold_w + 1
                        if cut > cur_i0:
                            finalize(cut)
                        state = "REST"
                        cur_i0 = cut
                        cur_cls = "REST"
                        continue

        # finalize tail
        finalize(N)

        # merge adjacent same class (FSM-level)
        merged: List[Segment] = []
        for s in segments:
            if merged and merged[-1].cls == s.cls and (not (merged[-1].protected_rest or s.protected_rest)):
                merged[-1].t1 = max(merged[-1].t1, s.t1)
                merged[-1].i1 = max(merged[-1].i1, s.i1)
            else:
                merged.append(s)

        # NEW: split strength segments by "protected rest runs" inside them
        merged2 = self._split_strength_on_protected_rest(
            merged, rest_top1=rest_top1, p_rest=p_rest, thr_rest_end=float(thr_rest_end), hop_s=float(hop_s), win_s=float(win_s)
        )

        # final merge adjacent same class again (but never across protected rest)
        merged3: List[Segment] = []
        for s in merged2:
            if merged3 and merged3[-1].cls == s.cls and (not (merged3[-1].protected_rest or s.protected_rest)):
                merged3[-1].t1 = max(merged3[-1].t1, s.t1)
                merged3[-1].i1 = max(merged3[-1].i1, s.i1)
            else:
                merged3.append(s)

        return [s.to_dict() for s in merged3]
