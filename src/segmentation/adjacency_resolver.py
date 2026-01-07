# src/segmentation/adjacency_resolver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import sys


@dataclass
class AdjacencyResolverConfig:
    # --- New preferred knobs ---
    max_gap_s: float = 0.1
    rest_bridge_s: float = 6.0

    blip_max_s: float = 6.0
    blip_max_mean: float = 0.45
    blip_max_peak: float = 0.65
    blip_max_margin: float = 0.03

    debug: bool = False

    # --- Legacy knobs (accepted but not used directly) ---
    score_margin: Optional[float] = None
    winner_min_mean: Optional[float] = None
    min_combined_mean_conf: Optional[float] = None  # <-- ADD THIS

    def __post_init__(self) -> None:
        if self.score_margin is not None:
            try:
                sm = float(self.score_margin)
                self.blip_max_margin = float(min(self.blip_max_margin, sm))
            except Exception:
                pass

        # winner_min_mean: previously used for "winner_mean<thr => REST"
        # We do NOT do that anymore. We only use it to define blips more aggressively.
        if self.winner_min_mean is not None:
            try:
                wm = float(self.winner_min_mean)
                self.blip_max_mean = float(min(self.blip_max_mean, wm))
            except Exception:
                pass

        # min_combined_mean_conf: legacy "combined winner mean" threshold.
        # Also NOT used for REST-collapsing anymore; only tighten blip definition.
        if self.min_combined_mean_conf is not None:
            try:
                mc = float(self.min_combined_mean_conf)
                self.blip_max_mean = float(min(self.blip_max_mean, mc))
            except Exception:
                pass

        # normalize ranges
        self.max_gap_s = float(max(0.0, self.max_gap_s))
        self.rest_bridge_s = float(max(0.0, self.rest_bridge_s))
        self.blip_max_s = float(max(0.0, self.blip_max_s))
        self.blip_max_mean = float(min(max(self.blip_max_mean, 0.0), 1.0))
        self.blip_max_peak = float(min(max(self.blip_max_peak, 0.0), 1.0))
        self.blip_max_margin = float(self.blip_max_margin)


def _segment_window_mask(t0s: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return (t0s >= (t0 - 1e-9)) & (t0s < (t1 - 1e-9))


def _mean_peak_margin(
    probs: np.ndarray,
    cls_idx: int,
    win_mask: np.ndarray,
) -> Tuple[float, float, float]:
    if not np.any(win_mask):
        return 0.0, 0.0, 0.0
    P = probs[win_mask]
    p_cls = P[:, cls_idx]
    mean_c = float(np.mean(p_cls))
    peak_c = float(np.max(p_cls))

    P2 = P.copy()
    P2[:, cls_idx] = -1.0
    best_other = np.max(P2, axis=1)
    margin = float(np.mean(p_cls - best_other))
    return mean_c, peak_c, margin

def resolve_adjacent_strength(
    segments: Sequence[Dict],
    probs_s: np.ndarray,
    t0s: np.ndarray,
    classes: Sequence[str],
    strength_classes: Sequence[str],
    cfg: Optional[AdjacencyResolverConfig] = None,
) -> List[Dict]:
    cfg = cfg or AdjacencyResolverConfig()

    classes_u = [str(c) for c in classes]
    idx = {c: i for i, c in enumerate(classes_u)}

    strength_set = {str(s).upper() for s in strength_classes}
    rest_names = {"REST", "PAUSE"}

    def is_rest(c: str) -> bool:
        return str(c).upper() in rest_names

    def is_strength(c: str) -> bool:
        cu = str(c).upper()
        return (cu in strength_set) and (cu not in rest_names)

    def dur(seg: Dict) -> float:
        return float(max(0.0, float(seg["t1"]) - float(seg["t0"])))

    def dbg(msg: str):
        if cfg.debug:
            print(msg, file=sys.stderr)

    def merge_two(a: Dict, b: Dict, cls: str) -> Dict:
        out = dict(a)
        out["class"] = cls
        out["t0"] = float(a["t0"])
        out["t1"] = float(b["t1"])
        return out

    def score(seg: Dict) -> Tuple[float, float, float]:
        cls = str(seg["class"])
        if cls not in idx:
            return 0.0, 0.0, 0.0
        m = _segment_window_mask(t0s, float(seg["t0"]), float(seg["t1"]))
        return _mean_peak_margin(probs_s, idx[cls], m)

    out: List[Dict] = []
    i = 0
    n = len(segments)

    while i < n:
        s = dict(segments[i])
        cls = str(s["class"])

        if not is_strength(cls):
            out.append(s)
            i += 1
            continue

        # strength + REST(short) + strength
        if i + 2 < n:
            mid = segments[i + 1]
            nxt = segments[i + 2]
            if is_rest(str(mid["class"])) and is_strength(str(nxt["class"])):
                gap1 = float(mid["t0"]) - float(s["t1"])
                gap2 = float(nxt["t0"]) - float(mid["t1"])
                rest_dur = dur(mid)
                if gap1 <= cfg.max_gap_s and gap2 <= cfg.max_gap_s and rest_dur <= cfg.rest_bridge_s:
                    cls2 = str(nxt["class"])
                    if cls2 == cls:
                        merged = dict(s)
                        merged["t1"] = float(nxt["t1"])
                        dbg(f"[DBG ADJ] bridge {cls} + REST({rest_dur:.1f}s) + {cls2} => MERGE")
                        out.append(merged)
                        i += 3
                        continue
                    else:
                        # different exercises: keep boundary
                        out.append(s)
                        i += 1
                        continue

        # strength + strength (direct)
        if i + 1 < n and is_strength(str(segments[i + 1]["class"])):

            # ---------- local helpers (keep inside to avoid changing other file parts) ----------
            def _stability_for(seg: Dict, cls_name: str) -> float:
                """
                Stability in [0..1] based on top-1 support and longest contiguous top-1 run.
                Higher => more consistent / less jitter.
                """
                if cls_name not in idx:
                    return 0.0
                m = _segment_window_mask(t0s, float(seg["t0"]), float(seg["t1"]))
                if not np.any(m):
                    return 0.0

                P = probs_s[m]  # [T, C]
                c = idx[cls_name]

                top1 = np.argmax(P, axis=1)
                is_top1 = (top1 == c)
                top1_rate = float(np.mean(is_top1))

                best = 0
                cur = 0
                for v in is_top1:
                    if v:
                        cur += 1
                        if cur > best:
                            best = cur
                    else:
                        cur = 0
                longest_run = float(best)
                nwin = float(len(is_top1))
                if nwin <= 0:
                    return 0.0

                return 0.6 * top1_rate + 0.4 * (longest_run / nwin)

            def _consistency_for(seg: Dict, cls_name: str) -> float:
                """
                Konstanz der Ausführung über das Segment:
                - hoch, wenn Klasse häufig Top1 ist
                - hoch, wenn p_cls wenig schwankt (niedrige std)
                Ergebnis grob in [0..1], als Tie-Breaker nutzbar.
                """
                if cls_name not in idx:
                    return 0.0
                m = _segment_window_mask(t0s, float(seg["t0"]), float(seg["t1"]))
                if not np.any(m):
                    return 0.0

                P = probs_s[m]
                c = idx[cls_name]
                p = P[:, c]

                top1 = np.argmax(P, axis=1)
                top1_rate = float(np.mean(top1 == c))

                stdp = float(np.std(p))  # 0..~0.5 typisch
                # std normalisieren grob (clamp)
                stdn = min(max(stdp / 0.25, 0.0), 1.0)

                # hohe top1_rate + niedrige std => hohe Konstanz
                return 0.7 * top1_rate + 0.3 * (1.0 - stdn)
                                        
            def _logsum_cls_for(seg: Dict, cls_name: str) -> float:
                """Evidence score for 'cls_name' over seg interval (higher = better)."""
                if cls_name not in idx:
                    return -1e9
                m = _segment_window_mask(t0s, float(seg["t0"]), float(seg["t1"]))
                if not np.any(m):
                    return -1e9
                eps = 1e-6
                p = probs_s[m, idx[cls_name]]
                return float(np.sum(np.log(p + eps)))

            def _to_rest(seg: Dict) -> Dict:
                seg2 = dict(seg)
                seg2["class"] = "REST"
                return seg2

            def _is_confident_set(seg: Dict) -> bool:
                m, p, g = score(seg)
                return (p >= 0.70) or (g >= 0.05)

            def _count_nearby_same_class(i_center: int, target_cls: str, look: int = 4) -> int:
                c = 0
                for k in range(1, look + 1):
                    j = i_center - k
                    if j < 0:
                        break
                    sj = dict(segments[j])
                    if str(sj["class"]) == target_cls and is_strength(target_cls) and _is_confident_set(sj):
                        c += 1
                for k in range(1, look + 1):
                    j = i_center + k
                    if j >= n:
                        break
                    sj = dict(segments[j])
                    if str(sj["class"]) == target_cls and is_strength(target_cls) and _is_confident_set(sj):
                        c += 1
                return c

            def _exists_before(i_center: int, target_cls: str, look: int = 6) -> bool:
                for k in range(1, look + 1):
                    j = i_center - k
                    if j < 0:
                        break
                    sj = dict(segments[j])
                    if str(sj["class"]) == target_cls and is_strength(target_cls) and _is_confident_set(sj):
                        return True
                return False

            def _exists_after(i_center: int, target_cls: str, look: int = 6) -> bool:
                for k in range(1, look + 1):
                    j = i_center + k
                    if j >= n:
                        break
                    sj = dict(segments[j])
                    if str(sj["class"]) == target_cls and is_strength(target_cls) and _is_confident_set(sj):
                        return True
                return False
            # -----------------------------------------------------------------------------------

            b = dict(segments[i + 1])
            cls_b = str(b["class"])
            gap = float(b["t0"]) - float(s["t1"])

            if gap <= cfg.max_gap_s:

                # same class -> merge
                if cls_b == cls:
                    merged = merge_two(s, b, cls)
                    dbg(f"[DBG ADJ] {cls} + {cls_b} (same) gap={gap:.3f} => MERGE")
                    out.append(merged)
                    i += 2
                    continue

                # compute scores for blip detection (unchanged logic)
                ma, pa, ga = score(s)
                mb, pb, gb = score(b)

                a_is_blip = (
                    dur(s) <= cfg.blip_max_s
                    and ma <= cfg.blip_max_mean
                    and pa <= cfg.blip_max_peak
                    and ga <= cfg.blip_max_margin
                )
                b_is_blip = (
                    dur(b) <= cfg.blip_max_s
                    and mb <= cfg.blip_max_mean
                    and pb <= cfg.blip_max_peak
                    and gb <= cfg.blip_max_margin
                )

                # 1) blip cases: keep existing behavior
                if a_is_blip and not b_is_blip:
                    dbg(
                        f"[DBG ADJ] {cls} looks like blip (dur={dur(s):.1f}s mean={ma:.3f} peak={pa:.3f} margin={ga:.3f}) "
                        f"=> merge into {cls_b}"
                    )
                    b2 = dict(b)
                    b2["t0"] = float(s["t0"])
                    out.append(b2)
                    i += 2
                    continue

                if b_is_blip and not a_is_blip:
                    dbg(
                        f"[DBG ADJ] {cls_b} looks like blip (dur={dur(b):.1f}s mean={mb:.3f} peak={pb:.3f} margin={gb:.3f}) "
                        f"=> absorb into {cls}"
                    )
                    merged = merge_two(s, b, cls)
                    out.append(merged)
                    i += 2
                    continue

                if a_is_blip and b_is_blip:
                    winner = cls if pa >= pb else cls_b
                    dbg(f"[DBG ADJ] both blips => choose {winner}")
                    merged = merge_two(s, b, winner)
                    out.append(merged)
                    i += 2
                    continue

                # 2) MUTEX: two non-blip different strength classes
                #    Keep only one class; demote the other to REST.
                #    Apply repeat-prior only when not clear, and disable it on A-before/B-after edge case.

                sa = _logsum_cls_for(s, cls)
                sb = _logsum_cls_for(b, cls_b)
                raw_logdiff = abs(sa - sb)

                # stability-weighted clarity
                alpha = float(getattr(cfg, "stability_alpha", 1.0))
                stab_a = _stability_for(s, cls)
                stab_b = _stability_for(b, cls_b)
                if sa >= sb:
                    stab_delta = stab_a - stab_b
                else:
                    stab_delta = stab_b - stab_a
                logdiff_eff = raw_logdiff + alpha * max(0.0, stab_delta)

                UNCLEAR_LOGDIFF = float(getattr(cfg, "unclear_logdiff", 1.0))

                dbg(
                    f"[DBG ADJ] mutex stats {cls}->{cls_b}: "
                    f"sa={sa:.3f} sb={sb:.3f} raw={raw_logdiff:.3f} "
                    f"stab_a={stab_a:.3f} stab_b={stab_b:.3f} stab_delta={stab_delta:.3f} "
                    f"logdiff_eff={logdiff_eff:.3f} thr={UNCLEAR_LOGDIFF:.3f}"
                )

                # Evaluate repeat-prior only if not clear (using logdiff_eff)
                if logdiff_eff <= UNCLEAR_LOGDIFF:

                    a_before = _exists_before(i, cls, look=6)
                    b_after  = _exists_after(i + 1, cls_b, look=6)

                    # NEW: disable repeat-prior on "A before, B after" edge case
                    if a_before and b_after:
                        # Repeat-prior disabled (we do NOT want "2-3 set" heuristic here),
                        # but we still need a tie-breaker for near-equal log evidence.
                        TIE_RAW = float(getattr(cfg, "tie_raw_logdiff", 0.08))  # near-tie threshold

                        if raw_logdiff <= TIE_RAW:
                            # tie-break by stability (consistency) only
                            if stab_b > stab_a:
                                dbg(
                                    f"[DBG ADJ] A-before/B-after + near-tie(raw={raw_logdiff:.3f}) -> "
                                    f"tie-break by stability: keep {cls_b} (stab {stab_b:.3f}>{stab_a:.3f})"
                                )
                                out.append(_to_rest(s))
                                out.append(b)
                            else:
                                dbg(
                                    f"[DBG ADJ] A-before/B-after + near-tie(raw={raw_logdiff:.3f}) -> "
                                    f"tie-break by stability: keep {cls} (stab {stab_a:.3f}>={stab_b:.3f})"
                                )
                                out.append(s)
                                out.append(_to_rest(b))
                            i += 2
                            continue

                        dbg(
                            f"[DBG ADJ] mutex unclear (logdiff_eff={logdiff_eff:.3f}) but A-before/B-after detected "
                            f"({cls} before, {cls_b} after) -> disable repeat-prior, fallback to evidence"
                        )
                    else:
                        # Only now compute repeat counts
                        cnt_a = _count_nearby_same_class(i, cls, look=4)
                        cnt_b = _count_nearby_same_class(i + 1, cls_b, look=4)
                        dbg(f"[DBG ADJ] repeat-prior counts {cls}={cnt_a} {cls_b}={cnt_b} (look=4)")

                        # If one class repeats confidently nearby and the other doesn't, pick that one
                        if cnt_a != cnt_b:
                            if cnt_a > cnt_b:
                                dbg(
                                    f"[DBG ADJ] mutex unclear (logdiff_eff={logdiff_eff:.3f}) + repeat-prior: "
                                    f"prefer {cls} (nearby_conf={cnt_a}>{cnt_b})"
                                )
                                out.append(s)
                                out.append(_to_rest(b))
                            else:
                                dbg(
                                    f"[DBG ADJ] mutex unclear (logdiff_eff={logdiff_eff:.3f}) + repeat-prior: "
                                    f"prefer {cls_b} (nearby_conf={cnt_b}>{cnt_a})"
                                )
                                out.append(_to_rest(s))
                                out.append(b)

                            i += 2
                            continue
                        else:
                            dbg(
                                f"[DBG ADJ] mutex unclear (logdiff_eff={logdiff_eff:.3f}) but repeat-prior tie "
                                f"(nearby_conf {cls}={cnt_a}, {cls_b}={cnt_b}) -> fallback to evidence"
                            )

                # fallback: pure evidence
                # -------- fallback (mit Konsistenz-Tie-Breaker) --------
                tie_raw = float(getattr(cfg, "tie_raw_logdiff", 0.25))   # nur bei nahe gleicher Evidenz
                beta   = float(getattr(cfg, "tie_consistency_beta", 0.8)) # Gewichtung Konstanz im Tie

                raw = abs(sa - sb)

                if raw <= tie_raw:
                    ca = _consistency_for(s, cls)
                    cb = _consistency_for(b, cls_b)

                    score_a = sa + beta * ca
                    score_b = sb + beta * cb

                    if score_a >= score_b:
                        dbg(
                            f"[DBG ADJ] tie-break {cls}->{cls_b}: raw={raw:.3f} "
                            f"sa={sa:.3f} sb={sb:.3f} ca={ca:.3f} cb={cb:.3f} "
                            f"=> keep {cls}"
                        )
                        out.append(s)
                        out.append(_to_rest(b))
                    else:
                        dbg(
                            f"[DBG ADJ] tie-break {cls}->{cls_b}: raw={raw:.3f} "
                            f"sa={sa:.3f} sb={sb:.3f} ca={ca:.3f} cb={cb:.3f} "
                            f"=> keep {cls_b}"
                        )
                        out.append(_to_rest(s))
                        out.append(b)

                    i += 2
                    continue

                # sonst: pure evidence wie bisher
                if sa >= sb:
                    dbg(
                        f"[DBG ADJ] mutex {cls} -> {cls_b}: keep {cls} (logsum {sa:.1f} >= {sb:.1f}), "
                        f"demote {cls_b} to REST"
                    )
                    out.append(s)
                    out.append(_to_rest(b))
                else:
                    dbg(
                        f"[DBG ADJ] mutex {cls} -> {cls_b}: keep {cls_b} (logsum {sb:.1f} > {sa:.1f}), "
                        f"demote {cls} to REST"
                    )
                    out.append(_to_rest(s))
                    out.append(b)

                i += 2
                continue

        # default fallthrough
        out.append(s)
        i += 1

    return out


