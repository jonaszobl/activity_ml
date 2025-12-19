# src/segmentation/postprocessing.py
import sys
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from ..features import rfft_band_features
from .reps import (
    select_rep_signal,
    estimate_rep_period_acf,  # bleibt für andere Stellen
    mad,
    _acf_primary_peak,        # NEU: wie im Original
)



# ---------- Probs glätten & Fenster-Segmentierung ----------
def smooth_probs_over_time(probs: np.ndarray, k: int = 5) -> np.ndarray:
    # leichte Tempoglatte der Probs; verhindert Flackern
    if k <= 1:
        return probs
    N, C = probs.shape
    out = np.empty_like(probs, dtype=float)
    for c in range(C):
        x = probs[:, c]
        csum = np.cumsum(np.insert(x, 0, 0.0))
        y = (csum[k:] - csum[:-k]) / k
        pad = np.full(k - 1, y[0] if len(y) else 0.0)
        out[:, c] = np.concatenate([pad, y])
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s  # normieren


def debounce_labels(cls_idx: np.ndarray, min_run: int = 3) -> np.ndarray:
    # entprellt schnelle Wechsel; sehr simpler Automat
    cls_idx = cls_idx.astype(int)
    out = np.empty_like(cls_idx)
    current = cls_idx[0]
    run = 1
    out[0] = current
    for i in range(1, len(cls_idx)):
        if cls_idx[i] == current:
            out[i] = current
            run = 1
        else:
            run += 1
            if run >= min_run:
                current = cls_idx[i]
                out[i] = current
                run = 1
            else:
                out[i] = current
    return out


def segment_from_window_preds(t0s, cls_idx, classes, win_s):
    # fensterklassen → zusammenhängende Segmente
    segments = []
    i = 0
    while i < len(cls_idx):
        k = int(cls_idx[i])
        j = i + 1
        while j < len(cls_idx) and int(cls_idx[j]) == k:
            j += 1
        t0 = float(t0s[i])
        t1 = float(t0s[j - 1]) + float(win_s)
        segments.append({
            "t0": t0,
            "t1": t1,
            "duration_s": float(t1 - t0),
            "class": str(classes[k]),
            "i0": int(i),
            "i1": int(j)
        })
        i = j
    return segments


def merge_short_segments(segments, min_len_s: float, prefer: str = "neighbor"):
    # zu kurze Segmente in Nachbarn mergen; reduziert Fetzen
    if not segments:
        return segments
    segs = [dict(s) for s in segments]
    i = 0
    while i < len(segs):
        s = segs[i]
        dur = float(s["t1"] - s["t0"])
        if dur < min_len_s and len(segs) > 1:
            left = segs[i - 1] if i - 1 >= 0 else None
            right = segs[i + 1] if i + 1 < len(segs) else None
            target = None
            if prefer == "prev" and left:
                target = ("prev", left)
            elif prefer == "next" and right:
                target = ("next", right)
            else:
                if left and right:
                    len_left = left["t1"] - left["t0"]
                    len_right = right["t1"] - right["t0"]
                    target = ("prev", left) if len_left <= len_right else ("next", right)
                elif left:
                    target = ("prev", left)
                elif right:
                    target = ("next", right)

            if target:
                side, _ = target
                if side == "prev" and left:
                    left["t1"] = s["t1"]; left["duration_s"] = float(left["t1"] - left["t0"])
                    del segs[i]; i = max(i - 1, 0); continue
                if side == "next" and right:
                    right["t0"] = s["t0"]; right["duration_s"] = float(right["t1"] - right["t0"])
                    del segs[i]; continue
        i += 1
    return segs


# ---------- Kraft vs. Alltagsklassen ----------
def strength_classes_from(M):
    # Filter gegen Alltagsklassen -> nur echte Kraftübungen
    exclude = {"REST", "PAUSE", "WALKING", "RUNNING"}
    return {c for c in M["classes"] if str(c).upper() not in exclude}


def seconds_to_hms(sec: float) -> str:
    # display helper; bleibt simple
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


# ---------- Post-Filter Defaults (Basis) ----------
POST_DEFAULTS = dict(
    # Basis-Werte; dienen als Untergrenze bevor Klassentuning greift
    min_strength_duration_s = 8.0,   # <8 s: sehr wahrscheinlich kein echter Satz
    min_rest_between_sets_s = 10.0,  # Pflicht-Ruhe zwischen Sets
    acf_peak_thr            = 0.18,  # Periodizität muss spürbar sein
    band_ratio_thr          = 0.35,  # Anteil 0.3–3 Hz hoch genug
    std_thr_g               = 0.05,  # Mindest-Amplitude (g) nach Highpass+Smooth
    min_rep_density         = 0.25,  # reps/duration_s
    conf_thr                = 0.50   # mittlere Segment-Confidence
)
# vorher strengere Schwellen getestet -> zu viele False REST → gnädiger gesetzt

# Rudern add-on, da rep_density rauswirft...
RUDERN_DENSITY_ALPHA = 0.70   # 70% der erwarteten Dichte reicht fürs Gate
RUDERN_ACF_TRUST     = 0.12   # ab dieser ACF-Höhe traut die Logik der Periode


# ============================================
# Klassenspezifische Schwellen (Hyperfunktion)
# ============================================
CLASS_THRESH = {

    # --- Brust / Fly ---
    "CABLE_FLY_CHEST": dict(
        acf_peak_thr=0.13,      # weiche Bewegung; Peak eher klein
        band_ratio_thr=0.25,    # wenig 0.3–3Hz-Energie
        std_thr_g=0.030,        # kaum G-Ausschlag am HG
        min_rep_density=0.17,   # ~1 rep in ~6s ok
        conf_thr=0.40,          # Modell teils unsicher → gnädiger
        min_strength_duration_s=10.0
    ),

    # --- Bankdrücken Langhantel ---
    "BENCH_BB": dict(
        acf_peak_thr=0.16,      # klarer Rhythmus, Auflage dämpft
        band_ratio_thr=0.32,
        std_thr_g=0.045,
        min_rep_density=0.22,   # ~1 rep / 4.5s
        conf_thr=0.45,
        min_strength_duration_s=8.0
    ),

    # --- Bankdrücken Kurzhanteln ---
    "BENCH_DB": dict(
        acf_peak_thr=0.15,      # ungleichmäßiger → etwas kleiner
        band_ratio_thr=0.28,
        std_thr_g=0.040,
        min_rep_density=0.20,
        conf_thr=0.40,
        min_strength_duration_s=9.0
    ),

    # --- Trizeps-Kabelzug ---
    "TRICEPS_PULLDOWN": dict(
        acf_peak_thr=0.15,      # schnell/gleichmäßig, kleine Amplitude
        band_ratio_thr=0.30,
        std_thr_g=0.038,
        min_rep_density=0.23,   # ~1 rep / 4.3s
        conf_thr=0.45,
        min_strength_duration_s=8.0
    ),

    # --- Schulterdrücken ---
    "SHOULDER_PRESS": dict(
        acf_peak_thr=0.17,      # Totpunkt oben -> Peak kleiner
        band_ratio_thr=0.33,
        std_thr_g=0.042,
        min_rep_density=0.22,
        conf_thr=0.45,
        min_strength_duration_s=9.0
    ),

    # --- Seitheben Kabel ---
    "LATERAL_RAISE_CABLE": dict(
        acf_peak_thr=0.12,      # kleine Range → schwacher Peak
        band_ratio_thr=0.24,
        std_thr_g=0.028,
        min_rep_density=0.18,
        conf_thr=0.35,
        min_strength_duration_s=9.0
    ),

    # --- Bizeps einarmig KH ---
    "BIZEPS_CURL_H": dict(
        acf_peak_thr=0.16,
        band_ratio_thr=0.30,
        std_thr_g=0.040,
        min_rep_density=0.22,
        conf_thr=0.45,
        min_strength_duration_s=8.0
    ),

    # --- Bizeps beidarmig ---
    "BIZEPS_CURL": dict(
        acf_peak_thr=0.15,
        band_ratio_thr=0.28,
        std_thr_g=0.038,
        min_rep_density=0.21,
        conf_thr=0.45,
        min_strength_duration_s=8.0
    ),

    # --- Rudern (Kabel/Maschine) ---
    "RUDERN": dict(
        acf_peak_thr=0.05,      # träge Bewegung → Peak flach
        band_ratio_thr=0.26,
        std_thr_g=0.040,
        min_rep_density=0.21,   # 1 rep ~6–7s ok
        conf_thr=0.42,          # toleranter
        min_strength_duration_s=10.0
    ),
}


def thresholds_for_class(name: str, base: dict):
    """
    Mischt Basis-Defaults mit Klassentunings aus CLASS_THRESH.
    -> erlaubt Feintuning je Übung ohne Re-Training
    """
    u = str(name).upper()
    overrides = CLASS_THRESH.get(u, {})
    cfg = dict(base)
    cfg.update({k: overrides[k] for k in overrides})
    return cfg


# ---------- Guards (pro Segment) ----------
def _segment_signal_guards(ax, ay, az, fs):
    # einfache Messgrößen je Segment; Gate für Postfilter
    sig = select_rep_signal(ax, ay, az, fs)
    std_best = float(np.std(sig)) if len(sig) else 0.0
    _, acf_peak = _acf_primary_peak(sig, fs, 0.4, 3.0)   # <- wieder wie früher
    _, band_ratio, _, _, _ = rfft_band_features(sig, fs, 0.3, 3.0)
    return std_best, float(acf_peak), float(band_ratio)



# ---------- Zentrales Postprocessing ----------
def apply_post_filters(df: pd.DataFrame,
                       segments: List[Dict[str, Any]],
                       probs_s: np.ndarray,
                       classes,
                       fs: float,
                       strength_classes,
                       cfg=POST_DEFAULTS):
    # setzt harte Regeln + Heuristiken nach Klassifizierung
    if not segments:
        return segments

    class_to_idx = {c: i for i, c in enumerate(classes)}
    sc_upper = {s.upper() for s in strength_classes}

    def is_strength(c):
        return str(c).upper() in sc_upper

    def mean_conf_for_segment(seg):
        # mittlere Modell-Confidence über Fenster
        if "i0" in seg and "i1" in seg and probs_s is not None:
            ci = class_to_idx.get(seg["class"], None)
            if ci is None:
                return 0.0
            i0, i1 = int(seg["i0"]), int(seg["i1"])
            i1 = max(i0+1, i1)
            p = probs_s[i0:i1, ci]
            if p.size > 0:
                return float(np.mean(p))
        return 0.0

    t = df["t"].to_numpy(float)
    ax = df["ax"].to_numpy(float)
    ay = df["ay"].to_numpy(float)
    az = df["az"].to_numpy(float)

    # 1) Harte Regeln & Signal-Guards
    for s in segments:
        dur = float(s["duration_s"])
        # — dynamische Schwellen für diese Klasse —
        cfg_c = thresholds_for_class(s["class"], cfg)

        # a) Übung ohne Reps -> REST
        if is_strength(s["class"]) and int(s.get("reps", 0)) == 0:
            s["class"] = "REST"; s["reps"] = 0; continue
        # b) Mindestdauer
        if is_strength(s["class"]) and dur < cfg_c["min_strength_duration_s"]:
            s["class"] = "REST"; s["reps"] = 0; continue
        # c) Guards
        if is_strength(s["class"]):
            mask = (t >= s["t0"]) & (t <= s["t1"])
            if np.count_nonzero(mask) >= int(0.8*fs):
                std_best, acf_peak, band_ratio = _segment_signal_guards(
                    ax[mask], ay[mask], az[mask], fs
                )
                # Basis: gemessene Dichte aus gezählten Reps
                rep_density = (s.get("reps", 0) / max(1e-9, dur))

                # Nur für RUDERN: ACF-basierte Proxy-Dichte als „Rettungsleine“
                if str(s["class"]).upper() == "RUDERN":
                    sig = select_rep_signal(ax[mask], ay[mask], az[mask], fs)
                    p_est = estimate_rep_period_acf(sig, fs, min_s=0.6, max_s=3.5)
                    if np.isfinite(p_est) and p_est > 0 and acf_peak >= RUDERN_ACF_TRUST:
                        exp_density = 1.0 / p_est
                        proxy_density = RUDERN_DENSITY_ALPHA * exp_density
                        rep_density = max(rep_density, proxy_density)

                mc = mean_conf_for_segment(s)  # aufzeichnung für tuning
                if str(s["class"]).upper() == "RUDERN":
                    t0 = s.get("t0", 0.0); t1 = s.get("t1", 0.0)
                    def fmt(sec):
                        sec = int(round(sec)); m = sec // 60; ss = sec % 60
                        return f"{m:02d}:{ss:02d}"
                    print(
                        f"[DBG RUDERN] {fmt(t0)}→{fmt(t1)} ({dur:5.1f}s) | "
                        f"reps={s.get('reps',0):2d} dens={rep_density:.3f} "
                        f"std={std_best:.3f} acf={acf_peak:.3f} band={band_ratio:.3f} "
                        f"mc={mc:.3f}",
                        file=sys.stderr
                    )
                if (std_best < cfg_c["std_thr_g"] or 
                    acf_peak < cfg_c["acf_peak_thr"] or 
                    band_ratio < cfg_c["band_ratio_thr"] or
                    rep_density < cfg_c["min_rep_density"]):
                    s["class"] = "REST"; s["reps"] = 0; continue
            else:
                s["class"] = "REST"; s["reps"] = 0; continue
        # d) Confidence (falls vorhanden)
        if is_strength(s["class"]):
            mc = mean_conf_for_segment(s)
            if mc < cfg_c["conf_thr"]:
                s["class"] = "REST"; s["reps"] = 0; continue

    # 2) Pflicht-Rest zwischen zwei Kraft-Segmenten
    i = 1
    while i < len(segments):
        a, b = segments[i-1], segments[i]
        if is_strength(a["class"]) and is_strength(b["class"]):
            gap = float(b["t0"] - a["t1"])
            if gap < cfg["min_rest_between_sets_s"]:
                dens_a = a.get("reps",0) / max(1e-9, a["duration_s"])
                dens_b = b.get("reps",0) / max(1e-9, b["duration_s"])

                # Bevorzugt höheren Dichte-Satz; bei ~gleich -> längerer gewinnt
                if dens_a > dens_b + 0.02:
                    turn_rest = b
                elif dens_b > dens_a + 0.02:
                    turn_rest = a
                else:
                    turn_rest = a if a["duration_s"] < b["duration_s"] else b
                turn_rest["class"] = "REST"
                turn_rest["reps"]  = 0
                i = max(1, i-1)
                continue
        i += 1

    # 3) REST-Glättung & Merges
    out = []
    for s in segments:
        if out and out[-1]["class"] == "REST" and s["class"] == "REST":
            out[-1]["t1"] = max(out[-1]["t1"], s["t1"])
            out[-1]["duration_s"] = float(out[-1]["t1"] - out[-1]["t0"])
        else:
            out.append(s)

    # 4) REST-Fetzen (<2s) zwischen gleichen Übungen entfernen
    j = 1
    while j+1 < len(out):
        prev, cur, nxt = out[j-1], out[j], out[j+1]
        if (cur["class"] == "REST" and cur["duration_s"] < 2.0 and
            is_strength(prev["class"]) and is_strength(nxt["class"]) and
            prev["class"] == nxt["class"]):
            prev["t1"] = nxt["t1"]
            prev["duration_s"] = float(prev["t1"] - prev["t0"])
            prev["reps"] = int(prev.get("reps",0) + nxt.get("reps",0))
            del out[j:j+2]
            j = max(1, j-1)
            continue
        j += 1

    # 5) „Singleton“-Sets (kurz & einmalig) zu REST kippen
    MIN_TOTAL_PER_CLASS_S = {
        "CABLE_FLY_CHEST": 35.0,
        "BENCH_BB": 35.0,
        "BENCH_DB": 35.0,
    }
    dur_by_class = defaultdict(float)
    for s in out:
        if s["class"] != "REST":
            dur_by_class[s["class"]] += s["duration_s"]
    singletons = {c for c, min_tot in MIN_TOTAL_PER_CLASS_S.items()
                  if dur_by_class.get(c, 0.0) > 0.0 and dur_by_class[c] < min_tot}

    if singletons:
        tmp = []
        for s in out:
            if s["class"] in singletons:
                s = dict(s)
                s["class"] = "REST"; s["reps"] = 0
            if tmp and tmp[-1]["class"] == "REST" and s["class"] == "REST":
                tmp[-1]["t1"] = max(tmp[-1]["t1"], s["t1"])
                tmp[-1]["duration_s"] = float(tmp[-1]["t1"] - tmp[-1]["t0"])
            else:
                tmp.append(s)
        out = tmp

    return out
