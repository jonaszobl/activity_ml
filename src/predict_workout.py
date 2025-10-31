#-------------------------------------------
# v0.1.3
# Prediction mit Klassenspezifischen Schwellen
# -> vorher fälschlich erkannte Übungen bei gehen -> sollte ruhen sein
#--------------------------------------------

import json
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .utils_jsonl import read_jsonl
from .features import build_windows, rfft_band_features 

# ---------- Utils / berechnungen... ----------
def softmax(z):
    # softmax mit numerischer Stabilisierung (Shift um max)
    z = np.asarray(z, float)
    m = np.max(z)
    e = np.exp(z - m)
    s = np.sum(e)
    return e / s if s > 0 else np.ones_like(e) / len(e)

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    # gleitender mittelwert in O(n) via cumsum
    if k <= 1:
        return x.astype(float, copy=True)
    k = int(k)
    x = x.astype(float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[k:] - c[:-k]) / k
    pad = np.full(k - 1, y[0] if len(y) else 0.0, dtype=float)
    return np.concatenate([pad, y]).astype(float)

def median(a):
    # robuster Mittelpunkt (Median)
    a = np.asarray(a, float)
    if len(a) == 0:
        return 0.0
    return float(np.median(a))

def mad(a):
    # (Median Absolute Deviation) – guter Streuungsindikator
    m = median(a)
    return float(np.median(np.abs(np.asarray(a, float) - m)))

# ---------- Peak-basierte Zählung (optional) ----------
def count_peaks(signal: np.ndarray, fs: float,
                min_separation_s: float = 0.4,
                thresh_mode: str = "median_mad",
                prominence: float = 0.0) -> int:
    # sehr simpler peak-Zähler mit minabstand + adaptiver Schwelle
    if len(signal) < 3:
        return 0
    if thresh_mode == "median_mad":
        th = median(signal) + 0.5 * mad(signal)  # konservative Schwelle
    else:
        th = median(signal)

    min_dist = max(1, int(round(min_separation_s * fs)))
    last = -min_dist
    peaks = 0
    for i in range(1, len(signal) - 1):
        if i - last < min_dist:
            continue
        s0, s1, s2 = signal[i - 1], signal[i], signal[i + 1]
        if s1 > th and s1 > s0 and s1 > s2:
            if prominence > 0.0:
                # „how much above baseline?“ – gegen Mini-Peaks/Lärm
                left = max(0, i - min_dist)
                right = min(len(signal) - 1, i + min_dist)
                base = max(median(signal[left:i]), median(signal[i:right + 1]))
                if (s1 - base) < prominence:
                    continue
            peaks += 1
            last = i
    return peaks

# ---------- Pair-Zählung (empfohlen) ----------
def count_reps_peak_trough(signal: np.ndarray, fs: float,
                           k: float = 0.6,
                           min_rep_s: float = 0.6,
                           max_rep_s: float = 4.0) -> int:
    # zählt wh als peak Trough-Paare; robust g. Drift/Offsets
    n = len(signal)
    if n < 3:
        return 0
    med = median(signal)
    m = mad(signal)
    if m < 1e-6:
        # fallback wenn MAD tot ist -> eher unüblich bei übungen
        m = np.std(signal) * 0.8
    if m < 1e-6:
        return 0

    up = med + k * m      # obere Schwelle
    down = med - k * m    # untere Schwelle

    last_ext_t = None
    last_ext_type = None
    reps = 0

    # reps zählen -> noch ungenau / überarbeiten!!!

    for i in range(1, n - 1):
        s0, s1, s2 = signal[i - 1], signal[i], signal[i + 1]
        t = i / fs
        if s1 > up and s1 > s0 and s1 > s2:
            # peak gefunden
            if last_ext_type == "trough":
                dt = t - last_ext_t
                if min_rep_s <= dt <= max_rep_s:
                    reps += 1
            last_ext_t = t
            last_ext_type = "peak"
        elif s1 < down and s1 < s0 and s1 < s2:
            # trough gefunden
            if last_ext_type == "peak":
                dt = t - last_ext_t
                if min_rep_s <= dt <= max_rep_s:
                    reps += 1
            last_ext_t = t
            last_ext_type = "trough"
    return reps

# ---------- Signalwalh & Periodenschätzung ----------
def highpass_ma(x: np.ndarray, fs: float, cutoff_s: float = 0.7) -> np.ndarray:
    # sehr simpler Highpass: x - moving_avg (Trend weg)
    k = max(1, int(round(cutoff_s * fs)))
    trend = moving_average(x, k)
    y = x.astype(float) - trend
    return y

def _acf_primary_peak(signal: np.ndarray, fs: float,
                      min_s: float, max_s: float):
    # ACF: finde ersten „bedeutsamen“ Peak -> grobe Periodenabschätzung
    n = len(signal)
    if n < int(fs * min_s) + 3:
        return 0.0, 0.0
    x = signal - np.mean(signal)
    std = np.std(x)
    if not np.isfinite(std) or std < 1e-8:
        return 0.0, 0.0

    ac = np.correlate(x, x, mode="full")[n-1:]
    ac /= (ac[0] + 1e-12)

    min_lag = max(1, int(round(min_s * fs)))
    max_lag = min(n - 2, int(round(max_s * fs)))
    if max_lag <= min_lag:
        return 0.0, 0.0

    lag = None
    # first local max (bewusst konservativ), sonst globaler im Fenster
    for i in range(min_lag + 1, max_lag - 1):
        if ac[i] > ac[i - 1] and ac[i] > ac[i + 1]:
            lag = i
            break
    if lag is None:
        lag = min_lag + int(np.argmax(ac[min_lag:max_lag]))
    return (lag / fs), float(ac[lag])

def estimate_rep_period_acf(signal: np.ndarray, fs: float,
                            min_s: float = 0.4, max_s: float = 3.0) -> float:
    # nur die Periode zurück (Wrapper)
    p, _ = _acf_primary_peak(signal, fs, min_s, max_s)
    return p

def select_rep_signal(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float) -> np.ndarray:
    # Wähle bestes „Reps“-Signal aus (ax/ay/az/magnitude)
    # Scoring primär via ACF-Peak, sonst std-Fallback.
    xhp = highpass_ma(ax, fs, 0.7)
    yhp = highpass_ma(ay, fs, 0.7)
    zhp = highpass_ma(az, fs, 0.7)
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    maghp = highpass_ma(mag, fs, 0.7)

    k = max(1, int(round(0.16 * fs)))  # leichter Smooth (ca.160 ms)
    xs = moving_average(xhp, k)
    ys = moving_average(yhp, k)
    zs = moving_average(zhp, k)
    ms = moving_average(maghp, k)

    cands = [xs, ys, zs, ms]
    scores = []
    for sig in cands:
        _, peak = _acf_primary_peak(sig, fs, 0.4, 3.0)
        scores.append(peak if np.isfinite(peak) else 0.0)
    best = int(np.argmax(scores))
    if scores[best] < 0.05:
        # schwacher ACF-Peak -> nimm die Achse mit größter Std -> fallback
        stds = np.array([np.std(c) for c in cands])
        best = int(np.argmax(stds))
    return cands[best]

# ---------- Modell / IO ----------
def load_model(path="artifacts/model.json"):
    # leichtgewichtiges JSON-Modell laden (keine sklearn runtime notwendig)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_features(X: np.ndarray, M: dict):
    # reine NumPy-Inferenz: z-transform -> linear -> softmax
    mean = np.asarray(M["scaler_mean"], float)
    scale = np.asarray(M["scaler_scale"], float)
    Xn = (X - mean) / scale
    W = np.asarray(M["W"], float)     # [C,F]
    b = np.asarray(M["b"], float)     # [C]
    logits = Xn @ W.T + b             # [N,C]
    probs = np.apply_along_axis(softmax, 1, logits)
    cls_idx = np.argmax(probs, axis=1)
    return cls_idx, probs

# ---------- DataFrame Helpers ----------
def ensure_time_column_df(df: pd.DataFrame) -> pd.DataFrame:
    # stelle sicher, dass 't' existiert (oder aus t_rel abgeleitet wird)
    if "t" in df.columns:
        t = pd.to_numeric(df["t"], errors="coerce")
    elif "t_rel" in df.columns:
        t_rel = pd.to_numeric(df["t_rel"], errors="coerce")
        t = t_rel - t_rel.iloc[0]  # 0-basieren
    else:
        raise KeyError("Weder 't' noch 't_rel' in Datei gefunden.")
    df = df.copy()
    df["t"] = t
    need = {"ax", "ay", "az"}
    missing = [c for c in need if c not in df.columns]
    assert not missing, f"Spalten fehlen: {missing}"
    df = df.dropna(subset=["t"])
    return df

# ---------- Post-Processing für stabilere Segmente ----------
def smooth_probs_over_time(probs: np.ndarray, k: int = 5) -> np.ndarray:
    # Zeitglättung der Klassenwahrscheinlichkeiten (moving avg über k)
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
    return out / s

def debounce_labels(cls_idx: np.ndarray, min_run: int = 3) -> np.ndarray:
    # min-Laufzeit bis zum Klassenwechsel (gegen kurze Zaps)
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
    # fenstervorhersagen in zusammenhängende Segmente packen
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
    # zu kurze Segmente an Nachbarn mergen (problem mit zu häufigen wechseln...)
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
                # default: an den kürzeren Nachbarn anhängen (min damage)
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

def strength_classes_from(M):
    # Klassenset um „Nicht-Kraft“ bereinigen (dort keine Reps etc.)
    exclude = {"REST", "PAUSE", "WALKING", "RUNNING"}
    return {c for c in M["classes"] if str(c).upper() not in exclude}

def seconds_to_hms(sec: float) -> str:
    # (m:ss bzw. h:mm:ss)
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

# ---------- Klassen-Heuristiken (für Reps) ----------
def class_name_key(name: str) -> str:
    # gruppierung diverser Schreibweisen auf gemeinsame keys
    u = str(name).upper()
    if "TRICEPS" in u and "PULL" in u:
        return "TRICEPS_PULLDOWN"
    if "FLY" in u or ("CABLE" in u and "CHEST" in u):
        return "CABLE_FLY"
    if "BENCH" in u or "BANKDR" in u:
        return "BENCH_PRESS"
    if "SHOULDER" in u and "PRESS" in u:
        return "SHOULDER_PRESS"
    return "DEFAULT"

def rep_params_for_class(name: str, base_k: float, base_min: float, base_max: float):
    # per-Klasse leichte anpassungen der rep-Parameter (k, min/max Dauer)
    key = class_name_key(name)
    if key == "CABLE_FLY":
        # fly ist langsamer + „weicher“ -> k runter, max etwas großzügiger
        k = max(0.30, base_k - 0.20)
        return k, base_min, max(base_max, base_min + 2.6)
    if key == "TRICEPS_PULLDOWN":
        # trizeps-Zug oft sauberer Rhythmus -> k leicht rauf, min leicht höher
        k = min(1.0, base_k + 0.10)
        return k, max(base_min, 0.70), base_max
    if key == "SHOULDER_PRESS":
        return base_k, base_min, base_max
    if key == "BENCH_PRESS":
        return base_k, base_min, base_max
    return base_k, base_min, base_max

# ---------- Post-Filter Defaults (Basis) ----------
POST_DEFAULTS = dict(
    # Basis-Schwellen zur FP-Reduktion.
    # Note: wurden iterativ getuned -> starke Probleme ohne tuning bezüglich herumgehen
    min_strength_duration_s = 8.0,   # <8s -> meist kein echter Satz sondern Bewegungsvorbereitung
    min_rest_between_sets_s = 10.0,  # harte Ruhepflicht zw. Sets
    acf_peak_thr            = 0.18,  # Periodizität muss spürbar sein (darunter viel Rauschen gesehen)
    band_ratio_thr          = 0.35,  # Anteil 0.3–3 Hz: unterhalb oft Walking/Random Move
    std_thr_g               = 0.05,  # Mindest-Amplitude (nach HP+Smooth)
    min_rep_density         = 0.25,  # reps/sec – zu dünn ⇒ eher kein Set
    conf_thr                = 0.50   # mittlere Modell-Confidence pro Segment
)

# ============================================
# Klassenspezifische Schwellen
# frühere Zahlen, was kaputtging, und warum es jetzt so ist.
# ============================================
CLASS_THRESH = {

    # --- Brust / Cable Fly ---
    # Historie:
    #   v0: acf_peak_thr=0.20 war zu hart -> echte, „weiche“ Sätze (langsamer, gleichmäßig)
    #       wurden oft zu REST gedreht (false negatives).
    #   v1: band_ratio_thr=0.30 -> immer noch Outfilter bei kontrollierten Reps.
    # Current-Tuning:
    "CABLE_FLY_CHEST": dict(
        acf_peak_thr=0.13,      # runtergesetzt (Fly hat flachere Peaks)
        band_ratio_thr=0.25,    # weniger Bandenergie akzeptieren (slow & controlled)
        std_thr_g=0.030,        # Handgelenk bewegt sich wenig -> niedrige Amplitude zulassen
        min_rep_density=0.17,   # ~1 rep pro ~6s ok (bewusst entspannt)
        conf_thr=0.40,          # Modell häufig unsicher ->gnädiger
        min_strength_duration_s=10.0  # Sätze dauern eher länger
    ),

    # --- Bankdrücken Langhantel ---
    # Historie:
    #   v0: std_thr_g=0.055 -> kurze, saubere Sätze ok; aber langsame lange Sätze fielen raus.
    #   v1: acf_peak_thr=0.18 -> etwas zu viel REST bei schweren Sets (langsamer Rhythmus).
    "BENCH_BB": dict(
        acf_peak_thr=0.16,      # Periodizität etwas lockern ggü. v1
        band_ratio_thr=0.32,    # moderat – zu tief ließ Walking durch
        std_thr_g=0.045,        # minimal geringer, damit schwere reps (kontrolliert) drin bleiben
        min_rep_density=0.22,   # ~1 rep pro 4.5s
        conf_thr=0.45,
        min_strength_duration_s=8.0
    ),

    # --- Bankdrücken Kurzhanteln ---
    # Historie:
    #   v0: band_ratio_thr=0.32 (wie BB) -> zu viele Drops bei DB (mehr Stabilisationsarbeit).
    "BENCH_DB": dict(
        acf_peak_thr=0.15,      # DB ungleichmäßiger -> Peak kleiner ok
        band_ratio_thr=0.28,    # etwas entspannter als BB
        std_thr_g=0.040,        # durch Stabilisierung geringere Amplituden
        min_rep_density=0.20,
        conf_thr=0.40,
        min_strength_duration_s=9.0
    ),

    # --- Trizeps-Kabelzug ---
    # Historie:
    #   v0: std_thr_g=0.030 ->ließ zu viel „Kabelzappeln“ durch (FP).
    #   v1: std_thr_g=0.040 -> echte leichte Sätze fielen teils weg.
    "TRICEPS_PULLDOWN": dict(
        acf_peak_thr=0.15,      # guter Rhythmus, aber Peak nicht riesig
        band_ratio_thr=0.30,
        std_thr_g=0.038,        # sweet spot zw. v0/v1 (FP runter, FN runter)
        min_rep_density=0.23,   # ~1 rep pro 4.3s
        conf_thr=0.45,
        min_strength_duration_s=8.0
    ),

    # --- Schulterdrücken ---
    # Historie:
    #   v0: acf_peak_thr=0.20 -> viele langsame reps (oben Halt) fälschlich verworfen.
    "SHOULDER_PRESS": dict(
        acf_peak_thr=0.17,      # Peak moderat
        band_ratio_thr=0.33,
        std_thr_g=0.042,        # etwas mehr Bewegung als Fly, weniger als BB
        min_rep_density=0.22,
        conf_thr=0.45,
        min_strength_duration_s=9.0
    ),

    # --- Seitheben (Kabel) ---
    # Historie:
    #   v0: std_thr_g=0.035 -> ließ Tritt-/Alltagsbewegung eher rein (zu viele FP im Sitzen).
    "LATERAL_RAISE_CABLE": dict(
        acf_peak_thr=0.12,      # kleine Range -> Peak schwach
        band_ratio_thr=0.24,    # Bandanteil niedrig halten, sonst REST→SET flips
        std_thr_g=0.028,        # sehr kleine Amplitude tolerieren (aber mit Guards oben!)
        min_rep_density=0.18,
        conf_thr=0.35,          # Modell unsicher -> etwas toleranter
        min_strength_duration_s=9.0
    ),

    # --- Bizeps einarmig KH ---
    # Historie:
    #   v0: band_ratio_thr=0.33 -> zu streng, echte langsame reps fielen raus.
    "BIZEPS_CURL_H": dict(
        acf_peak_thr=0.16,
        band_ratio_thr=0.30,    # leicht abgesenkt ggü. v0
        std_thr_g=0.040,
        min_rep_density=0.22,
        conf_thr=0.45,
        min_strength_duration_s=8.0
    ),

    # --- Bizeps beidarmig ---
    # Historie:
    #   v0: std_thr_g=0.045 -> zu hart, clean reps (sauber, wenig Schwung) gedroppt.
    "BIZEPS_CURL": dict(
        acf_peak_thr=0.15,      # gleichmäßig, Peak etwas kleiner
        band_ratio_thr=0.28,
        std_thr_g=0.038,        # etwas weicher als v0
        min_rep_density=0.21,
        conf_thr=0.45,
        min_strength_duration_s=8.0
    ),

    # --- Rudern (Kabel/Maschine) ---
    # Historie:
    #   v0: band_ratio_thr=0.26 -> bad: zu viele FP bei Sitzen → runter?
    #   v1: std_thr_g=0.030 -> dann zu viele FPs beim Gehen. Jetzt Mittelweg.
    "RUDERN": dict(
        acf_peak_thr=0.12,      # träge Bewegung -> Peak flacher ok
        band_ratio_thr=0.22,    # tiefer, um echte langsame Züge nicht zu killen
        std_thr_g=0.035,        # nicht zu niedrig (sonst Geh-FP), nicht zu hoch (sonst FN)
        min_rep_density=0.15,   # 1 rep ~6–7s akzeptieren
        conf_thr=0.35,          # oft unsicher -> toleranter
        min_strength_duration_s=9.0
    ),
}

def thresholds_for_class(name: str, base: dict):
    """
    Mix aus Basis-Defaults und Klassen-Overrides.
    So können wir heuristisch feintunen (A/B über CLI möglich), ohne Modell neu zu trainieren.
    """
    u = str(name).upper()
    overrides = CLASS_THRESH.get(u, {})
    cfg = dict(base)
    cfg.update({k: overrides[k] for k in overrides})
    return cfg

# ---------- Guards (pro Segment) ----------
def _segment_signal_guards(ax, ay, az, fs):
    # Schutzschranken je Segment: bestes Signal + std, ACF-Peak, Band-Ratio
    sig = select_rep_signal(ax, ay, az, fs)
    std_best = float(np.std(sig)) if len(sig) else 0.0
    _, acf_peak = _acf_primary_peak(sig, fs, 0.4, 3.0)
    _, band_ratio, _, _, _ = rfft_band_features(sig, fs, 0.3, 3.0)
    return std_best, float(acf_peak), float(band_ratio)

# ---------- Zentrales Postprocessing ----------
def apply_post_filters(df, segments, probs_s, classes, fs, strength_classes, cfg=POST_DEFAULTS):
    # hier widr aussortiert/vereint: harte Regeln, Signalqualitätschecks, Confidence, Merges
    if not segments:
        return segments

    class_to_idx = {c: i for i, c in enumerate(classes)}
    sc_upper = {s.upper() for s in strength_classes}

    def is_strength(c):
        return str(c).upper() in sc_upper

    def mean_conf_for_segment(seg):
        # mittlere (geglättete) Confidence der vorhergesagten Klasse
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

    # 1) Regeln/Guards je Segment anwenden
    for s in segments:
        dur = float(s["duration_s"])
        cfg_c = thresholds_for_class(s["class"], cfg)

        # a) Kraft ohne Reps → ziemlich sicher kein Set -> REST
        if is_strength(s["class"]) and int(s.get("reps", 0)) == 0:
            s["class"] = "REST"; s["reps"] = 0; continue
        # b) Mindestdauer nicht erreicht → REST
        if is_strength(s["class"]) and dur < cfg_c["min_strength_duration_s"]:
            s["class"] = "REST"; s["reps"] = 0; continue
        # c) Signal-Guards (Std/Periodizität/Bandanteil/Dichte)
        if is_strength(s["class"]):
            mask = (t >= s["t0"]) & (t <= s["t1"])
            if np.count_nonzero(mask) >= int(0.8*fs):
                std_best, acf_peak, band_ratio = _segment_signal_guards(ax[mask], ay[mask], az[mask], fs)
                rep_density = (s.get("reps", 0) / max(1e-9, dur))
                if (std_best < cfg_c["std_thr_g"] or 
                    acf_peak < cfg_c["acf_peak_thr"] or 
                    band_ratio < cfg_c["band_ratio_thr"] or
                    rep_density < cfg_c["min_rep_density"]):
                    s["class"] = "REST"; s["reps"] = 0; continue
            else:
                # zu wenig Samples im Segment → lieber REST (conservative)
                s["class"] = "REST"; s["reps"] = 0; continue
        # d) Confidence-Floor (Modell unsicher → REST)
        if is_strength(s["class"]):
            mc = mean_conf_for_segment(s)
            if mc < cfg_c["conf_thr"]:
                s["class"] = "REST"; s["reps"] = 0; continue

    # 2) Pflicht-Ruhe zwischen zwei Kraft-Segmenten (gegen Set-Fragmente)
    i = 1
    while i < len(segments):
        a, b = segments[i-1], segments[i]
        if is_strength(a["class"]) and is_strength(b["class"]):
            gap = float(b["t0"] - a["t1"])
            if gap < cfg["min_rest_between_sets_s"]:
                # das schwächere/kurze zum REST drehen (heuristic „pick loser“)
                ka = (a.get("reps",0), a["duration_s"])
                kb = (b.get("reps",0), b["duration_s"])
                turn_rest = a if (ka < kb) else b
                turn_rest["class"] = "REST"
                turn_rest["reps"]  = 0
                i = max(1, i-1)
                continue
        i += 1

    # 3) REST zusammenführen (Back-to-Back)
    out = []
    for s in segments:
        if out and out[-1]["class"] == "REST" and s["class"] == "REST":
            out[-1]["t1"] = max(out[-1]["t1"], s["t1"])
            out[-1]["duration_s"] = float(out[-1]["t1"] - out[-1]["t0"])
        else:
            out.append(s)

    # 4) Sehr kurze REST-Fetzen (<2s) zwischen gleichen Übungen entfernen (Set join)
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

    # 5) Singleton-Kraftblitze (gesamt zu kurz) -> REST (FP-Cleanup)
    from collections import defaultdict
    MIN_TOTAL_PER_CLASS_S = {
        # kurze einmalige Blitze dieser Klassen -> in Praxis meist Fehlklassis
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

# ---------- Hauptprogramm ----------
def main():
    ap = argparse.ArgumentParser(description="Segmentierung & Erkennung eines Workouts aus JSONL.")
    ap.add_argument("infile", help="Pfad zur JSONL-Datei (t oder t_rel).")
    ap.add_argument("--model", default="artifacts/model.json", help="Pfad zum exportierten Modell.")
    # Stabilisierung:
    ap.add_argument("--prob_smooth_k", type=int, default=5, help="Gleitendes Mittel über k Fenster (Probs). 1=aus.")
    ap.add_argument("--debounce_run", type=int, default=3, help="Minimale Fensteranzahl für Klassenwechsel.")
    ap.add_argument("--merge_min_s", type=float, default=4.0, help="Segmente kürzer als diese Dauer werden gemergt.")
    # Rep-Counting:
    ap.add_argument("--smooth_sec", type=float, default=0.2, help="Glättung der Basis-Achsen (nur für peaks-Mode).")
    ap.add_argument("--min_peak_sep", type=float, default=0.4, help="Min. Peak-Abstand (s) für 'peaks'-Modus.")
    ap.add_argument("--rep_mode", choices=["peaks", "pair"], default="pair",
                    help="Wdh.-Zählung: 'peaks' (alt) oder 'pair' (Peak↔Trough).")
    ap.add_argument("--rep_min_s", type=float, default=0.6, help="Basis-Min. Dauer einer Wdh. (s) im 'pair'-Modus.")
    ap.add_argument("--rep_max_s", type=float, default=4.0, help="Basis-Max. Dauer einer Wdh. (s) im 'pair'-Modus.")
    ap.add_argument("--rep_k", type=float, default=0.6, help="Basis-Schwellfaktor k*MAD im 'pair'-Modus.")
    # ACF-Adaptivität:
    ap.add_argument("--acf_enable", action="store_true", help="Nutze Autokorrelation zur Anpassung von Min/Max.")
    ap.add_argument("--acf_min_s", type=float, default=0.45, help="ACF: minimale Periodensuche (s).")
    ap.add_argument("--acf_max_s", type=float, default=3.0, help="ACF: maximale Periodensuche (s).")
    ap.add_argument("--acf_band", type=float, nargs=2, default=[0.6, 1.8],
                    help="Skalierung der ACF-Periode -> [min,max] Faktor (z. B. 0.6 1.8).")
    # Mindest-Wdh. als Übungs-Kriterium:
    ap.add_argument("--min_reps", type=int, default=4, help="Unterhalb nicht als Übung zählen.")
    ap.add_argument("--below_min_policy", choices=["rest", "keep", "drop"], default="rest",
                    help="Was tun bei <min_reps in Kraftsegmenten: 'rest' umlabeln, 'keep' belassen, 'drop' verwerfen.")
    # Post-Filter Overrides (optional) – Basiswerte
    ap.add_argument("--post_min_strength_sec", type=float, default=POST_DEFAULTS["min_strength_duration_s"])
    ap.add_argument("--post_min_rest_between_sec", type=float, default=POST_DEFAULTS["min_rest_between_sets_s"])
    ap.add_argument("--post_acf_peak_thr", type=float, default=POST_DEFAULTS["acf_peak_thr"])
    ap.add_argument("--post_band_ratio_thr", type=float, default=POST_DEFAULTS["band_ratio_thr"])
    ap.add_argument("--post_std_thr_g", type=float, default=POST_DEFAULTS["std_thr_g"])
    ap.add_argument("--post_min_rep_density", type=float, default=POST_DEFAULTS["min_rep_density"])
    ap.add_argument("--post_conf_thr", type=float, default=POST_DEFAULTS["conf_thr"])

    args = ap.parse_args()

    infile = Path(args.infile)
    assert infile.exists(), f"Datei nicht gefunden: {infile}"

    # 1) Modell laden
    M = load_model(args.model)
    classes = M["classes"]
    fs   = float(M["meta"]["fs_hz"])
    winS = float(M["meta"]["win_s"])
    hopS = float(M["meta"]["hop_s"])
    strength_classes = strength_classes_from(M)

    # 2) Workout laden
    rows = list(read_jsonl(str(infile)))
    assert rows, "Datei leer?"
    df = pd.DataFrame(rows)
    df = ensure_time_column_df(df).sort_values("t").reset_index(drop=True)

    # 3) Fenster + Features wie im Training (wichtig: gleiche fs/win/hop)
    X, _, t0s, feat_names = build_windows(df, fs, winS, hopS)
    assert X.shape[0] > 0, "Keine Fenster erzeugt – ist die Datei lang genug?"

    model_feats = M.get("feature_names", [])
    assert len(model_feats) == X.shape[1], (
        f"Feature-Anzahl passt nicht (Datei: {X.shape[1]} vs. Modell: {len(model_feats)}). "
        f"Hinweis: Nach Änderungen an features.py neu trainieren."
    )
    if model_feats and feat_names != model_feats:
        print("[WARN] Feature-Namen/Reihenfolge weichen vom Modell ab. Trainiere Modell ggf. neu.", file=sys.stderr)

    # 4) Vorhersagen je Fenster
    cls_idx, probs = predict_features(X, M)

    # 4a) Wahrscheinlichkeiten über Zeit glätten (stabilere Segmente)
    probs_s = smooth_probs_over_time(probs, k=max(1, int(args.prob_smooth_k)))
    cls_idx = np.argmax(probs_s, axis=1)

    # 4b) Entprellen der Klassenlabels
    cls_idx = debounce_labels(cls_idx, min_run=max(1, int(args.debounce_run)))

    # 5) Fenster → Segmente mappen
    segments = segment_from_window_preds(t0s, cls_idx, classes, winS)

    # 5a) Mini-Segmente mergen (gegen Segment-Zerfaserung)
    if args.merge_min_s and args.merge_min_s > 0:
        segments = merge_short_segments(segments, min_len_s=float(args.merge_min_s), prefer="neighbor")

    # 6) Rep-Counting je Segment (nur Kraftklassen)
    results = []
    t  = df["t"].to_numpy(float)
    ax = df["ax"].to_numpy(float)
    ay = df["ay"].to_numpy(float)
    az = df["az"].to_numpy(float)

    smooth_k = max(1, int(round(args.smooth_sec * fs)))
    az_smooth = moving_average(az, smooth_k)

    for seg in segments:
        seg_class = seg["class"]
        mask = (t >= seg["t0"]) & (t <= seg["t1"])

        reps = 0
        if seg_class in strength_classes and np.count_nonzero(mask) > 3:
            if args.rep_mode == "pair":
                # bestes Reps-Signal wählen
                sig = select_rep_signal(ax[mask], ay[mask], az[mask], fs)
                # per-Klasse Basisschwellen anpassen
                k0, min0, max0 = rep_params_for_class(seg_class,
                                                      base_k=float(args.rep_k),
                                                      base_min=float(args.rep_min_s),
                                                      base_max=float(args.rep_max_s))

                # optional: ACF-basierte Adaption der min/max Rep-Dauer
                if args.acf_enable:
                    p = estimate_rep_period_acf(sig, fs,
                                                min_s=float(args.acf_min_s),
                                                max_s=float(args.acf_max_s))
                    if p > 0:
                        lo_fac, hi_fac = float(args.acf_band[0]), float(args.acf_band[1])
                        min_s = max(0.35, min(p * lo_fac, max0))
                        max_s = max(min0, min(p * hi_fac, max0))
                    else:
                        min_s, max_s = min0, max0
                else:
                    min_s, max_s = min0, max0

                if min_s >= max_s:
                    # sanity fallback (sollte praktisch nicht passieren)
                    min_s = min0; max_s = max0

                reps = count_reps_peak_trough(sig, fs, k=k0, min_rep_s=min_s, max_rep_s=max_s)

                if reps == 0:
                    # zweiter Versuch mit etwas softerem k (gegen knappe Fälle)
                    k_try = max(0.25, k0 - 0.1)
                    reps = count_reps_peak_trough(sig, fs, k=k_try, min_rep_s=min_s, max_rep_s=max_s)
            else:
                # legacy: peaks-mode über geglättete az
                az_seg = az_smooth[mask]
                reps = count_peaks(
                    az_seg, fs,
                    min_separation_s=float(args.min_peak_sep),
                    thresh_mode="median_mad",
                    prominence=0.5 * mad(az_seg)
                )

        out_class = seg_class
        if seg_class in strength_classes:
            # Mindest-Reps-Policy
            if args.below_min_policy == "drop" and reps < int(args.min_reps):
                continue
            if args.below_min_policy == "rest" and reps < int(args.min_reps):
                out_class = "REST"
                reps = 0

        seg_out = {
            "t0": float(seg["t0"]),
            "t1": float(seg["t1"]),
            "duration_s": float(seg["t1"] - seg["t0"]),
            "class": out_class,
            "reps": int(reps) if out_class not in {"REST", "PAUSE", "WALKING", "RUNNING"} else 0,
            "i0": int(seg["i0"]),
            "i1": int(seg["i1"]),
        }
        results.append(seg_out)

    # 7) Post-Filter anwenden (inkl. evtl. CLI-Overrides auf Basiswerte)
    POST_DEFAULTS.update(dict(
        min_strength_duration_s = float(args.post_min_strength_sec),
        min_rest_between_sets_s = float(args.post_min_rest_between_sec),
        acf_peak_thr            = float(args.post_acf_peak_thr),
        band_ratio_thr          = float(args.post_band_ratio_thr),
        std_thr_g               = float(args.post_std_thr_g),
        min_rep_density         = float(args.post_min_rep_density),
        conf_thr                = float(args.post_conf_thr),
    ))
    results = apply_post_filters(df, results, probs_s, classes, fs, strength_classes, cfg=POST_DEFAULTS)

    # 8) Ausgabe (human readable)
    print("\nVorhersage-Segmente:")
    for s in results:
        dur = s["duration_s"]
        def hhmmss(sec):
            sec = int(round(sec))
            h = sec // 3600; m = (sec % 3600) // 60; sc = sec % 60
            return f"{h:d}:{m:02d}:{sc:02d}" if h > 0 else f"{m:d}:{sc:02d}"
        line = f"- {s['class']:20s}  {hhmmss(s['t0'])} → {hhmmss(s['t1'])}  ({dur:5.1f}s)"
        if s["reps"] > 0:
            line += f"  | reps: {s['reps']}"
        print(line)

    # 9) JSON-Export (für Frontend/Logs)
    out_json = infile.with_suffix(".pred.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "model_version": M.get("version", "v1"),
            "fs_hz": fs,
            "win_s": winS,
            "hop_s": hopS,
            "segments": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\nErgebnis gespeichert: {out_json}")

if __name__ == "__main__":
    main()
