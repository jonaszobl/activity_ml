# src/predict_workout.py
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .utils_jsonl import read_jsonl
from .features import build_windows


# ---------- Mathe / Utils ----------
def softmax(z):
    z = np.asarray(z, float)
    m = np.max(z)
    e = np.exp(z - m)
    s = np.sum(e)
    return e / s if s > 0 else np.ones_like(e) / len(e)

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Einfaches gleitendes Mittel (Fenster k, unzentriert)."""
    if k <= 1:
        return x.astype(float, copy=True)
    k = int(k)
    x = x.astype(float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[k:] - c[:-k]) / k
    pad = np.full(k - 1, y[0] if len(y) else 0.0, dtype=float)
    return np.concatenate([pad, y]).astype(float)

def median(a):
    a = np.asarray(a, float)
    if len(a) == 0:
        return 0.0
    return float(np.median(a))

def mad(a):
    """Median Absolute Deviation."""
    m = median(a)
    return float(np.median(np.abs(np.asarray(a, float) - m)))


# ---------- Peak-basierte Zählung (optional) ----------
def count_peaks(signal: np.ndarray, fs: float,
                min_separation_s: float = 0.4,
                thresh_mode: str = "median_mad",
                prominence: float = 0.0) -> int:
    """
    Einfache Peak-Erkennung (für --rep_mode peaks).
    """
    if len(signal) < 3:
        return 0
    if thresh_mode == "median_mad":
        th = median(signal) + 0.5 * mad(signal)
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
    """
    Zählt Wiederholungen als Peak↔Trough-Transitions mit dynamischen Schwellwerten.
    """
    n = len(signal)
    if n < 3:
        return 0
    med = median(signal)
    m = mad(signal)
    # Falls sehr geringe Amplitude: Fallback auf Std
    if m < 1e-6:
        m = np.std(signal) * 0.8
    if m < 1e-6:
        return 0

    up = med + k * m
    down = med - k * m

    last_ext_t = None
    last_ext_type = None  # "peak" | "trough"
    reps = 0

    for i in range(1, n - 1):
        s0, s1, s2 = signal[i - 1], signal[i], signal[i + 1]
        t = i / fs
        # lokaler Peak über up
        if s1 > up and s1 > s0 and s1 > s2:
            if last_ext_type == "trough":
                dt = t - last_ext_t
                if min_rep_s <= dt <= max_rep_s:
                    reps += 1
            last_ext_t = t
            last_ext_type = "peak"
        # lokales Minimum unter down
        elif s1 < down and s1 < s0 and s1 < s2:
            if last_ext_type == "peak":
                dt = t - last_ext_t
                if min_rep_s <= dt <= max_rep_s:
                    reps += 1
            last_ext_t = t
            last_ext_type = "trough"
    return reps


# ---------- Signalwahl & Periodenschätzung ----------
def highpass_ma(x: np.ndarray, fs: float, cutoff_s: float = 0.7) -> np.ndarray:
    """
    Sehr einfacher Highpass: remove gravity via moving-average subtraction.
    cutoff_s ≈ Fensterlänge für Trend (Gravitation) – 0.6–1.0s funktioniert gut.
    """
    k = max(1, int(round(cutoff_s * fs)))
    trend = moving_average(x, k)
    y = x.astype(float) - trend
    return y

def _acf_primary_peak(signal: np.ndarray, fs: float,
                      min_s: float, max_s: float):
    """
    Liefert (period_s, peak_val) des ersten signifikanten ACF-Peaks im Suchfenster.
    """
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

    # Erster lokaler Max im Fenster
    lag = None
    for i in range(min_lag + 1, max_lag - 1):
        if ac[i] > ac[i - 1] and ac[i] > ac[i + 1]:
            lag = i
            break
    if lag is None:
        lag = min_lag + int(np.argmax(ac[min_lag:max_lag]))
    return (lag / fs), float(ac[lag])

def estimate_rep_period_acf(signal: np.ndarray, fs: float,
                            min_s: float = 0.4, max_s: float = 3.0) -> float:
    p, _ = _acf_primary_peak(signal, fs, min_s, max_s)
    return p

def select_rep_signal(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float) -> np.ndarray:
    """
    Wählt pro Segment das „beste“ Signal:
    - Kandidaten: highpass+smooth für ax, ay, az, sowie Magnitude sqrt(ax^2+ay^2+az^2)
    - Bewertung: ACF-Peakhöhe; Fallback: größte Std
    """
    # Highpass
    xhp = highpass_ma(ax, fs, 0.7)
    yhp = highpass_ma(ay, fs, 0.7)
    zhp = highpass_ma(az, fs, 0.7)
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    maghp = highpass_ma(mag, fs, 0.7)

    # leichte Glättung
    k = max(1, int(round(0.16 * fs)))
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
    if scores[best] < 0.05:  # sehr schwache Periodizität → Fallback auf Std
        stds = np.array([np.std(c) for c in cands])
        best = int(np.argmax(stds))
    return cands[best]


# ---------- Modell / IO ----------
def load_model(path="artifacts/model.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_features(X: np.ndarray, M: dict):
    # Standardisieren wie im Training
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
    """Sorgt dafür, dass eine Spalte 't' (Sekunden, relativ zum Start) existiert."""
    if "t" in df.columns:
        t = pd.to_numeric(df["t"], errors="coerce")
    elif "t_rel" in df.columns:
        t_rel = pd.to_numeric(df["t_rel"], errors="coerce")
        t = t_rel - t_rel.iloc[0]  # relativ zum Dateistart
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

def strength_classes_from(M):
    exclude = {"REST", "PAUSE", "WALKING", "RUNNING"}
    return {c for c in M["classes"] if str(c).upper() not in exclude}

def seconds_to_hms(sec: float) -> str:
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


# ---------- Klassen-Heuristiken ----------
def class_name_key(name: str) -> str:
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
    """
    Klassenabhängige Feineinstellungen:
    - Cable Fly: kleinere Amplituden -> niedrigere Schwelle (k - 0.20), etwas längere max-Dauer
    - Triceps Pulldown: oft schnell & „wobbelig“ -> höhere Schwelle (k + 0.10), längere min-Dauer
    """
    key = class_name_key(name)
    if key == "CABLE_FLY":
        k = max(0.30, base_k - 0.20)
        return k, base_min, max(base_max, base_min + 2.6)  # etwas großzügiger oben
    if key == "TRICEPS_PULLDOWN":
        k = min(1.0, base_k + 0.10)
        return k, max(base_min, 0.70), base_max
    if key == "SHOULDER_PRESS":
        return base_k, base_min, base_max
    if key == "BENCH_PRESS":
        return base_k, base_min, base_max
    return base_k, base_min, base_max


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
    args = ap.parse_args()

    infile = Path(args.infile)
    assert infile.exists(), f"Datei nicht gefunden: {infile}"

    # 1) Modell laden
    M = load_model(args.model)
    classes = M["classes"]
    fs   = float(M["meta"]["fs_hz"])
    winS = float(M["meta"]["win_s"])
    hopS = float(M["meta"]["hop_s"])

    # 2) Workout laden
    rows = list(read_jsonl(str(infile)))
    assert rows, "Datei leer?"
    df = pd.DataFrame(rows)
    df = ensure_time_column_df(df).sort_values("t").reset_index(drop=True)

    # 3) Fenster + Features wie im Training
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

    # 4a) Probs glätten
    probs_s = smooth_probs_over_time(probs, k=max(1, int(args.prob_smooth_k)))
    cls_idx = np.argmax(probs_s, axis=1)

    # 4b) Entprellen
    cls_idx = debounce_labels(cls_idx, min_run=max(1, int(args.debounce_run)))

    # 5) Segmentierung
    segments = segment_from_window_preds(t0s, cls_idx, classes, winS)

    # 5a) Mini-Segmente mergen
    if args.merge_min_s and args.merge_min_s > 0:
        segments = merge_short_segments(segments, min_len_s=float(args.merge_min_s), prefer="neighbor")

    # 6) Rep-Counting
    strength_classes = strength_classes_from(M)
    results = []

    t  = df["t"].to_numpy(float)
    ax = df["ax"].to_numpy(float)
    ay = df["ay"].to_numpy(float)
    az = df["az"].to_numpy(float)

    # für peaks-Mode: leichte Glättung der Z-Achse (Kompatibilität)
    smooth_k = max(1, int(round(args.smooth_sec * fs)))
    az_smooth = moving_average(az, smooth_k)

    for seg in segments:
        seg_class = seg["class"]
        mask = (t >= seg["t0"]) & (t <= seg["t1"])

        reps = 0
        if seg_class in strength_classes and np.count_nonzero(mask) > 3:
            if args.rep_mode == "pair":
                # --- adaptive Signalwahl + Periodenschätzung ---
                sig = select_rep_signal(ax[mask], ay[mask], az[mask], fs)

                # Klassen-basierte Offsets
                k0, min0, max0 = rep_params_for_class(seg_class,
                                                      base_k=float(args.rep_k),
                                                      base_min=float(args.rep_min_s),
                                                      base_max=float(args.rep_max_s))

                # ACF-basierte Anpassung der erlaubten Rep-Dauer
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

                # Sicherheit: min<max
                if min_s >= max_s:
                    min_s = min0
                    max_s = max0

                reps = count_reps_peak_trough(sig, fs, k=k0, min_rep_s=min_s, max_rep_s=max_s)

                # Für sehr kleine Amplituden (z. B. Cable Fly) als zweite Chance Schwelle etwas senken
                if reps == 0:
                    k_try = max(0.25, k0 - 0.1)
                    reps = count_reps_peak_trough(sig, fs, k=k_try, min_rep_s=min_s, max_rep_s=max_s)

            else:
                # 'peaks'-Modus (mit Prominenz)
                az_seg = az_smooth[mask]
                reps = count_peaks(
                    az_seg, fs,
                    min_separation_s=float(args.min_peak_sep),
                    thresh_mode="median_mad",
                    prominence=0.5 * mad(az_seg)
                )

        # Mindest-Reps-Politik
        out_class = seg_class
        if seg_class in strength_classes:
            if args.below_min_policy == "drop" and reps < int(args.min_reps):
                # Segment komplett verwerfen
                continue
            if args.below_min_policy == "rest" and reps < int(args.min_reps):
                out_class = "REST"
                reps = 0

        seg_out = {
            "t0": float(seg["t0"]),
            "t1": float(seg["t1"]),
            "duration_s": float(seg["t1"] - seg["t0"]),
            "class": out_class,
            "reps": int(reps) if out_class not in {"REST", "PAUSE", "WALKING", "RUNNING"} else 0
        }
        results.append(seg_out)

    # 7) Ausgabe
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

    # 8) JSON-Export
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
