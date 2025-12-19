# src/segmentation/reps.py
import numpy as np


# ---------- Mathe / Utils ----------
def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    # simpler gleitender Mittelwert; für Rohachsen & Probs
    if k <= 1:
        return x.astype(float, copy=True)
    k = int(k)
    x = x.astype(float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[k:] - c[:-k]) / k
    pad = np.full(k - 1, y[0] if len(y) else 0.0, dtype=float)
    return np.concatenate([pad, y]).astype(float)


def median(a):
    # kleine robuste Helfer... nicht übertreiben
    a = np.asarray(a, float)
    if len(a) == 0:
        return 0.0
    return float(np.median(a))


def mad(a):
    # MAD statt STD -> robuster gg. Ausreißer
    m = median(a)
    return float(np.median(np.abs(np.asarray(a, float) - m)))


# ---------- Peak-basierte Zählung (optional) ----------
def count_peaks(signal: np.ndarray, fs: float,
                min_separation_s: float = 0.4,
                thresh_mode: str = "median_mad",
                prominence: float = 0.0) -> int:
    # alt/legacy; bleibt drin als fallback
    if len(signal) < 3:
        return 0
    if thresh_mode == "median_mad":
        th = median(signal) + 0.5 * mad(signal)  # vorher 0.3 -> höher, um Rauschen zu ignorieren
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
    # einfache Peak↔Trough-Paarlogik; robust genug für Basiszählung
    n = len(signal)
    if n < 3:
        return 0
    med = median(signal)
    m = mad(signal)
    if m < 1e-6:
        m = np.std(signal) * 0.8  # fallback wenn MAD zu klein
    if m < 1e-6:
        return 0

    up = med + k * m
    down = med - k * m  # k kann noch verfeinert werden…

    last_ext_t = None
    last_ext_type = None  # "peak" | "trough"
    reps = 0

    for i in range(1, n - 1):
        s0, s1, s2 = signal[i - 1], signal[i], signal[i + 1]
        t = i / fs
        if s1 > up and s1 > s0 and s1 > s2:
            if last_ext_type == "trough":
                dt = t - last_ext_t
                if min_rep_s <= dt <= max_rep_s:
                    reps += 1
            last_ext_t = t
            last_ext_type = "peak"
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
    # HP via MA-Subtraktion; simpel & schnell
    k = max(1, int(round(cutoff_s * fs)))
    trend = moving_average(x, k)
    y = x.astype(float) - trend
    return y


def _acf_primary_peak(signal: np.ndarray, fs: float,
                      min_s: float, max_s: float):
    # ACF-Peak grob suchen -> liefert p und peak-höhe
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
    for i in range(min_lag + 1, max_lag - 1):
        if ac[i] > ac[i - 1] and ac[i] > ac[i + 1]:
            lag = i
            break
    if lag is None:
        lag = min_lag + int(np.argmax(ac[min_lag:max_lag]))
    return (lag / fs), float(ac[lag])


def estimate_rep_period_acf(signal: np.ndarray, fs: float,
                            min_s: float = 0.4, max_s: float = 3.0) -> float:
    # hilfsfunktion um p zu schätzen; schlank gehalten
    p, _ = _acf_primary_peak(signal, fs, min_s, max_s)
    return p


def select_rep_signal(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float) -> np.ndarray:
    # Wahl der „besten“ Achse -> ACF-Score, Fallback STD
    xhp = highpass_ma(ax, fs, 0.7)
    yhp = highpass_ma(ay, fs, 0.7)
    zhp = highpass_ma(az, fs, 0.7)
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    maghp = highpass_ma(mag, fs, 0.7)

    k = max(1, int(round(0.16 * fs)))  # kurzer MA -> glättet leicht
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
    if scores[best] < 0.05:  # vorher 0.03 -> anheben, um Zufall zu vermeiden
        stds = np.array([np.std(c) for c in cands])
        best = int(np.argmax(stds))
    return cands[best]


# ---------- Klassen-Heuristiken (für Reps) ----------
def class_name_key(name: str) -> str:
    # mapping für rep-params; minimal gehalten
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
    # leichte Klassentunings; konservativ
    key = class_name_key(name)
    if key == "CABLE_FLY":
        k = max(0.30, base_k - 0.20)          # vorher base_k -> abgesenkt wegen sanfter Bewegung
        return k, base_min, max(base_max, base_min + 2.6)
    if key == "TRICEPS_PULLDOWN":
        k = min(1.0, base_k + 0.10)           # wurde angehoben da sonst Unterzählung
        return k, max(base_min, 0.70), base_max
    if key == "SHOULDER_PRESS":
        return base_k, base_min, base_max
    if key == "BENCH_PRESS":
        return base_k, base_min, base_max
    return base_k, base_min, base_max  # fallback
