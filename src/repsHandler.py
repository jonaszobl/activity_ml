# src/repsHandler.py
# ------------------------------------------------------------
# Per-Übung Wiederholungszähler mit Registry + Default-Handler.
# Öffentliche API: count_reps_for_segment(sig, fs, segment_class, k, min_s, max_s) -> int
# ------------------------------------------------------------
# wurde hinzugefügt um klare Basis zu haben, Feintuning sitzt in den spezifischen Handlern

from typing import Callable, Dict, Optional, List, Tuple
import numpy as np

try:
    from .rep_gap_repair import find_and_repair_rep_gaps
except Exception:
    # Fallback, (Modul auf Railway anders definiert...)
    from rep_gap_repair import find_and_repair_rep_gaps  # aufruf von rep_gap_repair

# ---------- kleine robuste Helfer ----------
def _median(a: np.ndarray) -> float:
    if a.size == 0: return 0.0  # lieber 0 als crash
    return float(np.median(a))

def _mad(a: np.ndarray) -> float:
    # MAD statt STD -> robuster gegen Ausreißer
    m = _median(a)
    return float(np.median(np.abs(a.astype(float) - m)))

def _class_name_key(name: str) -> str:
    # Normalisierung verschiedener Klassenbezeichnungen
    u = str(name).upper()
    if "TRICEPS" in u and "PULL" in u: return "TRICEPS_PULLDOWN"
    if "FLY" in u or ("CABLE" in u and "CHEST" in u): return "CABLE_FLY_CHEST"
    if "BENCH" in u or "BANKDR" in u:
        # falls "BENCH_BB"/"BENCH_DB" getrennt geführt wird, hier verfeinern
        return u  # nicht zusammenziehen
    if "SHOULDER" in u and "PRESS" in u: return "SHOULDER_PRESS"
    if "BIZEPS" in u and "H" in u: return "BIZEPS_CURL_H"
    if "BICEP" in u and "HAMMER" in u: return "BIZEPS_CURL_H"
    if "RUDERN" in u or "ROW" in u: return "RUDERN"
    if "LATERAL" in u and ("RAISE" in u or "SEIT" in u): return "LATERAL_RAISE_CABLE"
    return u

# ---------- Registry Grundgerüst ----------
RepFn = Callable[[np.ndarray, float, float, float, float], int]
_REP_HANDLERS: Dict[str, RepFn] = {}

def register_rep_handler(class_key: str):
    # Decorator zum Registrieren eines spezifischen Handlers
    key = _class_name_key(class_key)
    def _wrap(fn: RepFn):
        _REP_HANDLERS[key] = fn
        return fn
    return _wrap

def _get_handler(class_name: str) -> Optional[RepFn]:
    return _REP_HANDLERS.get(_class_name_key(class_name), None)

# ---- Periodenschätzung + Grid-Fit ----
def _estimate_period(signal: np.ndarray, fs: float, lo: float = 0.6, hi: float = 3.5) -> Tuple[float, float]:
    """Return (p_est, acf_peak). Fallback: 0 wenn unzuverlässig.
    # hilfsfunktion um die periodische Dauer (s/rep) grob zu schätzen
    """
    n = signal.size
    if n < int(hi*fs)+3:
        return 0.0, 0.0
    x = signal.astype(float)
    x -= float(np.mean(x))
    den = float(np.dot(x, x)) + 1e-12  # schutz gegen 0
    if den <= 1e-12:
        return 0.0, 0.0
    ac = np.correlate(x, x, mode="full")[n-1:] / den
    min_lag = max(1, int(round(lo*fs)))
    max_lag = min(n-2, int(round(hi*fs)))
    if max_lag <= min_lag:
        return 0.0, 0.0
    lag = None
    for i in range(min_lag+1, max_lag-1):
        if ac[i] > ac[i-1] and ac[i] > ac[i+1]:
            lag = i; break
    if lag is None:
        j = int(np.argmax(ac[min_lag:max_lag]))
        lag = min_lag + j
    return (float(lag)/fs, float(ac[lag]))

def _snap_min_around(t_s: float, signal: np.ndarray, fs: float, w_s: float) -> float:
    """Snap um t_s ans lokale Minimum in ±w_s.
    # wurde hinzugefügt weil Minima bei vielen Übungen stabilere Anker sind
    """
    if signal is None or fs <= 0: return t_s
    w = max(1, int(round(w_s*fs)))
    j = int(round(t_s*fs))
    L = signal.size
    a = max(1, j-w); b = min(L-2, j+w)
    if b <= a+1: return t_s
    seg = signal[a:b+1]
    k = int(np.argmin(seg))
    return float(a+k)/fs

def _cadence_grid_fit(candidates_s: List[float], *, p: float, T: float, signal: np.ndarray, fs: float,
                      snap_w_s: float = 0.10, tol_frac: float = 0.35) -> List[float]:
    """
    Lege ein Raster φ + k·p über [0,T] und wähle pro Zelle genau einen Zeitpunkt.
    # hilft, Unterzählung zu reduzieren; Kandidaten in Zellen snappen falls nötig
    """
    if not np.isfinite(p) or p <= 1e-6:
        return sorted(candidates_s)

    cand = np.array(sorted(candidates_s), float)
    if cand.size == 0:
        n_bins = max(1, int(round(T / max(p, 1e-6))))
        phi = 0.0
    else:
        phases = np.mod(cand, p)
        phi = float(np.median(phases)) % p
        n_bins = max(1, int(round((T - phi) / p)) + 1)

    out: List[float] = []
    tol = tol_frac * p
    for kbin in range(n_bins):
        t_grid = phi + kbin*p
        if t_grid < -1e-6 or t_grid > T + 1e-6:
            continue
        pick = None; dmin = 1e9
        for t in cand:
            d = abs(t - t_grid)
            if d < dmin:
                dmin = d; pick = t
        if pick is not None and dmin <= tol:
            out.append(float(pick))
        else:
            out.append(_snap_min_around(t_grid, signal, fs, snap_w_s))
    return sorted(out)

def _plausible_clamp(times_s: List[float], *, T: float, min_rep_s: float, max_rep_s: float,
                     signal: np.ndarray, fs: float, p: float) -> List[float]:
    """Clamp der Anzahl der Reps in [T/max_rep_s, T/min_rep_s] via Insert/Merge.
    # sanity check, falls Grid-Fit zu viel/zu wenig erzeugt hat
    """
    if T <= 0: return times_s
    lo = int(np.floor(T / max(1e-9, max_rep_s)))
    hi = int(np.ceil (T / max(1e-9, min_rep_s)))
    lo = max(1, lo)

    cur = sorted(times_s)
    n = len(cur)
    if p <= 0:
        diffs = np.diff(cur) if n >= 2 else np.array([])
        p = float(np.median(diffs)) if diffs.size else max_rep_s  # kann noch verfeinert werden...

    while n < lo:
        # zu wenig -> Zellen auffüllen
        if n == 0:
            cur = _cadence_grid_fit([], p=p, T=T, signal=signal, fs=fs)
        else:
            cur = _cadence_grid_fit(cur, p=p, T=T, signal=signal, fs=fs)
        n = len(cur)

    def _merge_once(arr: List[float]) -> List[float]:
        if len(arr) < 2: return arr
        arr = sorted(arr)
        dif = np.diff(arr)
        j = int(np.argmin(dif))
        mid = 0.5*(arr[j] + arr[j+1])
        mid = _snap_min_around(mid, signal, fs, 0.08)  # snap aufs Minimum
        out = arr[:j] + [mid] + arr[j+2:]
        return out

    while len(cur) > hi:
        # zu viel -> nahe Nachbarn mergen
        cur = _merge_once(cur)

    return sorted(cur)

# ---------- Default-Handler (PAIR mit MAD-Hysterese) ----------
def _pair_count_default(sig: np.ndarray, fs: float, k: float, min_rep_s: float, max_rep_s: float, *, cls_name: str = "") -> int:
    """
    Generic, schneller Pair-Zähler (Peak↔Trough über MAD-Hysterese).
    # bewusst simpel gehalten; spezifische Handler übernehmen die Details
    """
    n = sig.size
    if n < 3: return 0
    med = _median(sig); m = _mad(sig)
    if not np.isfinite(m) or m < 1e-6:
        m = max(1e-6, float(np.std(sig) * 0.8))  # fallback falls MAD zu klein
    if m < 1e-6: return 0

    up = med + k*m
    dn = med - k*m
    min_rep = int(round(min_rep_s * fs))
    max_rep = int(round(max_rep_s * fs))

    phase = "seek_up"
    t_up = None
    reps = 0
    anchors_s: List[float] = []  # wurde hinzugefügt weil Gap-Repair Anker braucht

    def is_peak(x0, x1, x2):   return (x1 > x0 and x1 > x2)
    def is_trough(x0, x1, x2): return (x1 < x0 and x1 < x2)

    # Zyklus: +Crossing -> local max -> -Crossing -> local min
    i = 1
    while i < n-1:
        s0, s1, s2 = sig[i-1], sig[i], sig[i+1]
        if phase == "seek_up":
            if s0 <= up < s1:
                t_up = i
                phase = "seek_peak"
        elif phase == "seek_peak":
            if is_peak(s0, s1, s2):
                phase = "seek_down"
        elif phase == "seek_down":
            if s0 >= dn > s1:
                phase = "seek_trough"
        elif phase == "seek_trough":
            if is_trough(s0, s1, s2):
                if t_up is not None:
                    dur = i - t_up
                    if min_rep <= dur <= max_rep:
                        reps += 1
                        anchors_s.append(i / float(fs))  # Anker setzen
                phase = "seek_up"
        i += 1

    # aufruf von rep_gap_repair (konservativ)
    if len(anchors_s) >= 2:
        repair = find_and_repair_rep_gaps(
            anchors_s,
            fs=fs,
            segment_class=cls_name,
            signal=sig,
            search_window_s=0.12,  # klein halten -> weniger Fehl-Snaps
            hi_factor=1.65,        # Lücke ~65% länger als erwartet => Kandidat
            max_inserts_per_gap=3
        )
        reps = len(repair["rep_ts_aug"])
    return reps

@register_rep_handler("BIZEPS_CURL_H")
def _count_hammer(sig: np.ndarray, fs: float, k: float, min_rep_s: float, max_rep_s: float) -> int:
    """
    Hammer Curls – konservativer Zähler; mit Anker + Gap-Repair.
    # vorher Doppelzählungen -> strenger + Reparatur der Lücken
    """
    n = sig.size
    if n < 7:
        return 0

    # 0) Leichte Glättung (konservativ)
    SMOOTH_S = 0.08  # nicht zu groß, sonst Peaks verwaschen
    k_smooth = max(1, int(round(SMOOTH_S * fs)))
    if k_smooth > 1:
        c = np.cumsum(np.insert(sig.astype(float), 0, 0.0))
        sig = (c[k_smooth:] - c[:-k_smooth]) / k_smooth
        n = sig.size
        if n < 7:
            return 0

    # 1) robuste Stats
    med = _median(sig)
    m = _mad(sig)
    if not np.isfinite(m) or m < 1e-6:
        m = max(1e-6, float(np.std(sig) * 0.8))
    if m < 1e-6:
        return 0

    # 2) Hysterese
    k_eff = max(0.36, 0.80 * k)  # wurde hinzugefügt weil die reps zu mild gezählt wurden
    up = med + k_eff * m
    dn = med - k_eff * m

    def is_peak(x0, x1, x2):   return (x1 > x0 and x1 > x2)
    def is_trough(x0, x1, x2): return (x1 < x0 and x1 < x2)

    # 3) Periode via ACF
    def _acf_primary_period(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
        L = x.size
        if L < int(hi*fs) + 3:
            return 0.0
        x0 = x.astype(float) - float(np.mean(x))
        den = float(np.dot(x0, x0)) + 1e-12
        ac = np.correlate(x0, x0, mode="full")[L-1:] / den
        min_lag = max(1, int(round(lo*fs)))
        max_lag = min(L-2, int(round(hi*fs)))
        if max_lag <= min_lag:
            return 0.0
        lag = None
        for i in range(min_lag+1, max_lag-1):
            if ac[i] > ac[i-1] and ac[i] > ac[i+1]:
                lag = i; break
        if lag is None:
            lag = min_lag + int(np.argmax(ac[min_lag:max_lag]))
        return float(lag) / fs

    p = _acf_primary_period(sig, fs, lo=0.6, hi=2.2)

    if p > 0:
        refractory = max(int(0.28 * fs), int(0.28 * p * fs))
        min_r = max(int(round(min_rep_s * fs)), int(0.50 * p * fs))
        max_r = max(int(round(max_rep_s * fs)), int(1.60 * p * fs))
    else:
        refractory = int(round(0.30 * fs))
        min_r = int(round(min_rep_s * fs))
        max_r = int(round(max_rep_s * fs))

    PROM_MIN = 0.28 * m
    SYM_LO, SYM_HI = 0.60, 1.90
    MIN_PK_TROUGH = int(round(0.16 * fs))
    CURV_MIN = 0.08 * m
    SNAP_S = 0.05
    snap_w = max(1, int(round(SNAP_S * fs)))
    MONO_DESC_S = 0.08
    MONO_DESC_MIN_NEG_RATIO = 0.7  # Abstieg soll überwiegend monoton sein

    phase = "seek_up"
    t_up = t_peak = t_trough = None
    v_peak = v_trough = None
    last_trough = -10**9
    reps = 0
    anchors_s: List[float] = []  # Anker für Gap-Repair

    i = 1
    while i < n-1:
        s0, s1, s2 = sig[i-1], sig[i], sig[i+1]

        if phase == "seek_up":
            if s0 <= up < s1:
                t_up = i
                phase = "seek_peak"

        elif phase == "seek_peak":
            if s1 > s0 and s1 > s2:
                t_peak, v_peak = i, s1
                if (v_peak - med) < PROM_MIN:
                    phase = "seek_up"
                else:
                    phase = "seek_down"

        elif phase == "seek_down":
            if s0 >= dn > s1:
                phase = "seek_trough"

        elif phase == "seek_trough":
            if s1 < s0 and s1 < s2:
                left  = max(1, i - snap_w)
                right = min(n-2, i + snap_w)
                j = left + int(np.argmin(sig[left:right+1]))  # Snap aufs Minimum
                curv = sig[j-1] - 2*sig[j] + sig[j+1]
                t_trough, v_trough = j, sig[j]

                # Monotonie-Check vor trough
                w = max(1, int(round(MONO_DESC_S * fs)))
                a = max(1, j - w)
                desc = sig[a:j]
                if desc.size >= 3:
                    neg_steps = np.sum(np.diff(desc) <= 0.0)
                    neg_ratio = float(neg_steps) / max(1, desc.size - 1)
                else:
                    neg_ratio = 1.0

                if ((med - v_trough) < PROM_MIN or
                    (t_peak is not None and (j - t_peak) < MIN_PK_TROUGH) or
                    (curv < CURV_MIN) or
                    (neg_ratio < MONO_DESC_MIN_NEG_RATIO)):
                    phase = "seek_up"; i += refractory; continue

                if t_up is not None:
                    dur = j - t_up
                    if (min_r <= dur <= max_r) and (j - last_trough) >= refractory:
                        peak_amp   = abs((v_peak if v_peak is not None else med) - med)
                        trough_amp = abs(med - (v_trough if v_trough is not None else med))
                        ratio = (peak_amp / max(1e-9, trough_amp)) if trough_amp > 0 else 99.0
                        if SYM_LO <= ratio <= SYM_HI:
                            reps += 1
                            last_trough = j
                            anchors_s.append(j / float(fs))
                phase = "seek_up"; i = max(i, j) + refractory; continue

        # Watchdog gegen Hängenbleiben
        if phase != "seek_up" and t_up is not None and (i - t_up) > int(1.15 * max(1, max_r)):
            phase = "seek_up"; t_up = None

        i += 1

    # Gap-Repair am Ende (nur wenn >=2 Anker)
    if len(anchors_s) >= 2:
        repair = find_and_repair_rep_gaps(
            anchors_s,
            fs=fs,
            segment_class="BIZEPS_CURL_H",
            signal=sig,
            search_window_s=0.12,
            hi_factor=1.65,
            max_inserts_per_gap=3
        )
        reps = len(repair["rep_ts_aug"])

    return int(reps)

# ---------- Spezifischer Handler: Rudern (Kabel/Maschine) ----------
@register_rep_handler("RUDERN")
def _count_row(sig: np.ndarray, fs: float, k: float, min_rep_s: float, max_rep_s: float) -> int:
    """
    Rudern: period-adaptive Pair-Zählung; mit Grid-Fit und Plausibilitäts-Clamp.
    # Ziel: Unterzählung reduzieren, Doppelzählungen vermeiden
    """
    n = sig.size
    if n < 7:
        return 0
    T = float(n) / float(fs)

    # realistische Grenzen unabhängig von predict_workout
    MIN_EFF = 1.4   # s/rep
    MAX_EFF = 3.5   # s/rep

    # 0) leichte Glättung
    k_smooth = max(1, int(round(0.10 * fs)))
    if k_smooth > 1:
        c = np.cumsum(np.insert(sig.astype(float), 0, 0.0))
        sig = (c[k_smooth:] - c[:-k_smooth]) / k_smooth
        n = sig.size
        if n < 7:
            return 0

    # 1) robuste Stats
    med = _median(sig)
    m = _mad(sig)
    if not np.isfinite(m) or m < 1e-6:
        m = max(1e-6, float(np.std(sig) * 0.8))
    if m < 1e-6:
        return 0

    k_eff = max(0.36, 0.75 * k)  # vorher 2.5 -> niedriger da reps zu hart gezählt
    up = med + k_eff * m
    dn = med - k_eff * m

    def is_peak(x0, x1, x2):   return (x1 > x0 and x1 > x2)
    def is_trough(x0, x1, x2): return (x1 < x0 and x1 < x2)

    prom_min = 0.18 * m
    sym_lo, sym_hi = 0.55, 2.00
    min_pk_trough = int(round(0.12 * fs))

    # 2) Pair-Automat (Anker sammeln)
    phase = "seek_up"
    t_up = t_peak = None
    v_peak = None
    i = 1
    refractory = int(round(0.22 * fs))
    min_r = int(round(MIN_EFF * fs))
    max_r = int(round(MAX_EFF * fs))
    anchors: List[float] = []
    reps_pair = 0

    while i < n-1:
        s0, s1, s2 = sig[i-1], sig[i], sig[i+1]

        if phase == "seek_up":
            if s0 <= up < s1:
                t_up = i
                phase = "seek_peak"

        elif phase == "seek_peak":
            if is_peak(s0, s1, s2):
                t_peak, v_peak = i, s1
                if (v_peak - med) < prom_min:
                    phase = "seek_up"
                else:
                    phase = "seek_down"

        elif phase == "seek_down":
            if s0 >= dn > s1:
                phase = "seek_trough"

        elif phase == "seek_trough":
            if is_trough(s0, s1, s2):
                j_left  = max(1, i - int(round(0.06*fs)))
                j_right = min(n-2, i + int(round(0.06*fs)))
                j = j_left + int(np.argmin(sig[j_left:j_right+1]))
                if (med - sig[j]) < 0.15*m or (t_peak is not None and (j - t_peak) < min_pk_trough):
                    phase = "seek_up"; i += refractory; continue
                if t_up is not None:
                    dur = j - t_up
                    if (min_r <= dur <= max_r):
                        peak_amp   = abs((v_peak if v_peak is not None else med) - med)
                        trough_amp = abs(med - sig[j])
                        ratio = (peak_amp / max(1e-9, trough_amp)) if trough_amp > 0 else 99.0
                        if sym_lo <= ratio <= sym_hi:
                            reps_pair += 1
                            anchors.append(j / float(fs))
                            i += int(round(0.18*fs))  # kleine Pause gegen Doppelzählung
                            phase = "seek_up"; continue
                phase = "seek_up"; i += int(round(0.10*fs)); continue

        if phase != "seek_up" and t_up is not None and (i - t_up) > int(1.20 * max(1, max_r)):
            phase = "seek_up"; t_up = None  # watchdog

        i += 1

    # 3) Periode p (ACF, Fallback: Median der Ankerabstände)
    p_acf, acp = _estimate_period(sig, fs, lo=0.6, hi=3.5)
    if not np.isfinite(p_acf) or p_acf <= 0.0 or acp < 0.12:
        if len(anchors) >= 2:
            dif = np.diff(np.array(sorted(anchors), float))
            p_acf = float(np.median(dif)) if dif.size else 0.0  # kann noch verfeinert werden...

    if p_acf > 0:
        refractory = max(refractory, int(0.25 * p_acf * fs))
        min_r = max(min_r, int(0.45 * p_acf * fs))
        max_r = max(max_r, int(1.80 * p_acf * fs))

    # 4) Cadence-Grid-Fit
    p_used = float(p_acf if p_acf > 0 else max(MIN_EFF, 1.8))  # soft default 1.8s
    anchors_grid = _cadence_grid_fit(
        candidates_s=anchors,
        p=p_used,
        T=T,
        signal=sig,
        fs=fs,
        snap_w_s=0.10,
        tol_frac=0.35
    )

    # 5) Plausibilitäts-Clamp
    if len(anchors_grid) >= 2:
        p_for_clamp = float(np.median(np.diff(anchors_grid)))
    else:
        p_for_clamp = p_used

    anchors_final = _plausible_clamp(
        anchors_grid,
        T=T,
        min_rep_s=MIN_EFF,
        max_rep_s=MAX_EFF,
        signal=sig,
        fs=fs,
        p=p_for_clamp
    )

    reps_final = max(reps_pair, len(anchors_final))  # konservativ: nimm das größere
    return int(reps_final)


@register_rep_handler("TRICEPS_PULLDOWN")
def _count_triceps_pulldown(sig: np.ndarray, fs: float, k: float, min_rep_s: float, max_rep_s: float) -> int:
    """
    TRICEPS_PULLDOWN – trough-anchored Zählung mit Snap aufs lokale Minimum.
    # jetzt inkl. Anker + Gap-Repair um Unterzählung zu vermeiden
    """
    n = sig.size
    if n < 7:
        return 0

    # 0) Glättung
    SMOOTH_S = 0.10  # minimal breiter als Hammer Curls
    k_smooth = max(1, int(round(SMOOTH_S * fs)))
    if k_smooth > 1:
        c = np.cumsum(np.insert(sig.astype(float), 0, 0.0))
        sig = (c[k_smooth:] - c[:-k_smooth]) / k_smooth
        n = sig.size
        if n < 7:
            return 0

    # 1) robuste Stats
    med = _median(sig)
    m = _mad(sig)
    if not np.isfinite(m) or m < 1e-6:
        m = max(1e-6, float(np.std(sig) * 0.8))
    if m < 1e-6:
        return 0

    # 2) Hysterese
    k_eff = max(0.28, 0.60 * k)  # wurde hinzugefügt weil die reps zu mild gezählt wurden
    up = med + k_eff * m
    dn = med - k_eff * m

    def is_peak(x0, x1, x2):   return (x1 > x0 and x1 > x2)
    def is_trough(x0, x1, x2): return (x1 < x0 and x1 < x2)

    # 3) Perioden-Schätzung
    def _acf_primary_period(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
        L = x.size
        if L < int(hi*fs) + 3:
            return 0.0
        x0 = x.astype(float) - float(np.mean(x))
        den = float(np.dot(x0, x0)) + 1e-12
        ac = np.correlate(x0, x0, mode="full")[L-1:] / den
        min_lag = max(1, int(round(lo*fs)))
        max_lag = min(L-2, int(round(hi*fs)))
        if max_lag <= min_lag:
            return 0.0
        lag = None
        for i in range(min_lag+1, max_lag-1):
            if ac[i] > ac[i-1] and ac[i] > ac[i+1]:
                lag = i; break
        if lag is None:
            lag = min_lag + int(np.argmax(ac[min_lag:max_lag]))
        return float(lag) / fs

    p = _acf_primary_period(sig, fs, lo=0.5, hi=2.5)

    # 4) Adaptives Zeitfenster & Refractory
    if p > 0:
        refractory = max(int(0.2 * fs), int(0.25 * p * fs))
        min_r = max(int(round(min_rep_s * fs)), int(0.45 * p * fs))
        max_r = max(int(round(max_rep_s * fs)), int(1.70 * p * fs))
    else:
        refractory = int(round(0.18 * fs))
        min_r = int(round(min_rep_s * fs))
        max_r = int(round(max_rep_s * fs))

    PROM_MIN = 0.10 * m
    SYM_LO, SYM_HI = 0.45, 2.30
    MIN_PK_TROUGH = int(round(0.10 * fs))
    CURV_MIN = 0.04 * m
    SNAP_S = 0.12
    snap_w = max(1, int(round(SNAP_S * fs)))

    # kleines Tuning bei sehr kurzer p
    if p > 0 and p < 0.90:
        PROM_MIN *= 0.90
        refractory = max(int(0.16 * fs), int(0.20 * p * fs))
        MIN_PK_TROUGH = max(int(round(0.10 * fs)), MIN_PK_TROUGH - int(round(0.02 * fs)))

    phase = "seek_up"
    t_up = t_peak = t_trough = None
    v_peak = v_trough = None
    last_i = -10**9
    reps = 0
    anchors_s: List[float] = []  # Anker für Gap-Repair

    i = 1
    while i < n-1:
        s0, s1, s2 = sig[i-1], sig[i], sig[i+1]

        if phase == "seek_up":
            if s0 <= up < s1:
                t_up = i
                phase = "seek_peak"

        elif phase == "seek_peak":
            if is_peak(s0, s1, s2):
                t_peak, v_peak = i, s1
                if (v_peak - med) < PROM_MIN:
                    phase = "seek_up"
                else:
                    phase = "seek_down"

        elif phase == "seek_down":
            if s0 >= dn > s1:
                phase = "seek_trough"

        elif phase == "seek_trough":
            if is_trough(s0, s1, s2):
                left  = max(1, i - snap_w)
                right = min(n-2, i + snap_w)
                j = left + int(np.argmin(sig[left:right+1]))
                curv = sig[j-1] - 2*sig[j] + sig[j+1]
                t_trough, v_trough = j, sig[j]

                if (med - v_trough) < PROM_MIN or (t_peak is not None and (j - t_peak) < MIN_PK_TROUGH) or (curv < CURV_MIN):
                    phase = "seek_up"; i += refractory; continue

                if t_up is not None:
                    dur = j - t_up
                    if (min_r <= dur <= max_r) and (j - last_i >= refractory):
                        peak_amp   = abs((v_peak if v_peak is not None else med) - med)
                        trough_amp = abs(med - v_trough)
                        ratio = (peak_amp / max(1e-9, trough_amp)) if trough_amp > 0 else 99.0
                        if SYM_LO <= ratio <= SYM_HI:
                            reps += 1
                            last_i = j
                            anchors_s.append(j / float(fs))  # Anker setzen
                phase = "seek_up"; i = max(i, j) + refractory; continue

        if phase != "seek_up" and t_up is not None and (i - t_up) > int(1.20 * max(1, max_r)):
            phase = "seek_up"; t_up = None

        i += 1

    # 7) Fallback (Periodik klar, aber unterzählt) – leichtes Nachziehen
    if p > 0:
        exp_reps = int(round(n / float(fs) / max(1e-9, p)))
        if reps <= max(2, exp_reps - 2):
            up2 = med + max(0.24, k_eff - 0.10) * m
            dn2 = med - max(0.24, k_eff - 0.10) * m
            refractory2 = max(int(0.15 * fs), int(0.20 * p * fs))
            min_r2 = max(int(0.40 * p * fs), int(round(min_rep_s * fs)))
            max_r2 = max(int(1.85 * p * fs), int(round(max_rep_s * fs)))
            PROM_MIN2 = 0.12 * m
            CURV_MIN2 = 0.06 * m
            MIN_PK_TROUGH2 = int(round(0.10 * fs))

            phase = "seek_up"
            t_up = t_peak = t_trough = None
            v_peak = v_trough = None
            last_i2 = -10**9
            add = 0

            i = 1
            while i < n-1:
                s0, s1, s2 = sig[i-1], sig[i], sig[i+1]
                if phase == "seek_up":
                    if s0 <= up2 < s1:
                        t_up = i; phase = "seek_peak"
                elif phase == "seek_peak":
                    if is_peak(s0, s1, s2):
                        t_peak, v_peak = i, s1
                        if (v_peak - med) < PROM_MIN2:
                            phase = "seek_up"
                        else:
                            phase = "seek_down"
                elif phase == "seek_down":
                    if s0 >= dn2 > s1:
                        phase = "seek_trough"
                elif phase == "seek_trough":
                    if is_trough(s0, s1, s2):
                        left  = max(1, i - snap_w)
                        right = min(n-2, i + snap_w)
                        j = left + int(np.argmin(sig[left:right+1]))
                        curv = sig[j-1] - 2*sig[j] + sig[j+1]

                        if (med - sig[j]) < PROM_MIN2 or (t_peak is not None and (j - t_peak) < MIN_PK_TROUGH2) or (curv < CURV_MIN2):
                            phase = "seek_up"; i += refractory2; continue

                        if t_up is not None:
                            dur = j - t_up
                            if (min_r2 <= dur <= max_r2) and (j - last_i2 >= refractory2):
                                peak_amp   = abs((v_peak if v_peak is not None else med) - med)
                                trough_amp = abs(med - sig[j])
                                ratio = (peak_amp / max(1e-9, trough_amp)) if trough_amp > 0 else 99.0
                                if 0.50 <= ratio <= 2.10:
                                    add += 1
                                    last_i2 = j
                        phase = "seek_up"; i = max(i, j) + refractory2; continue
                i += 1

            reps = min(exp_reps, reps + min(5, add))  # Deckel gegen Überkompensation

    # Gap-Repair am Ende (nutzt die Trough-Anker aus Haupt-Pass)
    if len(anchors_s) >= 2:
        repair = find_and_repair_rep_gaps(
            anchors_s,
            fs=fs,
            segment_class="TRICEPS_PULLDOWN",
            signal=sig,
            search_window_s=0.12,
            hi_factor=1.60,       # Trizeps: Minima sind oft sehr klar -> leicht offensiver
            max_inserts_per_gap=3
        )
        reps = max(reps, len(repair["rep_ts_aug"]))  # sicherheit: nimm das größere

    return int(reps)


# ---------- Öffentliche API ----------
def count_reps_for_segment(
    sig: np.ndarray,
    fs: float,
    segment_class: str,
    k: float,
    min_rep_s: float,
    max_rep_s: float
) -> int:
    """
    Universelle Entry-Funktion. Wählt ggf. spezifischen Handler,
    sonst Default-Handler. Gibt nur die gezählten reps (int) zurück.
    # API bewusst schlank gehalten
    """
    handler = _get_handler(segment_class)
    if handler is None:
        # Default mit Klassenname zur Auto-Ankerwahl im Repair
        return _pair_count_default(sig, fs, k, min_rep_s, max_rep_s, cls_name=segment_class)
    return handler(sig, fs, k, min_rep_s, max_rep_s)
