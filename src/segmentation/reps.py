# src/segmentation/reps.py
import numpy as np

# ============================================================
# Reps / Signal-Utilities (universell, personenrobust)
# ============================================================
# Ziele:
# - Rep-Counting robust gg. unterschiedliche Amplituden/Trageposition
# - Periodenschätzung über ACF, Bandenergie & SNR relativ zur Baseline
# - Debug-Metriken liefern (für Tuning)
# - Anti-Explosion: niemals absurd hohe Reps aus Noise
#
# UPDATE (Fixes aus Debug-Session):
# - Erweitertes Debug: warum reps==0? (threshold vs timing)
# - Robustere Periodenschätzung: ACF + domf-Konsens, engerer Bereich default
# - Soft-Threshold Fallbacks:
#   * Wenn Signal periodisch wirkt (acf/br ok), aber Pair-Counter 0 -> fallback count_peaks
#   * Wenn Pair-Counter sehr niedrig vs plausibel -> optional fallback (konservativ)
# - K niemals 0 setzen (das erzeugt up==down und kollabiert die Detektion)
# - Adaptive Threshold: wenn zu wenige crossings -> k schrittweise reduzieren
# ============================================================

DBG_REPS = False  # kann in predict_workout.py gesetzt werden


# ---------- Basics ----------
def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.astype(float, copy=True)
    k = int(k)
    x = x.astype(float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[k:] - c[:-k]) / k
    pad = np.full(k - 1, y[0] if len(y) else 0.0, dtype=float)
    return np.concatenate([pad, y]).astype(float)


def median(a) -> float:
    a = np.asarray(a, float)
    if a.size == 0:
        return 0.0
    return float(np.median(a))


def mad(a) -> float:
    a = np.asarray(a, float)
    if a.size == 0:
        return 0.0
    m = median(a)
    return float(np.median(np.abs(a - m)))


def count_peaks(
    signal: np.ndarray,
    fs: float,
    min_separation_s: float = 0.4,
    thresh_mode: str = "median_mad",
    prominence: float = 0.0
) -> int:
    """
    Legacy Peak-Counter als Fallback.
    Bleibt drin, damit alte Imports/CLI weiterhin funktionieren.
    """
    signal = np.asarray(signal, float)
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
    return int(peaks)


def count_reps_peak_trough(
    signal: np.ndarray,
    fs: float,
    k: float = 0.6,
    min_rep_s: float = 0.6,
    max_rep_s: float = 4.0
) -> int:
    """
    Legacy Peak↔Trough Pair-Counting (alte API).
    Wird für Kompatibilität beibehalten.
    """
    signal = np.asarray(signal, float)
    n = len(signal)
    if n < 3:
        return 0

    med = median(signal)
    m = mad(signal)
    if m < 1e-6:
        m = float(np.std(signal)) * 0.8
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

    return int(reps)


def robust_z(x: np.ndarray) -> np.ndarray:
    """Robuste Normalisierung: (x-med)/(MAD+eps)."""
    x = np.asarray(x, float)
    m0 = median(x)
    s0 = mad(x)
    if s0 < 1e-9:
        s0 = float(np.std(x)) * 0.8
    if s0 < 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - m0) / (s0 + 1e-12)


# ---------- Highpass / Smoothing ----------
def highpass_ma(x: np.ndarray, fs: float, cutoff_s: float = 0.7) -> np.ndarray:
    k = max(1, int(round(cutoff_s * fs)))
    trend = moving_average(x, k)
    return x.astype(float) - trend


def smooth_hp(x: np.ndarray, fs: float) -> np.ndarray:
    """Highpass + leichte Glättung (≈160ms)."""
    hp = highpass_ma(x, fs, 0.7)
    k = max(1, int(round(0.16 * fs)))
    return moving_average(hp, k)


# ---------- ACF period ----------
def _acf_primary_peak(signal: np.ndarray, fs: float, min_s: float, max_s: float):
    n = len(signal)
    if n < int(fs * min_s) + 3:
        return 0.0, 0.0
    x = signal - np.mean(signal)
    std = np.std(x)
    if not np.isfinite(std) or std < 1e-8:
        return 0.0, 0.0

    ac = np.correlate(x, x, mode="full")[n - 1:]
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


def estimate_rep_period_acf(signal: np.ndarray, fs: float, min_s: float = 0.4, max_s: float = 3.0) -> float:
    p, _ = _acf_primary_peak(signal, fs, min_s, max_s)
    return float(p)


# ---------- FFT band ratio (lightweight, no extra deps) ----------
def _rfft_band_ratio(x: np.ndarray, fs: float, band_lo: float = 0.3, band_hi: float = 3.0):
    """
    Returns: band_energy, band_ratio, dominant_frequency_in_band
    """
    n = len(x)
    if n < 8:
        return 0.0, 0.0, 0.0

    idx = np.arange(n)
    win = 0.54 - 0.46 * np.cos(2 * np.pi * idx / max(1, n - 1))
    x0 = (x - np.mean(x)) * win

    spec = np.fft.rfft(x0)
    pwr = (np.abs(spec) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    total = float(np.sum(pwr))
    mask = (freqs >= band_lo) & (freqs <= band_hi)
    band = float(np.sum(pwr[mask])) if np.any(mask) else 0.0
    ratio = band / (total + 1e-12)

    domf = 0.0
    if np.any(mask):
        pwr_band = pwr[mask]
        if pwr_band.size > 0 and np.max(pwr_band) > 0:
            domf = float(freqs[mask][int(np.argmax(pwr_band))])

    return float(band), float(ratio), float(domf)


# ---------- Better signal selection ----------
def select_rep_signal(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float) -> np.ndarray:
    """
    Universellere Signalwahl:
    Score = 2.0*ACF_peak + 1.2*band_ratio + 0.2*std_robust
    """
    ax = np.asarray(ax, float)
    ay = np.asarray(ay, float)
    az = np.asarray(az, float)
    mag = np.sqrt(ax * ax + ay * ay + az * az)

    cands_raw = [ax, ay, az, mag]
    cands = [smooth_hp(s, fs) for s in cands_raw]

    scores = []
    for sig in cands:
        rz = robust_z(sig)
        std_r = float(np.std(rz))
        _, acf_peak = _acf_primary_peak(rz, fs, 0.45, 2.5)  # enger: weniger "zu langsame" Peaks
        _, br, _ = _rfft_band_ratio(rz, fs, 0.3, 3.0)
        score = 2.0 * acf_peak + 1.2 * br + 0.2 * std_r
        scores.append(score)

    best = int(np.argmax(scores))
    return cands[best]


# ---------- Helper: robust period from ACF + domf ----------
def _period_from_domf(domf: float) -> float:
    if not np.isfinite(domf) or domf <= 1e-6:
        return 0.0
    return float(1.0 / domf)


def estimate_period_consensus(
    rz: np.ndarray,
    fs: float,
    acf_min_s: float,
    acf_max_s: float,
    band_lo: float = 0.3,
    band_hi: float = 3.0,
    consensus_tol: float = 0.35
):
    """
    Returns: period_s, acf_peak, band_ratio, domf, p_dom, ok_period
    Strategy:
    - get p_acf + peak strength
    - get domf in band -> p_dom
    - if both exist and agree within tol => use average
    - else prefer domf-derived if acf looks suspiciously slow (near acf_max_s)
      or if acf_peak weak; otherwise use acf.
    """
    p_acf, acf_peak = _acf_primary_peak(rz, fs, acf_min_s, acf_max_s)
    _, br, domf = _rfft_band_ratio(rz, fs, band_lo, band_hi)
    p_dom = _period_from_domf(domf)

    p = float(p_acf) if np.isfinite(p_acf) else 0.0
    ok = False

    if p_acf > 0 and p_dom > 0:
        rel = abs(p_acf - p_dom) / max(1e-9, min(p_acf, p_dom))
        if rel <= consensus_tol:
            p = 0.5 * (p_acf + p_dom)
            ok = True
        else:
            # disagreement: choose more plausible
            # if ACF pushes to very slow edge, domf is often better
            if (p_acf >= 0.85 * acf_max_s) or (acf_peak < 0.22):
                p = p_dom
            else:
                p = p_acf
            ok = True
    elif p_dom > 0:
        p = p_dom
        ok = True
    elif p_acf > 0:
        p = p_acf
        ok = True

    return float(p), float(acf_peak), float(br), float(domf), float(p_dom), bool(ok)


# ---------- Rep Counting (adaptive, robust) ----------
def count_reps_adaptive(
    signal: np.ndarray,
    fs: float,
    base_k: float = 0.7,
    min_s: float = 0.55,
    max_s: float = 3.5,

    # tighter default ACF search to avoid "p=2.8s" for fast reps
    acf_min_s: float = 0.50,
    acf_max_s: float = 2.50,

    # reliability / trimming
    trim_s: float = 0.6,          # slightly less aggressive than 0.8
    min_dur_s: float = 10.0,      # too short => unreliable
    min_acf_peak: float = 0.12,   # periodicity must be visible
    min_band_ratio: float = 0.70, # rep-band energy must dominate

    # NEW: threshold adaptation / fallbacks
    min_crossings_each: int = 2,  # need at least N peaks and troughs over thresholds
    k_floor: float = 0.35,        # never go below this
    k_step: float = 0.10,         # when too few crossings, relax threshold
    fallback_peak_mode: str = "median_mad",
    fallback_peak_prom: float = 0.0,
    fallback_when_pair_zero: bool = True,
):
    """
    Adaptive Rep-Counting with diagnostics & fallbacks.

    Returns: reps:int, debug:dict
      debug['ok'] indicates periodicity gate passed (i.e., reps are meaningful).
      debug['method'] indicates which method produced final reps.
    """
    sig = np.asarray(signal, float)
    if sig.size < 3 or fs <= 0:
        return 0, dict(ok=False, reason="too_short_or_fs", method="none")

    # ---- Trim transitions (start/end) ----
    n = int(sig.size)
    trim_n = int(round(trim_s * fs))
    if 2 * trim_n < n - 3:
        sig_use = sig[trim_n: n - trim_n]
    else:
        sig_use = sig

    rz = robust_z(sig_use)
    dur_s = float(len(rz) / float(fs)) if fs > 0 else 0.0
    if dur_s < float(min_dur_s):
        return 0, dict(ok=False, reason="segment_too_short", dur_s=dur_s, method="none")

    # Period + metrics (ACF + domf consensus)
    p, acf_peak, br, domf, p_dom, p_ok = estimate_period_consensus(
        rz, fs, acf_min_s, acf_max_s, 0.3, 3.0, consensus_tol=0.35
    )

    std_r = float(np.std(rz))
    snr = float(br * std_r)

    # ---- Reliability gate ----
    if (not np.isfinite(acf_peak)) or (not np.isfinite(br)) or (acf_peak < float(min_acf_peak)) or (br < float(min_band_ratio)):
        return 0, dict(
            ok=False,
            reason="not_periodic_enough",
            method="none",
            dur_s=float(dur_s),
            acf_peak=float(acf_peak),
            band_ratio=float(br),
            std_r=float(std_r),
            snr=float(snr),
            period_s=float(p),
            domf=float(domf),
            p_dom=float(p_dom),
        )

    # min/max rep duration (hard floors, but derived from period if plausible)
    min_rep_s = float(min_s)
    max_rep_s = float(max_s)
    if p_ok and np.isfinite(p) and 0.35 <= float(p) <= 4.0:
        # more permissive than before: don't explode, but allow faster reps
        min_rep_s = max(0.45, 0.50 * float(p))
        max_rep_s = min(5.00, 2.20 * float(p))
        if min_rep_s >= max_rep_s:
            min_rep_s = float(min_s)
            max_rep_s = float(max_s)

    # Adaptive k from SNR, but NEVER set k=0
    k = float(base_k)
    if snr < 0.10:
        k = max(k_floor, base_k - 0.25)
    elif snr < 0.20:
        k = max(k_floor, base_k - 0.15)
    elif snr > 0.45:
        k = min(1.00, base_k + 0.10)
    k = max(k_floor, float(k))

    m = mad(rz)
    if m < 1e-6:
        # fallback: use std as scale
        m = float(np.std(rz)) * 0.8
    if m < 1e-6:
        return 0, dict(ok=False, reason="scale_zero", dur_s=float(dur_s), method="none")

    med = median(rz)

    # ------------------------------------------------------------
    # Pair-counter with threshold adaptation:
    # - if too few crossings (peaks/troughs), relax k stepwise
    # ------------------------------------------------------------
    def _pair_count_with_k(k_use: float):
        up = med + k_use * m
        down = med - k_use * m

        last_ext_t = None
        last_ext_type = None
        reps_raw = 0

        n_peaks = 0
        n_troughs = 0
        n_pairs_ok = 0
        n_pairs_fast = 0
        n_pairs_slow = 0

        for i in range(1, len(rz) - 1):
            s0, s1, s2 = rz[i - 1], rz[i], rz[i + 1]
            t = i / fs

            if s1 > up and s1 > s0 and s1 > s2:
                n_peaks += 1
                if last_ext_type == "trough":
                    dt = t - last_ext_t
                    if dt < min_rep_s:
                        n_pairs_fast += 1
                    elif dt > max_rep_s:
                        n_pairs_slow += 1
                    else:
                        reps_raw += 1
                        n_pairs_ok += 1
                last_ext_t = t
                last_ext_type = "peak"

            elif s1 < down and s1 < s0 and s1 < s2:
                n_troughs += 1
                if last_ext_type == "peak":
                    dt = t - last_ext_t
                    if dt < min_rep_s:
                        n_pairs_fast += 1
                    elif dt > max_rep_s:
                        n_pairs_slow += 1
                    else:
                        reps_raw += 1
                        n_pairs_ok += 1
                last_ext_t = t
                last_ext_type = "trough"

        return int(reps_raw), int(n_peaks), int(n_troughs), int(n_pairs_ok), int(n_pairs_fast), int(n_pairs_slow), float(up), float(down)

    k_used = float(k)
    reps_raw, n_peaks, n_troughs, n_pairs_ok, n_pairs_fast, n_pairs_slow, up, down = _pair_count_with_k(k_used)

    # relax threshold if we basically never crossed
    relax_steps = 0
    while (n_peaks < min_crossings_each or n_troughs < min_crossings_each) and (k_used - k_step) >= k_floor and relax_steps < 5:
        k_used = max(k_floor, k_used - k_step)
        reps_raw, n_peaks, n_troughs, n_pairs_ok, n_pairs_fast, n_pairs_slow, up, down = _pair_count_with_k(k_used)
        relax_steps += 1

    reps_pair = int(reps_raw)

    # Anti-explosion limiter (rate-based)
    reps_limited = int(reps_pair)
    max_reps = 0
    if dur_s > 0:
        max_rate = 1.0 / max(0.45, min_rep_s)  # conservative but tied to min_rep_s
        max_reps = int(np.floor(dur_s * max_rate)) + 1
        reps_limited = int(min(reps_limited, max_reps))

    method = "pair"
    reps_final = int(reps_limited)

    # ------------------------------------------------------------
    # Fallback when pair returns 0 but periodic evidence is strong
    # (this addresses your "0 reps obwohl acf/br top" case)
    # ------------------------------------------------------------
    fb_reps = 0
    fb_max = 0
    if fallback_when_pair_zero and reps_final == 0:
        # Use peak counting on same rz with separation near expected min_rep_s
        fb_sep = float(min_rep_s)
        fb_reps = int(count_peaks(rz, fs, min_separation_s=fb_sep, thresh_mode=fallback_peak_mode, prominence=fallback_peak_prom))
        # peaks are "one phase" counts; for many exercises it's close enough.
        # keep conservative: cap with same limiter
        if dur_s > 0:
            fb_max = int(np.floor(dur_s * (1.0 / max(0.45, min_rep_s)))) + 1
            fb_reps = int(min(fb_reps, fb_max))

        if fb_reps > 0:
            reps_final = int(fb_reps)
            method = "fallback_peaks"

    dbg = dict(
        ok=True,
        method=str(method),
        reps=int(reps_final),
        reps_pair=int(reps_pair),
        reps_raw=int(reps_raw),
        reps_fallback=int(fb_reps),
        k=float(k),
        k_used=float(k_used),
        relax_steps=int(relax_steps),
        acf_peak=float(acf_peak),
        period_s=float(p),
        p_dom=float(p_dom),
        band_ratio=float(br),
        domf=float(domf),
        std_r=float(std_r),
        snr=float(snr),
        min_rep_s=float(min_rep_s),
        max_rep_s=float(max_rep_s),
        up=float(up),
        down=float(down),
        dur_s=float(dur_s),
        max_reps=int(max_reps),
        n_peaks=int(n_peaks),
        n_troughs=int(n_troughs),
        n_pairs_ok=int(n_pairs_ok),
        n_pairs_fast=int(n_pairs_fast),
        n_pairs_slow=int(n_pairs_slow),
        trim_s=float(trim_s),
    )
    return int(reps_final), dbg
