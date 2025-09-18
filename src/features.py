# src/features.py
import math
from typing import Tuple, List
import numpy as np
import pandas as pd

# ---------- leichte Helfer ----------
def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x.astype(float, copy=True)
    x = x.astype(float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[k:] - c[:-k]) / k
    pad = np.full(k-1, y[0] if len(y) else 0.0)
    return np.concatenate([pad, y])

def highpass_ma(x: np.ndarray, fs: float, cutoff_s: float = 0.7) -> np.ndarray:
    k = max(1, int(round(cutoff_s*fs)))
    trend = moving_average(x, k)
    return x.astype(float) - trend

def zero_cross_rate(x: np.ndarray) -> float:
    if len(x) < 2: return 0.0
    s = np.sign(x)
    for i in range(1, len(s)):
        if s[i] == 0: s[i] = s[i-1]
    return float(np.sum(s[1:] * s[:-1] < 0)) / max(1, len(x)-1)

def slope_sign_changes(x: np.ndarray) -> float:
    if len(x) < 3: return 0.0
    d1 = np.diff(x)
    s1 = np.sign(d1)
    for i in range(1, len(s1)):
        if s1[i] == 0: s1[i] = s1[i-1]
    return float(np.sum(s1[1:] * s1[:-1] < 0)) / max(1, len(d1)-1)

def waveform_length(x: np.ndarray) -> float:
    if len(x) < 2: return 0.0
    return float(np.sum(np.abs(np.diff(x))) / (len(x)-1))

def hjorth_params(x: np.ndarray) -> tuple:
    if len(x) < 3: return 0.0, 0.0, 0.0
    x = x.astype(float)
    var0 = np.var(x)
    dx = np.diff(x)
    var1 = np.var(dx)
    ddx = np.diff(dx)
    var2 = np.var(ddx)
    if var0 <= 1e-12: return 0.0, 0.0, 0.0
    mob = math.sqrt(var1/var0) if var1 > 0 else 0.0
    comp = math.sqrt((var2/var1)) / mob if var1 > 1e-12 and mob > 0 else 0.0
    return float(var0), float(mob), float(comp)

def rfft_band_features(x: np.ndarray, fs: float,
                       band_lo: float = 0.3, band_hi: float = 3.0):
    n = len(x)
    if n < 4:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    win = 0.54 - 0.46 * np.cos(2*np.pi*np.arange(n)/(n-1))
    xw = (x - np.mean(x)) * win
    spec = np.fft.rfft(xw)
    pwr = (np.abs(spec)**2) / n
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    total = float(np.sum(pwr))
    mask = (freqs >= band_lo) & (freqs <= band_hi)
    band = float(np.sum(pwr[mask])) if np.any(mask) else 0.0
    ratio = (band / total) if total > 1e-12 else 0.0

    if np.any(mask) and np.max(pwr[mask]) > 0:
        idx_band = np.argmax(pwr[mask])
        domf_band = float(freqs[mask][idx_band])
        domp_band = float(pwr[mask][idx_band] / (total + 1e-12))
    else:
        domf_band, domp_band = 0.0, 0.0

    centroid = float(np.sum(freqs * pwr) / (total + 1e-12))
    return band, ratio, domf_band, domp_band, centroid

def autocorr_lag(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag: return 0.0
    x0 = x - x.mean()
    num = float(np.dot(x0[:-lag], x0[lag:]))
    den = max(1.0, (len(x0) - lag))
    return num / den

def acf_primary_period(x: np.ndarray, fs: float,
                       min_s: float = 0.4, max_s: float = 3.0) -> tuple:
    n = len(x)
    if n < int(fs*min_s) + 3: return 0.0, 0.0
    y = x - np.mean(x)
    std = np.std(y)
    if not np.isfinite(std) or std < 1e-8: return 0.0, 0.0
    ac = np.correlate(y, y, mode="full")[n-1:]
    ac = ac / (ac[0] + 1e-12)
    min_lag = max(1, int(round(min_s*fs)))
    max_lag = min(n-2, int(round(max_s*fs)))
    if max_lag <= min_lag: return 0.0, 0.0
    lag = None
    for i in range(min_lag+1, max_lag-1):
        if ac[i] > ac[i-1] and ac[i] > ac[i+1]:
            lag = i; break
    if lag is None:
        lag = min_lag + int(np.argmax(ac[min_lag:max_lag]))
    return float(lag/fs), float(ac[lag])

def robust_stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        return dict(med=0, mad=0, p10=0, p25=0, p75=0, p90=0, iqr=0, ptp=0)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    q10, q25, q75, q90 = np.percentile(x, [10,25,75,90]).astype(float)
    return dict(
        med=med, mad=mad, p10=q10, p25=q25, p75=q75, p90=q90,
        iqr=float(q75 - q25), ptp=float(np.ptp(x))
    )

def tilt_angles(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float, lp_s: float = 0.7):
    k = max(1, int(round(lp_s*fs)))
    gx = moving_average(ax, k); gy = moving_average(ay, k); gz = moving_average(az, k)
    eps = 1e-9
    pitch = np.arctan2(-gx, np.sqrt(gy*gy + gz*gz) + eps)
    roll  = np.arctan2(gy, gz + eps)
    return pitch, roll

# ---------- Feature-Extraktion pro Fenster ----------
def window_features(df_win: pd.DataFrame, fs: float) -> Tuple[np.ndarray, List[str]]:
    ax = df_win["ax"].to_numpy(float)
    ay = df_win["ay"].to_numpy(float)
    az = df_win["az"].to_numpy(float)
    mag = np.sqrt(ax*ax + ay*ay + az*az)

    # highpass + slight smooth
    ax_hp = moving_average(highpass_ma(ax, fs, 0.7), max(1, int(round(0.16*fs))))
    ay_hp = moving_average(highpass_ma(ay, fs, 0.7), max(1, int(round(0.16*fs))))
    az_hp = moving_average(highpass_ma(az, fs, 0.7), max(1, int(round(0.16*fs))))
    mag_hp = moving_average(highpass_ma(mag, fs, 0.7), max(1, int(round(0.16*fs))))

    axes_hp = [ax_hp, ay_hp, az_hp]
    best = axes_hp[int(np.argmax([np.var(a) for a in axes_hp]))]

    def stats_full(arr: np.ndarray, pre: str):
        feats, names = [], []
        mu = float(arr.mean()) if len(arr) else 0.0
        sd = float(arr.std()) if len(arr) else 0.0
        mn = float(arr.min()) if len(arr) else 0.0
        mx = float(arr.max()) if len(arr) else 0.0
        var = float(arr.var()) if len(arr) else 0.0
        rms = float(np.sqrt(np.mean(arr*arr))) if len(arr) else 0.0
        rob = robust_stats(arr)
        ac1 = autocorr_lag(arr, 1); ac2 = autocorr_lag(arr, 2)
        wl = waveform_length(arr)
        zcr = zero_cross_rate(arr)
        ssc = slope_sign_changes(arr)
        act, mob, comp = hjorth_params(arr)
        feats += [mu, sd, mn, mx, var, rms,
                  rob["med"], rob["mad"], rob["p10"], rob["p25"], rob["p75"], rob["p90"], rob["iqr"], rob["ptp"],
                  ac1, ac2, wl, zcr, ssc, act, mob, comp]
        names += [f"{pre}_mean", f"{pre}_std", f"{pre}_min", f"{pre}_max", f"{pre}_var", f"{pre}_rms",
                  f"{pre}_med", f"{pre}_mad", f"{pre}_p10", f"{pre}_p25", f"{pre}_p75", f"{pre}_p90", f"{pre}_iqr", f"{pre}_ptp",
                  f"{pre}_ac1", f"{pre}_ac2", f"{pre}_wl", f"{pre}_zcr", f"{pre}_ssc", f"{pre}_hj_act", f"{pre}_hj_mob", f"{pre}_hj_comp"]
        return feats, names

    feats, names = [], []
    f, n = stats_full(mag_hp, "maghp"); feats += f; names += n
    f, n = stats_full(best,   "besthp"); feats += f; names += n

    # Jerk (Magnitude)
    if len(mag_hp) >= 2:
        jerk = np.diff(mag_hp) * fs
        feats += [float(np.sqrt(np.mean(jerk*jerk))), float(np.mean(np.abs(jerk)))]
    else:
        feats += [0.0, 0.0]
    names += ["jerk_rms", "jerk_mav"]

    # Autokorrelation (Periode + Peak)
    per_s, ac_peak = acf_primary_period(best, fs, 0.4, 3.0)
    feats += [per_s, ac_peak]; names += ["acf_period_s", "acf_peak"]

    # Frequenzband 0.3–3 Hz
    band_e, band_ratio, domf, domp, centroid = rfft_band_features(best, fs, 0.3, 3.0)
    feats += [band_e, band_ratio, domf, domp, centroid]
    names += ["spec_band_energy", "spec_band_ratio", "spec_domf", "spec_domp", "spec_centroid"]

    # Cross-Axis-Korrelationen (absolut)
    def safe_corr(a, b):
        if len(a) != len(b) or len(a) < 3: return 0.0
        sa, sb = np.std(a), np.std(b)
        if sa < 1e-8 or sb < 1e-8: return 0.0
        return float(np.corrcoef(a, b)[0,1])
    c_xy = abs(safe_corr(ax_hp, ay_hp)); c_xz = abs(safe_corr(ax_hp, az_hp)); c_yz = abs(safe_corr(ay_hp, az_hp))
    feats += [c_xy, c_xz, c_yz]
    names += ["corr_xy_abs", "corr_xz_abs", "corr_yz_abs"]

    # Orientierung (Pitch/Roll)
    pitch, roll = tilt_angles(df_win["ax"].to_numpy(float),
                              df_win["ay"].to_numpy(float),
                              df_win["az"].to_numpy(float), fs, 0.7)
    feats += [float(np.mean(pitch)), float(np.std(pitch)), float(np.mean(roll)), float(np.std(roll))]
    names += ["tilt_pitch_mean", "tilt_pitch_std", "tilt_roll_mean", "tilt_roll_std"]

    # Baro (optional)
    if "baro" in df_win.columns and df_win["baro"].notna().any():
        baro = df_win["baro"].ffill().bfill().to_numpy(float)
        baro_mean = float(baro.mean()); baro_var = float(baro.var()); baro_ac1 = autocorr_lag(baro, 1)
        dhdt = float((baro[-1] - baro[0]) * fs / max(1, len(baro)))
    else:
        baro_mean = baro_var = baro_ac1 = dhdt = 0.0
    feats += [baro_mean, baro_var, baro_ac1, dhdt]
    names += ["baro_mean", "baro_var", "baro_ac1", "baro_dhdt"]

    # Steps-Delta (optional)
    if "steps" in df_win.columns and df_win["steps"].notna().any():
        steps = df_win["steps"].ffill().bfill().to_numpy(float)
        steps_delta = float(steps[-1] - steps[0])
    else:
        steps_delta = 0.0
    feats += [steps_delta]; names += ["steps_delta"]

    # HR mean/std (optional)
    if "hr" in df_win.columns and df_win["hr"].notna().any():
        hr = df_win["hr"].dropna().astype(float).to_numpy()
        feats += [float(hr.mean()), float(hr.std())]
    else:
        feats += [0.0, 0.0]
    names += ["hr_mean", "hr_std"]

    return np.asarray(feats, dtype=np.float32), names

# ---------- Fensterung über eine Datei / DataFrame ----------
def build_windows(df: pd.DataFrame, fs: float, win_s: float, hop_s: float):
    """
    Schneidet Fenster und erzeugt:
      X  : Feature-Matrix (N, F)
      y  : Fenster-Labels (Mehrheit im Fenster) – kann None enthalten
      t0s: Startzeit pro Fenster (Sekunden relativ in df['t'])
      names: Feature-Namen
    """
    df = df.sort_values("t").reset_index(drop=True)
    if "t" not in df.columns:
        if "t_rel" in df.columns:
            df = df.copy()
            df["t"] = pd.to_numeric(df["t_rel"], errors="coerce")
            df = df.dropna(subset=["t"]).reset_index(drop=True)
        else:
            raise KeyError("build_windows erwartet Spalte 't' oder 't_rel'.")

    win = int(round(win_s * fs))
    hop = int(round(hop_s * fs))
    if win <= 0 or hop <= 0:
        return np.zeros((0, 0), dtype=np.float32), [], [], []

    X, y, t0s, names = [], [], [], None
    i = 0
    n = len(df)
    while i + win <= n:
        dfw = df.iloc[i:i+win]
        feats, nms = window_features(dfw, fs)
        if names is None:
            names = nms
        # Mehrheitslabel im Fenster (falls vorhanden)
        lab = None
        if "label" in dfw.columns:
            labs = dfw["label"].dropna().astype(str).str.upper().values
            if len(labs):
                vals, cnts = np.unique(labs, return_counts=True)
                lab = str(vals[np.argmax(cnts)])
        X.append(feats); y.append(lab); t0s.append(float(dfw["t"].iloc[0]))
        i += hop

    X = np.vstack(X) if len(X) else np.zeros((0, len(names or [])), dtype=np.float32)
    return X, y, t0s, (names or [])
