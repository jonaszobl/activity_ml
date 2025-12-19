# -------------------------------------------------------------------
# Feature extraction / current setup:
# - 10 Hz sampling rate
# - 5s feature windows
# - 2.5s window hop (overlap)
# - used by training and prediction
# -------------------------------------------------------------------

import math
from typing import Tuple, List

import numpy as np
import pandas as pd


# ---------- Helper stuff ----------
def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """
    The signal gets smoothed over k data
    If k <= 1 just return the data as float without smoothing.
    That way you see the trend -> signal gets smoothed
    """
    if k <= 1:
        return x.astype(float, copy=True)

    x = x.astype(float)

    # cumulative sum trick for O(n) moving average
    c = np.cumsum(np.insert(x, 0, 0.0))
    # [1,2,3,4,5,6] -> 6-1, 5-2...
    y = (c[k:] - c[:-k]) / k

    # pad beginning to keep same length x has k-1 false starting numbers -> remove them
    if len(y) > 0:
        first_val = y[0]
    else:
        first_val = 0.0
    pad = np.full(k - 1, first_val)

    return np.concatenate([pad, y])


def highpass_ma(x: np.ndarray, fs: float, cutoff_s: float = 0.7) -> np.ndarray:
    """
    Very simple high-pass filter:
    highpass(x) = x - moving_average(x)
    -> removes slow trend (gravity etc.) that way just the "zappeln" is left -> important for eg. reps
    """
    k = int(round(cutoff_s * fs))
    if k < 1:
        k = 1
    trend = moving_average(x, k)
    x_float = x.astype(float)
    return x_float - trend


def zero_cross_rate(x: np.ndarray) -> float:
    """
    Count how often the signal changes direction from positiv to negativ -> crossing the zero line at the graph
    Counts changes of + and - signals 
    """
    n = len(x)
    if n < 2:
        return 0.0

    s = np.sign(x).astype(float)

    # replace zeros by previous sign so they don't create fake crossings
    for i in range(1, n):
        if s[i] == 0:
            s[i] = s[i - 1]

    crossings = np.sum(s[1:] * s[:-1] < 0)
    return float(crossings) / max(1, n - 1)


def slope_sign_changes(x: np.ndarray) -> float:
    """
    Shows when signal changes direction -> local maxima or minima
    -> 0.6 to 0.8 = +0.2 but to 0.4 = -0.2 -> would be no zcr change
    """
    if len(x) < 3:
        return 0.0

    d1 = np.diff(x)
    s1 = np.sign(d1).astype(float)

    for i in range(1, len(s1)):
        if s1[i] == 0:
            s1[i] = s1[i - 1]

    changes = np.sum(s1[1:] * s1[:-1] < 0)
    return float(changes) / max(1, len(d1) - 1)


def waveform_length(x: np.ndarray) -> float:
    """
    Sum of absolute differences.
    Often used as a simple "activity" / roughness measure.
    Shows how much movement there is
    """
    if len(x) < 2:
        return 0.0
    diffs = np.diff(x)
    return float(np.sum(np.abs(diffs)) / (len(x) - 1))


def hjorth_params(x: np.ndarray) -> tuple:
    """
    Hjorth parameters for time series:
    - activity: variance of the signal
    - mobility: sqrt(var(dx) / var(x))
    - complexity: compares second derivative to mobility
    """
    if len(x) < 3:
        return 0.0, 0.0, 0.0

    x = x.astype(float)

    var0 = np.var(x)            # activity
    dx = np.diff(x)
    var1 = np.var(dx)
    ddx = np.diff(dx)
    var2 = np.var(ddx)

    if var0 <= 1e-12:
        return 0.0, 0.0, 0.0

    if var1 > 0:
        mob = math.sqrt(var1 / var0)
    else:
        mob = 0.0

    if var1 > 1e-12 and mob > 0:
        comp = math.sqrt(var2 / var1) / mob
    else:
        comp = 0.0

    return float(var0), float(mob), float(comp)


def rfft_band_features(
    x: np.ndarray,
    fs: float,
    band_lo: float = 0.3,
    band_hi: float = 3.0
):
    """
    FFT-based features in the band [band_lo, band_hi] (usually 0.3..3 Hz).
    This is roughly the repetition band for strength exercises, because 3Hz seem to much but it also gets the little peaks
    """
    n = len(x)
    if n < 4:
        # too short -> just return zeros, same as before
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # simple Hamming-like window -> reduces spectrum noise, just take that as a fact
    idx = np.arange(n)
    win = 0.54 - 0.46 * np.cos(2 * np.pi * idx / (n - 1))

    x_centered = x - np.mean(x)
    xw = x_centered * win

    spec = np.fft.rfft(xw)
    pwr = (np.abs(spec) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    total = float(np.sum(pwr))

    mask = (freqs >= band_lo) & (freqs <= band_hi)
    if np.any(mask):
        band = float(np.sum(pwr[mask]))
    else:
        band = 0.0

    if total > 1e-12:
        ratio = band / total
    else:
        ratio = 0.0

    # dominant frequency in band
    if np.any(mask) and np.max(pwr[mask]) > 0:
        pwr_band = pwr[mask]
        idx_band = int(np.argmax(pwr_band))
        domf_band = float(freqs[mask][idx_band])
        domp_band = float(pwr_band[idx_band] / (total + 1e-12))
    else:
        domf_band = 0.0
        domp_band = 0.0

    # spectral centroid
    centroid = float(np.sum(freqs * pwr) / (total + 1e-12))

    return band, ratio, domf_band, domp_band, centroid


def autocorr_lag(x: np.ndarray, lag: int) -> float:
    """
    Basic autocorrelation at a specific lag (not normalized).
    For example compare a signal to a signal in 3sek to get similarities
    """
    if len(x) <= lag:
        return 0.0

    x0 = x - x.mean()
    num = float(np.dot(x0[:-lag], x0[lag:]))
    den = max(1.0, (len(x0) - lag))
    return num / den


def acf_primary_period(
    x: np.ndarray,
    fs: float,
    min_s: float = 0.4,
    max_s: float = 3.0
) -> tuple:
    """
    Try to find the first meaningful period using the autocorrelation function.
    Search lag in [min_s, max_s] (seconds).
    Returns:
      (period_in_seconds, peak_height)
    Example:
    lag=0   lag=1   lag=2   lag=3   lag=4   lag=5 ...
      |       |       |       |       |       |
     1.0     0.5     0.1     0.6     0.2     ...
    """
    n = len(x)
    if n < int(fs * min_s) + 3:
        return 0.0, 0.0

    y = x - np.mean(x)
    std = np.std(y)
    if (not np.isfinite(std)) or std < 1e-8:
        return 0.0, 0.0

    ac_full = np.correlate(y, y, mode="full")
    ac = ac_full[n - 1:]
    ac = ac / (ac[0] + 1e-12)

    min_lag = max(1, int(round(min_s * fs)))
    max_lag = min(n - 2, int(round(max_s * fs)))
    if max_lag <= min_lag:
        return 0.0, 0.0

    lag = None

    # look for the first local maximum
    for i in range(min_lag + 1, max_lag - 1):
        if ac[i] > ac[i - 1] and ac[i] > ac[i + 1]:
            lag = i
            break

    # if not found, fallback to argmax in the interval
    if lag is None:
        best_idx = int(np.argmax(ac[min_lag:max_lag]))
        lag = min_lag + best_idx

    period_s = float(lag / fs)
    peak_val = float(ac[lag])
    return period_s, peak_val


def robust_stats(x: np.ndarray) -> dict:
    """
    Robust statistics:
      median, MAD, some quantiles, IQR, peak-to-peak.
      Not sensitive to unexpected signals
    """
    if len(x) == 0:
        return dict(
            med=0, mad=0,
            p10=0, p25=0, p75=0, p90=0,
            iqr=0, ptp=0
        )

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))

    q10, q25, q75, q90 = np.percentile(x, [10, 25, 75, 90]).astype(float)

    stats = dict(
        med=med,
        mad=mad,
        p10=q10,
        p25=q25,
        p75=q75,
        p90=q90,
        iqr=float(q75 - q25),
        ptp=float(np.ptp(x))
    )
    return stats


def tilt_angles(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    fs: float,
    lp_s: float = 0.7
):
    """
    Estimate slow orientation from raw accel data.
    Just low-pass each axis and compute pitch/roll.
    """
    k = int(round(lp_s * fs))
    if k < 1:
        k = 1

    gx = moving_average(ax, k)
    gy = moving_average(ay, k)
    gz = moving_average(az, k)

    eps = 1e-9

    pitch = np.arctan2(-gx, np.sqrt(gy * gy + gz * gz) + eps)
    roll = np.arctan2(gy, gz + eps)

    return pitch, roll


# ---------- Feature extraction for a single window ----------
def window_features(df_win: pd.DataFrame, fs: float) -> Tuple[np.ndarray, List[str]]:
    """
    Compute all features for one time window.
    The logic is exactly the same as in the original version.
    """
    # get accel columns
    ax = df_win["ax"].to_numpy(float)
    ay = df_win["ay"].to_numpy(float)
    az = df_win["az"].to_numpy(float)

    # magnitude (overall acceleration)
    mag = np.sqrt(ax * ax + ay * ay + az * az)

    # high-pass + smoothing (~160 ms)
    smooth_k = max(1, int(round(0.16 * fs)))

    ax_hp = moving_average(highpass_ma(ax, fs, 0.7), smooth_k)
    ay_hp = moving_average(highpass_ma(ay, fs, 0.7), smooth_k)
    az_hp = moving_average(highpass_ma(az, fs, 0.7), smooth_k)
    mag_hp = moving_average(highpass_ma(mag, fs, 0.7), smooth_k)

    # choose axis with highest variance as "best" axis
    axes_hp = [ax_hp, ay_hp, az_hp]
    variances = [np.var(a) for a in axes_hp]
    best_idx = int(np.argmax(variances))
    best = axes_hp[best_idx]

    
    # new symmetry and highpass of the "best" axis
    if len(best) > 0:
        peak_max = float(np.max(best))
        peak_min = float(np.min(best))
        range_amp = peak_max - peak_min
        if range_amp <= 1e-9:
            symmetry_score = 0.0
        else:
            symmetry_score = abs(peak_max + peak_min) / (range_amp + 1e-9)
    else:
        symmetry_score = 0.0

    # compare highpass enregy to raw energy (more "Noise" = more likly DB)
    hp_energy = float(np.sum(best * best))
    raw_energy = float(np.sum(ax * ax + ay * ay + az * az))
    hp_ratio = hp_energy / (raw_energy + 1e-9)


    # new to better differ between db,bb,sp
    # dominant axis 
    var_x, var_y, var_z = variances
    var_sum = var_x + var_y + var_z + 1e-9  # Schutz gegen 0

    # varianz on each axis
    var_x_ratio = float(var_x / var_sum)
    var_y_ratio = float(var_y / var_sum)
    var_z_ratio = float(var_z / var_sum)

    # index of the dominant axis
    dominant_axis_idx = float(best_idx)

    # energie -> sum of the pows
    eng_x = float(np.sum(ax_hp * ax_hp))
    eng_y = float(np.sum(ay_hp * ay_hp))
    eng_z = float(np.sum(az_hp * az_hp))
    eng_total = eng_x + eng_y + eng_z + 1e-9  # Schutz gg. 0

    eng_x_ratio = eng_x / eng_total
    eng_y_ratio = eng_y / eng_total
    eng_z_ratio = eng_z / eng_total

    # repetition band per axis 0.3-3Hz
    band_x, band_ratio_x, _, _, _ = rfft_band_features(ax_hp, fs, 0.3, 3.0)
    band_y, band_ratio_y, _, _, _ = rfft_band_features(ay_hp, fs, 0.3, 3.0)
    band_z, band_ratio_z, _, _, _ = rfft_band_features(az_hp, fs, 0.3, 3.0)

    def stats_full(arr: np.ndarray, pre: str):
        """
        Collect a bunch of basic stats + some time series stuff for one signal.
        """
        feats = []
        names = []

        if len(arr) > 0:
            mu = float(arr.mean())
            sd = float(arr.std())
            mn = float(arr.min())
            mx = float(arr.max())
            var = float(arr.var())
            rms = float(np.sqrt(np.mean(arr * arr)))
        else:
            mu = sd = mn = mx = var = rms = 0.0

        rob = robust_stats(arr)

        ac1 = autocorr_lag(arr, 1)
        ac2 = autocorr_lag(arr, 2)
        wl = waveform_length(arr)
        zcr = zero_cross_rate(arr)
        ssc = slope_sign_changes(arr)
        act, mob, comp = hjorth_params(arr)

        feats += [
            mu, sd, mn, mx, var, rms,
            rob["med"], rob["mad"], rob["p10"], rob["p25"],
            rob["p75"], rob["p90"], rob["iqr"], rob["ptp"],
            ac1, ac2, wl, zcr, ssc, act, mob, comp
        ]

        names += [
            f"{pre}_mean", f"{pre}_std", f"{pre}_min", f"{pre}_max",
            f"{pre}_var", f"{pre}_rms",
            f"{pre}_med", f"{pre}_mad", f"{pre}_p10", f"{pre}_p25",
            f"{pre}_p75", f"{pre}_p90", f"{pre}_iqr", f"{pre}_ptp",
            f"{pre}_ac1", f"{pre}_ac2", f"{pre}_wl", f"{pre}_zcr",
            f"{pre}_ssc", f"{pre}_hj_act", f"{pre}_hj_mob", f"{pre}_hj_comp"
        ]

        return feats, names

    all_feats: List[float] = []
    all_names: List[str] = []

    # stats on magnitude (robust to orientation)
    f_mag, n_mag = stats_full(mag_hp, "maghp")
    all_feats += f_mag
    all_names += n_mag

    # stats on best axis
    f_best, n_best = stats_full(best, "besthp")
    all_feats += f_best
    all_names += n_best

    # new feature to differ between the hard to predict excercises like bb,db,sp
    # the strong axis should help differ them
    all_feats += [var_x_ratio, var_y_ratio, var_z_ratio, dominant_axis_idx]
    all_names += [
        "var_ratio_x_hp", "var_ratio_y_hp", "var_ratio_z_hp",
        "dominant_axis_hp",
    ]

    # Jerk (derivative of magnitude) -> suddent movement
    if len(mag_hp) >= 2:
        jerk = np.diff(mag_hp) * fs
        jerk_rms = float(np.sqrt(np.mean(jerk * jerk)))
        jerk_mav = float(np.mean(np.abs(jerk)))
    else:
        jerk_rms = 0.0
        jerk_mav = 0.0

    all_feats += [jerk_rms, jerk_mav]
    all_names += ["jerk_rms", "jerk_mav"]

    # Autocorrelation-based period
    per_s, ac_peak = acf_primary_period(best, fs, 0.4, 3.0)
    all_feats += [per_s, ac_peak]
    all_names += ["acf_period_s", "acf_peak"]

    # FFT band features (0.3–3 Hz)
    band_e, band_ratio, domf, domp, centroid = rfft_band_features(best, fs, 0.3, 3.0)
    all_feats += [band_e, band_ratio, domf, domp, centroid]
    all_names += ["spec_band_energy", "spec_band_ratio", "spec_domf", "spec_domp", "spec_centroid"]

    # Cross-axis correlations
    def safe_corr(a, b):
        if len(a) != len(b) or len(a) < 3:
            return 0.0
        sa = np.std(a)
        sb = np.std(b)
        if sa < 1e-8 or sb < 1e-8:
            return 0.0
        mat = np.corrcoef(a, b)
        return float(mat[0, 1])

    c_xy = abs(safe_corr(ax_hp, ay_hp))
    c_xz = abs(safe_corr(ax_hp, az_hp))
    c_yz = abs(safe_corr(ay_hp, az_hp))

    all_feats += [c_xy, c_xz, c_yz]
    all_names += ["corr_xy_abs", "corr_xz_abs", "corr_yz_abs"]

    # Orientation (pitch / roll) based on raw accel
    pitch, roll = tilt_angles(
        df_win["ax"].to_numpy(float),
        df_win["ay"].to_numpy(float),
        df_win["az"].to_numpy(float),
        fs,
        0.7
    )
    pitch_mean = float(np.mean(pitch))
    pitch_std = float(np.std(pitch))
    roll_mean = float(np.mean(roll))
    roll_std = float(np.std(roll))

    all_feats += [pitch_mean, pitch_std, roll_mean, roll_std]
    all_names += ["tilt_pitch_mean", "tilt_pitch_std", "tilt_roll_mean", "tilt_roll_std"]

    # Barometer: required now
    if "baro" not in df_win.columns:
        raise KeyError("Baro is missing!")

    # interpolate NaNs first so we have something reasonable
    baro_series = df_win["baro"].astype(float)
    baro_interp = baro_series.interpolate(limit_direction="both").to_numpy()
    if np.all(np.isnan(baro_interp)):
        baro_interp = np.zeros_like(baro_interp)

    # old style block kept (logic the same as original)
    if df_win["baro"].notna().any():
        baro_used = df_win["baro"].ffill().bfill().to_numpy(float)
        baro_mean = float(baro_used.mean())
        baro_var = float(baro_used.var())
        baro_ac1 = autocorr_lag(baro_used, 1)
        dhdt = float((baro_used[-1] - baro_used[0]) * fs / max(1, len(baro_used)))
    else:
        baro_mean = 0.0
        baro_var = 0.0
        baro_ac1 = 0.0
        dhdt = 0.0

    all_feats += [baro_mean, baro_var, baro_ac1, dhdt]
    all_names += ["baro_mean", "baro_var", "baro_ac1", "baro_dhdt"]

    # new to better differ between db,bb,sp
    all_feats += [
        eng_x_ratio, eng_y_ratio, eng_z_ratio,
        band_ratio_x, band_ratio_y, band_ratio_z,
    ]
    all_names += [
        "eng_ratio_x_hp", "eng_ratio_y_hp", "eng_ratio_z_hp",
        "spec_band_ratio_x_hp", "spec_band_ratio_y_hp", "spec_band_ratio_z_hp",
    ]

    # new feature for bench_db -> has more instability than bb and sp which are genreally simular 
    db_instability = eng_x / (eng_z + 1e-9)
    eng_var = float(np.var([eng_x, eng_y, eng_z]))

    # Tilt + Slope (Instability)
    instability_score = slope_sign_changes(mag_hp) * float(np.std(roll))

    # new to better recognize bench_db out of bb and sp
    all_feats += [
    db_instability,
    eng_var,
    instability_score
    ]
    all_names += [
        "db_instability_ratio",
        "eng_variance_hp",
        "tilt_slope_instability"
    ]

    all_feats += [symmetry_score, hp_ratio]
    all_names += ["symmetry_score_besthp", "hp_ratio_besthp"]

    feat_vec = np.asarray(all_feats, dtype=np.float32)
    return feat_vec, all_names


# ---------- Sliding windows over a dataframe ----------
def build_windows(
    df: pd.DataFrame,
    fs: float,
    win_s: float,
    hop_s: float
):
    """
    Cut overlapping windows and compute features.

    Returns:
      X    : feature matrix (N, F)
      y    : label per window (majority vote), can be None
      t0s  : start times of windows (seconds)
      names: feature names
    """
    # make sure we have time sorted
    df = df.sort_values("t").reset_index(drop=True)

    if "t" not in df.columns:
        if "t_rel" in df.columns:
            df = df.copy()
            df["t"] = pd.to_numeric(df["t_rel"], errors="coerce")
            df = df.dropna(subset=["t"]).reset_index(drop=True)
        else:
            # same error as original
            raise KeyError("build_windows erwartet Spalte 't' oder 't_rel'.")

    # convert window / hop length from seconds to samples
    win = int(round(win_s * fs))
    hop = int(round(hop_s * fs))

    if win <= 0 or hop <= 0:
        # invalid params -> return empty result (like before)
        return np.zeros((0, 0), dtype=np.float32), [], [], []

    X = []
    y = []
    t0s = []
    names = None

    i = 0
    n = len(df)

    while i + win <= n:
        dfw = df.iloc[i:i + win]

        feats, nms = window_features(dfw, fs)

        if names is None:
            # remember first feature name list
            names = nms

        # majority label in the window (if labels exist)
        lab = None
        if "label" in dfw.columns:
            labs = dfw["label"].dropna().astype(str).str.upper().values
            if len(labs) > 0:
                vals, cnts = np.unique(labs, return_counts=True)
                lab = str(vals[np.argmax(cnts)])

        X.append(feats)
        y.append(lab)
        t0s.append(float(dfw["t"].iloc[0]))

        i += hop

    if len(X) > 0:
        X_mat = np.vstack(X)
    else:
        X_mat = np.zeros((0, len(names or [])), dtype=np.float32)

    return X_mat, y, t0s, (names or [])
