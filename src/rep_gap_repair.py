# src/rep_gap_repair.py
# ------------------------------------------------------------
# Rep-Gap-Detector & -Repair
# Findet "Lücken" in den Wiederholungen anhand der lokalen Kadenz
# und schlägt Insert-Positionen für mutmaßlich übersehene Reps vor.
# Optional: Snap ans Signal (Peak/Trough) im engen Suchfenster.
#
# Öffentliche API:
# - find_and_repair_rep_gaps(rep_ts, *, fs, segment_class=None,
#                            signal=None, anchor="auto",
#                            search_window_s=0.12,
#                            hi_factor=1.65, max_inserts_per_gap=3)
#   -> dict mit:
#        "rep_ts_aug": sortierte Liste aller originalen + eingefügten Zeitpunkte (s)
#        "inserted_ts": Liste nur der eingefügten Zeitpunkte (s)
#        "gaps": Liste Dicts mit Diagnose je Lücke
#
# Annahme: rep_ts sind Anker der gezählten Wiederholungen (typ. Trough-Zeiten).
# ------------------------------------------------------------
# wurde hinzugefügt um Unterzählung robuster zu korrigieren; Snap bleibt konservativ

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any, Tuple
import numpy as np

# ---------- kleine robuste Helfer ----------
def _median(a: np.ndarray) -> float:
    if a.size == 0: return 0.0  # lieber 0 als crash
    return float(np.median(a))

def _mad(a: np.ndarray) -> float:
    # MAD statt STD -> robuster gegen Ausreißer
    m = _median(a)
    return float(np.median(np.abs(a.astype(float) - m)))

def _class_name_key(name: Optional[str]) -> str:
    # Normalisierung verschiedener Klassenbezeichnungen
    if not name: return ""
    u = str(name).upper()
    if "TRICEPS" in u and "PULL" in u: return "TRICEPS_PULLDOWN"
    if "FLY" in u or ("CABLE" in u and "CHEST" in u): return "CABLE_FLY_CHEST"
    if "BENCH" in u or "BANKDR" in u: return u
    if "SHOULDER" in u and "PRESS" in u: return "SHOULDER_PRESS"
    if ("BIZEPS" in u and "H" in u) or ("BICEP" in u and "HAMMER" in u): return "BIZEPS_CURL_H"
    if "RUDERN" in u or "ROW" in u: return "RUDERN"
    if "LATERAL" in u and ("RAISE" in u or "SEIT" in u): return "LATERAL_RAISE_CABLE"
    return u  # fallback

def _default_anchor_for_class(class_name: Optional[str]) -> Literal["trough","peak"]:
    # Anker je Klasse (heuristik) -> Trough bevorzugt wo Endlage "hart" ist
    key = _class_name_key(class_name)
    if key in ("TRICEPS_PULLDOWN","RUDERN","BIZEPS_CURL_H","CABLE_FLY_CHEST","LATERAL_RAISE_CABLE"):
        return "trough"
    if "PRESS" in key or "BENCH" in key:
        return "peak"
    return "trough"  # default

# ---------- Kernlogik: erwartete Kadenz lokal schätzen ----------
def _local_expected_interval(diffs: np.ndarray, i: int, half_window: int = 2) -> float:
    # hilfsfunktion um die lokale rep-dauer über nachbarschaften zu schätzen
    L = diffs.size
    if L == 0: return 0.0
    lo = max(0, i - half_window)
    hi = min(L - 1, i + half_window)
    neigh = []
    for j in range(lo, hi + 1):
        if j == i: continue
        neigh.append(diffs[j])
    if not neigh:
        return float(np.median(diffs))  # fallback auf globalen median
    neigh = np.asarray(neigh, float)
    return float(np.median(neigh))

# ---------- Snap im Signal: in kleinem Fenster auf Extremum rasten ----------
def _snap_time_to_signal(
    t_s: float,
    signal: np.ndarray,
    fs: float,
    anchor: Literal["trough", "peak"],
    search_window_s: float
) -> Tuple[float, float]:
    # snap aufs lokale Minimum/Maximum im kleinen Fenster
    # wurde hinzugefügt weil Min/Max als Anker stabiler -> weniger Drift
    if signal is None or fs <= 0:
        return t_s, 0.0
    w = max(1, int(round(search_window_s * fs)))
    center = int(round(t_s * fs))
    L = signal.size
    a = max(1, center - w)
    b = min(L - 2, center + w)
    if b <= a + 1:
        return t_s, 0.0
    seg = signal[a:b+1].astype(float)
    if anchor == "trough":
        k = int(np.argmin(seg))
    else:
        k = int(np.argmax(seg))
    j = a + k
    curv = float(signal[j-1] - 2*signal[j] + signal[j+1]) if 1 <= j < L-1 else 0.0
    return (j / float(fs)), abs(curv)  # curv als grobe „Schärfe“ (kann noch verfeinert werden...)

# ---------- ACF-Periode (für globales p & Clamp) ----------
def _estimate_period(signal: Optional[np.ndarray], fs: float, lo: float = 0.6, hi: float = 3.5) -> Tuple[float, float]:
    # hilfsfunktion um grundtakt (s/rep) zu schätzen; bewusst schlank
    if signal is None or fs <= 0:
        return 0.0, 0.0
    n = signal.size
    if n < int(hi*fs)+3:
        return 0.0, 0.0
    x = signal.astype(float) - float(np.mean(signal))
    den = float(np.dot(x, x)) + 1e-12  # schutz gegen 0
    if den <= 1e-12: return 0.0, 0.0
    ac = np.correlate(x, x, mode="full")[n-1:] / den
    min_lag = max(1, int(round(lo*fs)))
    max_lag = min(n-2, int(round(hi*fs)))
    if max_lag <= min_lag: return 0.0, 0.0
    lag = None
    for i in range(min_lag+1, max_lag-1):
        if ac[i] > ac[i-1] and ac[i] > ac[i+1]:
            lag = i; break
    if lag is None:
        lag = min_lag + int(np.argmax(ac[min_lag:max_lag]))
    return (float(lag)/fs, float(ac[lag]))

# ---------- Cadence-Grid-Fit (φ + k·p) ----------
def _cadence_grid_fit(times: List[float], *, p: float, T: float,
                      signal: Optional[np.ndarray], fs: float,
                      anchor: Literal["trough","peak"], snap_w_s: float = 0.10,
                      tol_frac: float = 0.35) -> List[float]:
    # rastert auf φ + k·p und snapt in Zellen -> füllt Unterzählungen sanft
    if not np.isfinite(p) or p <= 1e-6 or T <= 0:
        return sorted(times)
    arr = np.array(sorted(times), float)
    phi = float(np.median(np.mod(arr, p))) if arr.size else 0.0  # median statt mean -> robuster
    phi = phi % p
    n_bins = max(1, int(round((T - phi) / p)) + 1)
    out = []
    tol = tol_frac * p
    for k in range(n_bins):
        t_grid = phi + k*p
        if t_grid < -1e-6 or t_grid > T + 1e-6:
            continue
        pick = None; dmin = 1e9
        for t in arr:
            d = abs(t - t_grid)
            if d < dmin:
                dmin = d; pick = t
        if pick is not None and dmin <= tol:
            out.append(float(pick))  # bestehender kandidat reicht
        else:
            t_sn, _ = _snap_time_to_signal(t_grid, signal, fs, anchor, snap_w_s) if signal is not None else (t_grid, 0.0)
            out.append(float(t_sn))  # snap ans extremum
    # Dedupe gegen zu enge Punkte
    out = sorted(out)
    min_d = 0.35 * max(p, 1e-6)  # vorher 0.25 -> höher gesetzt, da sonst Doppelungen
    dedup = [out[0]] if out else []
    for t in out[1:]:
        if abs(t - dedup[-1]) >= min_d:
            dedup.append(t)
    return dedup

# ---------- Hauptfunktion ----------
def find_and_repair_rep_gaps(
    rep_ts: List[float],
    *,
    fs: float,
    segment_class: Optional[str] = None,
    signal: Optional[np.ndarray] = None,
    anchor: Literal["auto","trough","peak"] = "auto",
    search_window_s: float = 0.12,
    hi_factor: float = 1.65,
    lo_factor: float = 0.55,
    max_inserts_per_gap: int = 3,
    # --- NEU: globale Plausibilitäts-Grenzen ---
    min_rep_s: Optional[float] = None,
    max_rep_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    rep_ts: Liste der bereits erkannten Rep-Ankerzeiten (Sekunden), streng monoton steigend.
    fs: Samplingrate (Hz), nur nötig falls 'signal' für Snap verwendet wird.

    Idee:
    - Diffs = t[i+1]-t[i]
    - Erwartetes Intervall lokal via Median der Nachbarn
    - Wenn diff > hi_factor * expected → Lücke; Anzahl Missing ~ round(diff/expected)-1 (>=1)
    - Optional: Snap vorgeschlagene Insert-Times ans Signal (lokales Min/Max in engem Fenster)

    hi_factor default 1.65: bewusst konservativ (≈ 65% länger als "erwartet" ⇒ Lücke).
    lo_factor wird derzeit nicht für Inserts genutzt, kann aber in der Diagnose helfen (z.B. Doppelzählung).

    NEU: Am Ende Plausibilitäts-Clamp anhand Segmentdauer T und min/max Dauer pro Rep.
    """
    # vorverarbeitung: sortieren + defensiv kopieren
    rep_ts = list(sorted(float(t) for t in rep_ts))
    out = {
        "rep_ts_aug": rep_ts.copy(),
        "inserted_ts": [],
        "gaps": []
    }
    if len(rep_ts) < 2:
        return out  # zu wenig anker -> nichts zu tun

    # debug-ausgabe (kann später entfernt werden)
    print("Gap filler called")

    diffs = np.diff(np.asarray(rep_ts, float))
    g_med = _median(diffs)  # globaler median
    g_mad = _mad(diffs)     # globaler MAD (diagnose)

    # Ankerwahl je Klasse
    if anchor == "auto":
        anchor_eff: Literal["trough","peak"] = _default_anchor_for_class(segment_class)
    else:
        anchor_eff = anchor

    # Gaps durchgehen und ggf. inserts vorschlagen
    for i in range(diffs.size):
        dt = float(diffs[i])
        exp_local = _local_expected_interval(diffs, i, half_window=2)
        if exp_local <= 1e-9:
            exp_local = g_med if g_med > 0 else dt  # fallback falls lokal nichts trägt

        ratio = dt / max(1e-9, exp_local)
        gap_info: Dict[str, Any] = {
            "index": i,
            "t0": rep_ts[i],
            "t1": rep_ts[i+1],
            "dt": dt,
            "exp_local": exp_local,
            "ratio": ratio,
            "insert_candidates": []
        }

        if ratio > hi_factor:
            # lücke erkannt -> inserts schätzen
            n_missing = int(round(dt / exp_local)) - 1
            n_missing = max(1, min(max_inserts_per_gap, n_missing))  # deckeln gegen overshoot
            proposals = [rep_ts[i] + (k+1)*exp_local for k in range(n_missing)]  # linear zwischen t0..t1

            snapped: List[Tuple[float,float]] = []
            if signal is not None and fs > 0:
                # snap ans signal im engen fenster
                for t in proposals:
                    t_sn, sharp = _snap_time_to_signal(t, signal, fs, anchor_eff, search_window_s)
                    snapped.append((t_sn, sharp))
                # nahe duplikate rausfiltern
                keep = []
                min_dist = 0.25 * exp_local  # vorher 0.20 -> leicht erhöht wegen doppelzählung
                for t_sn, sharp in snapped:
                    if any(abs(t_sn - x) < min_dist for x in out["rep_ts_aug"]):
                        continue
                    keep.append((t_sn, sharp))
                snapped = keep
                proposed_times = [t for t, _ in snapped]
            else:
                proposed_times = proposals  # ohne signal -> ungesnappte vorschläge

            # vorschläge übernehmen
            for t in proposed_times:
                out["rep_ts_aug"].append(float(t))
                gap_info["insert_candidates"].append(float(t))

        # diagnosewerte anhängen
        gap_info["global_median"] = g_med
        gap_info["global_mad"] = g_mad
        out["gaps"].append(gap_info)

    # Final sortieren & inserted_ts bestimmen
    out["rep_ts_aug"].sort()
    orig_set = set(rep_ts)
    out["inserted_ts"] = [t for t in out["rep_ts_aug"] if t not in orig_set]

    # ---------- NEU: Cadence-Grid + Plausibilitäts-Clamp ----------
    # T bestimmen: bevorzugt aus Signal, sonst Spannweite der Zeitpunkte
    if signal is not None and fs > 0:
        T = float(len(signal)) / float(fs)
    else:
        T = float(max(out["rep_ts_aug"]) - min(out["rep_ts_aug"]))

    # min/max defaulten (vorsichtig wählen)
    min_rep_s_eff = float(min_rep_s if (min_rep_s and min_rep_s > 0) else 0.60)  # vorher 0.50 -> zu aggressiv
    max_rep_s_eff = float(max_rep_s if (max_rep_s and max_rep_s > 0) else 4.00)  # kann noch verfeinert werden...

    # globale Periode schätzen (ACF -> fallback median diff)
    p_est, acp = _estimate_period(signal, fs) if signal is not None else (0.0, 0.0)
    if not np.isfinite(p_est) or p_est <= 0.0 or acp < 0.12:
        dif = np.diff(np.array(out["rep_ts_aug"], float))
        p_est = float(np.median(dif)) if dif.size else 0.0  # fallback bei fehlender periodik

    # Grid-Fit anwenden
    out_grid = _cadence_grid_fit(
        out["rep_ts_aug"],
        p=float(p_est if p_est > 0 else max(min_rep_s_eff, 1.8)),  # soft default 1.8s
        T=T,
        signal=signal,
        fs=fs,
        anchor=anchor_eff,
        snap_w_s=0.10,
        tol_frac=0.35
    )

    # Clamp: floor(T/max_s) ≤ N ≤ ceil(T/min_s)
    lo = int(np.floor(T / max(1e-9, max_rep_s_eff)))
    hi = int(np.ceil (T / max(1e-9, min_rep_s_eff)))
    lo = max(1, lo)

    # auffüllen/mergen entlang des grids
    cur = list(out_grid)
    if len(cur) < lo:
        # zu wenig -> nochmal rasterung, füllt fehlende zellen sanft
        cur = _cadence_grid_fit(cur, p=float(p_est if p_est > 0 else max(min_rep_s_eff, 1.8)),
                                T=T, signal=signal, fs=fs, anchor=anchor_eff, snap_w_s=0.10, tol_frac=0.35)
    while len(cur) > hi and len(cur) >= 2:
        # zu viel -> engstes paar mergen (mittel) + snap
        cur = sorted(cur)
        d = np.diff(cur)
        j = int(np.argmin(d))
        mid = 0.5*(cur[j] + cur[j+1])
        if signal is not None and fs > 0:
            mid, _ = _snap_time_to_signal(mid, signal, fs, anchor_eff, 0.08)  # kleineres fenster
        cur = cur[:j] + [float(mid)] + cur[j+2:]

    out["rep_ts_aug"] = sorted(cur)
    out["inserted_ts"] = [t for t in out["rep_ts_aug"] if t not in orig_set]  # final inserted

    return out

# ---------- Komfort-Wrapper für typische Nutzung ----------
def repair_using_counter_anchors(
    *,
    signal: np.ndarray,
    fs: float,
    anchors_s: List[float],
    segment_class: Optional[str] = None,
    search_window_s: float = 0.12,
    hi_factor: float = 1.4,
    max_inserts_per_gap: int = 3,
    min_rep_s: Optional[float] = None,
    max_rep_s: Optional[float] = None
) -> Dict[str, Any]:
    """
    Wenn der Zähler bereits Ankerzeiten liefert (Trough/Peak je nach Klasse),
    kann direkt hiermit repariert und ans Signal gesnappt werden.
    """
    # aufruf von find_and_repair_rep_gaps (bequemer wrapper)
    return find_and_repair_rep_gaps(
        anchors_s,
        fs=fs,
        segment_class=segment_class,
        signal=signal,
        anchor="auto",
        search_window_s=search_window_s,
        hi_factor=hi_factor,              # vorher 1.65 -> hier etwas strenger, findet mehr lücken
        max_inserts_per_gap=max_inserts_per_gap,
        min_rep_s=min_rep_s,
        max_rep_s=max_rep_s
    )

# ---------- Mini Self-Test ----------
if __name__ == "__main__":
    # Synth-Beispiel: alle 2.0 s eine Rep, aber einmal „Lücke“ von 4.0 s
    # wurde hinzugefügt um quick-check zu haben; ersetzt keinen echten unit test
    true = np.arange(0.0, 20.0, 2.0)
    observed = true.tolist()
    observed.remove(10.0)
    res = find_and_repair_rep_gaps(observed, fs=50.0, min_rep_s=1.4, max_rep_s=3.5)
    print("Original:", np.round(observed, 2))
    print("Augment :", np.round(res["rep_ts_aug"], 2))
    print("Inserted:", np.round(res["inserted_ts"], 2))
