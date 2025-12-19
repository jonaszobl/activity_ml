import json
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .utils_jsonl import read_jsonl
from .features import build_windows
from .model import load_model, predict_features

from .segmentation.reps import (
    moving_average,
    count_peaks,
    count_reps_peak_trough,
    estimate_rep_period_acf,
    select_rep_signal,
    rep_params_for_class,
    mad,
)

from .segmentation.postprocessing import (
    smooth_probs_over_time,
    debounce_labels,
    segment_from_window_preds,
    merge_short_segments,
    strength_classes_from,
    apply_post_filters,
    POST_DEFAULTS,
)


# ---------- DataFrame Helpers ----------
def ensure_time_column_df(df: pd.DataFrame) -> pd.DataFrame:
    # Zeitspalte „t“ herstellen; robust gg. t_rel
    if "t" in df.columns:
        t = pd.to_numeric(df["t"], errors="coerce")
    elif "t_rel" in df.columns:
        t_rel = pd.to_numeric(df["t_rel"], errors="coerce")
        t = t_rel - t_rel.iloc[0]
    else:
        raise KeyError("Weder 't' noch 't_rel' in Datei gefunden.")
    df = df.copy()
    df["t"] = t
    need = {"ax", "ay", "az"}
    missing = [c for c in need if c not in df.columns]
    assert not missing, f"Spalten fehlen: {missing}"
    df = df.dropna(subset=["t"])

    # hr/steps in den Rohdaten ignorieren, falls vorhanden
    for col in ("hr", "steps"):
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


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
    probs_s = smooth_probs_over_time(probs, k=max(1, int(args.prob_smooth_k)))  # mildert Zappeln

    # 4b) Entprellen
    cls_idx = np.argmax(probs_s, axis=1)
    cls_idx = debounce_labels(cls_idx, min_run=max(1, int(args.debounce_run)))  # vorher nur Argmax -> flackerte

    # 5) Segmentierung
    segments = segment_from_window_preds(t0s, cls_idx, classes, winS)

    # 5a) Mini-Segmente mergen
    if args.merge_min_s and args.merge_min_s > 0:
        segments = merge_short_segments(segments, min_len_s=float(args.merge_min_s), prefer="neighbor")

    # 6) Rep-Counting (BASIS – steuert Klassifizierung/Postfilter)
    results = []
    t  = df["t"].to_numpy(float)
    ax = df["ax"].to_numpy(float)
    ay = df["ay"].to_numpy(float)
    az = df["az"].to_numpy(float)

    smooth_k = max(1, int(round(args.smooth_sec * fs)))
    az_smooth = moving_average(az, smooth_k)  # nur für peaks-mode relevant

    for seg in segments:
        seg_class = seg["class"]
        mask = (t >= seg["t0"]) & (t <= seg["t1"])

        base_reps = 0
        if seg_class in strength_classes and np.count_nonzero(mask) > 3:
            if args.rep_mode == "pair":
                sig = select_rep_signal(ax[mask], ay[mask], az[mask], fs)
                k0, min0, max0 = rep_params_for_class(seg_class,
                                                      base_k=float(args.rep_k),
                                                      base_min=float(args.rep_min_s),
                                                      base_max=float(args.rep_max_s))

                if args.acf_enable:
                    p = estimate_rep_period_acf(sig, fs,
                                                min_s=float(args.acf_min_s),
                                                max_s=float(args.acf_max_s))
                    if p > 0:
                        lo_fac, hi_fac = float(args.acf_band[0]), float(args.acf_band[1])
                        min_s = max(0.35, min(p * lo_fac, max0))  # untergrenze 0.35s -> schützt gg. Doppelzählung
                        max_s = max(min0, min(p * hi_fac, max0))
                    else:
                        min_s, max_s = min0, max0
                else:
                    min_s, max_s = min0, max0

                if min_s >= max_s:
                    min_s = min0
                    max_s = max0  # safety gegen verdrehte parameter

                base_reps = count_reps_peak_trough(sig, fs, k=k0, min_rep_s=min_s, max_rep_s=max_s)

                if base_reps == 0:
                    k_try = max(0.25, k0 - 0.1)  # vorher k0 -> abgesenkt um zu strenge Schwelle zu relaxen
                    base_reps = count_reps_peak_trough(sig, fs, k=k_try, min_rep_s=min_s, max_rep_s=max_s)
            else:
                az_seg = az_smooth[mask]
                base_reps = count_peaks(
                    az_seg, fs,
                    min_separation_s=float(args.min_peak_sep),
                    thresh_mode="median_mad",
                    prominence=0.5 * mad(az_seg)  # zu kleine Peaks raus
                )

        out_class = seg_class
        if seg_class in strength_classes:
            if args.below_min_policy == "drop" and base_reps < int(args.min_reps):
                continue
            if args.below_min_policy == "rest" and base_reps < int(args.min_reps):
                out_class = "REST"
                base_reps = 0

        seg_out = {
            "t0": float(seg["t0"]),
            "t1": float(seg["t1"]),
            "duration_s": float(seg["t1"] - seg["t0"]),

            "class": out_class,
            "reps": int(base_reps) if out_class not in {"REST", "PAUSE", "WALKING", "RUNNING"} else 0,  # ALT/basis
            "i0": int(seg["i0"]),
            "i1": int(seg["i1"]),
        }
        results.append(seg_out)

    # 7) Post-Filter anwenden (mit evtl. CLI-Overrides auf Basis-Defaults)
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

    # 7b) Reps "verfeinern": Handler ist optional -> hier: reps_refined = Basiszählung
    for s in results:
        if s["class"] == "REST":
            s["reps_refined"] = 0
        else:
            s["reps_refined"] = int(s.get("reps", 0))

    # 8) Ausgabe
    print("\nVorhersage-Segmente:")
    for s in results:
        dur = s["duration_s"]

        def hhmmss(sec):
            # mini-display helper; lässt Stunden weg wenn unnötig
            sec = int(round(sec))
            h = sec // 3600
            m = (sec % 3600) // 60
            sc = sec % 60
            return f"{h:d}:{m:02d}:{sc:02d}" if h > 0 else f"{m:d}:{sc:02d}"

        line = f"- {s['class']:20s}  {hhmmss(s['t0'])} → {hhmmss(s['t1'])}  ({dur:5.1f}s)"
        if s["class"] not in {"REST", "PAUSE", "WALKING", "RUNNING"}:
            # reps: NEU (ALT) — neu kommt aus Handler, alt ist Basiszählung
            line += f"  | reps: {int(s['reps_refined'])} ({int(s['reps'])})"
        print(line)

    # 9) JSON-Export
    out_json = infile.with_suffix(".pred.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "model_version": M.get("version", "v1"),
            "fs_hz": fs,
            "win_s": winS,
            "hop_s": hopS,
            "segments": [
                {
                    "t0": s["t0"],
                    "t1": s["t1"],
                    "duration_s": s["duration_s"],
                    "class": s["class"],
                    "reps": int(s["reps"]),                 # alt/basis
                    "reps_refined": int(s["reps_refined"])  # neu/handler (hier = basis)
                }
                for s in results
            ]
        }, f, ensure_ascii=False, indent=2)
    print(f"\nErgebnis gespeichert: {out_json}")


if __name__ == "__main__":
    main()
