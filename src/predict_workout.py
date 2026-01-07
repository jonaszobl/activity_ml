# src/predict_workout.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils_jsonl import read_jsonl
from .model import load_model, predict_features
from .features import build_windows

from .segmentation.postprocessing import (
    smooth_probs_over_time,
    debounce_labels,
    merge_short_segments,
    strength_classes_from,
)

from .segmentation.decoder import StateMachineSegmenter, DecoderConfig
from .segmentation.exercise_gate import exercise_gate, ExerciseGateConfig
from .segmentation.adjacency_resolver import resolve_adjacent_strength, AdjacencyResolverConfig

from .segmentation.reps import (
    moving_average,      # only for peaks-mode smoothing
    count_peaks,         # legacy
    mad,                 # peaks prominence
    select_rep_signal,
    count_reps_adaptive, # adaptive
)


# ---------------- Helpers ----------------

def ensure_time_column_df(df: pd.DataFrame) -> pd.DataFrame:
    if "t" in df.columns:
        t = pd.to_numeric(df["t"], errors="coerce")
    elif "t_rel" in df.columns:
        t_rel = pd.to_numeric(df["t_rel"], errors="coerce")
        t = t_rel - t_rel.iloc[0]
    else:
        raise KeyError("Weder 't' noch 't_rel' in Datei gefunden.")

    df = df.copy()
    df["t"] = t
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    need = {"ax", "ay", "az"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Spalten fehlen: {missing}")

    # hr/steps ignorieren, falls vorhanden
    for col in ("hr", "steps"):
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def merge_consecutive_rest(segments):
    if not segments:
        return []
    out = [dict(segments[0])]
    for seg in segments[1:]:
        cur = dict(seg)
        prev = out[-1]
        if prev["class"] == "REST" and cur["class"] == "REST":
            prev["t1"] = max(float(prev["t1"]), float(cur["t1"]))
            prev["duration_s"] = float(prev["t1"] - prev["t0"])
            # keep i0 from first, i1 from last
            prev["i1"] = max(int(prev.get("i1", prev["i0"] + 1)), int(cur.get("i1", cur["i0"] + 1)))
        else:
            out.append(cur)
    return out


def seconds_to_hms(sec: float) -> str:
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Workout-Prediction (prod): window->decoder->merge->gate->adj, output kompatibel.")
    ap.add_argument("infile", help="Pfad zur JSONL-Datei (t oder t_rel).")
    ap.add_argument("--model", default="artifacts/model.json", help="Pfad zum exportierten Modell.")

    # window-level stabilization (wie debug_pipeline, aber prod-tauglich)
    ap.add_argument("--prob_smooth_k", type=int, default=5)
    ap.add_argument("--debounce_run", type=int, default=3)
    ap.add_argument("--merge_min_s", type=float, default=4.0)

    ap.add_argument("--no_smooth", action="store_true")
    ap.add_argument("--no_debounce", action="store_true")

    # exercise gate (wie debug_pipeline defaults)
    ap.add_argument("--gate_min_set_s", type=float, default=18.0)
    ap.add_argument("--gate_min_mean", type=float, default=0.55)
    ap.add_argument("--gate_min_peak", type=float, default=0.70)
    ap.add_argument("--no_gate", action="store_true")

    # adjacency resolver (wie debug_pipeline defaults)
    ap.add_argument("--adj_rest_bridge_s", type=float, default=6.0)
    ap.add_argument("--adj_score_margin", type=float, default=0.10)
    ap.add_argument("--adj_min_combined_mean", type=float, default=0.55)
    ap.add_argument("--no_adj", action="store_true")

    # rep counting (bleibt wie vorher im Output)
    ap.add_argument("--rep_mode", choices=["peaks", "pair"], default="pair")
    ap.add_argument("--smooth_sec", type=float, default=0.2)   # peaks-mode smoothing
    ap.add_argument("--min_peak_sep", type=float, default=0.4) # peaks-mode

    ap.add_argument("--rep_min_s", type=float, default=0.4)
    ap.add_argument("--rep_max_s", type=float, default=3.5)
    ap.add_argument("--rep_k", type=float, default=0.7)
    ap.add_argument("--acf_min_s", type=float, default=0.45)
    ap.add_argument("--acf_max_s", type=float, default=3.0)

    args = ap.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        raise FileNotFoundError(infile)

    # 1) Model
    M = load_model(args.model)
    classes = M["classes"]
    fs = float(M["meta"]["fs_hz"])
    winS = float(M["meta"]["win_s"])
    hopS = float(M["meta"]["hop_s"])
    strength_classes = strength_classes_from(M)

    # 2) Data
    rows = list(read_jsonl(str(infile)))
    if not rows:
        raise RuntimeError("Datei leer?")
    df = ensure_time_column_df(pd.DataFrame(rows))

    # 3) Windows + predict
    X, _, t0s, feat_names = build_windows(df, fs, winS, hopS)
    if X.shape[0] == 0:
        raise RuntimeError("Keine Fenster erzeugt – ist die Datei lang genug?")

    model_feats = M.get("feature_names", [])
    if model_feats and len(model_feats) != X.shape[1]:
        raise RuntimeError(
            f"Feature-Anzahl passt nicht (Datei: {X.shape[1]} vs. Modell: {len(model_feats)}). "
            f"Hinweis: Nach Änderungen an features.py neu trainieren."
        )
    if model_feats and feat_names != model_feats:
        # prod: nur warnen, kein debug-spam
        print("[WARN] Feature-Namen/Reihenfolge weichen vom Modell ab. Trainiere Modell ggf. neu.")

    cls_idx, probs = predict_features(X, M)

    # 4) smooth + debounce (window level)
    probs_s = probs
    if not args.no_smooth:
        probs_s = smooth_probs_over_time(probs_s, k=max(1, int(args.prob_smooth_k)))

    cls_idx2 = np.argmax(probs_s, axis=1)
    if not args.no_debounce:
        cls_idx2 = debounce_labels(cls_idx2, min_run=max(1, int(args.debounce_run)))

    # 5) Decoder -> segments (wie debug_pipeline, aber debug=False)
    dec_cfg = DecoderConfig(
        start_hold_w=2,
        end_hold_w=3,
        q_strength_start=0.55,
        q_rest_end=0.70,
        q_motion_low=0.40,
        min_set_s=12.0,
        min_rest_s=20.0,
        debug=False,
    )
    decoder = StateMachineSegmenter(classes=classes, strength_classes=strength_classes, cfg=dec_cfg)
    segments = decoder.decode(df=df, probs_s=probs_s, t0s=np.asarray(t0s, float), win_s=winS, hop_s=hopS, fs=fs)

    # 6) merge short (wie debug_pipeline)
    if args.merge_min_s and args.merge_min_s > 0:
        segments = merge_short_segments(segments, min_len_s=float(args.merge_min_s), prefer="neighbor")

    # 7) exercise gate (wie debug_pipeline)
    if not args.no_gate:
        gcfg = ExerciseGateConfig(
            min_set_s=float(args.gate_min_set_s),
            min_mean_conf=float(args.gate_min_mean),
            min_peak_conf=float(args.gate_min_peak),
            debug=False,
        )
        segments = exercise_gate(
            segments=segments,
            probs_s=probs_s,
            t0s=np.asarray(t0s, float),
            classes=classes,
            strength_classes=strength_classes,
            cfg=gcfg,
        )
        # cleanup: merge again (debug_pipeline macht min_len_s=0.0)
        segments = merge_short_segments(segments, min_len_s=0.0, prefer="neighbor")

    # 8) adjacency resolver (wie debug_pipeline)
    if not args.no_adj:
        acfg = AdjacencyResolverConfig(
            max_gap_s=0.1,
            rest_bridge_s=float(args.adj_rest_bridge_s),
            score_margin=float(args.adj_score_margin),
            min_combined_mean_conf=float(args.adj_min_combined_mean),
            debug=False,
        )
        segments = resolve_adjacent_strength(
            segments=segments,
            probs_s=probs_s,
            t0s=np.asarray(t0s, float),
            classes=classes,
            strength_classes=strength_classes,
            cfg=acfg,
        )
        segments = merge_short_segments(segments, min_len_s=0.0, prefer="neighbor")
        segments = merge_consecutive_rest(segments)

    # 9) Rep counting + output shaping (kompatibel zu altem predict_workout Output)
    t = df["t"].to_numpy(float)
    ax = df["ax"].to_numpy(float)
    ay = df["ay"].to_numpy(float)
    az = df["az"].to_numpy(float)

    smooth_k = max(1, int(round(float(args.smooth_sec) * fs)))
    az_smooth = moving_average(az, smooth_k)  # peaks-mode

    results = []
    for seg in segments:
        seg_class = str(seg["class"])
        mask = (t >= float(seg["t0"])) & (t <= float(seg["t1"]))

        reps = 0
        if seg_class in strength_classes and np.count_nonzero(mask) > 3:
            if args.rep_mode == "pair":
                sig = select_rep_signal(ax[mask], ay[mask], az[mask], fs)
                reps, _dbg = count_reps_adaptive(
                    sig,
                    fs,
                    base_k=float(args.rep_k),
                    min_s=float(args.rep_min_s),
                    max_s=float(args.rep_max_s),
                    acf_min_s=float(args.acf_min_s),
                    acf_max_s=float(args.acf_max_s),
                )
            else:
                az_seg = az_smooth[mask]
                reps = count_peaks(
                    az_seg,
                    fs,
                    min_separation_s=float(args.min_peak_sep),
                    thresh_mode="median_mad",
                    prominence=0.5 * mad(az_seg),
                )

        out = {
            "t0": float(seg["t0"]),
            "t1": float(seg["t1"]),
            "duration_s": float(seg["t1"] - seg["t0"]),
            "class": seg_class,
            "reps": int(reps) if seg_class in strength_classes else 0,
            "reps_refined": int(reps) if seg_class in strength_classes else 0,
        }
        results.append(out)

    # 10) Console output (wie vorher)
    print("\nVorhersage-Segmente:")
    for s in results:
        line = (
            f"- {s['class']:20s}  {seconds_to_hms(s['t0'])} → {seconds_to_hms(s['t1'])}  ({s['duration_s']:5.1f}s)"
        )
        if s["class"] not in {"REST", "PAUSE", "WALKING", "RUNNING"}:
            line += f"  | reps: {int(s['reps_refined'])} ({int(s['reps'])})"
        print(line)

    # 11) JSON export (wie vorher)
    out_json = infile.with_suffix(".pred.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_version": M.get("version", "v1"),
                "fs_hz": fs,
                "win_s": winS,
                "hop_s": hopS,
                "segments": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nErgebnis gespeichert: {out_json}")


if __name__ == "__main__":
    main()
