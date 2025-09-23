import io
import json
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import pandas as pd

from src.predict_workout import (  # type: ignore
    load_model, ensure_time_column_df, predict_features,
    smooth_probs_over_time, debounce_labels, segment_from_window_preds,
    merge_short_segments, strength_classes_from, select_rep_signal,
    estimate_rep_period_acf, count_reps_peak_trough, moving_average, mad
)
from src.features import build_windows  # type: ignore
from src.utils_jsonl import read_jsonl_from_io

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.json")

app = FastAPI(title="Bangle Workout API", version="1.0")

# ✅ CORS: Wildcard NUR ohne Credentials!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,     # <- wichtig
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "service": "bangle-workout-api"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

class PredictResponse(BaseModel):
    model_version: str
    fs_hz: float
    win_s: float
    hop_s: float
    segments: list
    model_config = {"protected_namespaces": ()}


def _run_predict(df: pd.DataFrame, M: dict,
                 prob_smooth_k=5, debounce_run=3, merge_min_s=4.0,
                 rep_mode="pair", rep_min_s=0.6, rep_max_s=4.0, rep_k=0.6,
                 acf_enable=True, acf_min_s=0.45, acf_max_s=3.0, acf_band=(0.6,1.8),
                 min_reps=4, below_min_policy="rest"):

    classes = M["classes"]
    fs   = float(M["meta"]["fs_hz"])
    winS = float(M["meta"]["win_s"])
    hopS = float(M["meta"]["hop_s"])

    df = ensure_time_column_df(df).sort_values("t").reset_index(drop=True)
    X, _, t0s, feat_names = build_windows(df, fs, winS, hopS)
    if X.shape[0] == 0:
        return [], fs, winS, hopS

    model_feats = M.get("feature_names", [])
    assert len(model_feats) == X.shape[1], "Feature count mismatch – Modell neu trainieren?"
    if model_feats and feat_names != model_feats:
        # Warnung, aber wir rechnen weiter
        pass

    # Fensterweise Klassifikation
    cls_idx, probs = predict_features(X, M)
    # Probs glätten
    probs_s = smooth_probs_over_time(probs, k=max(1, int(prob_smooth_k)))
    cls_idx = np.argmax(probs_s, axis=1)
    # Entprellen
    cls_idx = debounce_labels(cls_idx, min_run=max(1, int(debounce_run)))

    # Segmentierung
    segments = segment_from_window_preds(t0s, cls_idx, classes, winS)
    if merge_min_s and merge_min_s > 0:
        segments = merge_short_segments(segments, min_len_s=float(merge_min_s), prefer="neighbor")

    # Reps je Segment (nur Kraftklassen)
    strength_classes = strength_classes_from(M)
    t  = df["t"].to_numpy(float)
    ax = df["ax"].to_numpy(float)
    ay = df["ay"].to_numpy(float)
    az = df["az"].to_numpy(float)

    # für peaks-Mode: leichte Glättung der Z-Achse (Kompatibilität)
    smooth_k = max(1, int(round(0.2 * fs)))
    az_smooth = moving_average(az, smooth_k)

    results = []
    for seg in segments:
        seg_class = seg["class"]
        mask = (t >= seg["t0"]) & (t <= seg["t1"])
        reps = 0

        if seg_class in strength_classes and np.count_nonzero(mask) > 3:
            if rep_mode == "pair":
                sig = select_rep_signal(ax[mask], ay[mask], az[mask], fs)

                # Klassenspezifische Parameter aus deinem Python:
                from src.predict_workout import rep_params_for_class
                k0, min0, max0 = rep_params_for_class(seg_class, base_k=float(rep_k),
                                                      base_min=float(rep_min_s), base_max=float(rep_max_s))

                if acf_enable:
                    p = estimate_rep_period_acf(sig, fs, min_s=float(acf_min_s), max_s=float(acf_max_s))
                    if p > 0:
                        lo, hi = acf_band
                        min_s = max(0.35, min(p*lo, max0))
                        max_s = max(min0, min(p*hi, max0))
                    else:
                        min_s, max_s = min0, max0
                else:
                    min_s, max_s = min0, max0

                if min_s >= max_s:
                    min_s, max_s = min0, max0

                reps = count_reps_peak_trough(sig, fs, k=k0, min_rep_s=min_s, max_rep_s=max_s)
                if reps == 0:
                    k_try = max(0.25, k0 - 0.1)
                    reps = count_reps_peak_trough(sig, fs, k=k_try, min_rep_s=min_s, max_rep_s=max_s)
            else:
                # Peaks-Mode
                from src.predict_workout import count_peaks
                reps = count_peaks(
                    az_smooth[mask], fs,
                    min_separation_s=0.4,
                    thresh_mode="median_mad",
                    prominence=0.5 * mad(az_smooth[mask])
                )

        out_class = seg_class
        if seg_class in strength_classes:
            if below_min_policy == "drop" and reps < int(min_reps):
                continue
            if below_min_policy == "rest" and reps < int(min_reps):
                out_class = "REST"
                reps = 0

        results.append({
            "t0": float(seg["t0"]),
            "t1": float(seg["t1"]),
            "duration_s": float(seg["t1"] - seg["t0"]),
            "class": out_class,
            "reps": int(reps) if out_class not in {"REST", "PAUSE", "WALKING", "RUNNING"} else 0
        })

    return results, fs, winS, hopS

@app.on_event("startup")
def _load_model_once():
    # Modell lazy global load
    global MODEL_OBJ
    MODEL_OBJ = load_model(MODEL_PATH)

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    workout_file: UploadFile = File(...),   # erwartet workout.txt (JSONL)
    # Optional: Parametrisierung (Defaults wie in deinem Script)
    prob_smooth_k: int = Form(5),
    debounce_run: int = Form(3),
    merge_min_s: float = Form(4.0),
    rep_mode: str = Form("pair"),
    rep_min_s: float = Form(0.6),
    rep_max_s: float = Form(4.0),
    rep_k: float = Form(0.6),
    acf_enable: bool = Form(True),
    acf_min_s: float = Form(0.45),
    acf_max_s: float = Form(3.0),
    acf_band: str = Form("0.6,1.8"),
    min_reps: int = Form(4),
    below_min_policy: str = Form("rest")
):
    # parse acf_band
    try:
        lo, hi = [float(x) for x in acf_band.split(",")]
        acf_band_tuple = (lo, hi)
    except Exception:
        acf_band_tuple = (0.6, 1.8)

    # JSONL lesen
    data = await workout_file.read()
    io_buf = io.StringIO(data.decode("utf-8", errors="ignore"))
    rows = list(read_jsonl_from_io(io_buf))
    if not rows:
        return PredictResponse(
            model_version=MODEL_OBJ.get("version", "v1"),
            fs_hz=float(MODEL_OBJ["meta"]["fs_hz"]),
            win_s=float(MODEL_OBJ["meta"]["win_s"]),
            hop_s=float(MODEL_OBJ["meta"]["hop_s"]),
            segments=[]
        )

    df = pd.DataFrame(rows)
    segments, fs, winS, hopS = _run_predict(
        df, MODEL_OBJ,
        prob_smooth_k=prob_smooth_k, debounce_run=debounce_run, merge_min_s=merge_min_s,
        rep_mode=rep_mode, rep_min_s=rep_min_s, rep_max_s=rep_max_s, rep_k=rep_k,
        acf_enable=acf_enable, acf_min_s=acf_min_s, acf_max_s=acf_max_s, acf_band=acf_band_tuple,
        min_reps=min_reps, below_min_policy=below_min_policy
    )

    return PredictResponse(
        model_version=MODEL_OBJ.get("version", "v1"),
        fs_hz=float(fs), win_s=float(winS), hop_s=float(hopS),
        segments=segments
    )
