import io
import os
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd

# --- Imports gemäß deiner Struktur ---
from src.model.classifier import load_model, predict_features  # <- korrektes Modul
from src.predict_workout import (
    smooth_probs_over_time,
    debounce_labels,
    segment_from_window_preds,
    merge_short_segments,
    strength_classes_from,
    select_rep_signal,
    estimate_rep_period_acf,
    count_reps_peak_trough,
    moving_average,
    mad,
    rep_params_for_class,
)

# Post-Filter liegen bei dir in src/segmentation/postprocessing.py
from src.segmentation.postprocessing import apply_post_filters, POST_DEFAULTS

# build_windows liegt bei dir in src/features/legacy_features.py
from src.features.legacy_features import build_windows


# Falls du utils_jsonl.read_jsonl_from_io nicht verwenden willst:
def read_jsonl_from_io(fobj: io.StringIO):
    for line in fobj:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        yield json.loads(s)


# --- robuster Modellpfad (Repo-Root/artifacts/model.json) ---
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "artifacts" / "model.json"
MODEL_PATH = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL))

app = FastAPI(title="Bangle Workout API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in Prod enger setzen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_OBJ = None
MODEL_READY = False


@app.get("/", include_in_schema=False)
@app.get("/health", include_in_schema=False)
@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok", "ready": bool(MODEL_READY)}


@app.get("/readyz", include_in_schema=False)
def readyz():
    return {"ready": bool(MODEL_READY)}


class PredictResponse(BaseModel):
    model_version: str
    fs_hz: float
    win_s: float
    hop_s: float
    segments: list
    model_config = {"protected_namespaces": ()}


def _run_predict(
    df: pd.DataFrame,
    M: dict,
    prob_smooth_k=5,
    debounce_run=3,
    merge_min_s=4.0,
    rep_mode="pair",
    rep_min_s=0.6,
    rep_max_s=4.0,
    rep_k=0.6,
    acf_enable=False,
    acf_min_s=0.45,
    acf_max_s=3.0,
    acf_band=(0.6, 1.8),
    min_reps=1,
    below_min_policy="keep",
    # Post-Filter Overrides
    post_min_strength_sec=POST_DEFAULTS["min_strength_duration_s"],
    post_min_rest_between_sec=POST_DEFAULTS["min_rest_between_sets_s"],
    post_acf_peak_thr=POST_DEFAULTS["acf_peak_thr"],
    post_band_ratio_thr=POST_DEFAULTS["band_ratio_thr"],
    post_std_thr_g=POST_DEFAULTS["std_thr_g"],
    post_min_rep_density=POST_DEFAULTS["min_rep_density"],
    post_conf_thr=POST_DEFAULTS["conf_thr"],
):
    classes = M["classes"]
    fs = float(M["meta"]["fs_hz"])
    winS = float(M["meta"]["win_s"])
    hopS = float(M["meta"]["hop_s"])

    # --- robustes Zeit/Spalten-Handling (Meta-Zeilen raus) ---
    if "type" in df.columns:
        df = df[~(df["type"].astype(str) == "meta")].copy()

    if "t" not in df.columns:
        if "t_rel" in df.columns:
            t_rel = pd.to_numeric(df["t_rel"], errors="coerce")
            valid = t_rel[t_rel.notna()]
            if not valid.empty:
                t0 = valid.iloc[0]
                df["t"] = t_rel - t0
            else:
                df["t"] = np.nan
        else:
            raise HTTPException(status_code=400, detail="No time column: expected 't' or 't_rel'")

    for col in ["t", "ax", "ay", "az"]:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing column '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["t", "ax", "ay", "az"]).sort_values("t").reset_index(drop=True)

    # --- Feature-Fenster ---
    X, _, t0s, _feat_names = build_windows(df, fs, winS, hopS)
    if X.shape[0] == 0:
        return [], fs, winS, hopS

    model_feats = M.get("feature_names", [])
    assert len(model_feats) == X.shape[1], "Feature count mismatch – Modell neu trainieren?"

    # --- Fensterweise Klassifikation ---
    cls_idx, probs = predict_features(X, M)
    probs_s = smooth_probs_over_time(probs, k=max(1, int(prob_smooth_k)))
    cls_idx = np.argmax(probs_s, axis=1)
    cls_idx = debounce_labels(cls_idx, min_run=max(1, int(debounce_run)))

    # --- Segmentierung ---
    segments = segment_from_window_preds(t0s, cls_idx, classes, winS)
    if merge_min_s and merge_min_s > 0:
        segments = merge_short_segments(segments, min_len_s=float(merge_min_s), prefer="neighbor")

    # --- Reps je Segment ---
    strength_classes = strength_classes_from(M)
    t = df["t"].to_numpy(float)
    ax = df["ax"].to_numpy(float)
    ay = df["ay"].to_numpy(float)
    az = df["az"].to_numpy(float)

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
                k0, min0, max0 = rep_params_for_class(
                    seg_class,
                    base_k=float(rep_k),
                    base_min=float(rep_min_s),
                    base_max=float(rep_max_s),
                )

                if acf_enable:
                    p = estimate_rep_period_acf(sig, fs, min_s=float(acf_min_s), max_s=float(acf_max_s))
                    if p > 0:
                        lo, hi = acf_band
                        min_s = max(0.35, min(p * lo, max0))
                        max_s = max(min0, min(p * hi, max0))
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
                # optionaler legacy mode: Peaks (wenn du ihn wirklich noch brauchst)
                from src.predict_workout import count_peaks
                reps = count_peaks(
                    az_smooth[mask],
                    fs,
                    min_separation_s=0.4,
                    thresh_mode="median_mad",
                    prominence=0.5 * mad(az_smooth[mask]),
                )

        out_class = seg_class
        if seg_class in strength_classes:
            if below_min_policy == "drop" and reps < int(min_reps):
                continue
            if below_min_policy == "rest" and reps < int(min_reps):
                out_class = "REST"
                reps = 0

        results.append(
            {
                "t0": float(seg["t0"]),
                "t1": float(seg["t1"]),
                "duration_s": float(seg["t1"] - seg["t0"]),
                "class": out_class,
                "reps": int(reps) if out_class not in {"REST", "PAUSE", "WALKING", "RUNNING"} else 0,
                "i0": int(seg["i0"]),
                "i1": int(seg["i1"]),
            }
        )

    # --- Post-Filter anwenden ---
    pf_cfg = dict(POST_DEFAULTS)
    pf_cfg.update(
        dict(
            min_strength_duration_s=float(post_min_strength_sec),
            min_rest_between_sets_s=float(post_min_rest_between_sec),
            acf_peak_thr=float(post_acf_peak_thr),
            band_ratio_thr=float(post_band_ratio_thr),
            std_thr_g=float(post_std_thr_g),
            min_rep_density=float(post_min_rep_density),
            conf_thr=float(post_conf_thr),
        )
    )
    results = apply_post_filters(df, results, probs_s, classes, fs, strength_classes, cfg=pf_cfg)

    return results, fs, winS, hopS


@app.on_event("startup")
def _load_model_once():
    global MODEL_OBJ, MODEL_READY
    try:
        MODEL_OBJ = load_model(MODEL_PATH)
        MODEL_READY = True
        print(f"[startup] model loaded: {MODEL_PATH}")
    except Exception as e:
        MODEL_READY = False
        print(f"[startup] model load failed: {MODEL_PATH} -> {e}")


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    workout_file: UploadFile = File(...),
    prob_smooth_k: int = Form(5),
    debounce_run: int = Form(3),
    merge_min_s: float = Form(4.0),
    rep_mode: str = Form("pair"),
    rep_min_s: float = Form(0.6),
    rep_max_s: float = Form(4.0),
    rep_k: float = Form(0.6),
    acf_enable: bool = Form(False),
    acf_min_s: float = Form(0.45),
    acf_max_s: float = Form(3.0),
    acf_band: str = Form("0.6,1.8"),
    min_reps: int = Form(1),
    below_min_policy: str = Form("keep"),
    post_min_strength_sec: float = Form(8.0),
    post_min_rest_between_sec: float = Form(10.0),
    post_acf_peak_thr: float = Form(0.18),
    post_band_ratio_thr: float = Form(0.35),
    post_std_thr_g: float = Form(0.05),
    post_min_rep_density: float = Form(0.25),
    post_conf_thr: float = Form(0.50),
):
    if not MODEL_READY or MODEL_OBJ is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        lo, hi = [float(x) for x in acf_band.split(",")]
        acf_band_tuple = (lo, hi)
    except Exception:
        acf_band_tuple = (0.6, 1.8)

    data = await workout_file.read()
    text = data.decode("utf-8", errors="ignore")
    print(f"[predict] received {len(data)} bytes from {workout_file.filename}")

    # JSONL tolerant einlesen
    rows = []
    try:
        rows = list(read_jsonl_from_io(io.StringIO(text)))
    except Exception:
        rows = []

    if not rows:
        for ln in text.splitlines():
            s = ln.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                pass

    if not rows:
        try:
            obj = json.loads(text)
            rows = obj if isinstance(obj, list) else [obj]
        except Exception:
            rows = []

    if not rows:
        return PredictResponse(
            model_version=(MODEL_OBJ or {}).get("version", "v1"),
            fs_hz=float((MODEL_OBJ or {"meta": {"fs_hz": 50}})["meta"]["fs_hz"]),
            win_s=float((MODEL_OBJ or {"meta": {"win_s": 2}})["meta"]["win_s"]),
            hop_s=float((MODEL_OBJ or {"meta": {"hop_s": 0.5}})["meta"]["hop_s"]),
            segments=[],
        )

    df = pd.DataFrame(rows)
    print("[predict] df columns:", df.columns.tolist(), "len:", len(df))

    segments, fs, winS, hopS = _run_predict(
        df,
        MODEL_OBJ,
        prob_smooth_k=prob_smooth_k,
        debounce_run=debounce_run,
        merge_min_s=merge_min_s,
        rep_mode=rep_mode,
        rep_min_s=rep_min_s,
        rep_max_s=rep_max_s,
        rep_k=rep_k,
        acf_enable=acf_enable,
        acf_min_s=acf_min_s,
        acf_max_s=acf_max_s,
        acf_band=acf_band_tuple,
        min_reps=min_reps,
        below_min_policy=below_min_policy,
        post_min_strength_sec=post_min_strength_sec,
        post_min_rest_between_sec=post_min_rest_between_sec,
        post_acf_peak_thr=post_acf_peak_thr,
        post_band_ratio_thr=post_band_ratio_thr,
        post_std_thr_g=post_std_thr_g,
        post_min_rep_density=post_min_rep_density,
        post_conf_thr=post_conf_thr,
    )

    return PredictResponse(
        model_version=MODEL_OBJ.get("version", "v1"),
        fs_hz=float(fs),
        win_s=float(winS),
        hop_s=float(hopS),
        segments=segments,
    )
