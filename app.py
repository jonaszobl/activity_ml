# app.py  (aligned 1:1 with src/predict_workout.py console pipeline)

import io
import os
import json
import hashlib
from pathlib import Path
from functools import lru_cache

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import pandas as pd

# IMPORTANT: match console pipeline imports
from src.model.classifier import load_model, predict_features
from src.features import build_windows  # <-- SAME as console (NOT legacy_features)

from src.segmentation.postprocessing import (
    smooth_probs_over_time,
    debounce_labels,
    merge_short_segments,
    strength_classes_from,
)

from src.segmentation.decoder import StateMachineSegmenter, DecoderConfig
from src.segmentation.exercise_gate import exercise_gate, ExerciseGateConfig
from src.segmentation.adjacency_resolver import resolve_adjacent_strength, AdjacencyResolverConfig

from src.segmentation.reps import (
    moving_average,      # only for peaks-mode smoothing
    count_peaks,         # legacy
    mad,                 # peaks prominence
    select_rep_signal,
    count_reps_adaptive, # adaptive
)


# ---------------- Parsing helpers ----------------

def read_jsonl_from_io(fobj: io.StringIO):
    for line in fobj:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        yield json.loads(s)


def _parse_workout_bytes(data: bytes) -> list[dict]:
    text = data.decode("utf-8", errors="ignore")

    # tolerant JSONL parsing
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

    # fallback: JSON array / object
    if not rows:
        try:
            obj = json.loads(text)
            rows = obj if isinstance(obj, list) else [obj]
        except Exception:
            rows = []

    return rows


# ---------------- Console-identical preprocessing ----------------

def ensure_time_column_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    IDENTICAL to src/predict_workout.py (console).
    (Do not make baro mandatory here; console doesn't.)
    """
    if "t" in df.columns:
        t = pd.to_numeric(df["t"], errors="coerce")
    elif "t_rel" in df.columns:
        t_rel = pd.to_numeric(df["t_rel"], errors="coerce")
        t = t_rel - t_rel.iloc[0]
    else:
        raise HTTPException(status_code=400, detail="Neither 't' nor 't_rel' found in file")

    df = df.copy()
    df["t"] = t
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    need = {"ax", "ay", "az"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    # hr/steps ignorieren, falls vorhanden (console behavior)
    for col in ("hr", "steps"):
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def merge_consecutive_rest(segments):
    """
    IDENTICAL to src/predict_workout.py (console).
    """
    if not segments:
        return []
    out = [dict(segments[0])]
    for seg in segments[1:]:
        cur = dict(seg)
        prev = out[-1]
        if prev["class"] == "REST" and cur["class"] == "REST":
            prev["t1"] = max(float(prev["t1"]), float(cur["t1"]))
            prev["duration_s"] = float(prev["t1"] - prev["t0"])
            prev["i1"] = max(int(prev.get("i1", prev["i0"] + 1)), int(cur.get("i1", cur["i0"] + 1)))
        else:
            out.append(cur)
    return out


# ---------------- Model download/cache ----------------

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "artifacts" / "model.json"
MODEL_PATH = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL))

ARTIFACT_DL_TIMEOUT_S = float(os.environ.get("ARTIFACT_DL_TIMEOUT_S", "15"))
ARTIFACT_MAX_BYTES = int(os.environ.get("ARTIFACT_MAX_BYTES", str(5_000_000)))  # 5 MB

MODEL_OBJ = None
MODEL_READY = False


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _download_signed_url_to_tmp(url: str) -> Path:
    url_hash = _sha256(url)[:16]
    out_path = Path("/tmp") / f"model_{url_hash}.json"

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    try:
        with requests.get(url, stream=True, timeout=ARTIFACT_DL_TIMEOUT_S) as r:
            if r.status_code in (401, 403):
                raise HTTPException(status_code=401, detail="Signed URL not authorized/expired")
            if r.status_code == 404:
                raise HTTPException(status_code=404, detail="Signed URL not found")
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"Signed URL fetch failed: {r.status_code}")

            total = 0
            tmp_path = out_path.with_suffix(".json.part")
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > ARTIFACT_MAX_BYTES:
                        raise HTTPException(status_code=413, detail="Artifact too large")
                    f.write(chunk)

            tmp_path.replace(out_path)
            return out_path
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Signed URL fetch error: {type(e).__name__}: {e}")


@lru_cache(maxsize=8)
def _load_model_from_signed_url_cached(url: str) -> dict:
    path = _download_signed_url_to_tmp(url)
    return load_model(str(path))


def resolve_model(artifact_url: str | None):
    if artifact_url:
        M = _load_model_from_signed_url_cached(artifact_url)
        return M, "own", f"signed_url_sha256:{_sha256(artifact_url)[:16]}"
    if not MODEL_READY or MODEL_OBJ is None:
        raise HTTPException(status_code=503, detail="Default model not loaded")
    return MODEL_OBJ, "universal", f"default:{MODEL_PATH}"


# ---------------- Core pipeline (console-identical) ----------------

def _run_predict_console_equivalent(df: pd.DataFrame, M: dict, args: dict):
    """
    Mirrors src/predict_workout.py steps 1..9 for identical results.
    """
    classes = M["classes"]
    fs = float(M["meta"]["fs_hz"])
    winS = float(M["meta"]["win_s"])
    hopS = float(M["meta"]["hop_s"])
    strength_classes = strength_classes_from(M)

    # 2) Data
    df = ensure_time_column_df(df)

    # 3) Windows + predict
    X, _, t0s, feat_names = build_windows(df, fs, winS, hopS)
    if X.shape[0] == 0:
        # console throws; API returns empty segments (more user-friendly)
        return [], fs, winS, hopS

    model_feats = M.get("feature_names", [])
    if model_feats and len(model_feats) != X.shape[1]:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Feature count mismatch (data: {X.shape[1]} vs model: {len(model_feats)}). "
                f"Retrain after changing features."
            ),
        )

    # NOTE: console only warns on feat_names mismatch; we keep silent here for API stability.

    _, probs = predict_features(X, M)

    # 4) smooth + debounce (window level)
    probs_s = probs
    if not bool(args.get("no_smooth", False)):
        probs_s = smooth_probs_over_time(probs_s, k=max(1, int(args.get("prob_smooth_k", 5))))

    cls_idx2 = np.argmax(probs_s, axis=1)
    if not bool(args.get("no_debounce", False)):
        cls_idx2 = debounce_labels(cls_idx2, min_run=max(1, int(args.get("debounce_run", 3))))

    # 5) Decoder -> segments (fixed defaults as console)
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
    segments = decoder.decode(
        df=df,
        probs_s=probs_s,
        t0s=np.asarray(t0s, float),
        win_s=winS,
        hop_s=hopS,
        fs=fs,
    )

    # 6) merge short
    merge_min_s = float(args.get("merge_min_s", 4.0))
    if merge_min_s and merge_min_s > 0:
        segments = merge_short_segments(segments, min_len_s=merge_min_s, prefer="neighbor")

    # 7) exercise gate
    if not bool(args.get("no_gate", False)):
        gcfg = ExerciseGateConfig(
            min_set_s=float(args.get("gate_min_set_s", 18.0)),
            min_mean_conf=float(args.get("gate_min_mean", 0.55)),
            min_peak_conf=float(args.get("gate_min_peak", 0.70)),
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
        segments = merge_short_segments(segments, min_len_s=0.0, prefer="neighbor")

    # 8) adjacency resolver
    if not bool(args.get("no_adj", False)):
        acfg = AdjacencyResolverConfig(
            max_gap_s=0.1,
            rest_bridge_s=float(args.get("adj_rest_bridge_s", 6.0)),
            score_margin=float(args.get("adj_score_margin", 0.10)),
            min_combined_mean_conf=float(args.get("adj_min_combined_mean", 0.55)),
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

    # 9) Rep counting + output shaping
    t = df["t"].to_numpy(float)
    ax = df["ax"].to_numpy(float)
    ay = df["ay"].to_numpy(float)
    az = df["az"].to_numpy(float)

    rep_mode = str(args.get("rep_mode", "pair"))
    smooth_sec = float(args.get("smooth_sec", 0.2))
    min_peak_sep = float(args.get("min_peak_sep", 0.4))

    rep_min_s = float(args.get("rep_min_s", 0.4))
    rep_max_s = float(args.get("rep_max_s", 3.5))
    rep_k = float(args.get("rep_k", 0.7))
    acf_min_s = float(args.get("acf_min_s", 0.45))
    acf_max_s = float(args.get("acf_max_s", 3.0))

    smooth_k = max(1, int(round(smooth_sec * fs)))
    az_smooth = moving_average(az, smooth_k)  # peaks-mode

    results = []
    for seg in segments:
        seg_class = str(seg["class"])
        # NOTE: keep console's inclusive end (<=)
        mask = (t >= float(seg["t0"])) & (t <= float(seg["t1"]))

        reps = 0
        if seg_class in strength_classes and np.count_nonzero(mask) > 3:
            if rep_mode == "pair":
                sig = select_rep_signal(ax[mask], ay[mask], az[mask], fs)
                reps, _dbg = count_reps_adaptive(
                    sig,
                    fs,
                    base_k=rep_k,
                    min_s=rep_min_s,
                    max_s=rep_max_s,
                    acf_min_s=acf_min_s,
                    acf_max_s=acf_max_s,
                )
            else:
                az_seg = az_smooth[mask]
                reps = count_peaks(
                    az_seg,
                    fs,
                    min_separation_s=min_peak_sep,
                    thresh_mode="median_mad",
                    prominence=0.5 * mad(az_seg),
                )

        results.append({
            "t0": float(seg["t0"]),
            "t1": float(seg["t1"]),
            "duration_s": float(seg["t1"] - seg["t0"]),
            "class": seg_class,
            "reps": int(reps) if seg_class in strength_classes else 0,
            "reps_refined": int(reps) if seg_class in strength_classes else 0,
        })

    return results, fs, winS, hopS


# ---------------- FastAPI ----------------

app = FastAPI(title="Bangle Workout API", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model_once():
    global MODEL_OBJ, MODEL_READY
    try:
        MODEL_OBJ = load_model(MODEL_PATH)
        MODEL_READY = True
        print(f"[startup] default model loaded: {MODEL_PATH}")
    except Exception as e:
        MODEL_READY = False
        print(f"[startup] default model load failed: {MODEL_PATH} -> {e}")


@app.get("/", include_in_schema=False)
@app.get("/health", include_in_schema=False)
@app.get("/healthz", include_in_schema=False)
@app.get("/healthhz", include_in_schema=False)  # Railway healthcheck alias
def healthz():
    return {"status": "ok", "ready": bool(MODEL_READY)}


@app.get("/readyz", include_in_schema=False)
def readyz():
    return {"ready": bool(MODEL_READY)}


class PredictResponse(BaseModel):
    model_version: str
    trained_on: str          # "own" | "universal"
    artifact_ref: str        # "default:..." oder "signed_url_sha256:..."
    fs_hz: float
    win_s: float
    hop_s: float
    segments: list
    model_config = {"protected_namespaces": ()}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    workout_file: UploadFile = File(...),
    artifact_url: str | None = Form(None),

    # EXACT same knobs/defaults as console
    prob_smooth_k: int = Form(5),
    debounce_run: int = Form(3),
    merge_min_s: float = Form(4.0),

    no_smooth: bool = Form(False),
    no_debounce: bool = Form(False),

    gate_min_set_s: float = Form(18.0),
    gate_min_mean: float = Form(0.55),
    gate_min_peak: float = Form(0.70),
    no_gate: bool = Form(False),

    adj_rest_bridge_s: float = Form(6.0),
    adj_score_margin: float = Form(0.10),
    adj_min_combined_mean: float = Form(0.55),
    no_adj: bool = Form(False),

    rep_mode: str = Form("pair"),         # "pair" | "peaks"
    smooth_sec: float = Form(0.2),
    min_peak_sep: float = Form(0.4),

    rep_min_s: float = Form(0.4),
    rep_max_s: float = Form(3.5),
    rep_k: float = Form(0.7),
    acf_min_s: float = Form(0.45),
    acf_max_s: float = Form(3.0),
):
    M, trained_on, artifact_ref = resolve_model(artifact_url)

    data = await workout_file.read()
    rows = _parse_workout_bytes(data)

    if not rows:
        return PredictResponse(
            model_version=M.get("version", "v1"),
            trained_on=trained_on,
            artifact_ref=artifact_ref,
            fs_hz=float(M["meta"]["fs_hz"]),
            win_s=float(M["meta"]["win_s"]),
            hop_s=float(M["meta"]["hop_s"]),
            segments=[],
        )

    df = pd.DataFrame(rows)

    args = dict(
        prob_smooth_k=prob_smooth_k,
        debounce_run=debounce_run,
        merge_min_s=merge_min_s,
        no_smooth=no_smooth,
        no_debounce=no_debounce,
        gate_min_set_s=gate_min_set_s,
        gate_min_mean=gate_min_mean,
        gate_min_peak=gate_min_peak,
        no_gate=no_gate,
        adj_rest_bridge_s=adj_rest_bridge_s,
        adj_score_margin=adj_score_margin,
        adj_min_combined_mean=adj_min_combined_mean,
        no_adj=no_adj,
        rep_mode=rep_mode,
        smooth_sec=smooth_sec,
        min_peak_sep=min_peak_sep,
        rep_min_s=rep_min_s,
        rep_max_s=rep_max_s,
        rep_k=rep_k,
        acf_min_s=acf_min_s,
        acf_max_s=acf_max_s,
    )

    try:
        segments, fs, winS, hopS = _run_predict_console_equivalent(df, M, args)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predict failed: {type(e).__name__}: {e}")

    return PredictResponse(
        model_version=M.get("version", "v1"),
        trained_on=trained_on,
        artifact_ref=artifact_ref,
        fs_hz=float(fs),
        win_s=float(winS),
        hop_s=float(hopS),
        segments=segments,
    )
