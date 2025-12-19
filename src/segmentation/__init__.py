# src/segmentation/__init__.py
from .reps import (
    moving_average,
    median,
    mad,
    count_peaks,
    count_reps_peak_trough,
    highpass_ma,
    estimate_rep_period_acf,
    select_rep_signal,
    rep_params_for_class,
)
