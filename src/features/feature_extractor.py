from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from . import legacy_features


@dataclass
class WindowedFeatures:
    X: np.ndarray              
    y: List[Optional[str]]    
    t0s: List[float]                  
    feature_names: List[str]           


class FeatureExtractor:
    def __init__(self, fs: float):
        self.fs = fs

    def build_windows(
        self,
        df: pd.DataFrame,
        win_s: float,
        hop_s: float,
    ) -> WindowedFeatures:
        X, y, t0s, names = legacy_features.build_windows(
            df=df,
            fs=self.fs,
            win_s=win_s,
            hop_s=hop_s,
        )
        return WindowedFeatures(
            X=X,
            y=y,
            t0s=t0s,
            feature_names=names,
        )
