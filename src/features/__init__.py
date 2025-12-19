from .feature_extractor import FeatureExtractor, WindowedFeatures
from .legacy_features import rfft_band_features  # <- HINZUGEFÜGT


def build_windows(df, fs, win_s, hop_s):
    """
    Gleiche Signatur wie die alte build_windows-Funktion.
    Intern wird nur der neue FeatureExtractor verwendet.
    """
    extractor = FeatureExtractor(fs=fs)
    wf = extractor.build_windows(df=df, win_s=win_s, hop_s=hop_s)
    return wf.X, wf.y, wf.t0s, wf.feature_names
