# src/segmentation/__init__.py

from .postprocessing import (
    smooth_probs_over_time,
    debounce_labels,
    merge_short_segments,
    strength_classes_from,
)

from .decoder import StateMachineSegmenter, DecoderConfig
from .exercise_gate import exercise_gate, ExerciseGateConfig
from .adjacency_resolver import resolve_adjacent_strength, AdjacencyResolverConfig

__all__ = [
    "smooth_probs_over_time",
    "debounce_labels",
    "merge_short_segments",
    "strength_classes_from",
    "StateMachineSegmenter",
    "DecoderConfig",
    "exercise_gate",
    "ExerciseGateConfig",
    "resolve_adjacent_strength",
    "AdjacencyResolverConfig",
]
