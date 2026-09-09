from collections.abc import Mapping, Sequence
import numpy as np


Bins = Sequence[tuple[float, float]]
Color = tuple[float, float, float]  # RGB
ColorA = tuple[float, float, float, float]  # RGBA
Palette = Mapping[str, Color | ColorA]
UnfoldedBinary = list[np.ndarray]  # list of int64 feature IDs per molecule
UnfoldedCount = list[tuple[np.ndarray, np.ndarray]]  # (int64 feature IDs, float32 values)
