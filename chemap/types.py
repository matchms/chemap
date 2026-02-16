from typing import Mapping, Sequence, Tuple, Union
import numpy as np


Bins = Sequence[Tuple[float, float]]
Color = Tuple[float, float, float]  # RGB
ColorA = Tuple[float, float, float, float]  # RGBA
Palette = Mapping[str, Union[Color, ColorA]]
UnfoldedBinary = list[np.ndarray]  # list of int64 feature IDs per molecule
UnfoldedCount = list[tuple[np.ndarray, np.ndarray]]  # (int64 feature IDs, float32 values)
