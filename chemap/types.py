from typing import Mapping, Sequence, Tuple, Union


Bins = Sequence[Tuple[float, float]]
Color = Tuple[float, float, float]  # RGB
ColorA = Tuple[float, float, float, float]  # RGBA
Palette = Mapping[str, Union[Color, ColorA]]
