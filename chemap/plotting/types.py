from typing import Mapping, Tuple, Union


Color = Tuple[float, float, float]  # RGB
ColorA = Tuple[float, float, float, float]  # RGBA
Palette = Mapping[str, Union[Color, ColorA]]
