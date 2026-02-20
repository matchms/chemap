from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class ClevelandStyle:
    """Styling defaults for a Cleveland-ish dot plot."""
    figsize: Tuple[float, float] = (9.0, 6.0)
    dpi: int = 600
    markersize: float = 7.0
    markeredgecolor: str = "white"
    connect_linestyle: str = ":"
    connect_linewidth: float = 2.0
    connect_alpha: float = 0.9
    grid_x_linestyle: str = ":"
    grid_y_linestyle: str = "-"
    grid_alpha_x: float = 0.6
    grid_alpha_y: float = 0.6


def cleveland_dotplot(
    *,
    # Data in "tidy" arrays
    row: Sequence[str],
    x: Sequence[float],
    color_group: Optional[Sequence[str]] = None,
    marker_group: Optional[Sequence[str]] = None,
    connect_group: Optional[Sequence[str]] = None,
    marker_zorder: Optional[Mapping[str, float]] = None,

    # Ordering / labels
    row_order: Optional[Sequence[str]] = None,
    row_label_fn=None,

    # Mappings
    color_map: Optional[Dict[str, str]] = None,
    marker_map: Optional[Dict[str, str]] = None,

    # Figure/axes
    title: str = "",
    xlabel: str = "",
    ax: Optional[Axes] = None,

    # Behavior
    connect: bool = True,
    show_zero_line_if_needed: bool = True,

    # Range line
    row_range: bool = True,
    row_range_color: str = "black",
    row_range_linewidth: float = 1.5,
    row_range_alpha: float = 0.75,

    # Legends
    show_legends: bool = True,
    color_legend_title: str = "Setting",
    marker_legend_title: str = "Variant",

    style: ClevelandStyle = ClevelandStyle(),
) -> Tuple[Figure, Axes]:
    """
    Generic Cleveland-ish dot plot.

    Parameters
    ----------
    row, x
        One entry per point. `row` defines which horizontal row the point belongs to,
        `x` is the numeric x-position.
    color_group
        Category that controls dot color (e.g., binary/count/logcount).
    marker_group
        Category that controls marker shape (e.g., dense/sparse/fixed).
    connect_group
        Group id used to connect points on the same row (e.g., same setting).
        Connection happens within (row, connect_group).
    connect
        If True, draw a line between min(x) and max(x) within each (row, connect_group).
    """

    # --- normalize inputs ---
    row = list(map(str, row))
    x = np.asarray(x, dtype=float)

    n = len(row)
    if x.shape[0] != n:
        raise ValueError("row and x must have the same length")

    if color_group is None:
        color_group = ["_"] * n
    else:
        color_group = list(map(str, color_group))
        if len(color_group) != n:
            raise ValueError("color_group must match length of row/x")

    if marker_group is None:
        marker_group = ["_"] * n
    else:
        marker_group = list(map(str, marker_group))
        if len(marker_group) != n:
            raise ValueError("marker_group must match length of row/x")

    if connect_group is None:
        connect_group = ["_"] * n
    else:
        connect_group = list(map(str, connect_group))
        if len(connect_group) != n:
            raise ValueError("connect_group must match length of row/x")

    # Default mappings
    if color_map is None:
        # Let matplotlib handle if not provided: we’ll still create a legend if asked.
        color_map = {}

    if marker_map is None:
        marker_map = {"_": "o"}  # default to circle

    if marker_zorder is None:
        marker_zorder = {}

    # Row order
    if row_order is None:
        # stable order by appearance
        seen = []
        seen_set = set()
        for r in row:
            if r not in seen_set:
                seen.append(r)
                seen_set.add(r)
        row_order = seen
    else:
        row_order = list(map(str, row_order))

    ypos = {r: i for i, r in enumerate(row_order)}
    y = np.array([ypos[r] for r in row], dtype=float)

    # Labels
    if row_label_fn is None:
        row_label_fn = lambda s: s  # noqa: E731

    # --- axes setup ---
    if ax is None:
        fig_h = max(2.5, len(row_order) * 0.28)
        fig, ax = plt.subplots(figsize=(style.figsize[0], fig_h), dpi=style.dpi)
    else:
        fig = ax.figure

    # Row-range indicator (min->max across ALL points in the row)
    if row_range:
        from collections import defaultdict
        xs_by_row = defaultdict(list)
        for r, xv in zip(row, x):
            xs_by_row[r].append(float(xv))

        for r in row_order:
            xs = xs_by_row.get(r, [])
            if len(xs) < 2:
                continue
            y0 = ypos[r]
            ax.plot(
                [min(xs), max(xs)],
                [y0, y0],
                color=row_range_color,
                linewidth=row_range_linewidth,
                alpha=row_range_alpha,
                zorder=0,  # behind connectors + points
            )

    # --- optional connectors ---
    if connect:
        # For each (row, connect_group), connect min->max x
        key_arr = list(zip(row, connect_group))
        # group indices by key
        from collections import defaultdict
        idx_by_key = defaultdict(list)
        for i, k in enumerate(key_arr):
            idx_by_key[k].append(i)

        for (r, cg), idxs in idx_by_key.items():
            if len(idxs) < 2:
                continue
            xs = x[idxs]
            y0 = ypos[r]
            col_key = color_group[idxs[0]]
            col = color_map.get(col_key, "gray")
            ax.plot(
                [float(xs.min()), float(xs.max())],
                [y0, y0],
                linestyle=style.connect_linestyle,
                linewidth=style.connect_linewidth,
                alpha=style.connect_alpha,
                color=col,
                zorder=1,
            )

    # --- plot dots ---
    # Plot by (marker_group, color_group) so style is consistent & legend-friendly
    uniq_marker = list(dict.fromkeys(marker_group))
    uniq_color = list(dict.fromkeys(color_group))

    for mg in uniq_marker:
        m = marker_map.get(mg, "o")
        z = float(marker_zorder.get(mg, 3.0))
        for cg in uniq_color:
            mask = [(marker_group[i] == mg and color_group[i] == cg) for i in range(n)]
            if not any(mask):
                continue
            xs = x[mask]
            ys = y[mask]
            col = color_map.get(cg, None)
            ax.plot(
                xs, ys,
                linestyle="None",
                marker=m,
                color=col,
                markersize=style.markersize,
                markeredgecolor=style.markeredgecolor,
                zorder=z,
                label=None,  # legends are handled manually
            )

    # zero line if needed
    if show_zero_line_if_needed and n > 0 and float(np.min(x)) <= 0.0:
        ax.axvline(0, linestyle="--", linewidth=1.5, color="black")

    # axes / labels
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([row_label_fn(r) for r in row_order])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle=style.grid_x_linestyle, alpha=style.grid_alpha_x)
    ax.grid(True, axis="y", linestyle=style.grid_y_linestyle, alpha=style.grid_alpha_y)
    ax.set_axisbelow(True)

    # --- legends ---
    if show_legends:
        # marker legend
        marker_handles = []
        for mg in uniq_marker:
            if mg == "_" and len(set(uniq_marker)) == 1:
                continue
            marker_handles.append(
                Line2D([0], [0], marker=marker_map.get(mg, "o"),
                       color="black", linestyle="None", label=str(mg))
            )

        # color legend
        color_handles = []
        for cg in uniq_color:
            if cg == "_" and len(set(uniq_color)) == 1:
                continue
            col = color_map.get(cg, "gray")
            color_handles.append(
                Line2D([0], [0], marker="o", color=col, linestyle="None", label=str(cg))
            )

        # Place legends similarly to your original if both exist
        if marker_handles and color_handles:
            leg1 = ax.legend(handles=marker_handles, loc="lower right",
                             title=marker_legend_title, frameon=True)
            ax.add_artist(leg1)
            ax.legend(handles=color_handles, loc="lower left",
                      title=color_legend_title, frameon=True)
        elif marker_handles:
            ax.legend(handles=marker_handles, loc="lower right",
                      title=marker_legend_title, frameon=True)
        elif color_handles:
            ax.legend(handles=color_handles, loc="lower left",
                      title=color_legend_title, frameon=True)

    return fig, ax
