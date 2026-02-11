from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from chemap.plotting import (
    LabelMapConfig,
    PaletteConfig,
    PresentPairsConfig,
    build_hier_label_map,
    build_selected_label_column,
    build_selected_palette,
    make_hier_palette,
    map_classes_to_display_labels,
    palette_from_cmap,
    sorted_present_pairs,
)
from .types import Color, ColorA, Palette


# ---------------------------------------------------------------------------
# Core scatter (base for later plotting functions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScatterStyle:
    figsize: Tuple[float, float] = (20, 20)
    title: str = "UMAP of embeddings"

    s: float = 5.0
    alpha: float = 0.25
    linewidths: float = 0.0

    legend_title: Optional[str] = None
    legend_loc: str = "lower left"
    legend_frameon: bool = False
    legend_ncol: int = 1
    legend_markersize: float = 8.0
    legend_alpha: float = 0.8

    hide_ticks: bool = True
    hide_axis_labels: bool = True


def _validate_required_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"data_plot is missing required columns: {missing}")


def _to_rgba(color: Union[Color, ColorA]) -> ColorA:
    return mcolors.to_rgba(color)


def _build_legend_handles(
    ordered_labels: Sequence[str],
    palette: Palette,
    *,
    fallback: Union[Color, ColorA] = (0.5, 0.5, 0.5, 1.0),
    markersize: float = 8.0,
    alpha: float = 0.8,
) -> List[Line2D]:
    handles: List[Line2D] = []
    for lbl in ordered_labels:
        col = palette.get(lbl, fallback)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=markersize,
                markerfacecolor=_to_rgba(col),
                markeredgecolor="none",
                label=str(lbl),
                alpha=alpha,
            )
        )
    return handles


def scatter_plot_base(
    data_plot: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    label_col: str,
    palette: Palette,
    legend_labels: Optional[Sequence[str]] = None,
    style: ScatterStyle = ScatterStyle(),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """A base scatter plot function that takes pre-mapped labels and a palette.
    This is not intended for direct use, but as a building block for the more user-friendly wrapper functions below.
    """
    _validate_required_columns(data_plot, [x_col, y_col, label_col])

    labels = data_plot[label_col].dropna().map(str)

    if legend_labels is None:
        legend_labels = sorted(labels.unique().tolist(), key=lambda s: s.lower())
    else:
        legend_labels = [str(x) for x in legend_labels]

    colors = data_plot[label_col].map(lambda v: palette.get(str(v), (0.5, 0.5, 0.5, 1.0)))

    if ax is None:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    ax.scatter(
        data_plot[x_col].to_numpy(),
        data_plot[y_col].to_numpy(),
        c=[_to_rgba(c) for c in colors.tolist()],
        s=style.s,
        alpha=style.alpha,
        linewidths=style.linewidths,
    )

    ax.set_title(style.title)

    if style.hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if style.hide_axis_labels:
        ax.set_xlabel("")
        ax.set_ylabel("")

    legend_title = style.legend_title if style.legend_title is not None else label_col
    handles = _build_legend_handles(
        legend_labels,
        palette,
        markersize=style.legend_markersize,
        alpha=style.legend_alpha,
    )

    ax.legend(
        handles=handles,
        title=legend_title,
        loc=style.legend_loc,
        frameon=style.legend_frameon,
        ncol=style.legend_ncol,
    )

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Wrapper for all classes (balanced / small label spaces)
#   - NO config objects exposed; parameters are direct kwargs.
# ---------------------------------------------------------------------------

def scatter_plot_all_classes(
    data_plot: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    class_col: str = "Class",
    subclass_col: str = "Subclass",
    palette_or_cmap: Union[Palette, str, mpl.colors.Colormap] = "viridis",
    # ordering options (same semantics as before)
    class_order: Optional[Sequence[str]] = None,
    subclass_order_within_class: Optional[Mapping[str, Sequence[str]]] = None,
    # cmap palette options (used only if palette_or_cmap is not a dict)
    cmap_single_position: float = 0.5,
    cmap_rgb_only: bool = False,
    # plotting style (surface the key knobs; advanced users can pass ScatterStyle via style=)
    figsize: Tuple[float, float] = (20, 20),
    title: str = "UMAP of embeddings",
    s: float = 5.0,
    alpha: float = 0.25,
    linewidths: float = 0.0,
    legend_title: Optional[str] = None,
    legend_loc: str = "lower left",
    legend_frameon: bool = False,
    legend_ncol: int = 1,
    legend_markersize: float = 8.0,
    legend_alpha: float = 0.8,
    hide_ticks: bool = True,
    hide_axis_labels: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes, Dict[str, Union[Color, ColorA]]]:
    """Balanced/small label-space scatter.

    Parameters
    ----------
    data_plot: pd.DataFrame
        DataFrame containing the data to plot. Must include columns specified by x_col, y_col
        and the label columns.
    x_col: str
        Name of the column in data_plot to use for x-axis values.
    y_col: str
        Name of the column in data_plot to use for y-axis values.
    class_col: str
        Name of the column in data_plot that contains the "Class" labels.
    subclass_col: str
        Name of the column in data_plot that contains the "Subclass" labels.
    palette_or_cmap: Union[Palette, str, mpl.colors.Colormap]
        Either a dict mapping subclass labels to colors, or a colormap name / object to generate
        a palette from. If a colormap is provided, a palette will be generated based on the present
        (Class, Subclass) pairs.
    class_order: Optional[Sequence[str]]
        Optional ordering of classes for palette generation. If None, classes will be sorted alphabetically.
    subclass_order_within_class: Optional[Mapping[str, Sequence[str]]]
        Optional ordering of subclasses within each class for palette generation. If None, subclasses will be sorted
        alphabetically within each class.
    cmap_single_position: float
        If palette_or_cmap is a colormap and there is only one present subclass, this position in the colormap
        will be used for the color. Must be between 0 and 1. Default is 0.5 (the middle of the colormap).
    cmap_rgb_only: bool
        If True and palette_or_cmap is a colormap, the generated palette will contain RGB tuples instead of RGBA.
        Default is False (RGBA).
    figsize: Tuple[float, float]
        Size of the figure to create.
    title: str
        Title of the plot.
    
    - If `palette_or_cmap` is a dict: uses it directly (Subclass -> color).
    - Else: creates a Subclass palette from a colormap using present (Class, Subclass) order.
    """
    _validate_required_columns(data_plot, [x_col, y_col, class_col, subclass_col])

    present = sorted_present_pairs(
        data_plot,
        config=PresentPairsConfig(
            class_col=class_col,
            subclass_col=subclass_col,
            class_order=class_order,
            subclass_order_within_class=subclass_order_within_class,
        ),
    )
    present_subclasses = present[subclass_col].dropna().map(str).tolist()

    if isinstance(palette_or_cmap, Mapping):
        palette: Dict[str, Union[Color, ColorA]] = {str(k): v for k, v in palette_or_cmap.items()}
    else:
        palette = palette_from_cmap(
            palette_or_cmap,
            present_subclasses,
            single_position=cmap_single_position,
            rgb_only=cmap_rgb_only,
        )

    style = ScatterStyle(
        figsize=figsize,
        title=title,
        s=s,
        alpha=alpha,
        linewidths=linewidths,
        legend_title=legend_title if legend_title is not None else subclass_col,
        legend_loc=legend_loc,
        legend_frameon=legend_frameon,
        legend_ncol=legend_ncol,
        legend_markersize=legend_markersize,
        legend_alpha=legend_alpha,
        hide_ticks=hide_ticks,
        hide_axis_labels=hide_axis_labels,
    )

    fig, ax = scatter_plot_base(
        data_plot,
        x_col=x_col,
        y_col=y_col,
        label_col=subclass_col,
        palette=palette,
        legend_labels=present_subclasses,
        style=style,
        ax=ax,
    )
    return fig, ax, palette


# ---------------------------------------------------------------------------
# Wrapper for selection of hierarchical labels
#   - NO config objects exposed; parameters are direct kwargs.
# ---------------------------------------------------------------------------

def scatter_plot_hierarchical_labels(
    data_plot: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    superclass_col: str = "Superclass",
    class_col: str = "Class",
    display_label_col: str = "display_label",
    inplace: bool = False,
    # label-mapping params
    low_superclass_thres: int = 2500,
    low_class_thres: int = 5000,
    max_superclass_size: int = 10_000,
    rare_label: str = "Rare Superclass or Unknown",
    sep: str = "->",
    top_k_classes: Optional[int] = None,
    # palette params
    other_suffix: str = "other",
    base_cmap: str = "tab20",
    neutral_rare: Color = (0.6, 0.6, 0.6),
    neutral_other: Color = (0.35, 0.35, 0.35),
    child_lighten_min: float = 0.15,
    child_lighten_max: float = 0.65,
    # plotting style
    figsize: Tuple[float, float] = (20, 20),
    title: str = "UMAP of embeddings",
    s: float = 2.0,
    alpha: float = 0.2,
    linewidths: float = 0.0,
    legend_title: str = "Class / Superclass",
    legend_loc: str = "lower left",
    legend_frameon: bool = False,
    legend_ncol: int = 1,
    legend_markersize: float = 8.0,
    legend_alpha: float = 0.8,
    hide_ticks: bool = True,
    hide_axis_labels: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes, Dict[str, str], Dict[str, Union[Color, ColorA]]]:
    """Hierarchical-label scatter (builds display labels and palette internally).
    
    Parameters
    ----------
    data_plot: pd.DataFrame
        DataFrame containing the data to plot. Must include columns specified by x_col, y_col
        and the label columns.
    x_col: str
        Name of the column in data_plot to use for x-axis values.
    y_col: str
        Name of the column in data_plot to use for y-axis values.
    superclass_col: str
        Name of the column in data_plot that contains the "Superclass" labels.
    class_col: str
        Name of the column in data_plot that contains the "Class" labels.
    display_label_col: str
        Name of the column to create in data_plot for the display labels (combination of class and superclass).
    inplace: bool
        Whether to modify data_plot in place or work on a copy.
    low_superclass_thres: int
        Superclasses with fewer than this many samples will be considered "low" and their classes may
        be collapsed into the rare_label, depending on other parameters.
    low_class_thres: int
        Classes with fewer than this many samples will be considered "low" and may be collapsed into the rare_label,
        depending on other parameters.
    max_superclass_size: int
        Superclasses with more than this many samples will be considered "huge" and all their classes will be
        collapsed into "other" (with display label superclass + sep + other_suffix).
    rare_label: str
        Display label to use for collapsed rare classes/superclasses.
    sep: str
        Separator to use when combining superclass and class into display labels.
    top_k_classes: Optional[int]
        If not None, only the top K most common classes within each superclass will be kept as explicit display labels
        (even if they are above the low_class_thres), and the rest will be collapsed into the rare_label.
    other_suffix: str
        Suffix to use for the "other" category when collapsing huge superclasses.
    figsize: Tuple[float, float]
        Size of the figure to create.
    title: str
        Title of the plot.
    """
    _validate_required_columns(data_plot, [x_col, y_col, superclass_col, class_col])

    df = data_plot if inplace else data_plot.copy()

    class_to_label, info = build_hier_label_map(
        df,
        config=LabelMapConfig(
            superclass_col=superclass_col,
            class_col=class_col,
            low_superclass_thres=low_superclass_thres,
            low_class_thres=low_class_thres,
            max_superclass_size=max_superclass_size,
            rare_label=rare_label,
            sep=sep,
            top_k_classes=top_k_classes,
        ),
    )

    df[display_label_col] = map_classes_to_display_labels(
        df[class_col],
        class_to_label,
        rare_label=rare_label,
    )

    label_to_color = make_hier_palette(
        df[display_label_col].unique(),
        config=PaletteConfig(
            sep=sep,
            rare_label=rare_label,
            other_suffix=other_suffix,
            base_cmap=base_cmap,
            child_lighten_min=child_lighten_min,
            child_lighten_max=child_lighten_max,
            neutral_rare=neutral_rare,
            neutral_other=neutral_other,
        ),
    )

    present_labels = sorted(df[display_label_col].dropna().map(str).unique().tolist(), key=str.lower)

    style = ScatterStyle(
        figsize=figsize,
        title=title,
        s=s,
        alpha=alpha,
        linewidths=linewidths,
        legend_title=legend_title,
        legend_loc=legend_loc,
        legend_frameon=legend_frameon,
        legend_ncol=legend_ncol,
        legend_markersize=legend_markersize,
        legend_alpha=legend_alpha,
        hide_ticks=hide_ticks,
        hide_axis_labels=hide_axis_labels,
    )

    fig, ax = scatter_plot_base(
        df,
        x_col=x_col,
        y_col=y_col,
        label_col=display_label_col,
        palette=label_to_color,
        legend_labels=present_labels,
        style=style,
        ax=ax,
    )

    return fig, ax, class_to_label, dict(label_to_color)


# ---------------------------------------------------------------------------
# Wrapper for few selected classes
# ---------------------------------------------------------------------------

def scatter_plot_selected_only(
    data_plot: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    class_col: str = "Class",
    subclass_col: str = "Subclass",
    selected_classes: Optional[Sequence[Any]] = None,
    selected_subclasses: Optional[Sequence[Any]] = None,
    selected_size_relative: float = 2.0,
    other_label: str = "other",
    other_color: Union[Color, ColorA] = (0.8, 0.8, 0.8, 0.1),
    palette_or_cmap: Union[Palette, str, Any] = "viridis",
    cmap_single_position: float = 0.5,
    cmap_rgb_only: bool = False,
    style: ScatterStyle = ScatterStyle(),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes, Dict[str, Union[Color, ColorA]]]:
    """Scatter plot where only a selected subset is colored; all other points are gray 'other'.

    Additionally, selected points get a larger marker size: `style.s * selected_size_relative`.

    Parameters
    ----------
    data_plot: pd.DataFrame
        DataFrame containing the data to plot. Must include columns specified by x_col, y_col
        and the label columns.
    x_col: str
        Name of the column in data_plot to use for x-axis values.
    y_col: str
        Name of the column in data_plot to use for y-axis values.
    class_col: str
        Name of the column in data_plot that contains the "Class" labels.
    subclass_col: str
        Name of the column in data_plot that contains the "Subclass" labels.
    selected_classes: Optional[Sequence[Any]]
        List of class labels to highlight. If None, no classes are highlighted.
    selected_subclasses: Optional[Sequence[Any]]
        List of subclass labels to highlight. If None, no subclasses are highlighted.
    selected_size_relative: float
        Factor by which to increase the marker size of selected points. Must be > 0.
    other_label: str
        Label to use for the 'other' category in the legend.
    """
    if selected_size_relative <= 0:
        raise ValueError("selected_size_relative must be > 0")

    for col in (x_col, y_col, class_col, subclass_col):
        if col not in data_plot.columns:
            raise KeyError(f"data_plot is missing required column: {col}")

    df = data_plot.copy()

    df["_selected_label"] = build_selected_label_column(
        df,
        class_col=class_col,
        subclass_col=subclass_col,
        selected_classes=selected_classes,
        selected_subclasses=selected_subclasses,
        other_label=other_label,
    )

    # Identify selected vs other (for size scaling)
    is_other = df["_selected_label"].astype(str).eq(other_label)
    sizes = np.where(is_other.to_numpy(), style.s, style.s * float(selected_size_relative))

    # Legend order: selected labels alphabetical, then other last
    present = df["_selected_label"].dropna().map(str).unique().tolist()
    present_non_other = sorted([p for p in present if p != other_label], key=str.lower)
    legend_labels = present_non_other + ([other_label] if other_label in present else [])

    palette = build_selected_palette(
        legend_labels,
        palette_or_cmap=palette_or_cmap,
        other_label=other_label,
        other_color=other_color,
        cmap_single_position=cmap_single_position,
        cmap_rgb_only=cmap_rgb_only,
    )

    colors = df["_selected_label"].map(lambda v: palette.get(str(v), other_color))

    if ax is None:
        fig, ax = plt.subplots(figsize=style.figsize)
    else:
        fig = ax.figure

    ax.scatter(
        df[x_col].to_numpy(),
        df[y_col].to_numpy(),
        c=[_to_rgba(c) for c in colors.tolist()],
        s=sizes,
        alpha=style.alpha,
        linewidths=style.linewidths,
    )

    ax.set_title(style.title)

    if style.hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if style.hide_axis_labels:
        ax.set_xlabel("")
        ax.set_ylabel("")

    from matplotlib.lines import Line2D

    handles: List[Line2D] = []
    for lbl in legend_labels:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=style.legend_markersize,
                markerfacecolor=_to_rgba(palette.get(lbl, other_color)),
                markeredgecolor="none",
                label=str(lbl),
                alpha=style.legend_alpha,
            )
        )

    ax.legend(
        handles=handles,
        title=style.legend_title if style.legend_title is not None else "selected",
        loc=style.legend_loc,
        frameon=style.legend_frameon,
        ncol=style.legend_ncol,
    )

    fig.tight_layout()
    return fig, ax, palette
