import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelMapConfig:
    """Configuration for hierarchical display label mapping.

    The goal is to map fine-grained classes into display labels suitable for plots:
    - small superclasses -> collapse into `rare_label`
    - normal superclasses -> show superclass-only label
    - huge superclasses -> show superclass->class for common classes, superclass->other for the rest
    """

    superclass_col: str = "Superclass"
    class_col: str = "Class"

    low_superclass_thres: int = 2500
    low_class_thres: int = 5000
    max_superclass_size: int = 10_000

    rare_label: str = "Rare Superclass or Unknown"
    sep: str = "->"

    # If provided, keep only top_k_classes as explicit labels in huge superclasses
    top_k_classes: Optional[int] = None


def _normalize_label_value(x: Any) -> Optional[str]:
    """Normalize a label cell value.

    Returns None for missing / unknown-like values, else a stripped string.
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() in {"unknown", "none", "nan", "null"}:
        return None
    return s


def build_hier_label_map(
    labels: pd.DataFrame,
    *,
    config: LabelMapConfig = LabelMapConfig(),
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int | str]]]:
    """Build a mapping from fine-grained class labels to display labels.

    Parameters
    ----------
    labels:
        DataFrame containing at least `config.superclass_col` and `config.class_col`.
        Each row is one observation with a superclass and class.
    config:
        Thresholds and string configuration.

    Returns
    -------
    class_to_label:
        Dict mapping original Class (string) -> display label.
        Notes:
        - Missing/unknown class/superclass entries are NOT included here (because there is no class key).
        - This is intended for mapping non-null class values, e.g. `df[class_col].map(class_to_label).fillna(rare_label)`.
    superclass_info:
        Dict with per-superclass debug info: count, branch, n_classes.

    Branches
    --------
    - "rare_superclass": superclass count < low_superclass_thres -> all its classes map to rare_label
    - "huge_superclass": superclass count > max_superclass_size -> common classes map to "super->class",
      others to "super->other"
    - "keep_superclass": otherwise -> all its classes map to "superclass"
    """
    if not isinstance(labels, pd.DataFrame):
        raise TypeError("labels must be a pandas DataFrame")

    missing_cols = [c for c in (config.superclass_col, config.class_col) if c not in labels.columns]
    if missing_cols:
        raise KeyError(f"labels is missing required columns: {missing_cols}")

    if config.low_superclass_thres < 0 or config.low_class_thres < 0 or config.max_superclass_size < 0:
        raise ValueError("Thresholds must be non-negative")
    if config.low_superclass_thres > config.max_superclass_size:
        raise ValueError("low_superclass_thres must be <= max_superclass_size")
    if config.top_k_classes is not None and config.top_k_classes <= 0:
        raise ValueError("top_k_classes must be a positive integer or None")

    df = labels[[config.superclass_col, config.class_col]].copy()
    df[config.superclass_col] = df[config.superclass_col].map(_normalize_label_value)
    df[config.class_col] = df[config.class_col].map(_normalize_label_value)

    # Any row with missing superclass OR class is treated as unknown/rare.
    # We remove them from counting; downstream mapping should handle them via fillna(rare_label).
    valid = df.dropna(subset=[config.superclass_col, config.class_col])

    # If there is nothing valid, return empty mapping with empty info.
    if valid.empty:
        return {}, {}

    superclass_counts = valid[config.superclass_col].value_counts(dropna=True)

    # MultiIndex Series: (superclass, class) -> count
    class_counts_by_super = (
        valid.groupby([config.superclass_col, config.class_col], sort=False)
        .size()
        .astype(int)
    )

    class_to_label: Dict[str, str] = {}
    superclass_info: Dict[str, Dict[str, int | str]] = {}

    for superclass, sc_count in superclass_counts.items():
        # Series indexed by class for this superclass
        # (safe even if missing, but should not happen since superclass from superclass_counts)
        cls_counts = class_counts_by_super.xs(superclass, level=0, drop_level=True)

        if int(sc_count) < config.low_superclass_thres:
            branch = "rare_superclass"
            for cls in cls_counts.index:
                class_to_label[str(cls)] = config.rare_label

        elif int(sc_count) > config.max_superclass_size:
            branch = "huge_superclass"

            # Determine which classes qualify as "explicit"
            if config.top_k_classes is not None and len(cls_counts) > config.top_k_classes:
                top_classes = set(cls_counts.sort_values(ascending=False).head(config.top_k_classes).index)
            else:
                top_classes = set(cls_counts.index)

            other_label = f"{superclass}{config.sep}other"
            for cls, c_count in cls_counts.items():
                cls_str = str(cls)
                if (cls not in top_classes) or (int(c_count) < config.low_class_thres):
                    class_to_label[cls_str] = other_label
                else:
                    class_to_label[cls_str] = f"{superclass}{config.sep}{cls_str}"

        else:
            branch = "keep_superclass"
            for cls in cls_counts.index:
                class_to_label[str(cls)] = str(superclass)

        superclass_info[str(superclass)] = {
            "count": int(sc_count),
            "branch": branch,
            "n_classes": int(len(cls_counts)),
        }

    return class_to_label, superclass_info


# ---------------------------------------------------------------------------
# Palette generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaletteConfig:
    """Configuration for hierarchical palette construction."""

    sep: str = "->"
    rare_label: str = "Rare Superclass or Unknown"
    other_suffix: str = "other"

    # Base colormap for distinct superclasses (matplotlib colormap name)
    base_cmap: str = "tab20"

    # How much to lighten children relative to base color (0=no change, 1=white)
    child_lighten_min: float = 0.15
    child_lighten_max: float = 0.65

    neutral_rare: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    neutral_other: Tuple[float, float, float] = (0.35, 0.35, 0.35)


_SUBLABEL_RE_CACHE: Dict[str, re.Pattern[str]] = {}


def _get_sub_re(sep: str) -> re.Pattern[str]:
    pat = _SUBLABEL_RE_CACHE.get(sep)
    if pat is None:
        pat = re.compile(rf"^(.*?){re.escape(sep)}(.*)$")
        _SUBLABEL_RE_CACHE[sep] = pat
    return pat


def _lighten_rgb(rgb: Tuple[float, float, float], amount: float) -> Tuple[float, float, float]:
    """Blend `rgb` towards white by `amount` in [0, 1]."""
    amount = float(np.clip(amount, 0.0, 1.0))
    r, g, b = rgb
    return (r + (1.0 - r) * amount, g + (1.0 - g) * amount, b + (1.0 - b) * amount)


def _get_cmap(cmap: Union[str, mpl.colors.Colormap]) -> mpl.colors.Colormap:
    """Matplotlib 3.7+ safe colormap retrieval."""
    if isinstance(cmap, str):
        return mpl.colormaps.get_cmap(cmap)
    return cmap


def _distinct_base_colors(n: int, cmap_name: str) -> list[Tuple[float, float, float]]:
    """Get n distinct colors from a matplotlib colormap, as RGB tuples.

    Uses the non-deprecated Matplotlib colormap registry API.
    """
    if n <= 0:
        return []
    cmap = _get_cmap(cmap_name)
    xs = np.linspace(0.0, 1.0, n)
    return [mcolors.to_rgb(cmap(float(x))) for x in xs]


def make_hier_palette(
    display_labels: Iterable[Any],
    *,
    config: PaletteConfig = PaletteConfig(),
) -> Dict[str, Tuple[float, float, float]]:
    """Create a hierarchical color palette for plot-ready display labels.

    Parameters
    ----------
    display_labels:
        Iterable of final labels used for plotting (e.g. the mapped labels).
        Can include:
        - superclass-only labels: "Lipids"
        - hierarchical labels: "Lipids->Fatty acids", "Lipids->other"
        - rare label: config.rare_label
    config:
        Palette rules and defaults.

    Returns
    -------
    label_to_color:
        Dict mapping each label (string) to an RGB tuple (floats in [0, 1]).

    Notes
    -----
    - Superclasses get distinct base colors from `config.base_cmap`.
    - Child labels under a superclass get progressively lighter variants of the base color.
    - "...->other" gets `config.neutral_other`.
    - `config.rare_label` gets `config.neutral_rare`.
    """
    # Normalize, drop NA, preserve uniqueness with stable ordering
    s = pd.Series(list(display_labels))
    s = s[~s.isna()].map(lambda x: str(x))
    seen: set[str] = set()
    unique_labels: list[str] = []
    for lab in s.tolist():
        if lab not in seen:
            seen.add(lab)
            unique_labels.append(lab)

    sub_re = _get_sub_re(config.sep)

    super_to_children: Dict[str, list[str]] = {}
    pure_super: set[str] = set()

    for lab in unique_labels:
        if lab == config.rare_label:
            continue
        m = sub_re.match(lab)
        if not m:
            pure_super.add(lab)
        else:
            sup = m.group(1)
            super_to_children.setdefault(sup, []).append(lab)

    base_supers = sorted(pure_super | set(super_to_children.keys()))
    base_colors = _distinct_base_colors(len(base_supers), config.base_cmap)
    super_to_base: Dict[str, Tuple[float, float, float]] = dict(zip(base_supers, base_colors, strict=True))

    label_to_color: Dict[str, Tuple[float, float, float]] = {}

    # Pure superclass colors
    for sup in pure_super:
        if sup in super_to_base:
            label_to_color[sup] = super_to_base[sup]

    # Children colors
    for sup, labs in super_to_children.items():
        base = super_to_base.get(sup, (0.0, 0.0, 0.0))

        other_label = f"{sup}{config.sep}{config.other_suffix}"

        # Stable ordering: alphabetical, but keep "other" last
        labs_sorted = sorted(labs, key=lambda x: (x == other_label, x))

        n = len(labs_sorted)
        if n == 0:
            continue

        # Lighten amounts from min..max across children
        amounts = np.linspace(config.child_lighten_min, config.child_lighten_max, n)

        for lab, amt in zip(labs_sorted, amounts, strict=True):
            if lab == other_label:
                label_to_color[lab] = config.neutral_other
            else:
                label_to_color[lab] = _lighten_rgb(base, float(amt))

    # Rare label color
    if config.rare_label in unique_labels:
        label_to_color[config.rare_label] = config.neutral_rare

    # Ensure valid RGB tuples
    return {k: mcolors.to_rgb(v) for k, v in label_to_color.items()}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def map_classes_to_display_labels(
    class_series: pd.Series,
    class_to_label: Mapping[str, str],
    *,
    rare_label: str = "Rare Superclass or Unknown",
) -> pd.Series:
    """Map a class series to display labels with a fallback for unknowns.

    This is the typical way to apply `build_hier_label_map()`.

    Parameters
    ----------
    class_series:
        Series of class labels (may contain NaN / unknown values).
    class_to_label:
        Mapping from class -> display label.
    rare_label:
        Label used for missing/unmapped classes.

    Returns
    -------
    Series of display labels.
    """
    if not isinstance(class_series, pd.Series):
        raise TypeError("class_series must be a pandas Series")
    normalized = class_series.map(_normalize_label_value)
    mapped = normalized.map(lambda x: class_to_label.get(x, None) if x is not None else None)
    return mapped.fillna(rare_label)


# ---------------------------------------------------------------------------
# Balanced / small-class helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PresentPairsConfig:
    """Configuration for extracting and sorting present (Class, Subclass) pairs."""

    class_col: str = "Class"
    subclass_col: str = "Subclass"

    # Optional explicit global ordering for classes.
    class_order: Optional[Sequence[str]] = None

    # Optional ordering for subclasses within a class:
    # { "Lipids": ["Fatty acids", "Steroids", ...], "Alkaloids": [...], ... }
    subclass_order_within_class: Optional[Mapping[str, Sequence[str]]] = None

    # If True, ensure class/subclass values are normalized by stripping whitespace.
    strip: bool = True

    # If True, subclasses must be non-null to be included (matches the prototype behavior).
    require_subclass: bool = True


def _normalize_for_sorting(x: Any, *, strip: bool = True) -> str:
    """Convert a value to a stable sort key (case-insensitive string)."""
    if pd.isna(x):
        return ""
    s = str(x)
    if strip:
        s = s.strip()
    return s.lower()


def sorted_present_pairs(
    data_plot: pd.DataFrame,
    *,
    config: PresentPairsConfig = PresentPairsConfig(),
) -> pd.DataFrame:
    """Return a sorted DataFrame of unique (Class, Subclass) pairs present in `data_plot`.

    This is useful for small or balanced label spaces where you want a stable legend order
    and an easy way to construct palettes for categorical plots.

    Parameters
    ----------
    data_plot:
        DataFrame containing at least `config.class_col` and `config.subclass_col`.
    config:
        Sorting and filtering configuration.

    Returns
    -------
    DataFrame with exactly two columns: [class_col, subclass_col], sorted.
    Only pairs that occur in `data_plot` are returned. If `require_subclass=True`,
    rows with missing subclass are ignored.

    Sorting rules
    -------------
    - Primary: class order
        - if `class_order` provided: categorical order (unknown classes go last)
        - else: alphabetical (case-insensitive)
    - Secondary: subclass order within class
        - if `subclass_order_within_class` provided:
            - subclasses listed for that class come first in the specified order
            - unlisted subclasses come after, alphabetically
        - else: alphabetical (case-insensitive)
    """
    if not isinstance(data_plot, pd.DataFrame):
        raise TypeError("data_plot must be a pandas DataFrame")

    missing_cols = [c for c in (config.class_col, config.subclass_col) if c not in data_plot.columns]
    if missing_cols:
        raise KeyError(f"data_plot is missing required columns: {missing_cols}")

    df = data_plot[[config.class_col, config.subclass_col]].copy()

    if config.require_subclass:
        df = df.dropna(subset=[config.subclass_col])

    # Deduplicate present pairs
    df = df.drop_duplicates()

    # Optional stripping (keeps original values, but affects sorting keys)
    class_vals = df[config.class_col].map(lambda x: str(x).strip() if (config.strip and not pd.isna(x)) else x)
    sub_vals = df[config.subclass_col].map(lambda x: str(x).strip() if (config.strip and not pd.isna(x)) else x)
    df[config.class_col] = class_vals
    df[config.subclass_col] = sub_vals

    # Build primary class sort key
    if config.class_order is not None:
        # Unknown classes become NaN in categorical; sort with na_position="last"
        df["_class_key"] = pd.Categorical(
            df[config.class_col],
            categories=list(config.class_order),
            ordered=True,
        )
        class_sort_cols = ["_class_key"]
        na_pos = "last"
    else:
        df["_class_key"] = df[config.class_col].map(lambda x: _normalize_for_sorting(x, strip=config.strip))
        class_sort_cols = ["_class_key"]
        na_pos = "last"

    # Build subclass sort key (per-class order supported)
    if config.subclass_order_within_class is not None:
        order_map = config.subclass_order_within_class

        # Precompute index maps for O(1) lookup
        index_maps: Dict[str, Dict[str, int]] = {}
        for cls, order in order_map.items():
            index_maps[str(cls)] = {str(lbl): i for i, lbl in enumerate(order)}

        def _sub_key(row: pd.Series) -> Tuple[int, Any]:
            cls = row[config.class_col]
            sub = row[config.subclass_col]
            cls_s = "" if pd.isna(cls) else str(cls)
            sub_s = "" if pd.isna(sub) else str(sub)

            idx_map = index_maps.get(cls_s)
            if idx_map is None:
                # No explicit order for this class -> alphabetical fallback
                return (1, _normalize_for_sorting(sub_s, strip=config.strip))

            if sub_s in idx_map:
                return (0, idx_map[sub_s])

            # Not listed -> after listed items, alphabetical among unlisted
            return (1, _normalize_for_sorting(sub_s, strip=config.strip))

        df["_sub_key"] = df.apply(_sub_key, axis=1)
    else:
        df["_sub_key"] = df[config.subclass_col].map(lambda x: _normalize_for_sorting(x, strip=config.strip))

    df = df.sort_values(class_sort_cols + ["_sub_key"], na_position=na_pos)
    df = df.drop(columns=["_class_key", "_sub_key"])
    return df


# ---------------------------------------------------------------------------
# Simple palette helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CmapPaletteConfig:
    """Configuration for generating a categorical palette from a colormap."""

    cmap: str = "viridis"

    # If n == 1, position within the colormap to sample.
    single_position: float = 0.5

    # If True, return RGB tuples; otherwise return RGBA tuples (matplotlib default).
    rgb_only: bool = False


def palette_from_cmap(
    labels: Sequence[Any],
    *,
    config: CmapPaletteConfig = CmapPaletteConfig(),
) -> Dict[str, Tuple[float, float, float] | Tuple[float, float, float, float]]:
    """Evenly distribute labels along a colormap.

    Parameters
    ----------
    labels:
        Sequence of labels. Labels are stringified for keys.
    config:
        Palette configuration including the colormap.

    Returns
    -------
    Dict label -> color, where color is RGBA by default (or RGB if `rgb_only=True`).

    Notes
    -----
    - If there is only one label, samples at `single_position` (default 0.5).
    - For multiple labels, samples evenly across [0, 1].
    """
    cmap = _get_cmap(config.cmap)

    seen: set[str] = set()
    uniq: list[str] = []
    for lbl in labels:
        if pd.isna(lbl):
            continue
        s = str(lbl)
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    n = len(uniq)
    if n == 0:
        return {}

    positions = np.array([float(np.clip(config.single_position, 0.0, 1.0))]) if n == 1 else np.linspace(0.0, 1.0, n)
    out: Dict[str, Any] = {lbl: cmap(float(pos)) for lbl, pos in zip(uniq, positions, strict=True)}

    if config.rgb_only:
        return {k: mcolors.to_rgb(v) for k, v in out.items()}
    return {k: mcolors.to_rgba(v) for k, v in out.items()}
