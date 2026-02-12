from typing import Tuple
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from chemap.plotting import (
    CmapPaletteConfig,
    LabelMapConfig,
    PaletteConfig,
    PresentPairsConfig,
    build_hier_label_map,
    make_hier_palette,
    map_classes_to_display_labels,
    palette_from_cmap,
    sorted_present_pairs,
)


def _is_rgb(t: Tuple[float, float, float]) -> bool:
    if not (isinstance(t, tuple) and len(t) == 3):
        return False
    return all(isinstance(x, (float, int)) and 0.0 <= float(x) <= 1.0 for x in t)


def _is_rgba(t: Tuple[float, float, float, float]) -> bool:
    if not (isinstance(t, tuple) and len(t) == 4):
        return False
    return all(isinstance(x, (float, int)) and 0.0 <= float(x) <= 1.0 for x in t)


# ---------------------------------------------------------------------
# build_hier_label_map / map_classes_to_display_labels
# ---------------------------------------------------------------------

def test_build_hier_label_map_empty_valid_returns_empty():
    df = pd.DataFrame({"Superclass": [None, np.nan], "Class": [None, "unknown"]})
    class_to_label, info = build_hier_label_map(df)
    assert class_to_label == {}
    assert info == {}


def test_build_hier_label_map_branches_rare_keep_huge():
    # Construct:
    # - RareSuper: total 2 (<3) -> all classes -> rare_label
    # - KeepSuper: total 4 (>=3 and <=6) -> all classes map to superclass
    # - HugeSuper: total 7 (>6) -> explicit labels for big classes; others -> other
    df = pd.DataFrame(
        {
            "Superclass": (
                ["RareSuper"] * 2
                + ["KeepSuper"] * 4
                + ["HugeSuper"] * 7
            ),
            "Class": (
                ["r1", "r2"]
                + ["k1", "k1", "k2", "k3"]     # mix, but all should map to "KeepSuper"
                + ["h1"] * 4 + ["h2"] * 2 + ["h3"] * 1  # huge
            ),
        }
    )

    cfg = LabelMapConfig(
        superclass_col="Superclass",
        class_col="Class",
        low_superclass_thres=3,
        low_class_thres=3,
        max_superclass_size=6,
        rare_label="RARE",
        sep="->",
        top_k_classes=None,
    )
    class_to_label, info = build_hier_label_map(df, config=cfg)

    # RareSuper classes -> RARE
    assert class_to_label["r1"] == "RARE"
    assert class_to_label["r2"] == "RARE"
    assert info["RareSuper"]["branch"] == "rare_superclass"

    # KeepSuper classes -> KeepSuper
    assert class_to_label["k1"] == "KeepSuper"
    assert class_to_label["k2"] == "KeepSuper"
    assert class_to_label["k3"] == "KeepSuper"
    assert info["KeepSuper"]["branch"] == "keep_superclass"

    # HugeSuper:
    # h1 count 4 >= low_class_thres -> explicit "HugeSuper->h1"
    # h2 count 2 < low_class_thres -> "HugeSuper->other"
    # h3 count 1 < low_class_thres -> "HugeSuper->other"
    assert class_to_label["h1"] == "HugeSuper->h1"
    assert class_to_label["h2"] == "HugeSuper->other"
    assert class_to_label["h3"] == "HugeSuper->other"
    assert info["HugeSuper"]["branch"] == "huge_superclass"


def test_build_hier_label_map_top_k_caps_explicit_classes_in_huge_superclass():
    df = pd.DataFrame(
        {
            "Superclass": ["Huge"] * 12,
            "Class": ["a"] * 6 + ["b"] * 3 + ["c"] * 2 + ["d"] * 1,
        }
    )
    cfg = LabelMapConfig(
        low_superclass_thres=1,
        max_superclass_size=5,  # so "Huge" is huge
        low_class_thres=1,      # so counts alone wouldn't collapse
        top_k_classes=2,        # only top 2 should remain explicit: a,b
        rare_label="RARE",
        sep="->",
    )
    class_to_label, info = build_hier_label_map(df, config=cfg)
    assert class_to_label["a"] == "Huge->a"
    assert class_to_label["b"] == "Huge->b"
    assert class_to_label["c"] == "Huge->other"
    assert class_to_label["d"] == "Huge->other"
    assert info["Huge"]["branch"] == "huge_superclass"


def test_map_classes_to_display_labels_fallback_for_missing_and_unmapped():
    class_to_label = {"A": "X", "B": "Y"}
    ser = pd.Series(["A", "B", "C", None, "unknown", "  "])
    out = map_classes_to_display_labels(ser, class_to_label, rare_label="RARE")
    assert out.tolist() == ["X", "Y", "RARE", "RARE", "RARE", "RARE"]


# ---------------------------------------------------------------------
# make_hier_palette
# ---------------------------------------------------------------------

def test_make_hier_palette_assigns_neutral_colors_for_rare_and_other():
    display_labels = [
        "Sup1",
        "Sup2->Child1",
        "Sup2->other",
        "Rare Superclass or Unknown",
    ]
    cfg = PaletteConfig(
        sep="->",
        rare_label="Rare Superclass or Unknown",
        other_suffix="other",
        base_cmap="tab20",
        neutral_rare=(0.9, 0.9, 0.9),
        neutral_other=(0.1, 0.1, 0.1),
        child_lighten_min=0.2,
        child_lighten_max=0.2,
    )
    pal = make_hier_palette(display_labels, config=cfg)

    assert pal["Rare Superclass or Unknown"] == (0.9, 0.9, 0.9)
    assert pal["Sup2->other"] == (0.1, 0.1, 0.1)

    # Everything should be valid RGB tuples in [0, 1]
    for k, v in pal.items():
        assert _is_rgb(v), f"{k} is not valid rgb: {v}"


def test_make_hier_palette_is_deterministic_given_same_inputs():
    labels = ["A", "B->c1", "B->c2", "B->other", "Rare Superclass or Unknown"]
    cfg = PaletteConfig(base_cmap="tab20")
    pal1 = make_hier_palette(labels, config=cfg)
    pal2 = make_hier_palette(labels, config=cfg)
    assert pal1 == pal2


# ---------------------------------------------------------------------
# sorted_present_pairs
# ---------------------------------------------------------------------

def test_sorted_present_pairs_default_alpha_sort():
    df = pd.DataFrame(
        {
            "Class": ["B", "A", "B", "A"],
            "Subclass": ["b2", "a2", "b1", "a1"],
        }
    )
    out = sorted_present_pairs(df, config=PresentPairsConfig())
    # Expect A first, then B; within each alpha by subclass
    assert out[["Class", "Subclass"]].values.tolist() == [
        ["A", "a1"],
        ["A", "a2"],
        ["B", "b1"],
        ["B", "b2"],
    ]


def test_sorted_present_pairs_respects_class_order_and_subclass_order_within_class():
    df = pd.DataFrame(
        {
            "Class": ["B", "A", "B", "A", "C"],
            "Subclass": ["b2", "a2", "b1", "a1", "c1"],
        }
    )
    cfg = PresentPairsConfig(
        class_order=["B", "A"],  # C should go last
        subclass_order_within_class={"B": ["b2", "b1"], "A": ["a2"]},
    )
    out = sorted_present_pairs(df, config=cfg)
    # B first (b2 then b1), then A (a2 then remaining a1 alpha), then C
    assert out[["Class", "Subclass"]].values.tolist() == [
        ["B", "b2"],
        ["B", "b1"],
        ["A", "a2"],
        ["A", "a1"],
        ["C", "c1"],
    ]


def test_sorted_present_pairs_drops_nan_subclass_when_required():
    df = pd.DataFrame({"Class": ["A", "A"], "Subclass": ["x", np.nan]})
    out = sorted_present_pairs(df, config=PresentPairsConfig(require_subclass=True))
    assert out.values.tolist() == [["A", "x"]]


# ---------------------------------------------------------------------
# palette_from_cmap
# ---------------------------------------------------------------------

def test_palette_from_cmap_empty_returns_empty():
    pal = palette_from_cmap([], config=CmapPaletteConfig(cmap="viridis"))
    assert pal == {}


def test_palette_from_cmap_returns_rgba_by_default():
    pal = palette_from_cmap(["a", "b", "c"], config=CmapPaletteConfig(cmap="viridis", rgb_only=False))
    assert set(pal.keys()) == {"a", "b", "c"}
    assert all(_is_rgba(tuple(v)) for v in pal.values())


def test_palette_from_cmap_returns_rgb_when_rgb_only_true():
    pal = palette_from_cmap(["a", "b"], config=CmapPaletteConfig(cmap="viridis", rgb_only=True))
    assert set(pal.keys()) == {"a", "b"}
    assert all(_is_rgb(tuple(v)) for v in pal.values())


def test_palette_from_cmap_single_label_uses_single_position():
    cfg = CmapPaletteConfig(cmap="viridis", single_position=0.25, rgb_only=False)
    pal = palette_from_cmap(["only"], config=cfg)

    cmap = mpl.colormaps.get_cmap("viridis")
    expected = mcolors.to_rgba(cmap(0.25))
    assert pal["only"] == expected
