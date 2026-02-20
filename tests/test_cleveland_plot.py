import matplotlib


matplotlib.use("Agg")  # must be set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from chemap.plotting import cleveland_dotplot


@pytest.fixture(autouse=True)
def _close_figures():
    """Ensure matplotlib figures are closed after each test."""
    yield
    plt.close("all")


def _find_lines(ax, *, zorder=None, linestyle=None, marker=None):
    """Helper: filter Line2D from axes by a few common properties."""
    lines = [ln for ln in ax.lines if isinstance(ln, Line2D)]
    if zorder is not None:
        lines = [ln for ln in lines if ln.get_zorder() == zorder]
    if linestyle is not None:
        lines = [ln for ln in lines if ln.get_linestyle() == linestyle]
    if marker is not None:
        lines = [ln for ln in lines if ln.get_marker() == marker]
    return lines


def _has_zero_vline(ax) -> bool:
    """Detect the zero reference line produced by ax.axvline(0, linestyle='--')."""
    for ln in ax.lines:
        if not isinstance(ln, Line2D):
            continue
        if ln.get_linestyle() != "--":
            continue
        xdata = np.asarray(ln.get_xdata(), dtype=float)
        if xdata.size == 2 and np.allclose(xdata, [0.0, 0.0]):
            return True
    return False


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="row and x must have the same length"):
        cleveland_dotplot(
            row=["A", "B"],
            x=[1.0],
            show_legends=False,
        )


def test_group_length_mismatch_raises():
    with pytest.raises(ValueError, match="color_group must match length"):
        cleveland_dotplot(
            row=["A", "B"],
            x=[1.0, 2.0],
            color_group=["binary"],  # mismatch
            show_legends=False,
        )

    with pytest.raises(ValueError, match="marker_group must match length"):
        cleveland_dotplot(
            row=["A", "B"],
            x=[1.0, 2.0],
            marker_group=["dense"],  # mismatch
            show_legends=False,
        )

    with pytest.raises(ValueError, match="connect_group must match length"):
        cleveland_dotplot(
            row=["A", "B"],
            x=[1.0, 2.0],
            connect_group=["g1"],  # mismatch
            show_legends=False,
        )


def test_default_row_order_is_stable_by_appearance():
    fig, ax = cleveland_dotplot(
        row=["B", "A", "B", "C"],
        x=[0.2, 0.1, 0.3, 0.4],
        show_legends=False,
        connect=False,
        row_range=False,
        show_zero_line_if_needed=False,
    )

    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["B", "A", "C"]


def test_row_range_indicator_drawn_per_row_with_2plus_points():
    # Row A has 3 points -> should get a range line.
    # Row B has 1 point -> no range line.
    fig, ax = cleveland_dotplot(
        row=["A", "A", "A", "B"],
        x=[10, 20, 5, 7],
        row_range=True,
        connect=False,
        show_legends=False,
        show_zero_line_if_needed=False,
    )

    # range lines have zorder=0 and default solid linestyle '-'
    range_lines = _find_lines(ax, zorder=0)
    assert len(range_lines) == 1

    ln = range_lines[0]
    xdata = np.asarray(ln.get_xdata(), dtype=float)
    ydata = np.asarray(ln.get_ydata(), dtype=float)

    assert np.allclose(xdata, [5.0, 20.0])
    # Row A is first (stable appearance order), so y should be 0
    assert np.allclose(ydata, [0.0, 0.0])


def test_connectors_drawn_within_row_and_connect_group_and_use_color_map():
    # Two points in (row=A, group=g1) => connector drawn.
    # One point in (row=A, group=g2) => no connector for that.
    color_map = {"binary": "crimson", "count": "teal"}

    fig, ax = cleveland_dotplot(
        row=["A", "A", "A"],
        x=[1.0, 3.0, 2.0],
        color_group=["binary", "binary", "count"],
        connect_group=["g1", "g1", "g2"],
        connect=True,
        row_range=False,
        show_legends=False,
        show_zero_line_if_needed=False,
        color_map=color_map,
    )

    connector_lines = _find_lines(ax, zorder=1, linestyle=":")
    assert len(connector_lines) == 1

    ln = connector_lines[0]
    xdata = np.asarray(ln.get_xdata(), dtype=float)
    ydata = np.asarray(ln.get_ydata(), dtype=float)

    assert np.allclose(xdata, [1.0, 3.0])
    assert np.allclose(ydata, [0.0, 0.0])  # row A at y=0

    # Connector color is taken from first element in that (row, connect_group)
    # Here that's 'binary' -> crimson.
    assert ln.get_color() == "crimson"


def test_marker_zorder_applied_per_marker_group():
    marker_map = {"dense": "o", "sparse": "^"}
    marker_zorder = {"dense": 3.0, "sparse": 5.0}

    fig, ax = cleveland_dotplot(
        row=["A", "A"],
        x=[1.0, 1.0],
        marker_group=["dense", "sparse"],
        color_group=["binary", "binary"],
        marker_map=marker_map,
        marker_zorder=marker_zorder,
        connect=False,
        row_range=False,
        show_legends=False,
        show_zero_line_if_needed=False,
    )

    # Marker lines have linestyle "None"
    dense_lines = _find_lines(ax, linestyle="None", marker="o")
    sparse_lines = _find_lines(ax, linestyle="None", marker="^")

    assert len(dense_lines) == 1
    assert len(sparse_lines) == 1

    assert dense_lines[0].get_zorder() == 3.0
    assert sparse_lines[0].get_zorder() == 5.0


def test_zero_line_added_only_when_min_x_leq_zero():
    # Case 1: includes negative -> should add zero vline
    fig, ax = cleveland_dotplot(
        row=["A", "B"],
        x=[-0.1, 0.2],
        connect=False,
        row_range=False,
        show_legends=False,
        show_zero_line_if_needed=True,
    )
    assert _has_zero_vline(ax) is True

    # Case 2: all positive -> should not add zero vline
    fig, ax = cleveland_dotplot(
        row=["A", "B"],
        x=[0.1, 0.2],
        connect=False,
        row_range=False,
        show_legends=False,
        show_zero_line_if_needed=True,
    )
    assert _has_zero_vline(ax) is False

    # Case 3: negative but disabled -> should not add
    fig, ax = cleveland_dotplot(
        row=["A", "B"],
        x=[-0.1, 0.2],
        connect=False,
        row_range=False,
        show_legends=False,
        show_zero_line_if_needed=False,
    )
    assert _has_zero_vline(ax) is False


def test_legends_created_when_enabled_and_groups_present():
    marker_map = {"dense": "o", "sparse": "^"}
    color_map = {"binary": "crimson", "count": "teal"}

    fig, ax = cleveland_dotplot(
        row=["A", "A", "B", "B"],
        x=[1.0, 2.0, 3.0, 4.0],
        marker_group=["dense", "sparse", "dense", "sparse"],
        color_group=["binary", "binary", "count", "count"],
        marker_map=marker_map,
        color_map=color_map,
        connect=False,
        row_range=False,
        show_legends=True,
        show_zero_line_if_needed=False,
    )

    # When both marker_handles and color_handles exist:
    # - one legend is added as an artist (ax.add_artist),
    # - the other is the "current" legend returned by ax.legend(...)
    legends = [ch for ch in ax.get_children() if isinstance(ch, Legend)]
    assert len(legends) >= 1
    assert ax.get_legend() is not None  # at least the last legend exists


def test_no_legends_when_disabled():
    fig, ax = cleveland_dotplot(
        row=["A", "B"],
        x=[1.0, 2.0],
        show_legends=False,
        connect=False,
        row_range=False,
        show_zero_line_if_needed=False,
    )

    legends = [ch for ch in ax.get_children() if isinstance(ch, Legend)]
    assert len(legends) == 0
    assert ax.get_legend() is None
