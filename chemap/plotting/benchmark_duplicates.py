from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
from chemap.benchmarking import compute_duplicate_max_mass_differences
from chemap.plotting.colormap_handling import n_colors_from_cmap
from chemap.plotting.colormaps import green_yellow_red
from chemap.types import Bins


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateBinResult:
    """Binned duplicate statistics for one experiment/dataset."""
    name: str
    bin_edges: Bins
    bin_labels: List[str]
    bin_counts: np.ndarray  # shape (n_bins,)
    total: int


def default_bins_da() -> Bins:
    """Default bins for maximum mass differences (Da)."""
    return [(0, 1), (1, 10), (10, 50), (50, 100), (100, 200), (200, 400), (400, np.inf)]


def _format_bin_labels(bins: Bins, unit: str = "Da") -> List[str]:
    labels: List[str] = []
    for low, high in bins:
        if np.isinf(high):
            labels.append(f"{low:g}-inf {unit}")
        else:
            labels.append(f"{low:g}-{high:g} {unit}")
    return labels


def compute_duplicate_bin_counts(
    duplicates: Sequence[Sequence[int]],
    masses: Sequence[float],
    *,
    bins: Optional[Bins] = None,
    unit: str = "Da",
    name: str = "experiment",
) -> DuplicateBinResult:
    """Compute counts of duplicate fingerprint groups binned by max mass difference.

    Parameters
    ----------
    duplicates:
        Output of `find_duplicates_with_hashing(fingerprints)`:
        iterable of groups; each group is a sequence of indices into `masses`.
        Only groups with len>=2 are typically present, but we don't assume it.
    masses:
        Per-compound masses aligned with fingerprint order.
    bins:
        Bins for maximum mass difference (low inclusive, high exclusive).
        Defaults to `default_bins_da()`.
    unit:
        Used only for bin labels.
    name:
        Label used for reporting/plotting.

    Returns
    -------
    DuplicateBinResult
    """
    if bins is None:
        bins = default_bins_da()

    masses_arr = np.asarray(masses, dtype=float)
    if masses_arr.ndim != 1:
        raise ValueError("masses must be a 1D sequence")

    max_diffs_arr = compute_duplicate_max_mass_differences(duplicates, masses_arr)

    # bin counts
    counts = np.zeros(len(bins), dtype=int)
    for i, (low, high) in enumerate(bins):
        if max_diffs_arr.size == 0:
            counts[i] = 0
            continue
        mask = (max_diffs_arr >= low) & (max_diffs_arr < high)
        counts[i] = int(np.count_nonzero(mask))

    labels = _format_bin_labels(bins, unit=unit)
    total = int(counts.sum())

    return DuplicateBinResult(
        name=name,
        bin_edges=bins,
        bin_labels=labels,
        bin_counts=counts,
        total=total,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_duplicate_bins(
    results: Sequence[DuplicateBinResult],
    *,
    figsize: Tuple[float, float] = (10, 6),
    sort_by_total: bool = True,
    cmap = green_yellow_red,
    bar_height: float = 0.5,
    show_totals: bool = True,
    totals_fmt: str = "{value:.0f}",
    xlabel: str = "Compounds with Fingerprint Duplicates",
    title: str = "Duplicate Statistics by Experiment",
    legend_title: str = "Maximum mass difference\n(for identical fingerprints)",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot stacked horizontal bars of duplicate counts across bins.

    Parameters
    ----------
    results:
        One DuplicateBinResult per experiment/dataset.
        All results should share the same bins (same length and meaning).
    sort_by_total:
        Sort experiments descending by total duplicates.
    cmap:
        hand over desired matplotlib colormap. Default is a green-yellow-red colormap.
    bar_height:
        Height of each horizontal bar.
    show_totals:
        Add total number at bar end.
    totals_fmt:
        Format string used for totals; available var is `value`.
    xlabel, title, legend_title:
        Plot labels.

    """
    if len(results) == 0:
        raise ValueError("results must be non-empty")

    # Validate consistent bins
    n_bins = len(results[0].bin_counts)
    for r in results:
        if len(r.bin_counts) != n_bins:
            raise ValueError("All results must use the same number of bins (bin_counts length mismatch)")
        if r.bin_labels != results[0].bin_labels:
            raise ValueError("All results must use the same bin_labels (bins mismatch)")

    res = list(results)
    if sort_by_total:
        res.sort(key=lambda r: r.total, reverse=True)

    bin_labels = res[0].bin_labels
    colors = n_colors_from_cmap(n_bins, cmap)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    y_positions = np.arange(len(res))
    left_stack = np.zeros(len(res), dtype=float)

    # Stack bars per bin
    for i, label in enumerate(bin_labels):
        values = np.array([r.bin_counts[i] for r in res], dtype=float)
        ax.barh(
            y_positions,
            values,
            height=bar_height,
            left=left_stack,
            label=label,
            color=colors[i],
        )
        left_stack += values

    # Totals
    if show_totals:
        for i, y in enumerate(y_positions):
            value = float(res[i].total)
            ax.text(value, y - 0.1, totals_fmt.format(value=value), fontsize=9)

    ax.set_axisbelow(True)
    ax.grid(which="major", color="#DDDDDD", linewidth=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r.name for r in res])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(title=legend_title, loc="upper right")

    return fig, ax


# ---------------------------------------------------------------------------
# End-to-end convenience wrapper for multiple experiments
# ---------------------------------------------------------------------------

def plot_duplicates_by_experiment(
    experiments: Mapping[str, Mapping[str, Any]],
    masses_arr: np.ndarray,
    *,
    bins: Optional[Bins] = None,
    unit: str = "Da",
    # plot options
    cmap = green_yellow_red,
    title: str = "Duplicate fingerprints plot",
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[plt.Axes] = None,
    sort_by_total: bool = True,
) -> Tuple[plt.Figure, plt.Axes, List[DuplicateBinResult]]:
    """Compute binned duplicate stats per experiment and plot them.

    Parameters
    ----------
    experiments:
        Mapping experiment_name -> dict-like with:
          - name: duplicates groups (from find_duplicates_with_hashing)
    bins, unit
        Passed to `compute_duplicate_bin_counts`.
    figsize, sort_by_total, cmap:
        Passed to `plot_duplicate_bins`.
    """
    results: List[DuplicateBinResult] = []
    for name, duplicates in experiments.items():
        res = compute_duplicate_bin_counts(
            duplicates,
            masses_arr,
            bins=bins,
            unit=unit,
            name=str(name),
        )
        results.append(res)

    fig, ax = plot_duplicate_bins(
        results,
        figsize=figsize,
        cmap=cmap,
        sort_by_total=sort_by_total,
        title=title,
        ax=ax,
    )
    return fig, ax, results
