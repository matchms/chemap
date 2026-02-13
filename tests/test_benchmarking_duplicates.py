
import matplotlib
import numpy as np
import pytest


matplotlib.use("Agg")  # headless backend for CI
from matplotlib import pyplot as plt
from chemap.benchmarking import (  # utility.py
    compute_compound_max_mass_differences,
    compute_duplicate_max_mass_differences,
)
from chemap.plotting.benchmark_duplicates import (  # benchmark_duplicates.py
    DuplicateBinResult,
    compute_duplicate_bin_counts,
    default_bins_da,
    plot_duplicate_bins,
    plot_duplicates_by_experiment,
)


# ---------------------------------------------------------------------
# utility.py tests
# ---------------------------------------------------------------------

def test_compute_compound_max_mass_differences_simple():
    masses = np.array([100.0, 105.0, 110.0])
    # min=100, max=110
    # for 100: max(0, 10)=10
    # for 105: max(5, 5)=5
    # for 110: max(10, 0)=10
    out = compute_compound_max_mass_differences(masses)
    assert np.allclose(out, np.array([10.0, 5.0, 10.0]))


def test_compute_compound_max_mass_differences_all_equal():
    masses = np.array([42.0, 42.0, 42.0])
    out = compute_compound_max_mass_differences(masses)
    assert np.allclose(out, np.array([0.0, 0.0, 0.0]))


def test_compute_duplicate_max_mass_differences_multiple_groups_concatenates():
    masses = np.array([10.0, 11.0, 30.0, 31.0, 32.0])

    duplicates = [
        [0, 1],       # masses [10, 11] -> max diffs: [1, 1]
        [2, 3, 4],    # masses [30,31,32] -> max diffs: [2,1,2]
    ]
    out = compute_duplicate_max_mass_differences(duplicates, masses)
    assert np.allclose(out, np.array([1.0, 1.0, 2.0, 1.0, 2.0]))


def test_compute_duplicate_max_mass_differences_empty_duplicates_returns_empty():
    masses = np.array([1.0, 2.0, 3.0])
    out = compute_duplicate_max_mass_differences([], masses)
    assert out.shape == (0,)
    assert out.dtype == float


def test_compute_duplicate_max_mass_differences_raises_on_out_of_range_index():
    masses = np.array([1.0, 2.0, 3.0])
    duplicates = [[0, 3]]  # 3 is out of range
    with pytest.raises(IndexError):
        _ = compute_duplicate_max_mass_differences(duplicates, masses)


# ---------------------------------------------------------------------
# benchmark_duplicates.py tests
# ---------------------------------------------------------------------

def test_compute_duplicate_bin_counts_basic_binning():
    # We craft duplicates + masses so that the produced max diffs are known.
    # masses indices: 0..4
    masses = np.array([100.0, 101.0, 200.0, 205.0, 210.0])

    duplicates = [
        [0, 1],        # masses [100,101] -> per-item max diffs [1,1]
        [2, 3, 4],     # masses [200,205,210] -> max diffs [10,5,10]
    ]
    # Combined max diffs: [1,1,10,5,10]
    bins = [(0, 1), (1, 10), (10, 50), (50, np.inf)]
    res = compute_duplicate_bin_counts(
        duplicates,
        masses,
        bins=bins,
        unit="Da",
        name="exp1",
    )

    assert isinstance(res, DuplicateBinResult)
    assert res.name == "exp1"
    assert res.bin_edges == bins
    assert res.bin_labels == ["0-1 Da", "1-10 Da", "10-50 Da", "50-inf Da"]

    # Bin membership with [low, high):
    # [0,1): none
    # [1,10): values 1,1,5 -> 3
    # [10,50): values 10,10 -> 2
    # [50,inf): none
    assert np.array_equal(res.bin_counts, np.array([0, 3, 2, 0], dtype=int))
    assert res.total == 5


def test_compute_duplicate_bin_counts_empty_maxdiffs_all_zero_counts():
    masses = np.array([100.0, 101.0, 102.0])
    duplicates: list[list[int]] = []
    bins = default_bins_da()
    res = compute_duplicate_bin_counts(duplicates, masses, bins=bins, name="empty")

    assert res.total == 0
    assert np.array_equal(res.bin_counts, np.zeros(len(bins), dtype=int))


def test_compute_duplicate_bin_counts_raises_if_masses_not_1d():
    masses = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        _ = compute_duplicate_bin_counts([[0, 1]], masses)


def test_plot_duplicate_bins_requires_nonempty():
    with pytest.raises(ValueError):
        _ = plot_duplicate_bins([])


def test_plot_duplicate_bins_raises_on_inconsistent_bin_labels():
    bins = [(0, 1), (1, np.inf)]
    r1 = DuplicateBinResult(
        name="a",
        bin_edges=bins,
        bin_labels=["0-1 Da", "1-inf Da"],
        bin_counts=np.array([1, 2]),
        total=3,
    )
    r2 = DuplicateBinResult(
        name="b",
        bin_edges=bins,
        bin_labels=["0-1 Da", "WRONG"],  # mismatch
        bin_counts=np.array([0, 1]),
        total=1,
    )
    with pytest.raises(ValueError):
        _ = plot_duplicate_bins([r1, r2])


def test_plot_duplicate_bins_raises_on_inconsistent_bin_counts_length():
    bins = [(0, 1), (1, np.inf)]
    r1 = DuplicateBinResult(
        name="a",
        bin_edges=bins,
        bin_labels=["0-1 Da", "1-inf Da"],
        bin_counts=np.array([1, 2]),
        total=3,
    )
    r2 = DuplicateBinResult(
        name="b",
        bin_edges=bins,
        bin_labels=["0-1 Da", "1-inf Da"],
        bin_counts=np.array([1, 2, 3]),  # length mismatch
        total=6,
    )
    with pytest.raises(ValueError):
        _ = plot_duplicate_bins([r1, r2])


def test_plot_duplicate_bins_sorts_by_total_desc_and_creates_axes():
    bins = [(0, 1), (1, np.inf)]
    r_low = DuplicateBinResult(
        name="low",
        bin_edges=bins,
        bin_labels=["0-1 Da", "1-inf Da"],
        bin_counts=np.array([0, 1]),
        total=1,
    )
    r_high = DuplicateBinResult(
        name="high",
        bin_edges=bins,
        bin_labels=["0-1 Da", "1-inf Da"],
        bin_counts=np.array([5, 5]),
        total=10,
    )

    fig, ax = plot_duplicate_bins([r_low, r_high], sort_by_total=True)

    # Y tick labels should be ["high", "low"] after sorting
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["high", "low"]

    # Smoke-check: there should be 2 bins * 2 experiments = 4 bar patches
    assert len(ax.patches) == 4

    # Cleanup
    plt.close(fig)


def test_plot_duplicates_by_experiment_happy_path():
    masses = np.array([100.0, 101.0, 200.0, 205.0, 210.0])
    duplicates_a = [[0, 1]]          # max diffs [1,1]
    duplicates_b = [[2, 3, 4]]       # max diffs [10,5,10]

    experiments = {
        "A": {"duplicates": duplicates_a, "masses": masses},
        "B": {"duplicates": duplicates_b, "masses": masses},
    }

    fig, ax, results = plot_duplicates_by_experiment(
        experiments,
        bins=[(0, 1), (1, 10), (10, np.inf)],
        sort_by_total=True,
    )

    # Should have one result per experiment
    assert {r.name for r in results} == {"A", "B"}

    # Totals: A => 2, B => 3
    totals = {r.name: r.total for r in results}
    assert totals["A"] == 2
    assert totals["B"] == 3

    plt.close(fig)


def test_plot_duplicates_by_experiment_missing_keys_raises():
    experiments = {"A": {"duplicates": [[0, 1]]}}  # missing masses
    with pytest.raises(KeyError):
        _ = plot_duplicates_by_experiment(experiments)
