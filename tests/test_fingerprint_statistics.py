import numpy as np
from chemap.fingerprint_statistics import (
    unfolded_count_fingerprint_bit_statistics,
    unfolded_fingerprint_bit_statistics,
)


def _as_index_fp(*bits: int) -> np.ndarray:
    """Helper: build one indices-only fingerprint."""
    return np.array(bits, dtype=np.int64)


# =============================================================================
# unfolded_fingerprint_bit_statistics (numba njit)
# =============================================================================

def test_sparse_bit_statistics_empty_input():
    keys, counts, first = unfolded_fingerprint_bit_statistics([])
    assert keys.dtype == np.int64
    assert counts.dtype == np.int32
    assert first.dtype == np.int32
    assert keys.size == 0
    assert counts.size == 0
    assert first.size == 0


def test_sparse_bit_statistics_single_fp_unique_bits_sorted_and_first_instance_zero():
    fps = [_as_index_fp(10, 3, 7)]
    keys, counts, first = unfolded_fingerprint_bit_statistics(fps)

    np.testing.assert_array_equal(keys, np.array([3, 7, 10], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([1, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(first, np.array([0, 0, 0], dtype=np.int32))


def test_sparse_bit_statistics_multiple_fps_counts_and_first_instance():
    fps = [
        _as_index_fp(3, 7, 10),
        _as_index_fp(7, 10),
        _as_index_fp(10, 42),
    ]
    keys, counts, first = unfolded_fingerprint_bit_statistics(fps)

    # Keys must be sorted ascending
    np.testing.assert_array_equal(keys, np.array([3, 7, 10, 42], dtype=np.int64))

    # counts: 3->1, 7->2, 10->3, 42->1
    np.testing.assert_array_equal(counts, np.array([1, 2, 3, 1], dtype=np.int32))

    # first instance indices: 3->0, 7->0, 10->0, 42->2
    np.testing.assert_array_equal(first, np.array([0, 0, 0, 2], dtype=np.int32))


def test_sparse_bit_statistics_duplicate_bits_within_one_fp_are_counted_multiple_times():
    # NOTE: current implementation counts every occurrence (no per-fingerprint dedup).
    fps = [
        _as_index_fp(5, 5, 5),
        _as_index_fp(5),
    ]
    keys, counts, first = unfolded_fingerprint_bit_statistics(fps)

    np.testing.assert_array_equal(keys, np.array([5], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([4], dtype=np.int32))
    np.testing.assert_array_equal(first, np.array([0], dtype=np.int32))


def test_sparse_bit_statistics_handles_empty_fingerprints_in_list():
    fps = [
        _as_index_fp(),          # empty fp at index 0
        _as_index_fp(1, 2),      # index 1
        _as_index_fp(),          # empty fp at index 2
        _as_index_fp(2, 3, 2),   # index 3 (note duplicate 2)
    ]
    keys, counts, first = unfolded_fingerprint_bit_statistics(fps)

    np.testing.assert_array_equal(keys, np.array([1, 2, 3], dtype=np.int64))
    # counts: 1->1, 2->(1 from fp1 + 2 from fp3) = 3, 3->1
    np.testing.assert_array_equal(counts, np.array([1, 3, 1], dtype=np.int32))
    # first: 1->1, 2->1, 3->3
    np.testing.assert_array_equal(first, np.array([1, 1, 3], dtype=np.int32))


# =============================================================================
# unfolded_count_fingerprint_bit_statistics (python wrapper)
# =============================================================================

def test_unfolded_count_bit_statistics_uses_only_keys_ignores_counts():
    # fingerprints are (keys, values). Only keys should be used.
    fps = [
        (np.array([2, 5], dtype=np.int64), np.array([100.0, 1.0], dtype=np.float32)),
        (np.array([5], dtype=np.int64), np.array([999.0], dtype=np.float32)),
        (np.array([2], dtype=np.int64), np.array([0.0], dtype=np.float32)),
    ]

    keys, counts, first = unfolded_count_fingerprint_bit_statistics(fps)

    np.testing.assert_array_equal(keys, np.array([2, 5], dtype=np.int64))
    np.testing.assert_array_equal(counts, np.array([2, 2], dtype=np.int32))
    np.testing.assert_array_equal(first, np.array([0, 0], dtype=np.int32))
