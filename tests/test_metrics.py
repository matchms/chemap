import numpy as np
import pytest
import scipy.sparse as sp
from numba.typed import List as NumbaList
from chemap.metrics import (
    tanimoto_distance_dense,
    tanimoto_distance_sparse,
    tanimoto_distance_sparse_binary,
    tanimoto_similarity,
    tanimoto_similarity_dense,
    tanimoto_similarity_matrix,
    tanimoto_similarity_matrix_dense,
    tanimoto_similarity_matrix_sparse,
    tanimoto_similarity_matrix_sparse_binary,
    tanimoto_similarity_sparse,
    tanimoto_similarity_sparse_binary,
)


def to_numba_list(arrs):
    """Convert a Python list of numpy arrays to numba.typed.List."""
    out = NumbaList()
    for a in arrs:
        out.append(a)
    return out


# ----------------------------------------
# Dense: single-pair similarity / distance
# ----------------------------------------

def test_tanimoto_similarity_dense_simple():
    # A = [1, 2, 3], B = [2, 1, 0]
    # min_sum = 1 + 1 + 0 = 2
    # max_sum = 2 + 2 + 3 = 7
    A = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    B = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    sim = tanimoto_similarity_dense(A, B)
    assert pytest.approx(sim, rel=1e-8) == 2.0 / 7.0

    # identical vectors should give 1.0
    v = np.array([0.0, 5.0, 2.0, 7.0], dtype=np.float32)
    assert tanimoto_similarity_dense(v, v) == pytest.approx(1.0)

    # one zero vector and one non-zero: similarity = 0
    zero = np.zeros(4, dtype=np.float32)
    nonzero = np.array([1.0, 0.0, 2.0, 3.0], dtype=np.float32)
    assert tanimoto_similarity_dense(zero, nonzero) == pytest.approx(0.0)


def test_tanimoto_distance_dense_simple():
    A = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    B = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    dist = tanimoto_distance_dense(A, B)
    assert pytest.approx(dist, rel=1e-8) == 1.0 - (2.0 / 7.0)


# ----------------------------------------
# Dense: batch all-vs-all matrix
# ----------------------------------------

def test_tanimoto_similarity_matrix_dense_shape_and_consistency():
    X = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
        ],
        dtype=np.float32,
    )
    S = tanimoto_similarity_matrix_dense(X, X)
    assert S.shape == (2, 2)
    assert S[0, 0] == pytest.approx(1.0)
    assert S[1, 1] == pytest.approx(1.0)
    assert S[0, 1] == pytest.approx(tanimoto_similarity_dense(X[0], X[1]))
    assert S[1, 0] == pytest.approx(tanimoto_similarity_dense(X[1], X[0]))


# ----------------------------------------
# Unfolded: binary
# ----------------------------------------

def test_tanimoto_unfolded_binary_simple():
    # bits1 = {0,2,5}, bits2 = {2,3,5,7}
    # intersection = {2,5} => 2
    # union = {0,2,3,5,7} => 5
    bits1 = np.array([0, 2, 5], dtype=np.int64)
    bits2 = np.array([2, 3, 5, 7], dtype=np.int64)

    sim = tanimoto_similarity_sparse_binary(bits1, bits2)
    dist = tanimoto_distance_sparse_binary(bits1, bits2)

    assert sim == pytest.approx(2.0 / 5.0)
    assert dist == pytest.approx(1.0 - 2.0 / 5.0)


def test_tanimoto_unfolded_binary_zero_cases():
    empty = np.array([], dtype=np.int64)
    bits = np.array([1, 10], dtype=np.int64)
    # empty vs empty -> similarity 1.0 (both all-zero)
    assert tanimoto_similarity_sparse_binary(empty, empty) == pytest.approx(1.0)
    # empty vs non-empty -> similarity 0.0
    assert tanimoto_similarity_sparse_binary(empty, bits) == pytest.approx(0.0)


# ----------------------------------------
# Unfolded: count/weight
# ----------------------------------------

@pytest.mark.parametrize(
    "bits1, vals1, bits2, vals2, expected",
    [
        # Example: v1 = [1,2,3] on bits [0,1,2], v2 = [2,1,1] on bits [1,2,3]
        # min_sum = min(2,2)+min(3,1)=2+1=3
        # max_sum = 1 + max(2,2) + max(3,1) + 1 = 1 + 2 + 3 + 1 = 7
        (
            np.array([0, 1, 2], dtype=np.int64),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([1, 2, 3], dtype=np.int64),
            np.array([2.0, 1.0, 1.0], dtype=np.float32),
            3.0 / 7.0,
        ),
        # identical
        (
            np.array([5, 10], dtype=np.int64),
            np.array([4.0, 6.0], dtype=np.float32),
            np.array([5, 10], dtype=np.int64),
            np.array([4.0, 6.0], dtype=np.float32),
            1.0,
        ),
        # no overlap
        (
            np.array([0, 2], dtype=np.int64),
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([1, 3], dtype=np.int64),
            np.array([1.0, 1.0], dtype=np.float32),
            0.0,
        ),
    ],
)
def test_tanimoto_unfolded_count(bits1, vals1, bits2, vals2, expected):
    sim = tanimoto_similarity_sparse(bits1, vals1, bits2, vals2)
    dist = tanimoto_distance_sparse(bits1, vals1, bits2, vals2)
    assert pytest.approx(sim, rel=1e-7) == expected
    assert pytest.approx(dist, rel=1e-7) == 1.0 - expected


# ----------------------------------------
# Sparse fixed-size CSR: single pair
# ----------------------------------------

def _row_slice(X: sp.csr_matrix, i: int):
    a0 = X.indptr[i]
    a1 = X.indptr[i + 1]
    return X.indices[a0:a1], X.data[a0:a1]


def test_tanimoto_sparse_matches_dense_on_small_example():
    # Small dense vectors -> CSR -> compare
    A = np.array([1.0, 2.0, 0.0, 3.0], dtype=np.float32)
    B = np.array([2.0, 1.0, 1.0, 0.0], dtype=np.float32)

    dense_sim = tanimoto_similarity_dense(A, B)

    X = sp.csr_matrix(np.vstack([A, B]))
    X.sort_indices()

    ind1, dat1 = _row_slice(X, 0)
    ind2, dat2 = _row_slice(X, 1)

    sparse_sim = tanimoto_similarity_sparse(ind1, dat1, ind2, dat2)
    sparse_dist = tanimoto_distance_sparse(ind1, dat1, ind2, dat2)

    assert sparse_sim == pytest.approx(dense_sim, rel=1e-8)
    assert sparse_dist == pytest.approx(1.0 - dense_sim, rel=1e-8)


# ----------------------------------------
# Unfolded matrix functions
# ----------------------------------------

def test_tanimoto_similarity_matrix_sparse_binary_small():
    refs = to_numba_list([
        np.array([0, 2, 5], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
    ])
    qs = to_numba_list([
        np.array([2, 5], dtype=np.int64),
        np.array([3], dtype=np.int64),
    ])

    S = tanimoto_similarity_matrix_sparse_binary(refs, qs)
    assert S.shape == (2, 2)

    # (0,0): {0,2,5} vs {2,5} => inter=2 union=3 => 2/3
    assert S[0, 0] == pytest.approx(2.0 / 3.0)
    # (1,1): {1,2} vs {3} => 0
    assert S[1, 1] == pytest.approx(0.0)


def test_tanimoto_similarity_matrix_sparse_small():
    ref_bits = to_numba_list([np.array([0, 1], np.int64), np.array([1, 2], np.int64)])
    ref_vals = to_numba_list([np.array([1.0, 2.0], np.float32), np.array([2.0, 3.0], np.float32)])
    qry_bits = to_numba_list([np.array([1, 2], np.int64)])
    qry_vals = to_numba_list([np.array([2.0, 3.0], np.float32)])

    S = tanimoto_similarity_matrix_sparse(ref_bits, ref_vals, qry_bits, qry_vals)
    assert S.shape == (2, 1)

    # ref0 dense [1,2,0], qry dense [0,2,3]
    # min_sum = 2, max_sum = 1+2+3=6 => 1/3
    assert S[0, 0] == pytest.approx(1.0 / 3.0)
    # ref1 equals qry => 1.0
    assert S[1, 0] == pytest.approx(1.0)


# ----------------------------------------
# Unified wrappers
# ----------------------------------------

def test_unified_tanimoto_similarity_dense_wrapper():
    A = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    B = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    sim = tanimoto_similarity(A, B, kind="dense")
    assert sim == pytest.approx(2.0 / 7.0)


def test_unified_tanimoto_similarity_unfolded_binary_wrapper():
    bits1 = np.array([0, 2, 5], dtype=np.int64)
    bits2 = np.array([2, 3, 5, 7], dtype=np.int64)
    sim = tanimoto_similarity(bits1, bits2, kind="unfolded-binary")
    assert sim == pytest.approx(2.0 / 5.0)


def test_unified_tanimoto_similarity_unfolded_count_wrapper():
    fp1 = (np.array([0, 1], dtype=np.int64), np.array([1.0, 2.0], dtype=np.float32))
    fp2 = (np.array([1, 2], dtype=np.int64), np.array([2.0, 3.0], dtype=np.float32))
    sim = tanimoto_similarity(fp1, fp2, kind="unfolded-count")
    assert sim == pytest.approx(1.0 / 3.0)


def test_unified_tanimoto_similarity_sparse_wrapper():
    A = sp.csr_matrix(np.array([[1.0, 2.0, 0.0]], dtype=np.float32))
    B = sp.csr_matrix(np.array([[0.0, 2.0, 3.0]], dtype=np.float32))
    sim = tanimoto_similarity(A, B, kind="sparse")
    assert sim == pytest.approx(1.0 / 3.0)


def test_unified_tanimoto_similarity_matrix_dense_wrapper():
    X = np.array([[1.0, 0.0, 2.0],
                  [0.0, 1.0, 3.0]], dtype=np.float32)
    S = tanimoto_similarity_matrix(X, X, kind="dense")
    assert S.shape == (2, 2)
    assert S[0, 0] == pytest.approx(1.0)
    assert S[1, 1] == pytest.approx(1.0)


def test_unified_tanimoto_similarity_matrix_dense_wrapper_mismatched_shapes():
    A = np.random.random((10, 3)).astype(np.float32)
    B = np.random.random((5, 4)).astype(np.float32)
    with pytest.raises(ValueError):
        _ = tanimoto_similarity_matrix(A, B, kind="dense")
