import numpy as np
import pytest
from chemap.metrics import (
    generalized_tanimoto_similarity,
    generalized_tanimoto_similarity_matrix,
    generalized_tanimoto_similarity_matrix_sparse,
    generalized_tanimoto_similarity_matrix_sparse_all_vs_all,
    generalized_tanimoto_similarity_sparse_numba,
    occupied_bit_mapping,
    sparse_fingerprint_to_csr,
)


# ----------------------------------------
# Tests for generalized_tanimoto_similarity (dense)
# ----------------------------------------

def test_generalized_tanimoto_similarity_simple():
    # A = [1, 2, 3], B = [2, 1, 0]
    # min_sum = 1 + 1 + 0 = 2
    # max_sum = 2 + 2 + 3 = 7
    # expected = 2 / 7
    A = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    B = np.array([2.0, 1.0, 0.0], dtype=np.float64)
    sim = generalized_tanimoto_similarity(A, B)
    assert pytest.approx(sim, rel=1e-8) == 2.0 / 7.0

    # identical vectors should give 1.0
    v = np.array([0.0, 5.0, 2.0, 7.0], dtype=np.float64)
    assert generalized_tanimoto_similarity(v, v) == pytest.approx(1.0)

    # one zero vector and one non-zero: similarity = 0
    zero = np.zeros(4, dtype=np.float64)
    nonzero = np.array([1.0, 0.0, 2.0, 3.0], dtype=np.float64)
    assert generalized_tanimoto_similarity(zero, nonzero) == pytest.approx(0.0)


def test_generalized_tanimoto_distance_dense_mismatched_shapes():
    # A has shape (10, 3), B has shape (5, 4) → columns do not match
    A = np.random.random((10, 3))
    B = np.random.random((5, 4))

    with pytest.raises(AssertionError) as excinfo:
        _ = generalized_tanimoto_similarity_matrix(A, B)

    msg = str(excinfo.value)
    assert "Vector sizes do not match" in msg


def test_generalized_tanimoto_implementation_consistency():
    """Test if sparse and dense implementation give the same results."""
    binary_vectors = np.array([
        [1, 4, 0, 0, 1, 0, 0, 1],
        [0, 3, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
    ], dtype=np.int32)

    generalized_tanimoto_dense = generalized_tanimoto_similarity_matrix(binary_vectors, binary_vectors)

    for i in range(len(binary_vectors)):
        for j in range(len(binary_vectors)):
            keys1 = np.where(binary_vectors[i] >= 1)[0]
            values1 = binary_vectors[i][keys1]
            keys2 = np.where(binary_vectors[j] >= 1)[0]
            values2 = binary_vectors[j][keys2]
            
            generalized_tanimoto_score = generalized_tanimoto_similarity(binary_vectors[i], binary_vectors[j])
            assert generalized_tanimoto_score == pytest.approx(generalized_tanimoto_dense[i, j])

            generalized_tanimoto_sparse = generalized_tanimoto_similarity_sparse_numba(keys1, values1, keys2, values2)
            assert generalized_tanimoto_sparse == pytest.approx(generalized_tanimoto_dense[i, j])


# ---------------------------------------------------
# Tests for generalized_tanimoto_similarity_sparse_numba (sparse)
# ---------------------------------------------------

@pytest.mark.parametrize("keys1, vals1, keys2, vals2, expected", [
    # Case 1: overlapping keys and non-overlapping keys
    (
        np.array([0, 1, 2], dtype=np.int64),
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([1, 2, 3], dtype=np.int64),
        np.array([2.0, 1.0, 1.0], dtype=np.float64),
        3.0 / 7.0  # computed by hand (see analysis)
    ),
    # Case 2: completely identical sparse vectors
    (
        np.array([5, 10], dtype=np.int64),
        np.array([4.0, 6.0], dtype=np.float64),
        np.array([5, 10], dtype=np.int64),
        np.array([4.0, 6.0], dtype=np.float64),
        1.0
    ),
    # Case 3: no overlap
    (
        np.array([0, 2], dtype=np.int64),
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([1, 3], dtype=np.int64),
        np.array([1.0, 1.0], dtype=np.float64),
        0.0
    ),
])
def test_generalized_tanimoto_similarity_sparse_numba(keys1, vals1, keys2, vals2, expected):
    sim = generalized_tanimoto_similarity_sparse_numba(keys1, vals1, keys2, vals2)
    assert pytest.approx(sim, rel=1e-8) == expected


# -------------------------------------------------------
# Tests for occupied_bit_mapping and sparse_fingerprint_to_csr
# -------------------------------------------------------

def test_occupied_bit_mapping_and_csr_conversion():
    # Create two tiny sparse fingerprints
    # fingerprint A has keys [2, 5] with values [1.0, 1.0]
    # fingerprint B has keys [3]    with values [2.0]
    fp_A = (np.array([2, 5], dtype=np.int64), np.array([1.0, 1.0], dtype=np.float64))
    fp_B = (np.array([3], dtype=np.int64), np.array([2.0], dtype=np.float64))
    fingerprints = [fp_A, fp_B]

    # occupied_bit_mapping should collect keys {2,3,5} and sort → [2, 3, 5]
    mapping = occupied_bit_mapping(fingerprints)
    assert mapping == {2: 0, 3: 1, 5: 2}

    # Now test sparse_fingerprint_to_csr
    X = sparse_fingerprint_to_csr(fingerprints, mapping)
    # Expect a 2×3 CSR matrix:
    # Row 0 (fp_A): at column index mapping[2]=0 → 1.0, mapping[5]=2 → 1.0
    # Row 1 (fp_B): at column index mapping[3]=1 → 2.0
    dense = X.toarray()
    expected_dense = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 2.0, 0.0]
    ], dtype=np.float32)
    np.testing.assert_allclose(dense, expected_dense)


# ---------------------------------------------------
# Tests for generalized_tanimoto_similarity_matrix_sparse_all_vs_all
# ---------------------------------------------------

def test_generalized_tanimoto_similarity_matrix_sparse_all_vs_all_two_vectors():
    # fingerprint 1: keys [0,1] with values [1,2] → dense [1,2,0]
    # fingerprint 2: keys [1,2] with values [2,3] → dense [0,2,3]
    fp1 = (np.array([0, 1], dtype=np.int64), np.array([1.0, 2.0], dtype=np.float64))
    fp2 = (np.array([1, 2], dtype=np.int64), np.array([2.0, 3.0], dtype=np.float64))
    fingerprints = [fp1, fp2]

    # Compute the 2×2 generalized_tanimoto matrix.
    S = generalized_tanimoto_similarity_matrix_sparse_all_vs_all(fingerprints)

    # Diagonals must be 1.0
    assert S.shape == (2, 2)
    assert S[0, 0] == pytest.approx(1.0)
    assert S[1, 1] == pytest.approx(1.0)

    # Off‐diagonal: 
    # v1 = [1,2,0], v2 = [0,2,3]
    # min_sum = 0 + 2 + 0 = 2
    # max_sum = 1 + 2 + 3 = 6 → 2/6 = 1/3
    expected_off = 1.0 / 3.0
    assert S[0, 1] == pytest.approx(expected_off)
    assert S[1, 0] == pytest.approx(expected_off)


# ---------------------------------------------------
# Tests for generalized_tanimoto_similarity_matrix_sparse (cross‐set)
# ---------------------------------------------------

def test_generalized_tanimoto_similarity_matrix_sparse_cross():
    # Using the same two fingerprints as above,
    # but treat them as two different sets.
    fp1 = (np.array([0, 1], dtype=np.int64), np.array([1.0, 2.0], dtype=np.float64))
    fp2 = (np.array([1, 2], dtype=np.int64), np.array([2.0, 3.0], dtype=np.float64))

    X1 = [fp1]
    X2 = [fp2]
    S_cross = generalized_tanimoto_similarity_matrix_sparse(X1, X2)

    # Should be 1×1 matrix whose single entry is same 1/3
    assert S_cross.shape == (1, 1)
    assert S_cross[0, 0] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------
# Test consistency between dense and sparse‐numba versions
# ---------------------------------------------------

def test_consistency_dense_and_sparse_numba():
    # Pick a random non‐negative integer vector of length 5
    rng = np.random.default_rng(42)
    A = rng.integers(low=0, high=5, size=5).astype(np.float64)
    B = rng.integers(low=0, high=5, size=5).astype(np.float64)

    # Compute dense version
    dens_sim = generalized_tanimoto_similarity(A, B)

    # Convert to sparse format: only keep indices where values > 0
    keys1 = np.nonzero(A)[0].astype(np.int64)
    vals1 = A[keys1]
    keys2 = np.nonzero(B)[0].astype(np.int64)
    vals2 = B[keys2]

    sparse_sim = generalized_tanimoto_similarity_sparse_numba(keys1, vals1, keys2, vals2)
    assert pytest.approx(dens_sim, rel=1e-8) == sparse_sim
