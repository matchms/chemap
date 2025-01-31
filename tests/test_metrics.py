import pytest
from metrics import jaccard_index_sparse, jaccard_similarity_matrix


def test_jaccard_sparse_basic():
    assert jaccard_index_sparse(np.array([1, 2, 3]), np.array([2, 3, 4])) == pytest.approx(2/4)
    assert jaccard_index_sparse(np.array([1, 3, 5]), np.array([2, 4, 6])) == 0.0
    assert jaccard_index_sparse(np.array([]), np.array([])) == 0.0
    assert jaccard_index_sparse(np.array([]), np.array([1, 2, 3])) == 0.0
    assert jaccard_index_sparse(np.array([1, 2, 3]), np.array([])) == 0.0
    assert jaccard_index_sparse(np.array([1, 2, 3]), np.array([1, 2, 3])) == 1.0


def test_jaccard_implementation_consistency():
    """Test if sparse and dense implementation give the same results."""
    binary_vectors = np.array([
        [1, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ], dtype=np.int32)

    jaccard_dense = jaccard_similarity_matrix(binary_vectors, binary_vectors)

    for i in range(len(binary_vectors)):
        for j in range(len(binary_vectors)):
            sparse1 = np.where(binary_vectors[i] == 1)[0]
            sparse2 = np.where(binary_vectors[j] == 1)[0]
            jaccard_sparse = jaccard_index_sparse(sparse1, sparse2)
            assert jaccard_sparse == pytest.approx(jaccard_dense[i, j])
