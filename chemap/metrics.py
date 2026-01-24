from typing import Callable, Literal, Optional, Sequence, Tuple, Union, overload
import numba
import numpy as np
import scipy.sparse as sp
from numba import prange
from sklearn.metrics import pairwise_distances


# ---------------------------
# Terminology / Types
# ---------------------------

# Unfolded inputs
UnfoldedBinary = np.ndarray
UnfoldedCount = Tuple[np.ndarray, np.ndarray]
UnfoldedFingerprint = Union[UnfoldedBinary, UnfoldedCount]

# Dense / sparse fixed-size
DenseVector = np.ndarray
DenseMatrix = np.ndarray
SparseMatrix = sp.csr_matrix


# ---------------------------
# Generalized Jaccard/Tanimoto
# ---------------------------

# "Generalized" means for nonnegative values:
# sim = sum(min(x,y)) / sum(max(x,y))
# For binary vectors this equals standard Jaccard.


# ---- Dense (single pair) ----

@numba.njit(cache=True, fastmath=True)
def tanimoto_similarity_dense(a: np.ndarray, b: np.ndarray) -> float:
    """
    Generalized Jaccard/Tanimoto similarity for dense 1D vectors (nonnegative values).
    Works for binary or count/weight vectors.

    sim = sum(min(a,b)) / sum(max(a,b))

    Parameters
    ----------
    a
        1D numpy array (vector).
    b
        1D numpy array (vector).
    """
    min_sum = 0.0
    max_sum = 0.0
    for i in range(a.shape[0]):
        ai = a[i]
        bi = b[i]
        if ai < bi:
            min_sum += ai
            max_sum += bi
        else:
            min_sum += bi
            max_sum += ai
    if max_sum == 0.0:
        return 1.0
    return min_sum / max_sum


@numba.njit(cache=True, fastmath=True)
def tanimoto_distance_dense(a: np.ndarray, b: np.ndarray) -> float:
    """Distance = 1 - similarity."""
    return 1.0 - tanimoto_similarity_dense(a, b)


# ---- Unfolded (single pair) ----
# Unfolded requires aligned bits. This implementation merges two sorted bit lists.
# For binary unfolded: values are implicitly 1.0.

@numba.njit(cache=True, fastmath=True)
def tanimoto_similarity_unfolded_binary(bits1: np.ndarray, bits2: np.ndarray) -> float:
    """
    Binary unfolded similarity: Jaccard on sorted unique bit arrays.

    Parameters
    ----------
    bits1
        1D numpy array of sorted bit indices (unique).
    bits2
        1D numpy array of sorted bit indices (unique).
    """
    i = 0
    j = 0
    inter = 0
    n1 = bits1.shape[0]
    n2 = bits2.shape[0]
    while i < n1 and j < n2:
        b1 = bits1[i]
        b2 = bits2[j]
        if b1 == b2:
            inter += 1
            i += 1
            j += 1
        elif b1 < b2:
            i += 1
        else:
            j += 1
    union = n1 + n2 - inter
    if union == 0:
        return 1.0
    return inter / union


@numba.njit(cache=True, fastmath=True)
def tanimoto_similarity_unfolded_count(
        bits1: np.ndarray, vals1: np.ndarray,
        bits2: np.ndarray, vals2: np.ndarray
        ) -> float:
    """
    Count/weight unfolded similarity between two sparse vectors given as sorted (bits, values).
    sim = sum(min)/sum(max)

    Parameters
    ----------
    bits1
        1D numpy array of sorted bit indices (unique) for vector 1.
    vals1
        1D numpy array of counts for vector 1.
    bits2
        1D numpy array of sorted bit indices (unique) for vector 2.
    vals2
        1D numpy array of counts for vector 2.
    """
    i = 0
    j = 0
    n1 = bits1.shape[0]
    n2 = bits2.shape[0]
    min_sum = 0.0
    max_sum = 0.0

    while i < n1 and j < n2:
        b1 = bits1[i]
        b2 = bits2[j]
        if b1 == b2:
            v1 = vals1[i]
            v2 = vals2[j]
            if v1 < v2:
                min_sum += v1
                max_sum += v2
            else:
                min_sum += v2
                max_sum += v1
            i += 1
            j += 1
        elif b1 < b2:
            max_sum += vals1[i]
            i += 1
        else:
            max_sum += vals2[j]
            j += 1

    while i < n1:
        max_sum += vals1[i]
        i += 1
    while j < n2:
        max_sum += vals2[j]
        j += 1

    if max_sum == 0.0:
        return 1.0
    return min_sum / max_sum


@numba.njit(cache=True, fastmath=True)
def tanimoto_distance_unfolded_count(
        bits1: np.ndarray, vals1: np.ndarray,
        bits2: np.ndarray, vals2: np.ndarray
        ) -> float:
    return 1.0 - tanimoto_similarity_unfolded_count(bits1, vals1, bits2, vals2)


@numba.njit(cache=True, fastmath=True)
def tanimoto_distance_unfolded_binary(bits1: np.ndarray, bits2: np.ndarray) -> float:
    return 1.0 - tanimoto_similarity_unfolded_binary(bits1, bits2)



@numba.njit
def generalized_tanimoto_similarity(A, B):
    """
    Calculate the generalized Tanimoto similarity between two count vectors.
    
    Parameters:
    A (array-like): First count vector.
    B (array-like): Second count vector.
    
    Returns:
    float: Tanimoto similarity.
    """
    
    min_sum = np.sum(np.minimum(A, B))
    max_sum = np.sum(np.maximum(A, B))
    
    return min_sum / max_sum


def generalized_tanimoto_similarity_matrix_sparse_all_vs_all(fingerprints) -> np.ndarray:
    """
    Calculate the generalized Tanimoto similarity between all sparse fingerprints.
    """
    mapping = occupied_bit_mapping(fingerprints)
    X = sparse_fingerprint_to_csr(fingerprints, mapping)

    # Precompute L1 norms.
    norms = np.array(X.sum(axis=1)).ravel()

    # Compute pairwise Manhattan distances.
    manhattan = pairwise_distances(X, metric='manhattan')

    return compute_generalized_tanimoto_from_manhattan_symmetric(norms, manhattan)


def generalized_tanimoto_similarity_matrix_sparse(fingerprints_1, fingerprints_2) -> np.ndarray:
    """
    Calculate the generalized Tanimoto similarity between sparse fingerprints_1 and fingerprints_2.
    """
    mapping = occupied_bit_mapping(fingerprints_1 + fingerprints_2)
    X1 = sparse_fingerprint_to_csr(fingerprints_1, mapping)
    X2 = sparse_fingerprint_to_csr(fingerprints_2, mapping)

    # Precompute L1 norms.
    norms1 = np.array(X1.sum(axis=1)).ravel()
    norms2 = np.array(X2.sum(axis=1)).ravel()

    # Compute pairwise Manhattan distances.
    manhattan = pairwise_distances(X1, X2, metric='manhattan')

    return compute_generalized_tanimoto_from_manhattan(norms1, norms2, manhattan)


def occupied_bit_mapping(fingerprints_sparse):
    # Collect all unique keys.
    all_keys = set()
    for keys, _ in fingerprints_sparse:
        all_keys.update(keys.tolist())

    # Create a mapping from original key to a new, contiguous index.
    sorted_keys = sorted(all_keys)
    return {old_key: new_key for new_key, old_key in enumerate(sorted_keys)}


def sparse_fingerprint_to_csr(fingerprints_sparse, mapping):
    # Build lists for constructing the sparse matrix.
    rows = []
    cols = []
    vals = []

    for i, (keys, values) in enumerate(fingerprints_sparse):
        new_keys = np.array([mapping[k] for k in keys], dtype=np.int32)
        rows.extend([i] * len(new_keys))
        cols.extend(new_keys.tolist())
        vals.extend(values.tolist())

    num_rows = len(fingerprints_sparse)
    num_cols = len(mapping)  # number of occupied features

    # Build the COO matrix and convert to CSR.
    X = sp.coo_matrix((vals, (rows, cols)), shape=(num_rows, num_cols), dtype=np.float32)
    return X.tocsr()


@numba.njit(parallel=True, fastmath=True)
def compute_generalized_tanimoto_from_manhattan_symmetric(norms, manhattan):
    n = norms.shape[0]
    tanimoto = np.empty((n, n), dtype=norms.dtype)
    for i in numba.prange(n):
        for j in range(n):
            union = norms[i] + norms[j] + manhattan[i, j]
            if union > 0:
                tanimoto[i, j] = (norms[i] + norms[j] - manhattan[i, j]) / union
            else:
                tanimoto[i, j] = 1.0
    return tanimoto


@numba.njit(parallel=True, fastmath=True)
def compute_generalized_tanimoto_from_manhattan(norms1, norms2, manhattan):
    n = norms1.shape[0]
    m = norms2.shape[0]
    tanimoto = np.empty((n, m), dtype=norms1.dtype)
    for i in numba.prange(n):
        for j in range(m):
            union = norms1[i] + norms2[j] + manhattan[i, j]
            if union > 0:
                tanimoto[i, j] = (norms1[i] + norms2[j] - manhattan[i, j]) / union
            else:
                tanimoto[i, j] = 1.0
    return tanimoto


@numba.njit
def generalized_tanimoto_similarity_sparse_numba(keys1, values1, keys2, values2) -> float:
    """
    Calculate the generalized Tanimoto similarity between two sparse count vectors.

    Parameters:
    keys1, values1 (array-like): Keys and values for the first sparse vector.
    keys2, values2 (array-like): Keys and values for the second sparse vector.
    """
    i, j = 0, 0
    min_sum, max_sum = 0.0, 0.0

    while i < len(keys1) and j < len(keys2):
        if keys1[i] == keys2[j]:
            min_sum += min(values1[i], values2[j])
            max_sum += max(values1[i], values2[j])
            i += 1
            j += 1
        elif keys1[i] < keys2[j]:
            max_sum += values1[i]
            i += 1
        else:
            max_sum += values2[j]
            j += 1

    # Add remaining values from both vectors
    while i < len(keys1):
        max_sum += values1[i]
        i += 1

    while j < len(keys2):
        max_sum += values2[j]
        j += 1

    return min_sum / max_sum


@numba.jit(nopython=True, fastmath=True, parallel=True)
def generalized_tanimoto_similarity_matrix(references: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Returns matrix of generalized Tanimoto similarity between all-vs-all vectors
    of references and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    assert references.shape[1] == queries.shape[1], "Vector sizes do not match!"

    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = np.zeros((size1, size2)) #, dtype=np.float32)
    for i in prange(size1):
        for j in range(size2):
            scores[i, j] = generalized_tanimoto_similarity(references[i, :], queries[j, :])
    return scores


@numba.jit(nopython=True, fastmath=True, parallel=True)
def generalized_tanimoto_similarity_matrix_sparse_numba(
    references: list, queries: list) -> np.ndarray:
    """Returns matrix of generalized Tanimoto similarity between all-vs-all vectors of references and queries.

    Parameters
    ----------
    references:
        List of sparse fingerprints (tuple of two arrays: keys and counts).
    queries
        List of sparse fingerprints (tuple of two arrays: keys and counts).

    Returns
    -------
    scores:
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = len(references)
    size2 = len(queries)
    scores = np.zeros((size1, size2))
    for i in prange(size1):
        for j in range(size2):
            scores[i, j] = generalized_tanimoto_similarity_sparse_numba(
                references[i][0], references[i][1],
                queries[j][0], queries[j][1])
    return scores


@numba.njit
def generalized_tanimoto_similarity_weighted(A, B, weights):
    """
    Calculate the weighted generarlized Tanimoto similarity between two count vectors.
    
    Parameters:
    ----------
        A (array-like): First count vector.
        B (array-like): Second count vector.
        weights: weights for every vector bit
    
    Returns:
    float: Tanimoto similarity.
    """
    
    min_sum = np.sum(np.minimum(A, B) * weights)
    max_sum = np.sum(np.maximum(A, B) * weights)
    
    return min_sum / max_sum


@numba.jit(nopython=True, fastmath=True, parallel=True)
def generalized_tanimoto_similarity_matrix_weighted(
        references: np.ndarray,
        queries: np.ndarray,
        weights: np.ndarray
        ) -> np.ndarray:
    """Returns matrix of generalized Tanimoto similarity between all-vs-all vectors of references and queries.

    Parameters
    ----------
    references
        Reference vectors as 2D numpy array. Expects that vector_i corresponds to
        references[i, :].
    queries
        Query vectors as 2D numpy array. Expects that vector_i corresponds to
        queries[i, :].

    Returns
    -------
    scores
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = references.shape[0]
    size2 = queries.shape[0]
    scores = np.zeros((size1, size2)) #, dtype=np.float32)
    for i in prange(size1):
        for j in range(size2):
            scores[i, j] = generalized_tanimoto_similarity_weighted(references[i, :], queries[j, :], weights)
    return scores
