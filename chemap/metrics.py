from typing import Literal, Optional, Tuple, Union
import numba
import numpy as np
import scipy.sparse as sp


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
def tanimoto_similarity_sparse_binary(bits1: np.ndarray, bits2: np.ndarray) -> float:
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
def tanimoto_distance_sparse_binary(bits1: np.ndarray, bits2: np.ndarray) -> float:
    return 1.0 - tanimoto_similarity_sparse_binary(bits1, bits2)


# ---- Sparse fixed-size CSR (single pair) ----
# Signature compatible with PyNNDescent / UMAP "sparse custom metric":
# metric(ind1, data1, ind2, data2) -> float
# where ind* are sorted column indices and data* are values.

@numba.njit(
    [
        "f4(i4[::1], f4[::1], i4[::1], f4[::1])",
        "f8(i4[::1], f8[::1], i4[::1], f8[::1])",
        "f4(i8[::1], f4[::1], i8[::1], f4[::1])",
        "f8(i8[::1], f8[::1], i8[::1], f8[::1])",
    ],
    cache=True,
    fastmath=True,
)
def tanimoto_distance_sparse(ind1, data1, ind2, data2) -> float:
    """
    Generalized Tanimoto distance for CSR row slices:
      dist = 1 - sum(min)/sum(max)

    Works for binary (data are ones) or count/weight (data are nonnegative).

    Parameters
    ----------
    ind1
        1D numpy array of sorted column indices for vector 1.   
    data1
        1D numpy array of values for vector 1.
    ind2
        1D numpy array of sorted column indices for vector 2.
    data2
        1D numpy array of values for vector 2.
    """
    i = 0
    j = 0
    n1 = ind1.shape[0]
    n2 = ind2.shape[0]
    min_sum = 0.0
    max_sum = 0.0

    while i < n1 and j < n2:
        c1 = ind1[i]
        c2 = ind2[j]
        if c1 == c2:
            v1 = data1[i]
            v2 = data2[j]
            if v1 < v2:
                min_sum += v1
                max_sum += v2
            else:
                min_sum += v2
                max_sum += v1
            i += 1
            j += 1
        elif c1 < c2:
            max_sum += data1[i]
            i += 1
        else:
            max_sum += data2[j]
            j += 1

    while i < n1:
        max_sum += data1[i]
        i += 1
    while j < n2:
        max_sum += data2[j]
        j += 1

    if max_sum == 0.0:
        return 0.0
    return 1.0 - (min_sum / max_sum)


@numba.njit(cache=True, fastmath=True)
def tanimoto_similarity_sparse(ind1, data1, ind2, data2) -> float:
    return 1.0 - tanimoto_distance_sparse(ind1, data1, ind2, data2)



# ---------------------------
# Pairwise matrices (batch)
# ---------------------------

# Dense batch: compute all-vs-all without sklearn, numba-parallel
@numba.njit(parallel=True, fastmath=True, cache=True)
def tanimoto_similarity_matrix_dense(references: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """
    Pairwise generalized Tanimoto similarity between two dense matrices.

    Parameters
    ----------
    references
        2D numpy array of shape (R, D).
    queries
        2D numpy array of shape (Q, D).
    """
    R = references.shape[0]
    Q = queries.shape[0]
    out = np.empty((R, Q), dtype=np.float32)
    for i in numba.prange(R):
        for j in range(Q):
            out[i, j] = tanimoto_similarity_dense(references[i], queries[j])
    return out


# Sparse batch: compute all-vs-all on CSR without densifying.
# This is O(R*Q*avg_nnz_merge) and can be expensive for large R,Q.
# For huge datasets prefer ANN (PyNNDescent/UMAP) with `tanimoto_distance_sparse`.
@numba.njit(parallel=True, fastmath=True, cache=True)
def tanimoto_similarity_matrix_sparse_binary(references, queries) -> np.ndarray:
    """
    Pairwise Tanimoto similarity between two sets of unfolded or sparse binary fingerprints.

    Parameters
    ----------
    references
        List of 1D numpy arrays of sorted bit indices (unique).
    queries
        List of 1D numpy arrays of sorted bit indices (unique).
    """
    R = len(references)
    Q = len(queries)
    out = np.empty((R, Q), dtype=np.float32)
    for i in numba.prange(R):
        for j in range(Q):
            out[i, j] = tanimoto_similarity_sparse_binary(references[i], queries[j])
    return out


@numba.njit(parallel=True, fastmath=True, cache=True)
def tanimoto_similarity_matrix_sparse(
        references_bits,
        references_vals,
        queries_bits,
        queries_vals
        ) -> np.ndarray:
    """
    Pairwise generalized Tanimoto similarity between two sets of unfolded count/weight fingerprints.

    Parameters
    ----------
    references_bits
        List of 1D numpy arrays of sorted bit indices (unique) for reference fingerprints.
    references_vals
        List of 1D numpy arrays of counts/weights for reference fingerprints.
    queries_bits
        List of 1D numpy arrays of sorted bit indices (unique) for query fingerprints.
    queries_vals
        List of 1D numpy arrays of counts/weights for query fingerprints.
    """
    R = len(references_bits)
    Q = len(queries_bits)
    out = np.empty((R, Q), dtype=np.float32)
    for i in numba.prange(R):
        for j in range(Q):
            out[i, j] = tanimoto_similarity_sparse(
                references_bits[i], references_vals[i],
                queries_bits[j], queries_vals[j],
            )
    return out


# ---------------------------
# High-level Python convenience wrappers
# ---------------------------

def _as_1xD_csr(x: Union[np.ndarray, sp.csr_matrix]) -> sp.csr_matrix:
    """Convert input to a 1xD csr_matrix."""
    if sp.isspmatrix_csr(x):
        if x.shape[0] == 1:
            return x
        if x.shape[0] != 1:
            raise ValueError("Expected a 1xD CSR row for single fingerprint.")
        return x
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Expected a 1D dense vector.")
    return sp.csr_matrix(x.reshape(1, -1))


def tanimoto_similarity(
    a: Union[DenseVector, sp.csr_matrix, UnfoldedFingerprint],
    b: Union[DenseVector, sp.csr_matrix, UnfoldedFingerprint],
    *,
    kind: Optional[Literal["dense", "sparse", "unfolded-binary", "unfolded-count"]] = None,
) -> float:
    """
    Function to compute Tanimoto similarity between two fingerprints/vectors. 
    Unified single-pair API.

    Parameters
    ----------
    a
        First fingerprint/vector.
    b
        Second fingerprint/vector.
    kind
        Optional specification of the representation type. If None, the function will attempt to infer the type.
        Can be one of:
        - kind="dense": expects 1D arrays same length
        - kind="sparse": expects 1xD csr_matrix for each
        - kind="unfolded-binary": expects 1D bit arrays (sorted unique)
        - kind="unfolded-count": expects (bits, counts) for each
    """
    if kind is None:
        # best-effort inference
        if sp.isspmatrix(a) or sp.isspmatrix(b):
            kind = "sparse"
        elif isinstance(a, tuple) or isinstance(b, tuple):
            kind = "unfolded-count"
        else:
            # could be dense or unfolded-binary; assume dense if numeric vector length is "large"
            # For safety in a library, caller should pass kind when ambiguous.
            kind = "dense"

    if kind == "dense":
        aa = np.asarray(a)  # type: ignore[arg-type]
        bb = np.asarray(b)  # type: ignore[arg-type]
        if aa.ndim != 1 or bb.ndim != 1 or aa.shape[0] != bb.shape[0]:
            raise ValueError("Dense vectors must be 1D and same length.")
        return float(tanimoto_similarity_dense(aa.astype(np.float32, copy=False),
                                              bb.astype(np.float32, copy=False)))

    if kind == "sparse":
        A = _as_1xD_csr(a)  # type: ignore[arg-type]
        B = _as_1xD_csr(b)  # type: ignore[arg-type]
        A.sort_indices()
        B.sort_indices()
        ind1 = A.indices[A.indptr[0]:A.indptr[1]]
        dat1 = A.data[A.indptr[0]:A.indptr[1]]
        ind2 = B.indices[B.indptr[0]:B.indptr[1]]
        dat2 = B.data[B.indptr[0]:B.indptr[1]]
        return float(tanimoto_similarity_sparse(ind1, dat1, ind2, dat2))

    if kind == "unfolded-binary":
        bits1 = np.asarray(a, dtype=np.int64)  # type: ignore[arg-type]
        bits2 = np.asarray(b, dtype=np.int64)  # type: ignore[arg-type]
        return float(tanimoto_similarity_sparse_binary(bits1, bits2))

    if kind == "unfolded-count":
        bits1, vals1 = a  # type: ignore[misc]
        bits2, vals2 = b  # type: ignore[misc]
        return float(tanimoto_similarity_sparse(
            np.asarray(bits1, dtype=np.int64),
            np.asarray(vals1, dtype=np.float32),
            np.asarray(bits2, dtype=np.int64),
            np.asarray(vals2, dtype=np.float32),
        ))

    raise ValueError(f"Unknown kind={kind!r}")


def tanimoto_similarity_matrix(
    references: Union[DenseMatrix, sp.csr_matrix],
    queries: Union[DenseMatrix, sp.csr_matrix],
    *,
    kind: Literal["dense", "sparse"] = "dense",
) -> np.ndarray:
    """
    Unified all-vs-all API for fixed-size representations.

    - kind="dense": references (R,D), queries (Q,D) -> (R,Q)
    - kind="sparse": references CSR (R,D), queries CSR (Q,D) -> (R,Q) via per-row merges

    Note: sparse all-vs-all is expensive for large R,Q; use ANN for kNN graphs instead.
    """
    if kind == "dense":
        R = np.asarray(references, dtype=np.float32)
        Q = np.asarray(queries, dtype=np.float32)
        if R.ndim != 2 or Q.ndim != 2 or R.shape[1] != Q.shape[1]:
            raise ValueError("Dense matrices must be 2D and share the same number of columns.")
        return tanimoto_similarity_matrix_dense(R, Q)

    if kind == "sparse":
        if not (sp.isspmatrix_csr(references) and sp.isspmatrix_csr(queries)):
            raise TypeError("For kind='sparse', references and queries must be CSR matrices.")
        A = references.copy()
        B = queries.copy()
        A.sort_indices()
        B.sort_indices()

        R = A.shape[0]
        Q = B.shape[0]
        out = np.empty((R, Q), dtype=np.float32)

        # Row-wise merge calling the numba sparse primitive
        for i in range(R):
            a0 = A.indptr[i]
            a1 = A.indptr[i + 1]
            ind1 = A.indices[a0:a1]
            dat1 = A.data[a0:a1]
            for j in range(Q):
                b0 = B.indptr[j]
                b1 = B.indptr[j + 1]
                ind2 = B.indices[b0:b1]
                dat2 = B.data[b0:b1]
                out[i, j] = tanimoto_similarity_sparse(ind1, dat1, ind2, dat2)
        return out

    raise ValueError(f"Unknown kind={kind!r}")
