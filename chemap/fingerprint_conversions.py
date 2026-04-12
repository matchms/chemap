from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.sparse as sp


# ---------------------------
# Types
# ---------------------------

CountFingerprint = Tuple[np.ndarray, np.ndarray]  # (bits, counts)
BinaryFingerprint = np.ndarray                    # (bits,)
FingerprintInput = Union[CountFingerprint, BinaryFingerprint]

TFTransform = Optional[Callable[[np.ndarray], np.ndarray]]  # for count fingerprints only


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Column vocabulary for unfolded fingerprints."""
    col_bits: np.ndarray  # shape (n_cols,), int64; original bit-id per column
    df: np.ndarray        # shape (n_cols,), int32; document frequency per column (occurrence across rows)
    bit_to_col: Optional[Dict[int, int]] = None  # optional (can be huge)


@dataclass(frozen=True, slots=True)
class MatrixWithVocab:
    """Convenience return type."""
    X: sp.csr_matrix
    vocab: Vocabulary
    idf: Optional[np.ndarray] = None  # shape (n_cols,), float32/float64


# ---------------------------
# Helper: thresholds
# ---------------------------

def _resolve_occurrence_thresholds(
    n_rows: int,
    min_occurrence: Optional[int],
    max_occurrence: Optional[Union[int, float]],
) -> tuple[Optional[int], Optional[int]]:
    if min_occurrence is not None:
        if not isinstance(min_occurrence, (int, np.integer)):
            raise TypeError("min_occurrence must be an int or None.")
        min_occ = int(min_occurrence)
        if min_occ < 1:
            raise ValueError("min_occurrence must be >= 1 when provided.")
    else:
        min_occ = None

    if max_occurrence is not None:
        if isinstance(max_occurrence, (int, np.integer)):
            max_occ = int(max_occurrence)
            if max_occ < 1:
                raise ValueError("max_occurrence (int) must be >= 1 when provided.")
        elif isinstance(max_occurrence, (float, np.floating)):
            frac = float(max_occurrence)
            if not (0.0 < frac < 1.0):
                raise ValueError("max_occurrence (float) must be in (0, 1).")
            # NLP-style: maximum document frequency as fraction of documents
            max_occ = int(np.floor(frac * n_rows))
            max_occ = max(1, max_occ)
        else:
            raise TypeError("max_occurrence must be an int, a float in (0,1), or None.")
    else:
        max_occ = None

    if min_occ is not None and max_occ is not None and min_occ > max_occ:
        raise ValueError("min_occurrence cannot be larger than max_occurrence after conversion.")

    return min_occ, max_occ


# ---------------------------
# Helper: input normalization & DF computation
# ---------------------------

def _infer_kind(item: FingerprintInput) -> Literal["count", "binary"]:
    if isinstance(item, tuple) and len(item) == 2:
        return "count"
    return "binary"


def _validate_row(
    row: FingerprintInput,
    row_idx: int,
    kind: Literal["count", "binary"],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Return (bits_i64, counts_or_none). Always 1D.
    """
    if kind == "count":
        bits, counts = row  # type: ignore[misc]
        bits_arr = np.asarray(bits, dtype=np.int64)
        counts_arr = np.asarray(counts)
        if bits_arr.ndim != 1 or counts_arr.ndim != 1:
            raise ValueError(f"Row {row_idx}: bits and counts must be 1D arrays.")
        if bits_arr.shape[0] != counts_arr.shape[0]:
            raise ValueError(
                f"Row {row_idx}: bits and counts must have same length; "
                f"got {bits_arr.shape[0]} vs {counts_arr.shape[0]}."
            )
        return bits_arr, counts_arr
    else:
        bits_arr = np.asarray(row, dtype=np.int64)  # type: ignore[arg-type]
        if bits_arr.ndim != 1:
            raise ValueError(f"Row {row_idx}: bits must be a 1D array.")
        return bits_arr, None


def _compute_df_and_order(
    fingerprints: Sequence[FingerprintInput],
    *,
    sort_bits: bool,
) -> tuple[Dict[int, int], Optional[Dict[int, int]], int, Literal["count", "binary"]]:
    """
    Compute document frequency df(bit) = number of rows where bit appears at least once.

    Parameters
    ----------
    fingerprints
        Sequence of unfolded fingerprints.
    sort_bits
        If True, sort bits in the vocabulary; else preserve first-seen order.
    Returns:
      df_dict : {bit_id -> df}
      order   : {bit_id -> first_seen_index} if sort_bits=False else None
      nnz_ub  : upper bound for nnz allocation (sum of raw row lengths)
      kind    : "count" or "binary"
    """
    n_rows = len(fingerprints)
    if n_rows == 0:
        return {}, None, 0, "binary"

    kind = _infer_kind(fingerprints[0])
    df: Dict[int, int] = {}
    order: Optional[Dict[int, int]] = {} if not sort_bits else None
    nnz_ub = 0

    for i, row in enumerate(fingerprints):
        row_kind = _infer_kind(row)
        if row_kind != kind:
            raise TypeError(
                "All fingerprints must have the same form: either (bits, counts) tuples "
                "or bit-only arrays."
            )

        bits_i64, _counts = _validate_row(row, i, kind)
        nnz_ub += int(bits_i64.size)

        if bits_i64.size == 0:
            continue

        # DF: presence once per row (sorted unique is fine)
        row_unique = np.unique(bits_i64)
        for b in row_unique:
            bi = int(b)
            df[bi] = df.get(bi, 0) + 1

        # ORDER: first-seen in original iteration order (NOT unique-sorted order)
        if order is not None:
            for b in bits_i64:
                bi = int(b)
                if bi not in order:
                    order[bi] = len(order)
    return df, order, nnz_ub, kind


def _build_vocab(
    df_dict: Dict[int, int],
    order: Optional[Dict[int, int]],
    *,
    n_rows: int,
    sort_bits: bool,
    return_bit_to_col: bool,
    min_occurrence: Optional[int],
    max_occurrence: Optional[Union[int, float]],
) -> Vocabulary:
    """
    Build filtered vocabulary (col_bits + df array + optional bit_to_col).
    """
    min_occ, max_occ = _resolve_occurrence_thresholds(n_rows, min_occurrence, max_occurrence)

    if not df_dict:
        col_bits = np.empty((0,), dtype=np.int64)
        df = np.empty((0,), dtype=np.int32)
        return Vocabulary(col_bits=col_bits, df=df, bit_to_col={} if return_bit_to_col else None)

    if sort_bits:
        all_bits = np.array(sorted(df_dict.keys()), dtype=np.int64)
    else:
        assert order is not None
        # Preserve first-seen order
        all_bits = np.empty((len(order),), dtype=np.int64)
        for bi, j in order.items():
            all_bits[j] = bi

    # df aligned to all_bits
    df_all = np.fromiter((df_dict[int(b)] for b in all_bits), dtype=np.int32, count=all_bits.size)

    if min_occ is None and max_occ is None:
        keep = np.ones(all_bits.shape[0], dtype=bool)
    else:
        keep = np.ones(all_bits.shape[0], dtype=bool)
        if min_occ is not None:
            keep &= (df_all >= min_occ)
        if max_occ is not None:
            keep &= (df_all <= max_occ)

    col_bits = all_bits[keep]
    df_kept = df_all[keep]

    bit_to_col: Optional[Dict[int, int]] = None
    if return_bit_to_col:
        bit_to_col = {int(b): int(j) for j, b in enumerate(col_bits)}

    return Vocabulary(col_bits=col_bits, df=df_kept, bit_to_col=bit_to_col)


# ---------------------------
# Case 1: CSR conversion (counts or binary)
# ---------------------------

def fingerprints_to_csr(
    fingerprints: Sequence[FingerprintInput],
    *,
    dtype: Union[np.dtype, type] = np.float32,
    sort_bits: bool = True,
    sort_indices_within_rows: bool = True,
    consolidate_duplicates_within_rows: bool = True,
    return_bit_to_col: bool = False,
    min_occurrence: Optional[int] = None,
    max_occurrence: Optional[Union[int, float]] = None,
    tf_transform: TFTransform = None,
) -> MatrixWithVocab:
    """
    Convert unfolded molecular fingerprints into a CSR sparse matrix.

    This function accepts either count fingerprints, given as ``(bits, counts)``
    tuples, or binary fingerprints, given as arrays of bit identifiers. Each
    unique retained bit becomes one column in the output matrix, and each input
    fingerprint becomes one row.

    For count fingerprints, matrix entries are the corresponding counts, optionally
    transformed within each row using ``tf_transform``. For binary fingerprints,
    retained entries are set to 1.

    Vocabulary construction is dataset-level and based on document frequency (DF),
    i.e. the number of fingerprints in which a bit occurs at least once. Optional
    ``min_occurrence`` and ``max_occurrence`` thresholds are applied to this DF
    before the matrix is built.

    Parameters
    ----------
    fingerprints
        Sequence of unfolded fingerprints. All entries must have the same form:
        either count fingerprints as ``(bits, counts)`` tuples or binary
        fingerprints as one-dimensional arrays of bit identifiers.
    dtype
        Data type of the matrix values.
    sort_bits
        If True, columns are ordered by ascending bit identifier. If False,
        columns follow the order in which bits are first encountered in the
        dataset.
    sort_indices_within_rows
        If True, sort column indices within each row of the CSR matrix.
    consolidate_duplicates_within_rows
        If True, repeated bit identifiers within a row are merged before values
        are written to the matrix. For count fingerprints, duplicate counts are
        summed before applying ``tf_transform``. For binary fingerprints, repeated
        occurrences are reduced to a single presence.
    return_bit_to_col
        If True, include a ``bit_to_col`` mapping in the returned vocabulary.
        This can be memory-intensive for very large vocabularies.
    min_occurrence
        Minimum document frequency required for a bit to be retained. Must be
        ``None`` or an integer >= 1.
    max_occurrence
        Maximum document frequency allowed for a bit to be retained. May be
        ``None``, an integer >= 1, or a float in ``(0, 1)`` interpreted as a
        fraction of the number of rows.
    tf_transform
        Optional transformation applied to count values within each row after
        optional duplicate consolidation. Examples include ``np.log1p`` or
        sublinear term-frequency transforms. Ignored for binary fingerprints.

    Returns
    -------
    MatrixWithVocab
        Object containing the CSR matrix and the corresponding vocabulary.
        ``idf`` is ``None`` for this function.

    Notes
    -----
    Document frequency is always computed as row-wise presence, not as the total
    number of occurrences across the dataset.

    Examples
    --------
    Count fingerprints::

        fps = [
            (np.array([2, 5, 5]), np.array([1, 2, 3])),
            (np.array([2, 9]),    np.array([4, 1])),
        ]
        out = fingerprints_to_csr(fps, tf_transform=np.log1p)

    Binary fingerprints::

        fps = [
            np.array([2, 5, 9]),
            np.array([2, 7]),
        ]
        out = fingerprints_to_csr(fps)
    """
    n_rows = len(fingerprints)
    if n_rows == 0:
        X = sp.csr_matrix((0, 0), dtype=dtype)
        vocab = Vocabulary(col_bits=np.empty((0,), np.int64), df=np.empty((0,), np.int32),
                           bit_to_col={} if return_bit_to_col else None)
        return MatrixWithVocab(X=X, vocab=vocab, idf=None)

    df_dict, order, nnz_ub, kind = _compute_df_and_order(
        fingerprints,
        sort_bits=sort_bits,
    )

    vocab = _build_vocab(
        df_dict,
        order,
        n_rows=n_rows,
        sort_bits=sort_bits,
        return_bit_to_col=return_bit_to_col,
        min_occurrence=min_occurrence,
        max_occurrence=max_occurrence,
    )

    n_cols = int(vocab.col_bits.size)
    ind_dtype = np.int32 if n_cols < np.iinfo(np.int32).max else np.int64

    # Upper bound allocation; we trim after filtering
    data = np.empty((nnz_ub,), dtype=dtype)
    indices = np.empty((nnz_ub,), dtype=ind_dtype)
    indptr = np.empty((n_rows + 1,), dtype=np.int64)
    indptr[0] = 0

    pos = 0
    col_bits = vocab.col_bits

    if sort_bits:
        # Fast vectorized mapping via searchsorted, with exact-match filtering
        for i, row in enumerate(fingerprints):
            bits_i64, counts = _validate_row(row, i, kind)

            if bits_i64.size == 0 or n_cols == 0:
                indptr[i + 1] = pos
                continue

            if consolidate_duplicates_within_rows:
                uniq_bits, inv = np.unique(bits_i64, return_inverse=True)
                if kind == "count":
                    summed = np.bincount(inv, weights=counts.astype(np.float64, copy=False))  # type: ignore[union-attr]
                    vals = summed.astype(dtype, copy=False)
                    if tf_transform is not None:
                        vals = tf_transform(vals)
                else:
                    uniq_bits = uniq_bits
                    vals = np.ones(uniq_bits.shape[0], dtype=dtype)
            else:
                uniq_bits = bits_i64
                if kind == "count":
                    vals = counts.astype(dtype, copy=False)  # type: ignore[union-attr]
                    if tf_transform is not None:
                        vals = tf_transform(vals)
                else:
                    vals = np.ones(bits_i64.shape[0], dtype=dtype)

            # map to kept vocabulary
            cols = np.searchsorted(col_bits, uniq_bits)
            in_bounds = cols < n_cols
            if not np.any(in_bounds):
                indptr[i + 1] = pos
                continue

            cols2 = cols[in_bounds]
            bits2 = uniq_bits[in_bounds]
            match = col_bits[cols2] == bits2
            if not np.any(match):
                indptr[i + 1] = pos
                continue

            cols3 = cols2[match].astype(ind_dtype, copy=False)
            vals3 = vals[in_bounds][match]

            if sort_indices_within_rows and cols3.size > 1:
                order_idx = np.argsort(cols3, kind="mergesort")
                cols3 = cols3[order_idx]
                vals3 = vals3[order_idx]

            k = int(cols3.size)
            indices[pos : pos + k] = cols3
            data[pos : pos + k] = vals3
            pos += k
            indptr[i + 1] = pos

    else:
        # Dict mapping; slower but preserves first-seen order
        mapping = vocab.bit_to_col or {int(b): int(j) for j, b in enumerate(col_bits)}

        for i, row in enumerate(fingerprints):
            bits_i64, counts = _validate_row(row, i, kind)

            if bits_i64.size == 0 or n_cols == 0:
                indptr[i + 1] = pos
                continue

            if consolidate_duplicates_within_rows:
                uniq_bits, inv = np.unique(bits_i64, return_inverse=True)
                if kind == "count":
                    summed = np.bincount(inv, weights=counts.astype(np.float64, copy=False))  # type: ignore[union-attr]
                    vals = summed.astype(dtype, copy=False)
                    if tf_transform is not None:
                        vals = tf_transform(vals)
                else:
                    vals = np.ones(uniq_bits.shape[0], dtype=dtype)
            else:
                uniq_bits = bits_i64
                if kind == "count":
                    vals = counts.astype(dtype, copy=False)  # type: ignore[union-attr]
                    if tf_transform is not None:
                        vals = tf_transform(vals)
                else:
                    vals = np.ones(bits_i64.shape[0], dtype=dtype)

            cols_tmp = []
            vals_tmp = []
            for t in range(int(uniq_bits.size)):
                bi = int(uniq_bits[t])
                j = mapping.get(bi, -1)
                if j != -1:
                    cols_tmp.append(j)
                    vals_tmp.append(vals[t])

            if not cols_tmp:
                indptr[i + 1] = pos
                continue

            cols3 = np.asarray(cols_tmp, dtype=ind_dtype)
            vals3 = np.asarray(vals_tmp, dtype=dtype)

            if sort_indices_within_rows and cols3.size > 1:
                order_idx = np.argsort(cols3, kind="mergesort")
                cols3 = cols3[order_idx]
                vals3 = vals3[order_idx]

            k = int(cols3.size)
            indices[pos : pos + k] = cols3
            data[pos : pos + k] = vals3
            pos += k
            indptr[i + 1] = pos

    # Trim
    indices = indices[:pos]
    data = data[:pos]

    X = sp.csr_matrix((data, indices, indptr), shape=(n_rows, n_cols), dtype=dtype)
    X.sum_duplicates()
    if sort_indices_within_rows:
        X.sort_indices()

    return MatrixWithVocab(X=X, vocab=vocab, idf=None)


# ---------------------------
# Case 2: TF-IDF / IDF weighting
# ---------------------------

def idf_normalized(df: np.ndarray, N: int) -> np.ndarray:
    """
    Compute normalized inverse document frequency values in the range [0, 1].

    The normalization is defined as::

        idf(df) = log(N / df) / log(N)

    where ``N`` is the number of rows and ``df`` is the document frequency of
    each retained bit. Bits present in every row receive an IDF of 0, while bits
    present in exactly one row receive the maximum value of 1.

    Parameters
    ----------
    df
        One-dimensional array of document frequencies for the retained columns.
    N
        Total number of rows (fingerprints) in the dataset.
    """
    df = np.asarray(df, dtype=np.float64)
    if N < 1:
        raise ValueError("N must be >= 1.")
    if df.size == 0:
        return df.astype(np.float32)

    # Avoid divide-by-zero (df should never be 0 for kept columns, but keep it safe)
    df = np.maximum(df, 1.0)
    max_idf = np.log(N / 1.0)
    if max_idf == 0.0:
        return np.zeros(df.shape[0], dtype=np.float32)
    return (np.log(N / df) / max_idf).astype(np.float32)


def fingerprints_to_tfidf(
    fingerprints: Sequence[FingerprintInput],
    *,
    dtype: Union[np.dtype, type] = np.float32,
    sort_bits: bool = True,
    sort_indices_within_rows: bool = True,
    consolidate_duplicates_within_rows: bool = True,
    return_bit_to_col: bool = False,
    min_occurrence: Optional[int] = None,
    max_occurrence: Optional[Union[int, float]] = None,
    tf_transform: TFTransform = None,
) -> MatrixWithVocab:
    """
    Convert unfolded fingerprints into a TF-IDF- or IDF-weighted CSR matrix.

    This function first builds the same filtered sparse representation as
    :func:`fingerprints_to_csr`, then multiplies each retained column by its
    normalized inverse document frequency (IDF).

    For count fingerprints, matrix entries are interpreted as term frequencies
    (TF), optionally transformed within each row using ``tf_transform``, and then
    multiplied by IDF. For binary fingerprints, each retained entry starts as 1
    and is then weighted only by IDF.

    Document frequency thresholds are applied before IDF is computed, so the
    returned IDF vector corresponds exactly to the retained vocabulary.

    Parameters
    ----------
    fingerprints
        Sequence of unfolded fingerprints, either all count fingerprints or all
        binary fingerprints.
    dtype
        Data type of the matrix values.
    sort_bits
        If True, columns are ordered by ascending bit identifier. If False,
        columns follow first-seen order in the dataset.
    sort_indices_within_rows
        If True, sort column indices within each row of the CSR matrix.
    consolidate_duplicates_within_rows
        If True, repeated bit identifiers within a row are merged before TF
        values are determined.
    return_bit_to_col
        If True, include a bit-to-column mapping in the returned vocabulary.
    min_occurrence
        Minimum document frequency required for a bit to be retained.
    max_occurrence
        Maximum document frequency allowed for a bit to be retained.
    tf_transform
        Optional transformation applied to count values within each row before
        IDF weighting. Ignored for binary fingerprints.
    """
    out = fingerprints_to_csr(
        fingerprints,
        dtype=dtype,
        sort_bits=sort_bits,
        sort_indices_within_rows=sort_indices_within_rows,
        consolidate_duplicates_within_rows=consolidate_duplicates_within_rows,
        return_bit_to_col=return_bit_to_col,
        min_occurrence=min_occurrence,
        max_occurrence=max_occurrence,
        tf_transform=tf_transform,
    )

    X = out.X.copy()
    N = X.shape[0]

    if N == 0:
        return MatrixWithVocab(X=X, vocab=out.vocab, idf=None)

    idf = idf_normalized(out.vocab.df, N)

    # Apply IDF per nonzero without building a diagonal matrix
    if X.nnz:
        X.data *= idf[X.indices]

    return MatrixWithVocab(X=X, vocab=out.vocab, idf=idf)


# ---------------------------
# Case 3: Folding (after filtering, after optional IDF)
# ---------------------------

def fold_csr_mod(
    X: sp.csr_matrix,
    n_folded_features: int,
    *,
    sort_indices: bool = True,
    sum_duplicates: bool = True,
) -> sp.csr_matrix:
    """
    Efficient folding for CSR matrices using modulo hashing on column indices.

    This is O(nnz) for the mapping plus the cost of summing duplicates per row.

    Parameters
    ----------
    X
        Input matrix in CSR format.
    n_folded_features
        Number of columns in the folded output matrix.
    sort_indices
        If True, sort column indices within each row of the output matrix.
    sum_duplicates
        If True, merge collisions created by folding by summing duplicate entries.
    """
    if not sp.isspmatrix_csr(X):
        raise TypeError("X must be a scipy.sparse.csr_matrix.")
    if n_folded_features < 1:
        raise ValueError("n_folded_features must be >= 1.")

    if X.nnz == 0:
        return sp.csr_matrix((X.shape[0], n_folded_features), dtype=X.dtype)

    # Map indices; keep same indptr/data arrays
    new_indices = (X.indices % n_folded_features).astype(X.indices.dtype, copy=False)
    Y = sp.csr_matrix((X.data.copy(), new_indices, X.indptr.copy()),
                      shape=(X.shape[0], n_folded_features),
                      dtype=X.dtype)

    if sum_duplicates:
        Y.sum_duplicates()
    if sort_indices:
        Y.sort_indices()
    return Y


def fingerprints_to_csr_folded(
    fingerprints: Sequence[FingerprintInput],
    *,
    n_folded_features: int,
    dtype: Union[np.dtype, type] = np.float32,
    sort_bits: bool = True,
    sort_indices_within_rows: bool = True,
    consolidate_duplicates_within_rows: bool = True,
    return_bit_to_col: bool = False,
    min_occurrence: Optional[int] = None,
    max_occurrence: Optional[Union[int, float]] = None,
    tf_transform: TFTransform = None,
) -> MatrixWithVocab:
    """
    Convert unfolded fingerprints to a sparse matrix and then fold the feature
    space using modulo hashing.

    This is a two-step operation:

    1. Build a filtered unfolded CSR matrix from the input fingerprints.
    2. Fold its columns into ``n_folded_features`` dimensions using
       :func:`fold_csr_mod`.

    The vocabulary stored in the returned object always describes the retained
    unfolded features before hashing-based folding. The matrix itself is the
    folded representation.

    Parameters
    ----------
    fingerprints
        Sequence of unfolded count or binary fingerprints.
    n_folded_features
        Number of columns in the folded output matrix.
    dtype
        Data type of the matrix values.
    sort_bits
        Controls how the unfolded vocabulary is ordered before folding.
    sort_indices_within_rows
        If True, sort column indices within each row before returning the folded
        matrix.
    consolidate_duplicates_within_rows
        If True, merge repeated bit identifiers within a row before writing the
        unfolded matrix.
    return_bit_to_col
        If True, include a bit-to-column mapping for the unfolded vocabulary.
    min_occurrence
        Minimum document frequency required for a bit to be retained before
        folding.
    max_occurrence
        Maximum document frequency allowed for a bit to be retained before
        folding.
    tf_transform
        Optional transformation applied to count values before folding. Ignored
        for binary fingerprints.
    """
    out = fingerprints_to_csr(
        fingerprints,
        dtype=dtype,
        sort_bits=sort_bits,
        sort_indices_within_rows=sort_indices_within_rows,
        consolidate_duplicates_within_rows=consolidate_duplicates_within_rows,
        return_bit_to_col=return_bit_to_col,
        min_occurrence=min_occurrence,
        max_occurrence=max_occurrence,
        tf_transform=tf_transform,
    )
    X_folded = fold_csr_mod(out.X, n_folded_features)
    return MatrixWithVocab(X=X_folded, vocab=out.vocab, idf=None)


def fingerprints_to_tfidf_folded(
    fingerprints: Sequence[FingerprintInput],
    *,
    n_folded_features: int,
    dtype: Union[np.dtype, type] = np.float32,
    sort_bits: bool = True,
    sort_indices_within_rows: bool = True,
    consolidate_duplicates_within_rows: bool = True,
    return_bit_to_col: bool = False,
    min_occurrence: Optional[int] = None,
    max_occurrence: Optional[Union[int, float]] = None,
    tf_transform: TFTransform = None,
) -> MatrixWithVocab:
    """
    Convert unfolded fingerprints to a TF-IDF- or IDF-weighted sparse matrix and
    then fold the feature space using modulo hashing.

    This function first constructs a filtered unfolded representation, computes
    normalized IDF weights for the retained vocabulary, applies those weights to
    the matrix values, and finally folds the weighted matrix into
    ``n_folded_features`` dimensions using modulo hashing.

    The returned vocabulary and IDF vector always refer to the retained unfolded
    features before folding. The matrix itself is the folded weighted
    representation.

    Parameters
    ----------
    fingerprints
        Sequence of unfolded count or binary fingerprints.
    n_folded_features
        Number of columns in the folded output matrix.
    dtype
        Data type of the matrix values.
    sort_bits
        Controls how the unfolded vocabulary is ordered before folding.
    sort_indices_within_rows
        If True, sort column indices within each row before returning the folded
        matrix.
    consolidate_duplicates_within_rows
        If True, merge repeated bit identifiers within a row before TF/IDF
        weighting is applied.
    return_bit_to_col
        If True, include a bit-to-column mapping for the unfolded vocabulary.
    min_occurrence
        Minimum document frequency required for a bit to be retained before
        weighting and folding.
    max_occurrence
        Maximum document frequency allowed for a bit to be retained before
        weighting and folding.
    tf_transform
        Optional transformation applied to count values before IDF weighting.
        Ignored for binary fingerprints.
    """
    out = fingerprints_to_tfidf(
        fingerprints,
        dtype=dtype,
        sort_bits=sort_bits,
        sort_indices_within_rows=sort_indices_within_rows,
        consolidate_duplicates_within_rows=consolidate_duplicates_within_rows,
        return_bit_to_col=return_bit_to_col,
        min_occurrence=min_occurrence,
        max_occurrence=max_occurrence,
        tf_transform=tf_transform,
    )
    X_folded = fold_csr_mod(out.X, n_folded_features)
    return MatrixWithVocab(X=X_folded, vocab=out.vocab, idf=out.idf)


def _restrict_vocab_top_k_frequency(
    vocab: Vocabulary,
    *,
    n_rows: int,
    max_features: int,
    exclude_constant_bits: bool = True,
    return_bit_to_col: bool = False,
) -> Vocabulary:
    """
    Keep the `max_features` most frequent bits from an existing vocabulary.

    Bits are ranked by descending document frequency (DF). Ties are resolved by
    the current column order of `vocab`, so ordering remains deterministic.

    Parameters
    ----------
    vocab
        Vocabulary describing the unfolded feature space.
    n_rows
        Number of fingerprints in the reference dataset used to compute DF.
    max_features
        Number of features to retain.
    exclude_constant_bits
        If True, remove bits with `df == n_rows` before ranking. Such bits are
        present in every fingerprint and therefore do not contribute to
        discrimination within this dataset.
    return_bit_to_col
        If True, build the `bit_to_col` mapping for the reduced vocabulary.

    Returns
    -------
    Vocabulary
        Reduced vocabulary containing at most `max_features` bits.
    """
    if max_features < 1:
        raise ValueError("max_features must be >= 1.")

    col_bits = vocab.col_bits
    df = vocab.df

    if col_bits.size == 0:
        return Vocabulary(
            col_bits=np.empty((0,), dtype=np.int64),
            df=np.empty((0,), dtype=np.int32),
            bit_to_col={} if return_bit_to_col else None,
        )

    keep = np.ones(df.shape[0], dtype=bool)
    if exclude_constant_bits:
        keep &= (df < n_rows)

    idx = np.nonzero(keep)[0]
    if idx.size == 0:
        return Vocabulary(
            col_bits=np.empty((0,), dtype=np.int64),
            df=np.empty((0,), dtype=np.int32),
            bit_to_col={} if return_bit_to_col else None,
        )

    # Stable sort: descending DF, ties keep existing order
    order = np.argsort(-df[idx], kind="mergesort")
    selected = idx[order[:max_features]]

    # Preserve the existing vocabulary order in the returned representation
    selected.sort()

    new_bits = col_bits[selected]
    new_df = df[selected]

    bit_to_col = None
    if return_bit_to_col:
        bit_to_col = {int(b): int(j) for j, b in enumerate(new_bits)}

    return Vocabulary(col_bits=new_bits, df=new_df, bit_to_col=bit_to_col)


def _fingerprints_to_csr_with_vocab(
    fingerprints: Sequence[FingerprintInput],
    vocab: Vocabulary,
    *,
    dtype: Union[np.dtype, type] = np.float32,
    sort_indices_within_rows: bool = True,
    consolidate_duplicates_within_rows: bool = True,
    tf_transform: TFTransform = None,
) -> sp.csr_matrix:
    """
    Build a CSR matrix from fingerprints using a fixed vocabulary.

    Only bits present in `vocab` are retained. Column positions are taken
    directly from `vocab.bit_to_col` if available, otherwise a mapping is built
    on the fly.
    """
    n_rows = len(fingerprints)
    n_cols = int(vocab.col_bits.size)

    if n_rows == 0 or n_cols == 0:
        return sp.csr_matrix((n_rows, n_cols), dtype=dtype)

    kind = _infer_kind(fingerprints[0])
    ind_dtype = np.int32 if n_cols < np.iinfo(np.int32).max else np.int64

    nnz_ub = 0
    for i, row in enumerate(fingerprints):
        row_kind = _infer_kind(row)
        if row_kind != kind:
            raise TypeError(
                "All fingerprints must have the same form: either (bits, counts) "
                "tuples or bit-only arrays."
            )
        bits_i64, _ = _validate_row(row, i, kind)
        nnz_ub += int(bits_i64.size)

    data = np.empty((nnz_ub,), dtype=dtype)
    indices = np.empty((nnz_ub,), dtype=ind_dtype)
    indptr = np.empty((n_rows + 1,), dtype=np.int64)
    indptr[0] = 0

    mapping = vocab.bit_to_col or {int(b): int(j) for j, b in enumerate(vocab.col_bits)}

    pos = 0
    for i, row in enumerate(fingerprints):
        bits_i64, counts = _validate_row(row, i, kind)

        if bits_i64.size == 0:
            indptr[i + 1] = pos
            continue

        if consolidate_duplicates_within_rows:
            uniq_bits, inv = np.unique(bits_i64, return_inverse=True)
            if kind == "count":
                summed = np.bincount(inv, weights=counts.astype(np.float64, copy=False))
                vals = summed.astype(dtype, copy=False)
                if tf_transform is not None:
                    vals = np.asarray(tf_transform(vals), dtype=dtype)
            else:
                vals = np.ones(uniq_bits.shape[0], dtype=dtype)
        else:
            uniq_bits = bits_i64
            if kind == "count":
                vals = counts.astype(dtype, copy=False)
                if tf_transform is not None:
                    vals = np.asarray(tf_transform(vals), dtype=dtype)
            else:
                vals = np.ones(bits_i64.shape[0], dtype=dtype)

        cols_tmp = []
        vals_tmp = []
        for t in range(int(uniq_bits.size)):
            j = mapping.get(int(uniq_bits[t]), -1)
            if j != -1:
                cols_tmp.append(j)
                vals_tmp.append(vals[t])

        if cols_tmp:
            cols = np.asarray(cols_tmp, dtype=ind_dtype)
            vals = np.asarray(vals_tmp, dtype=dtype)

            if sort_indices_within_rows and cols.size > 1:
                order = np.argsort(cols, kind="mergesort")
                cols = cols[order]
                vals = vals[order]

            k = int(cols.size)
            indices[pos:pos + k] = cols
            data[pos:pos + k] = vals
            pos += k

        indptr[i + 1] = pos

    X = sp.csr_matrix((data[:pos], indices[:pos], indptr), shape=(n_rows, n_cols), dtype=dtype)
    X.sum_duplicates()
    if sort_indices_within_rows:
        X.sort_indices()
    return X


def fingerprints_to_csr_frequency_folded(
    fingerprints: Sequence[FingerprintInput],
    *,
    n_frequency_features: int,
    dtype: Union[np.dtype, type] = np.float32,
    sort_bits: bool = True,
    sort_indices_within_rows: bool = True,
    consolidate_duplicates_within_rows: bool = True,
    return_bit_to_col: bool = False,
    min_occurrence: Optional[int] = None,
    max_occurrence: Optional[Union[int, float]] = None,
    tf_transform: TFTransform = None,
    exclude_constant_bits: bool = True,
) -> MatrixWithVocab:
    """
    Build a sparse fingerprint matrix using only the most frequent unfolded bits.

    This function first computes the unfolded vocabulary and document frequencies
    on the input dataset, optionally applies occurrence-based filtering, and then
    retains only the `n_frequency_features` most frequent remaining bits.

    Unlike modulo hashing, this representation does not introduce collisions.
    Instead, dimensionality is reduced by selecting a fixed number of high-frequency
    bits from the reference dataset.

    Parameters
    ----------
    fingerprints
        Sequence of unfolded count or binary fingerprints.
    n_frequency_features
        Number of frequent bits to retain.
    dtype
        Data type of the matrix values.
    sort_bits
        Controls the initial ordering of the unfolded vocabulary before frequency
        ranking is applied.
    sort_indices_within_rows
        If True, sort column indices within each row of the output matrix.
    consolidate_duplicates_within_rows
        If True, merge repeated bit identifiers within each row before values are
        written to the matrix.
    return_bit_to_col
        If True, include a bit-to-column mapping in the returned vocabulary.
    min_occurrence
        Minimum document frequency required for a bit to be considered.
    max_occurrence
        Maximum document frequency allowed for a bit to be considered.
    tf_transform
        Optional transformation applied to count values within each row. Ignored
        for binary fingerprints.
    exclude_constant_bits
        If True, exclude bits present in every fingerprint from frequency-based
        selection. Such bits carry no discriminative information within the
        reference dataset.

    Returns
    -------
    MatrixWithVocab
        Object containing the reduced CSR matrix and the selected vocabulary.
        `idf` is `None` for this function.
    """
    n_rows = len(fingerprints)
    if n_rows == 0:
        X = sp.csr_matrix((0, 0), dtype=dtype)
        vocab = Vocabulary(
            col_bits=np.empty((0,), dtype=np.int64),
            df=np.empty((0,), dtype=np.int32),
            bit_to_col={} if return_bit_to_col else None,
        )
        return MatrixWithVocab(X=X, vocab=vocab, idf=None)

    base = fingerprints_to_csr(
        fingerprints,
        dtype=dtype,
        sort_bits=sort_bits,
        sort_indices_within_rows=sort_indices_within_rows,
        consolidate_duplicates_within_rows=consolidate_duplicates_within_rows,
        return_bit_to_col=True,
        min_occurrence=min_occurrence,
        max_occurrence=max_occurrence,
        tf_transform=None,  # build vocab first; apply tf later during rebuild
    )

    reduced_vocab = _restrict_vocab_top_k_frequency(
        base.vocab,
        n_rows=n_rows,
        max_features=n_frequency_features,
        exclude_constant_bits=exclude_constant_bits,
        return_bit_to_col=return_bit_to_col,
    )

    X = _fingerprints_to_csr_with_vocab(
        fingerprints,
        Vocabulary(
            col_bits=reduced_vocab.col_bits,
            df=reduced_vocab.df,
            bit_to_col={int(b): int(j) for j, b in enumerate(reduced_vocab.col_bits)},
        ),
        dtype=dtype,
        sort_indices_within_rows=sort_indices_within_rows,
        consolidate_duplicates_within_rows=consolidate_duplicates_within_rows,
        tf_transform=tf_transform,
    )

    return MatrixWithVocab(X=X, vocab=reduced_vocab, idf=None)
