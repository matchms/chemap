import numpy as np
import pytest
import scipy.sparse as sp
from chemap.fingerprint_conversions import (
    Vocabulary,
    _fingerprints_to_csr_with_vocab,
    _restrict_vocab_top_k_frequency,
    fingerprints_to_csr,
    fingerprints_to_csr_folded,
    fingerprints_to_csr_frequency_folded,
    fingerprints_to_tfidf,
    fingerprints_to_tfidf_folded,
    fold_csr_mod,
    idf_normalized,
)


# ----------------------------
# Small helpers
# ----------------------------

def _assert_csr_shape_dtype(X: sp.csr_matrix, shape, dtype):
    assert sp.isspmatrix_csr(X)
    assert X.shape == shape
    assert X.dtype == np.dtype(dtype)


# ----------------------------
# idf_normalized
# ----------------------------

def test_idf_normalized_basic_properties():
    df = np.array([1, 2, 5, 10], dtype=np.int32)
    N = 10
    idf = idf_normalized(df, N)

    assert idf.shape == (4,)
    assert idf.dtype == np.float32
    assert idf[0] == pytest.approx(1.0)
    assert idf[-1] == pytest.approx(0.0)
    assert np.all(idf >= 0.0)
    assert np.all(idf <= 1.0)
    assert idf[0] >= idf[1] >= idf[2] >= idf[3]


def test_idf_normalized_empty():
    df = np.array([], dtype=np.int32)
    out = idf_normalized(df, 5)
    assert out.size == 0
    assert out.dtype == np.float32


def test_idf_normalized_invalid_N():
    with pytest.raises(ValueError):
        idf_normalized(np.array([1, 2], dtype=np.int32), 0)


def test_idf_normalized_single_row_returns_zeros():
    df = np.array([1], dtype=np.int32)
    out = idf_normalized(df, 1)
    np.testing.assert_allclose(out, np.array([0.0], dtype=np.float32))


def test_idf_normalized_clips_zero_df_defensively():
    out = idf_normalized(np.array([0, 1, 2], dtype=np.int32), 4)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


# ----------------------------
# fingerprints_to_csr
# ----------------------------

def test_fingerprints_to_csr_count_basic_sorted_vocab():
    fp_a = (np.array([2, 5], dtype=np.int64), np.array([1.0, 1.0], dtype=np.float32))
    fp_b = (np.array([3], dtype=np.int64), np.array([2.0], dtype=np.float32))

    out = fingerprints_to_csr([fp_a, fp_b], sort_bits=True, return_bit_to_col=True)
    X, vocab = out.X, out.vocab

    _assert_csr_shape_dtype(X, (2, 3), np.float32)
    np.testing.assert_array_equal(vocab.col_bits, np.array([2, 3, 5], dtype=np.int64))
    np.testing.assert_array_equal(vocab.df, np.array([1, 1, 1], dtype=np.int32))
    assert vocab.bit_to_col == {2: 0, 3: 1, 5: 2}

    expected = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(X.toarray(), expected)


def test_fingerprints_to_csr_binary_basic():
    fp_a = np.array([2, 5], dtype=np.int64)
    fp_b = np.array([3], dtype=np.int64)

    out = fingerprints_to_csr([fp_a, fp_b], sort_bits=True)
    X, vocab = out.X, out.vocab

    _assert_csr_shape_dtype(X, (2, 3), np.float32)
    np.testing.assert_array_equal(vocab.col_bits, np.array([2, 3, 5], dtype=np.int64))
    np.testing.assert_array_equal(vocab.df, np.array([1, 1, 1], dtype=np.int32))

    expected = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(X.toarray(), expected)


def test_fingerprints_to_csr_empty_input_returns_empty_matrix_and_vocab():
    out = fingerprints_to_csr([], sort_bits=True, return_bit_to_col=True)

    _assert_csr_shape_dtype(out.X, (0, 0), np.float32)
    assert out.idf is None
    assert out.vocab.col_bits.shape == (0,)
    assert out.vocab.df.shape == (0,)
    assert out.vocab.bit_to_col == {}


def test_fingerprints_to_csr_single_empty_row_binary():
    out = fingerprints_to_csr([np.array([], dtype=np.int64)], sort_bits=True)

    _assert_csr_shape_dtype(out.X, (1, 0), np.float32)
    np.testing.assert_array_equal(out.vocab.col_bits, np.array([], dtype=np.int64))
    np.testing.assert_array_equal(out.vocab.df, np.array([], dtype=np.int32))


def test_fingerprints_to_csr_consolidates_duplicates_in_row_counts():
    fp = (
        np.array([7, 7, 2], dtype=np.int64),
        np.array([1.0, 2.0, 5.0], dtype=np.float32),
    )
    out = fingerprints_to_csr([fp], sort_bits=True, consolidate_duplicates_within_rows=True)
    X, vocab = out.X, out.vocab

    np.testing.assert_array_equal(vocab.col_bits, np.array([2, 7], dtype=np.int64))
    np.testing.assert_allclose(X.toarray().ravel(), np.array([5.0, 3.0], dtype=np.float32))


def test_fingerprints_to_csr_duplicate_counts_same_final_matrix_even_without_preconsolidation():
    fp = (
        np.array([7, 7, 2], dtype=np.int64),
        np.array([1.0, 2.0, 5.0], dtype=np.float32),
    )

    out_a = fingerprints_to_csr(
        [fp],
        sort_bits=True,
        consolidate_duplicates_within_rows=True,
        tf_transform=None,
    )
    out_b = fingerprints_to_csr(
        [fp],
        sort_bits=True,
        consolidate_duplicates_within_rows=False,
        tf_transform=None,
    )

    np.testing.assert_allclose(out_a.X.toarray(), out_b.X.toarray())
    np.testing.assert_array_equal(out_a.vocab.col_bits, out_b.vocab.col_bits)


def test_fingerprints_to_csr_duplicate_counts_tf_transform_depends_on_preconsolidation():
    fp = (
        np.array([7, 7], dtype=np.int64),
        np.array([1.0, 2.0], dtype=np.float32),
    )

    out_a = fingerprints_to_csr(
        [fp],
        sort_bits=True,
        consolidate_duplicates_within_rows=True,
        tf_transform=np.log1p,
    )
    out_b = fingerprints_to_csr(
        [fp],
        sort_bits=True,
        consolidate_duplicates_within_rows=False,
        tf_transform=np.log1p,
    )

    # True: log1p(1+2) = log(4)
    # False: log1p(1) + log1p(2)
    expected_a = np.array([[np.log1p(3.0)]], dtype=np.float32)
    expected_b = np.array([[np.log1p(1.0) + np.log1p(2.0)]], dtype=np.float32)

    np.testing.assert_allclose(out_a.X.toarray(), expected_a, rtol=1e-6)
    np.testing.assert_allclose(out_b.X.toarray(), expected_b, rtol=1e-6)
    assert not np.allclose(out_a.X.toarray(), out_b.X.toarray())


def test_fingerprints_to_csr_binary_duplicates_become_counts_if_not_preconsolidated():
    fp = np.array([5, 5, 5], dtype=np.int64)

    out = fingerprints_to_csr(
        [fp],
        sort_bits=True,
        consolidate_duplicates_within_rows=False,
    )

    # Current implementation sums duplicates in the final CSR.
    np.testing.assert_array_equal(out.vocab.col_bits, np.array([5], dtype=np.int64))
    np.testing.assert_allclose(out.X.toarray(), np.array([[3.0]], dtype=np.float32))


def test_fingerprints_to_csr_tf_transform_applied_counts_only():
    fp = (np.array([1, 2], dtype=np.int64), np.array([1.0, 3.0], dtype=np.float32))
    out = fingerprints_to_csr([fp], sort_bits=True, tf_transform=np.log1p)
    X = out.X.toarray().ravel()

    np.testing.assert_allclose(
        X,
        np.log1p(np.array([1.0, 3.0], dtype=np.float32)),
        rtol=1e-6,
    )


def test_fingerprints_to_csr_raises_on_mixed_input_types():
    fps = [
        (np.array([1], dtype=np.int64), np.array([1.0], dtype=np.float32)),
        np.array([2], dtype=np.int64),
    ]
    with pytest.raises(TypeError):
        fingerprints_to_csr(fps)


@pytest.mark.parametrize(
    "row",
    [
        (np.array([[1, 2]], dtype=np.int64), np.array([1.0, 2.0], dtype=np.float32)),
        (np.array([1, 2], dtype=np.int64), np.array([[1.0, 2.0]], dtype=np.float32)),
        np.array([[1, 2]], dtype=np.int64),
    ],
)
def test_fingerprints_to_csr_rejects_non_1d_rows(row):
    with pytest.raises(ValueError):
        fingerprints_to_csr([row])


def test_fingerprints_to_csr_rejects_mismatched_bits_and_counts_lengths():
    fp = (
        np.array([1, 2], dtype=np.int64),
        np.array([1.0], dtype=np.float32),
    )
    with pytest.raises(ValueError):
        fingerprints_to_csr([fp])


def test_fingerprints_to_csr_respects_dtype_argument():
    fp = (np.array([1, 2], dtype=np.int64), np.array([1.0, 2.0], dtype=np.float32))
    out = fingerprints_to_csr([fp], dtype=np.float64)
    assert out.X.dtype == np.float64


# ----------------------------
# Occurrence filtering
# ----------------------------

def test_fingerprints_to_csr_min_occurrence_filters_rare_bits():
    fps = [
        (np.array([1, 2], np.int64), np.array([1.0, 1.0], np.float32)),
        (np.array([2], np.int64), np.array([2.0], np.float32)),
        (np.array([3], np.int64), np.array([3.0], np.float32)),
    ]
    out = fingerprints_to_csr(fps, min_occurrence=2, sort_bits=True)
    X, vocab = out.X, out.vocab

    np.testing.assert_array_equal(vocab.col_bits, np.array([2], dtype=np.int64))
    np.testing.assert_array_equal(vocab.df, np.array([2], dtype=np.int32))
    assert X.shape == (3, 1)
    np.testing.assert_allclose(X.toarray(), np.array([[1.0], [2.0], [0.0]], dtype=np.float32))


def test_fingerprints_to_csr_max_occurrence_int_filters_frequent_bits():
    fps = [
        np.array([1, 2], np.int64),
        np.array([2], np.int64),
        np.array([2, 3], np.int64),
    ]
    out = fingerprints_to_csr(fps, max_occurrence=2, sort_bits=True)
    X, vocab = out.X, out.vocab

    np.testing.assert_array_equal(vocab.col_bits, np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(vocab.df, np.array([1, 1], dtype=np.int32))
    np.testing.assert_allclose(
        X.toarray(),
        np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_fingerprints_to_csr_max_occurrence_fraction_filters_frequent_bits():
    fps = [
        np.array([1, 2], np.int64),
        np.array([2], np.int64),
        np.array([2, 3], np.int64),
        np.array([2], np.int64),
    ]
    out = fingerprints_to_csr(fps, max_occurrence=0.5, sort_bits=True)
    X, vocab = out.X, out.vocab

    np.testing.assert_array_equal(vocab.col_bits, np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(vocab.df, np.array([1, 1], dtype=np.int32))
    assert X.shape == (4, 2)


def test_fingerprints_to_csr_min_and_max_occurrence_together():
    fps = [
        np.array([1, 2], np.int64),
        np.array([2, 3], np.int64),
        np.array([2, 4], np.int64),
        np.array([2], np.int64),
    ]
    # df: 1->1, 2->4, 3->1, 4->1
    out = fingerprints_to_csr(
        fps,
        min_occurrence=1,
        max_occurrence=3,
        sort_bits=True,
    )
    np.testing.assert_array_equal(out.vocab.col_bits, np.array([1, 3, 4], dtype=np.int64))
    np.testing.assert_array_equal(out.vocab.df, np.array([1, 1, 1], dtype=np.int32))


def test_occurrence_threshold_validation():
    fps = [np.array([1], np.int64)]

    with pytest.raises(ValueError):
        fingerprints_to_csr(fps, min_occurrence=0)
    with pytest.raises(ValueError):
        fingerprints_to_csr(fps, max_occurrence=0)
    with pytest.raises(ValueError):
        fingerprints_to_csr(fps, max_occurrence=1.0)
    with pytest.raises(ValueError):
        fingerprints_to_csr(fps, max_occurrence=-0.1)
    with pytest.raises(ValueError):
        fingerprints_to_csr(fps, min_occurrence=3, max_occurrence=2)


def test_occurrence_threshold_type_validation():
    fps = [np.array([1], np.int64)]

    with pytest.raises(TypeError):
        fingerprints_to_csr(fps, min_occurrence="2")
    with pytest.raises(TypeError):
        fingerprints_to_csr(fps, max_occurrence="2")


# ----------------------------
# fingerprints_to_tfidf
# ----------------------------

def test_fingerprints_to_tfidf_binary_idf_only():
    fps = [
        np.array([1, 2], np.int64),
        np.array([2], np.int64),
    ]
    out = fingerprints_to_tfidf(fps, sort_bits=True)
    X, vocab, idf = out.X, out.vocab, out.idf

    assert idf is not None
    np.testing.assert_array_equal(vocab.col_bits, np.array([1, 2], dtype=np.int64))
    np.testing.assert_allclose(idf, np.array([1.0, 0.0], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(
        X.toarray(),
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )


def test_fingerprints_to_tfidf_count_tf_times_idf():
    fps = [
        (np.array([1, 2], np.int64), np.array([2.0, 1.0], np.float32)),
        (np.array([2], np.int64), np.array([3.0], np.float32)),
    ]
    out = fingerprints_to_tfidf(fps, sort_bits=True)
    X, idf = out.X.toarray(), out.idf

    assert idf is not None
    np.testing.assert_allclose(idf, np.array([1.0, 0.0], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(X, np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32))


def test_fingerprints_to_tfidf_respects_min_occurrence_before_idf():
    fps = [
        (np.array([1], np.int64), np.array([1.0], np.float32)),
        (np.array([2], np.int64), np.array([1.0], np.float32)),
        (np.array([2], np.int64), np.array([1.0], np.float32)),
    ]
    out = fingerprints_to_tfidf(fps, min_occurrence=2, sort_bits=True)
    expected_idf = np.log(3.0 / 2.0) / np.log(3.0)

    np.testing.assert_array_equal(out.vocab.col_bits, np.array([2], np.int64))
    assert out.idf is not None
    assert out.idf[0] == pytest.approx(expected_idf, rel=1e-6)


def test_fingerprints_to_tfidf_with_tf_transform():
    fps = [
        (np.array([1, 2], np.int64), np.array([1.0, 3.0], np.float32)),
        (np.array([2], np.int64), np.array([1.0], np.float32)),
    ]
    out = fingerprints_to_tfidf(fps, sort_bits=True, tf_transform=np.log1p)

    # df(1)=1 -> idf=1
    # df(2)=2 -> idf=0
    expected = np.array([[np.log1p(1.0), 0.0], [0.0, 0.0]], dtype=np.float32)

    assert out.idf is not None
    np.testing.assert_allclose(out.idf, np.array([1.0, 0.0], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(out.X.toarray(), expected, rtol=1e-6)


def test_fingerprints_to_tfidf_empty_input():
    out = fingerprints_to_tfidf([], sort_bits=True)
    _assert_csr_shape_dtype(out.X, (0, 0), np.float32)
    assert out.idf is None
    assert out.vocab.col_bits.size == 0
    assert out.vocab.df.size == 0


# ----------------------------
# fold_csr_mod
# ----------------------------

def test_fold_csr_mod_basic():
    X = sp.csr_matrix(np.array([[1, 0, 2, 0, 3, 0]], dtype=np.float32))
    Y = fold_csr_mod(X, n_folded_features=3)

    np.testing.assert_allclose(
        Y.toarray(),
        np.array([[1.0, 3.0, 2.0]], dtype=np.float32),
    )


def test_fold_csr_mod_sums_collisions():
    X = sp.csr_matrix(np.array([[1, 2, 3, 4]], dtype=np.float32))
    Y = fold_csr_mod(X, n_folded_features=2)

    # cols 0,2 -> folded col 0 ; cols 1,3 -> folded col 1
    np.testing.assert_allclose(Y.toarray(), np.array([[4.0, 6.0]], dtype=np.float32))


def test_fold_csr_mod_empty_matrix():
    X = sp.csr_matrix((3, 5), dtype=np.float32)
    Y = fold_csr_mod(X, n_folded_features=7)
    _assert_csr_shape_dtype(Y, (3, 7), np.float32)
    assert Y.nnz == 0


def test_fold_csr_mod_requires_csr():
    X = sp.csc_matrix(np.array([[1.0, 2.0]], dtype=np.float32))
    with pytest.raises(TypeError):
        fold_csr_mod(X, n_folded_features=2)


def test_fold_csr_mod_requires_positive_dimension():
    X = sp.csr_matrix(np.array([[1.0]], dtype=np.float32))
    with pytest.raises(ValueError):
        fold_csr_mod(X, n_folded_features=0)


# ----------------------------
# folded public APIs
# ----------------------------

def test_fingerprints_to_csr_folded_matches_manual_fold_after_filtering():
    fps = [
        np.array([1, 2, 10], np.int64),
        np.array([2, 10], np.int64),
    ]
    out = fingerprints_to_csr_folded(fps, n_folded_features=4, min_occurrence=2, sort_bits=True)
    Xf = out.X.toarray()

    expected = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(Xf, expected)


def test_fingerprints_to_tfidf_folded_dense_shape():
    fps = [
        (np.array([1, 2], np.int64), np.array([2.0, 1.0], np.float32)),
        (np.array([2], np.int64), np.array([3.0], np.float32)),
        (np.array([3], np.int64), np.array([1.0], np.float32)),
    ]
    out = fingerprints_to_tfidf_folded(
        fps,
        n_folded_features=8,
        min_occurrence=None,
        max_occurrence=None,
        sort_bits=True,
    )
    assert out.X.shape == (3, 8)
    assert out.idf is not None
    assert out.vocab.col_bits.ndim == 1
    assert out.vocab.df.ndim == 1


def test_fingerprints_to_tfidf_folded_can_be_densified_fixed_length():
    fps = [
        np.array([1, 2], np.int64),
        np.array([2], np.int64),
        np.array([3], np.int64),
    ]
    out = fingerprints_to_tfidf_folded(fps, n_folded_features=4096, sort_bits=True)
    dense = out.X.toarray()
    assert dense.shape == (3, 4096)
    assert dense.dtype == np.float32


# ----------------------------
# Order / stability
# ----------------------------

def test_sort_bits_true_is_deterministic_vocab_order():
    fps = [
        np.array([10, 2], np.int64),
        np.array([2, 5], np.int64),
    ]
    out = fingerprints_to_csr(fps, sort_bits=True)
    np.testing.assert_array_equal(out.vocab.col_bits, np.array([2, 5, 10], dtype=np.int64))


def test_sort_bits_false_preserves_first_seen_vocab_order():
    fps = [
        np.array([10, 2], np.int64),
        np.array([2, 5], np.int64),
    ]
    out = fingerprints_to_csr(fps, sort_bits=False)
    np.testing.assert_array_equal(out.vocab.col_bits, np.array([10, 2, 5], dtype=np.int64))


# ----------------------------
# _restrict_vocab_top_k_frequency
# ----------------------------

def test_restrict_vocab_top_k_frequency_keeps_top_k_by_df():
    vocab = Vocabulary(
        col_bits=np.array([10, 20, 30, 40], dtype=np.int64),
        df=np.array([3, 7, 5, 1], dtype=np.int32),
        bit_to_col={10: 0, 20: 1, 30: 2, 40: 3},
    )

    reduced = _restrict_vocab_top_k_frequency(
        vocab,
        n_rows=10,
        max_features=2,
        exclude_constant_bits=True,
        return_bit_to_col=True,
    )

    # Top-2 by df are bits 20 (7) and 30 (5)
    np.testing.assert_array_equal(reduced.col_bits, np.array([20, 30], dtype=np.int64))
    np.testing.assert_array_equal(reduced.df, np.array([7, 5], dtype=np.int32))
    assert reduced.bit_to_col == {20: 0, 30: 1}


def test_restrict_vocab_top_k_frequency_excludes_constant_bits_by_default():
    vocab = Vocabulary(
        col_bits=np.array([10, 20, 30], dtype=np.int64),
        df=np.array([5, 4, 3], dtype=np.int32),
        bit_to_col=None,
    )

    reduced = _restrict_vocab_top_k_frequency(
        vocab,
        n_rows=5,
        max_features=3,
        exclude_constant_bits=True,
        return_bit_to_col=False,
    )

    np.testing.assert_array_equal(reduced.col_bits, np.array([20, 30], dtype=np.int64))
    np.testing.assert_array_equal(reduced.df, np.array([4, 3], dtype=np.int32))
    assert reduced.bit_to_col is None


def test_restrict_vocab_top_k_frequency_can_keep_constant_bits_if_requested():
    vocab = Vocabulary(
        col_bits=np.array([10, 20, 30], dtype=np.int64),
        df=np.array([5, 4, 3], dtype=np.int32),
        bit_to_col=None,
    )

    reduced = _restrict_vocab_top_k_frequency(
        vocab,
        n_rows=5,
        max_features=2,
        exclude_constant_bits=False,
        return_bit_to_col=False,
    )

    np.testing.assert_array_equal(reduced.col_bits, np.array([10, 20], dtype=np.int64))
    np.testing.assert_array_equal(reduced.df, np.array([5, 4], dtype=np.int32))


def test_restrict_vocab_top_k_frequency_ties_preserve_existing_vocab_order():
    vocab = Vocabulary(
        col_bits=np.array([50, 10, 30, 20], dtype=np.int64),
        df=np.array([4, 4, 4, 1], dtype=np.int32),
        bit_to_col=None,
    )

    reduced = _restrict_vocab_top_k_frequency(
        vocab,
        n_rows=10,
        max_features=2,
        exclude_constant_bits=True,
        return_bit_to_col=False,
    )

    # Stable tie-breaking should keep the existing order among equal-df features.
    np.testing.assert_array_equal(reduced.col_bits, np.array([50, 10], dtype=np.int64))
    np.testing.assert_array_equal(reduced.df, np.array([4, 4], dtype=np.int32))


def test_restrict_vocab_top_k_frequency_handles_empty_vocab():
    vocab = Vocabulary(
        col_bits=np.array([], dtype=np.int64),
        df=np.array([], dtype=np.int32),
        bit_to_col=None,
    )

    reduced = _restrict_vocab_top_k_frequency(
        vocab,
        n_rows=5,
        max_features=3,
        exclude_constant_bits=True,
        return_bit_to_col=True,
    )

    np.testing.assert_array_equal(reduced.col_bits, np.array([], dtype=np.int64))
    np.testing.assert_array_equal(reduced.df, np.array([], dtype=np.int32))
    assert reduced.bit_to_col == {}


def test_restrict_vocab_top_k_frequency_all_features_removed_by_constant_filter():
    vocab = Vocabulary(
        col_bits=np.array([1, 2], dtype=np.int64),
        df=np.array([5, 5], dtype=np.int32),
        bit_to_col=None,
    )

    reduced = _restrict_vocab_top_k_frequency(
        vocab,
        n_rows=5,
        max_features=2,
        exclude_constant_bits=True,
        return_bit_to_col=True,
    )

    np.testing.assert_array_equal(reduced.col_bits, np.array([], dtype=np.int64))
    np.testing.assert_array_equal(reduced.df, np.array([], dtype=np.int32))
    assert reduced.bit_to_col == {}


def test_restrict_vocab_top_k_frequency_requires_positive_max_features():
    vocab = Vocabulary(
        col_bits=np.array([1], dtype=np.int64),
        df=np.array([1], dtype=np.int32),
        bit_to_col=None,
    )

    with pytest.raises(ValueError):
        _restrict_vocab_top_k_frequency(
            vocab,
            n_rows=3,
            max_features=0,
            exclude_constant_bits=True,
            return_bit_to_col=False,
        )


# ----------------------------
# _fingerprints_to_csr_with_vocab
# ----------------------------

def test_fingerprints_to_csr_with_vocab_binary_restricts_to_given_vocab():
    fps = [
        np.array([2, 5, 9], dtype=np.int64),
        np.array([5, 7], dtype=np.int64),
    ]
    vocab = Vocabulary(
        col_bits=np.array([5, 9], dtype=np.int64),
        df=np.array([2, 1], dtype=np.int32),
        bit_to_col={5: 0, 9: 1},
    )

    X = _fingerprints_to_csr_with_vocab(
        fps,
        vocab,
        dtype=np.float32,
        sort_indices_within_rows=True,
        consolidate_duplicates_within_rows=True,
        tf_transform=None,
    )

    _assert_csr_shape_dtype(X, (2, 2), np.float32)
    np.testing.assert_allclose(
        X.toarray(),
        np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )


def test_fingerprints_to_csr_with_vocab_counts_and_tf_transform():
    fps = [
        (np.array([2, 5, 5], dtype=np.int64), np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        (np.array([5, 7], dtype=np.int64), np.array([4.0, 1.0], dtype=np.float32)),
    ]
    vocab = Vocabulary(
        col_bits=np.array([5], dtype=np.int64),
        df=np.array([2], dtype=np.int32),
        bit_to_col={5: 0},
    )

    X = _fingerprints_to_csr_with_vocab(
        fps,
        vocab,
        dtype=np.float32,
        sort_indices_within_rows=True,
        consolidate_duplicates_within_rows=True,
        tf_transform=np.log1p,
    )

    # row0: 5 -> 2+3 = 5 -> log1p(5)
    # row1: 5 -> 4 -> log1p(4)
    np.testing.assert_allclose(
        X.toarray(),
        np.array([[np.log1p(5.0)], [np.log1p(4.0)]], dtype=np.float32),
        rtol=1e-6,
    )


def test_fingerprints_to_csr_with_vocab_builds_mapping_if_missing():
    fps = [np.array([9, 1], dtype=np.int64)]
    vocab = Vocabulary(
        col_bits=np.array([1, 9], dtype=np.int64),
        df=np.array([1, 1], dtype=np.int32),
        bit_to_col=None,
    )

    X = _fingerprints_to_csr_with_vocab(
        fps,
        vocab,
        dtype=np.float32,
        sort_indices_within_rows=True,
        consolidate_duplicates_within_rows=True,
        tf_transform=None,
    )

    np.testing.assert_allclose(X.toarray(), np.array([[1.0, 1.0]], dtype=np.float32))


def test_fingerprints_to_csr_with_vocab_empty_vocab_returns_zero_column_matrix():
    fps = [
        np.array([1, 2], dtype=np.int64),
        np.array([3], dtype=np.int64),
    ]
    vocab = Vocabulary(
        col_bits=np.array([], dtype=np.int64),
        df=np.array([], dtype=np.int32),
        bit_to_col={},
    )

    X = _fingerprints_to_csr_with_vocab(
        fps,
        vocab,
        dtype=np.float32,
        sort_indices_within_rows=True,
        consolidate_duplicates_within_rows=True,
        tf_transform=None,
    )

    _assert_csr_shape_dtype(X, (2, 0), np.float32)
    assert X.nnz == 0


def test_fingerprints_to_csr_with_vocab_rejects_mixed_input_types():
    fps = [
        np.array([1], dtype=np.int64),
        (np.array([1], dtype=np.int64), np.array([1.0], dtype=np.float32)),
    ]
    vocab = Vocabulary(
        col_bits=np.array([1], dtype=np.int64),
        df=np.array([1], dtype=np.int32),
        bit_to_col={1: 0},
    )

    with pytest.raises(TypeError):
        _fingerprints_to_csr_with_vocab(
            fps,
            vocab,
            dtype=np.float32,
            sort_indices_within_rows=True,
            consolidate_duplicates_within_rows=True,
            tf_transform=None,
        )


# ----------------------------
# fingerprints_to_csr_frequency_folded
# ----------------------------

def test_fingerprints_to_csr_frequency_folded_keeps_most_frequent_bits():
    fps = [
        np.array([1, 2, 3], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
        np.array([2], dtype=np.int64),
        np.array([4], dtype=np.int64),
    ]
    # df: 1->1, 2->3, 3->2, 4->1
    out = fingerprints_to_csr_frequency_folded(
        fps,
        n_frequency_features=2,
        sort_bits=True,
        exclude_constant_bits=True,
        return_bit_to_col=True,
    )

    np.testing.assert_array_equal(out.vocab.col_bits, np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(out.vocab.df, np.array([3, 2], dtype=np.int32))
    assert out.vocab.bit_to_col == {2: 0, 3: 1}
    np.testing.assert_allclose(
        out.X.toarray(),
        np.array(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_fingerprints_to_csr_frequency_folded_excludes_constant_bits_by_default():
    fps = [
        np.array([1, 2], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        np.array([1, 4], dtype=np.int64),
    ]
    # df: 1->3 (constant), 2->1, 3->1, 4->1
    out = fingerprints_to_csr_frequency_folded(
        fps,
        n_frequency_features=2,
        sort_bits=True,
        exclude_constant_bits=True,
    )

    # bit 1 should be excluded despite highest df
    np.testing.assert_array_equal(out.vocab.col_bits, np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(out.vocab.df, np.array([1, 1], dtype=np.int32))
    np.testing.assert_allclose(
        out.X.toarray(),
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_fingerprints_to_csr_frequency_folded_can_keep_constant_bits_if_requested():
    fps = [
        np.array([1, 2], dtype=np.int64),
        np.array([1, 3], dtype=np.int64),
        np.array([1, 4], dtype=np.int64),
    ]
    out = fingerprints_to_csr_frequency_folded(
        fps,
        n_frequency_features=2,
        sort_bits=True,
        exclude_constant_bits=False,
    )

    np.testing.assert_array_equal(out.vocab.col_bits, np.array([1, 2], dtype=np.int64))
    np.testing.assert_array_equal(out.vocab.df, np.array([3, 1], dtype=np.int32))
    np.testing.assert_allclose(
        out.X.toarray(),
        np.array(
            [
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_fingerprints_to_csr_frequency_folded_respects_occurrence_filters_before_topk():
    fps = [
        np.array([1, 2], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
        np.array([2, 4], dtype=np.int64),
        np.array([5], dtype=np.int64),
    ]
    # df: 1->1, 2->3, 3->1, 4->1, 5->1
    # min_occurrence=2 keeps only bit 2, then top-k selects from that.
    out = fingerprints_to_csr_frequency_folded(
        fps,
        n_frequency_features=3,
        min_occurrence=2,
        sort_bits=True,
        exclude_constant_bits=True,
    )

    np.testing.assert_array_equal(out.vocab.col_bits, np.array([2], dtype=np.int64))
    np.testing.assert_array_equal(out.vocab.df, np.array([3], dtype=np.int32))
    np.testing.assert_allclose(
        out.X.toarray(),
        np.array([[1.0], [1.0], [1.0], [0.0]], dtype=np.float32),
    )


def test_fingerprints_to_csr_frequency_folded_counts_and_tf_transform():
    fps = [
        (np.array([1, 2, 2], dtype=np.int64), np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        (np.array([2, 3], dtype=np.int64), np.array([4.0, 1.0], dtype=np.float32)),
        (np.array([2], dtype=np.int64), np.array([5.0], dtype=np.float32)),
    ]
    # df: 1->1, 2->3, 3->1
    out = fingerprints_to_csr_frequency_folded(
        fps,
        n_frequency_features=1,
        sort_bits=True,
        exclude_constant_bits=False,
        tf_transform=np.log1p,
    )

    np.testing.assert_array_equal(out.vocab.col_bits, np.array([2], dtype=np.int64))
    np.testing.assert_array_equal(out.vocab.df, np.array([3], dtype=np.int32))
    np.testing.assert_allclose(
        out.X.toarray(),
        np.array(
            [
                [np.log1p(5.0)],
                [np.log1p(4.0)],
                [np.log1p(5.0)],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
    )


def test_fingerprints_to_csr_frequency_folded_empty_input():
    out = fingerprints_to_csr_frequency_folded(
        [],
        n_frequency_features=10,
        sort_bits=True,
        exclude_constant_bits=True,
        return_bit_to_col=True,
    )

    _assert_csr_shape_dtype(out.X, (0, 0), np.float32)
    assert out.idf is None
    assert out.vocab.bit_to_col == {}


def test_fingerprints_to_csr_frequency_folded_all_removed_by_constant_filter():
    fps = [
        np.array([1, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
    ]
    out = fingerprints_to_csr_frequency_folded(
        fps,
        n_frequency_features=5,
        sort_bits=True,
        exclude_constant_bits=True,
        return_bit_to_col=True,
    )

    _assert_csr_shape_dtype(out.X, (2, 0), np.float32)
    np.testing.assert_array_equal(out.vocab.col_bits, np.array([], dtype=np.int64))
    np.testing.assert_array_equal(out.vocab.df, np.array([], dtype=np.int32))
    assert out.vocab.bit_to_col == {}
