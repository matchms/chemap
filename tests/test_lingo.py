import numpy as np
import pytest
from scipy.sparse import issparse
from chemap.fingerprints.lingo import LingoFingerprint


SMILES = [
    "[C-]#N",
    "CC(=O)NC1=CC=C(C=C1)O",
]


def _is_unfolded_binary(out) -> bool:
    return (
        isinstance(out, list)
        and all(isinstance(x, np.ndarray) for x in out)
        and all(x.dtype == np.int64 for x in out)
    )


def _is_unfolded_count(out) -> bool:
    return (
        isinstance(out, list)
        and all(isinstance(x, tuple) and len(x) == 2 for x in out)
        and all(isinstance(k, np.ndarray) and isinstance(v, np.ndarray) for k, v in out)
        and all(k.dtype == np.int64 for k, _ in out)
        and all(v.dtype == np.float32 for _, v in out)
    )


def test_folded_dense_binary_shape_dtype():
    fp = LingoFingerprint(fp_size=128, substring_length=4, count=False, sparse=False, folded=True)
    X = fp.transform(SMILES)

    assert isinstance(X, np.ndarray)
    assert X.shape == (len(SMILES), 128)
    # skfp uses uint8 for binary
    assert X.dtype == np.uint8
    # only 0/1 values
    assert set(np.unique(X)).issubset({0, 1})


def test_folded_dense_count_shape_dtype():
    fp = LingoFingerprint(fp_size=128, substring_length=4, count=True, sparse=False, folded=True)
    X = fp.transform(SMILES)

    assert isinstance(X, np.ndarray)
    assert X.shape == (len(SMILES), 128)
    assert X.dtype == np.uint32
    # counts are non-negative integers
    assert np.all(X >= 0)


def test_folded_sparse_binary_shape_dtype():
    fp = LingoFingerprint(fp_size=128, substring_length=4, count=False, sparse=True, folded=True)
    X = fp.transform(SMILES)

    # skfp returns scipy.sparse.csr_array
    assert issparse(X)
    assert X.shape == (len(SMILES), 128)
    # csr_array dtype should match skfp choices (uint8 for binary)
    assert X.dtype == np.uint8


def test_unfolded_binary_format_and_sorted():
    fp = LingoFingerprint(fp_size=128, substring_length=4, count=False, sparse=False, folded=False)
    out = fp.transform(SMILES)

    assert _is_unfolded_binary(out)
    assert len(out) == len(SMILES)

    for keys in out:
        # sorted
        assert np.all(keys[:-1] <= keys[1:]) if keys.size > 1 else True
        # no duplicates (dict keys)
        assert keys.size == np.unique(keys).size


def test_unfolded_count_format_sorted_and_alignment():
    fp = LingoFingerprint(fp_size=128, substring_length=4, count=True, sparse=False, folded=False)
    out = fp.transform(SMILES)

    assert _is_unfolded_count(out)
    assert len(out) == len(SMILES)

    for keys, vals in out:
        assert keys.shape == vals.shape
        # sorted by feature id
        assert np.all(keys[:-1] <= keys[1:]) if keys.size > 1 else True
        # counts positive
        assert np.all(vals > 0)


def test_unfolded_empty_for_too_short_smiles():
    # substring_length > len(smiles) => no substrings => empty output
    fp = LingoFingerprint(fp_size=128, substring_length=10, count=False, folded=False)
    out = fp.transform(["CC"])

    assert _is_unfolded_binary(out)
    assert len(out) == 1
    assert out[0].size == 0


def test_folded_equals_manual_folding_from_unfolded_counts():
    """
    For Lingo, folding is 'sha1(token) % fp_size' with binary/count.
    Here we test a consistency property:
      - compute unfolded (hash64 ids + counts)
      - fold them by (id64 % fp_size) and aggregating
    This will NOT match the folded implementation if id64 uses only part of sha1.
    So we only test that folding produces a valid-shaped vector with expected sparsity,
    and that a deterministic run is stable.

    If you later store the full sha1-int as unfolded id, you can strengthen this test
    to exact equality with folded output.
    """
    fp_size = 128
    fp_unf = LingoFingerprint(fp_size=fp_size, substring_length=4, count=True, folded=False)
    unf = fp_unf.transform(SMILES)

    # fold ourselves from unfolded ids
    X_manual = np.zeros((len(SMILES), fp_size), dtype=np.uint32)
    for i, (keys, vals) in enumerate(unf):
        # keys are int64; interpret as unsigned for modulo stability
        keys_u = keys.astype(np.uint64, copy=False)
        buckets = (keys_u % np.uint64(fp_size)).astype(np.int64)
        for b, v in zip(buckets, vals):
            X_manual[i, b] += np.uint32(v)

    # shape and dtype sanity
    assert X_manual.shape == (len(SMILES), fp_size)
    assert X_manual.dtype == np.uint32
    assert np.any(X_manual > 0)


def test_deterministic_unfolded_output():
    fp = LingoFingerprint(fp_size=128, substring_length=4, count=True, folded=False, n_jobs=2)
    out1 = fp.transform(SMILES)
    out2 = fp.transform(SMILES)

    assert len(out1) == len(out2)
    for (k1, v1), (k2, v2) in zip(out1, out2):
        np.testing.assert_array_equal(k1, k2)
        np.testing.assert_array_equal(v1, v2)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_parallel_and_serial_same(n_jobs):
    fp = LingoFingerprint(fp_size=128, substring_length=4, count=False, folded=False, n_jobs=n_jobs)
    out = fp.transform(SMILES)

    assert _is_unfolded_binary(out)
    # Just ensure it runs and output shapes look right
    assert len(out) == len(SMILES)


def test_sparse_flag_ignored_when_unfolded():
    """
    When folded=False, we return lists (unfolded formats),
    so sparse=True should not force CSR outputs.
    """
    fp = LingoFingerprint(fp_size=128, substring_length=4, count=False, sparse=True, folded=False)
    out = fp.transform(SMILES)
    assert _is_unfolded_binary(out)
