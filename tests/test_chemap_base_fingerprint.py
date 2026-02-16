from collections.abc import Sequence
from typing import Any
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Mol
from scipy.sparse import csr_array, issparse
from chemap.fingerprints import ChemapBaseFingerprint
from chemap.types import UnfoldedBinary, UnfoldedCount


SMILES = ["CCO", "CCN", "c1ccccc1"]


# -----------------------
# Minimal dummy subclasses
# -----------------------

class DummyFoldedFP(ChemapBaseFingerprint):
    """
    Implements only folded behavior (via BaseFingerprintTransformer pipeline).
    """
    def __init__(self, *, n_features_out=8, **kwargs: Any):
        super().__init__(n_features_out=n_features_out, **kwargs)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray:
        # Return a deterministic (n_samples, D) uint8 matrix.
        n = len(X)
        D = self.n_features_out
        out = np.zeros((n, D), dtype=np.uint8)
        # set one bit based on index to make rows non-identical
        for i in range(n):
            out[i, i % D] = 1
        return out


class DummyUnfoldedBinaryFP(ChemapBaseFingerprint):
    """
    Implements unfolded binary output.
    """
    def __init__(self, *, n_features_out=8, **kwargs: Any):
        super().__init__(n_features_out=8, **kwargs)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray:
        raise AssertionError("Folded path should not call _calculate_fingerprint when folded=False")

    def _calculate_unfolded(self, X_smiles: Sequence[str]) -> UnfoldedBinary:
        # One feature per molecule: feature id = len(smiles)
        return [np.array([len(s)], dtype=np.int64) for s in X_smiles]


class DummyUnfoldedCountFP(ChemapBaseFingerprint):
    """
    Implements unfolded count output.
    """
    def __init__(self, *, n_features_out=8, **kwargs: Any):
        super().__init__(n_features_out=8, count=True, **kwargs)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray:
        raise AssertionError("Folded path should not call _calculate_fingerprint when folded=False")

    def _calculate_unfolded(self, X_smiles: Sequence[str]) -> UnfoldedCount:
        # Two features per molecule: ids are [1, 2], values are [len, len+1]
        out: UnfoldedCount = []
        for s in X_smiles:
            keys = np.array([1, 2], dtype=np.int64)
            vals = np.array([float(len(s)), float(len(s) + 1)], dtype=np.float32)
            out.append((keys, vals))
        return out


class DummyFoldedFPReturningSparse(ChemapBaseFingerprint):
    def __init__(self, *, n_features_out=8, **kwargs):
        super().__init__(n_features_out=n_features_out, **kwargs)

    def _calculate_fingerprint(self, X):
        n = len(X)
        D = self.n_features_out
        dense = np.zeros((n, D), dtype=np.uint8)
        for i in range(n):
            dense[i, i % D] = 1
        return csr_array(dense) if self.sparse else dense


# -----------------------
# Helper checkers
# -----------------------

def _is_unfolded_binary(x) -> bool:
    return (
        isinstance(x, list)
        and all(isinstance(a, np.ndarray) and a.dtype == np.int64 for a in x)
    )


def _is_unfolded_count(x) -> bool:
    return (
        isinstance(x, list)
        and all(isinstance(t, tuple) and len(t) == 2 for t in x)
        and all(isinstance(k, np.ndarray) and k.dtype == np.int64 for k, _ in x)
        and all(isinstance(v, np.ndarray) and v.dtype == np.float32 for _, v in x)
    )


# -----------------------
# Tests
# -----------------------

def test_folded_true_calls_super_transform_dense():
    fp = DummyFoldedFP(folded=True, sparse=False, count=False)
    X = fp.transform(SMILES)

    assert isinstance(X, np.ndarray)
    assert X.shape == (len(SMILES), fp.n_features_out)
    assert X.dtype == np.uint8
    # each row has exactly one 1
    assert np.all(X.sum(axis=1) == 1)


def test_folded_true_sparse_when_subclass_returns_sparse():
    fp = DummyFoldedFPReturningSparse(folded=True, sparse=True, count=False)
    X = fp.transform(SMILES)
    assert issparse(X)
    assert X.shape == (len(SMILES), fp.n_features_out)
    assert X.dtype == np.uint8


def test_folded_false_binary_uses_ensure_smiles_accepts_mols():
    mols = [Chem.MolFromSmiles(s) for s in SMILES]
    assert all(m is not None for m in mols)

    fp = DummyUnfoldedBinaryFP(folded=False, count=False)
    out = fp.transform(mols)  # pass mols to force ensure_smiles conversion

    assert _is_unfolded_binary(out)
    assert len(out) == len(SMILES)
    # feature id == len(smiles)
    for s, arr in zip(SMILES, out):
        np.testing.assert_array_equal(arr, np.array([len(s)], dtype=np.int64))


def test_folded_false_count_uses_ensure_smiles_and_returns_float32_vals():
    fp = DummyUnfoldedCountFP(folded=False)
    out = fp.transform(SMILES)

    assert _is_unfolded_count(out)
    assert len(out) == len(SMILES)
    for s, (keys, vals) in zip(SMILES, out):
        np.testing.assert_array_equal(keys, np.array([1, 2], dtype=np.int64))
        np.testing.assert_array_equal(vals, np.array([float(len(s)), float(len(s) + 1)], dtype=np.float32))


def test_parallel_map_batch_size_none_is_ok():
    """
    Regression test: joblib Parallel rejects batch_size=None.
    ChemapBaseFingerprint should convert None -> 'auto'.
    """
    fp = DummyUnfoldedBinaryFP(folded=False, n_jobs=2, batch_size=None)
    out = fp.transform(SMILES)

    assert _is_unfolded_binary(out)
    assert len(out) == len(SMILES)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_parallel_map_deterministic(n_jobs):
    fp = DummyUnfoldedCountFP(folded=False, n_jobs=n_jobs)
    out1 = fp.transform(SMILES)
    out2 = fp.transform(SMILES)

    assert len(out1) == len(out2)
    for (k1, v1), (k2, v2) in zip(out1, out2):
        np.testing.assert_array_equal(k1, k2)
        np.testing.assert_array_equal(v1, v2)


def test_folded_false_default_base_raises_not_implemented():
    """
    If a subclass forgets to implement _calculate_unfolded, folded=False should raise.
    """
    class IncompleteFP(ChemapBaseFingerprint):
        def __init__(self):
            super().__init__(n_features_out=8, folded=False)

        def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray:
            return np.zeros((len(X), 8), dtype=np.uint8)

    fp = IncompleteFP()
    with pytest.raises(NotImplementedError):
        fp.transform(SMILES)
