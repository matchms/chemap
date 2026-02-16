import numpy as np
import pytest
import scipy.sparse as sp
from rdkit import Chem
from chemap.fingerprints import ElementCountFingerprint


def mol(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"RDKit failed to parse SMILES: {smiles}"
    return m


def assert_row(fp_out, row_idx: int, expected: dict, elem2idx: dict):
    """Helper: compare one output row with expected counts for selected elements."""
    row = fp_out[row_idx]
    for elem, count in expected.items():
        i = elem2idx[elem]
        assert float(row[i]) == pytest.approx(float(count))


def test_inferred_vocab_and_counts_implicit_h_dense():
    # ethanol, water, benzene
    X = [mol("CCO"), mol("O"), mol("c1ccccc1")]
    tr = ElementCountFingerprint(include_hs="implicit", unknown_policy="other", sparse=False)

    out = tr.fit_transform(X)
    assert out.dtype == np.float32
    assert out.shape[0] == 3

    # inferred elements should include at least C, O, H, Other (sorted + appended Other)
    assert "C" in tr.elements_
    assert "O" in tr.elements_
    assert "H" in tr.elements_
    assert "Other" in tr.elements_
    assert tr.elements_[-1] == "Other"

    elem2idx = tr._elem2idx_

    # ethanol: C2 O1 H6
    assert_row(out, 0, {"C": 2, "O": 1, "H": 6}, elem2idx)
    # water: O1 H2
    assert_row(out, 1, {"O": 1, "H": 2}, elem2idx)
    # benzene: C6 H6
    assert_row(out, 2, {"C": 6, "H": 6}, elem2idx)

    # "Other" should remain 0 for these molecules
    assert float(out[0, elem2idx["Other"]]) == 0.0
    assert float(out[1, elem2idx["Other"]]) == 0.0
    assert float(out[2, elem2idx["Other"]]) == 0.0


def test_include_hs_none_does_not_add_h_feature_when_inferred():
    X = [mol("CCO")]
    tr = ElementCountFingerprint(include_hs="none", unknown_policy="other", sparse=False)
    out = tr.fit_transform(X)

    assert "H" not in tr.elements_
    assert out.shape[1] == len(tr.elements_)
    elem2idx = tr._elem2idx_
    assert_row(out, 0, {"C": 2, "O": 1}, elem2idx)


def test_unknown_policy_other_accumulates_unknown_elements():
    # Fix vocabulary to only C (+Other)
    X = [mol("CCO")]
    tr = ElementCountFingerprint(elements=["C"], include_hs="none", unknown_policy="other", sparse=False)
    out = tr.fit_transform(X)

    assert tr.elements_ == ["C", "Other"]
    elem2idx = tr._elem2idx_

    # ethanol: C2, and O is unknown -> Other1
    assert_row(out, 0, {"C": 2, "Other": 1}, elem2idx)


def test_unknown_policy_ignore_drops_unknown_elements():
    X = [mol("CCO")]
    tr = ElementCountFingerprint(elements=["C"], include_hs="none", unknown_policy="ignore", sparse=False)
    out = tr.fit_transform(X)

    assert tr.elements_ == ["C"]
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(2.0)


def test_unknown_policy_error_raises_on_unknown():
    X = [mol("CCO")]
    tr = ElementCountFingerprint(elements=["C"], include_hs="none", unknown_policy="error", sparse=False)

    with pytest.raises(ValueError, match="Unknown element"):
        tr.fit_transform(X)


def test_sparse_output_is_csr_float32_and_matches_dense_values():
    X = [mol("CCO"), mol("O")]
    tr_dense = ElementCountFingerprint(include_hs="implicit", unknown_policy="other", sparse=False)
    tr_sparse = ElementCountFingerprint(include_hs="implicit", unknown_policy="other", sparse=True)

    dense = tr_dense.fit_transform(X)
    sparse = tr_sparse.fit_transform(X)

    assert sp.isspmatrix_csr(sparse)
    assert sparse.dtype == np.float32
    np.testing.assert_allclose(sparse.toarray(), dense, rtol=0, atol=0)


def test_transform_without_fit_behaves_like_fit_transform():
    X = [mol("CCO"), mol("O")]
    tr1 = ElementCountFingerprint(include_hs="implicit", unknown_policy="other", sparse=False)
    tr2 = ElementCountFingerprint(include_hs="implicit", unknown_policy="other", sparse=False)

    out1 = tr1.fit_transform(X)
    out2 = tr2.transform(X)  # should auto-fit internally

    # Since both infer the same vocab from same X, results should match
    assert tr1.elements_ == tr2.elements_
    np.testing.assert_allclose(out2, out1, rtol=0, atol=0)


def test_parallel_consistency():
    X = [mol("CCO"), mol("O"), mol("c1ccccc1")]
    tr1 = ElementCountFingerprint(include_hs="implicit", unknown_policy="other", sparse=False, n_jobs=1)
    tr2 = ElementCountFingerprint(include_hs="implicit", unknown_policy="other", sparse=False, n_jobs=2)

    out1 = tr1.fit_transform(X)
    out2 = tr2.fit_transform(X)

    assert tr1.elements_ == tr2.elements_
    np.testing.assert_allclose(out2, out1, rtol=0, atol=0)


def test_explicit_h_counts_once_with_addhs():
    # methane with explicit H atoms
    m = Chem.AddHs(mol("C"))  # CH4 explicit
    tr = ElementCountFingerprint(elements=["C", "H"], include_hs="explicit", unknown_policy="error", sparse=False)

    out = tr.fit_transform([m])
    elem2idx = tr._elem2idx_

    # Expected: C1 H4
    assert_row(out, 0, {"C": 1, "H": 4}, elem2idx)
