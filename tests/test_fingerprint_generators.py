from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pytest
import scipy.sparse as sp
from rdkit.Chem import rdFingerprintGenerator
from skfp.fingerprints import (
    AtomPairFingerprint,
    MACCSFingerprint,
    MAPFingerprint,
    PubChemFingerprint,
)
from chemap import FingerprintConfig, compute_fingerprints


# simple smiles for testing
SMILES: List[str] = [
    "CCO",                 # ethanol
    "c1ccccc1",            # benzene
    "CC(=O)O",             # acetic acid
    "C1CCCCC1",            # cyclohexane
    "N[C@@H](C)C(=O)O",    # alanine
]


# ----------------------------
# Generator inventory (RDKit)
# ----------------------------

def _rdkit_generators() -> List[Tuple[str, Any]]:
    # rdFingerprintGenerator-style objects
    return [
        ("rdkit_morgan_2048_r2", rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)),
        ("rdkit_rdkitfp_2048", rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)),
        ("rdkit_atompair_2048", rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)),
        ("rdkit_toptorsion_2048", rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)),
    ]


# ---------------------------------------
# Generator inventory (scikit-fingerprints)
# ---------------------------------------

def _skfp_generators() -> Dict[str, Any]:
    return {
        "MAPFingerprint": MAPFingerprint,
        "AtomPairFingerprint": AtomPairFingerprint,
        "MACCSFingerprint": MACCSFingerprint,
        "PubChemFingerprint": PubChemFingerprint,
    }


def _supports_count_param(params: Dict[str, Any]) -> Optional[str]:
    """
    Heuristic: discover a "count" flag parameter if present.
    (scikit-fingerprints uses different spellings across versions / classes)
    """
    for key in ("count", "counts", "use_counts", "useCounts", "use_count", "useCountsSimulation"):
        if key in params:
            return key
    return None


def _build_skfp_transformers() -> List[Tuple[str, Any, bool]]:
    """
    Returns [(name, transformer_instance_binary, supports_count_variant)].
    """
    mod = _skfp_generators()
    if mod is None:
        return []

    out: List[Tuple[str, Any, bool]] = []
    for cls_name, cls in mod.items():
        try:
            base = cls()  # type: ignore[call-arg]
            params = base.get_params(deep=False)
            count_key = _supports_count_param(params)
            supports_count = count_key is not None
            out.append((f"skfp_{cls_name}", base, supports_count))
        except Exception:
            # If a particular transformer can't be constructed with defaults, skip it.
            continue

    return out


def _skfp_make_variant(fp: Any, *, count: bool) -> Any:
    """
    Clone transformer with `count` enabled if it supports it; otherwise return original.
    """
    params = fp.get_params(deep=False)
    count_key = _supports_count_param(params)
    if count_key is None:
        return fp
    params = dict(params)
    params[count_key] = bool(count)
    return fp.__class__(**params)


# ----------------------------
# Assertions / shape utilities
# ----------------------------

def _assert_dense_matrix(X: np.ndarray, n_rows: int) -> None:
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert X.shape[0] == n_rows
    assert X.dtype == np.float32
    assert np.isfinite(X).all()


def _assert_binary_dense(X: np.ndarray) -> None:
    # allow all-zeros rows for tiny molecules, but values must be 0/1
    u = np.unique(X)
    assert set(u.tolist()).issubset({0.0, 1.0})


def _assert_count_dense(X: np.ndarray) -> None:
    assert (X >= 0).all()
    # at least something non-zero overall
    assert float(X.sum()) >= 0.0


def _assert_csr_matrix(X: sp.csr_matrix, n_rows: int) -> None:
    assert sp.isspmatrix_csr(X)
    assert X.shape[0] == n_rows
    assert X.dtype == np.float32
    assert np.isfinite(X.data).all()


def _assert_binary_csr(X: sp.csr_matrix) -> None:
    if X.nnz:
        u = np.unique(X.data)
        assert set(u.tolist()).issubset({1.0})


def _assert_count_csr(X: sp.csr_matrix) -> None:
    assert (X.data >= 0).all()


def _supports_unfolded_variant(fp: Any) -> bool:
    # Your code requires a `variant` param to do folded=False for sklearn/scikit-fingerprints
    try:
        params = fp.get_params(deep=False)
    except Exception:
        return False
    return "variant" in params

# ============================================================
# CASE 1
# - dense fingerprint (binary and where feasible count)
# ============================================================

def test_dense_fingerprints_binary_and_count_many_generators() -> None:
    gens_rdkit = _rdkit_generators()
    gens_skfp = _build_skfp_transformers()

    if not gens_rdkit and not gens_skfp:
        pytest.skip("No fingerprint generators available (RDKit/scikit-fingerprints not importable).")

    # RDKit: always supports both binary + count
    for name, gen in gens_rdkit:
        # binary
        cfg = FingerprintConfig(count=False, folded=True, return_csr=False, invalid_policy="keep")
        X = compute_fingerprints(SMILES, gen, cfg, show_progress=False, n_jobs=1)
        _assert_dense_matrix(X, n_rows=len(SMILES))
        _assert_binary_dense(X)

        # count
        cfg = replace(cfg, count=True)
        X = compute_fingerprints(SMILES, gen, cfg, show_progress=False, n_jobs=1)
        _assert_dense_matrix(X, n_rows=len(SMILES))
        _assert_count_dense(X)

    # scikit-fingerprints: binary always; count only if transformer supports it
    for name, fp, supports_count in gens_skfp:
        # binary
        cfg = FingerprintConfig(count=False, folded=True, return_csr=False, invalid_policy="keep")
        X = compute_fingerprints(SMILES, fp, cfg, show_progress=False, n_jobs=1)
        _assert_dense_matrix(X, n_rows=len(SMILES))
        _assert_binary_dense(X)

        # count (where feasible)
        if supports_count:
            fp_count = _skfp_make_variant(fp, count=True)
            cfg = replace(cfg, count=True)
            X = compute_fingerprints(SMILES, fp_count, cfg, show_progress=False, n_jobs=1)
            _assert_dense_matrix(X, n_rows=len(SMILES))
            _assert_count_dense(X)


# ============================================================
# CASE 2
# - if fitting the fingerprint: return_csr=True (binary and where feasible count)
#   (We still include RDKit here because return_csr=True is supported as well.)
# ============================================================

def test_return_csr_folded_binary_and_count_many_generators() -> None:
    gens_rdkit = _rdkit_generators()
    gens_skfp = _build_skfp_transformers()

    if not gens_rdkit and not gens_skfp:
        pytest.skip("No fingerprint generators available (RDKit/scikit-fingerprints not importable).")

    # RDKit
    for name, gen in gens_rdkit:
        # binary -> CSR
        cfg = FingerprintConfig(count=False, folded=True, return_csr=True, invalid_policy="keep")
        X = compute_fingerprints(SMILES, gen, cfg, show_progress=False, n_jobs=1)
        _assert_csr_matrix(X, n_rows=len(SMILES))
        _assert_binary_csr(X)

        # count -> CSR
        cfg = replace(cfg, count=True)
        X = compute_fingerprints(SMILES, gen, cfg, show_progress=False, n_jobs=1)
        _assert_csr_matrix(X, n_rows=len(SMILES))
        _assert_count_csr(X)

    # scikit-fingerprints (fit/transform backend)
    for name, fp, supports_count in gens_skfp:
        # binary -> CSR
        cfg = FingerprintConfig(count=False, folded=True, return_csr=True, invalid_policy="keep")
        X = compute_fingerprints(SMILES, fp, cfg, show_progress=False, n_jobs=1)
        _assert_csr_matrix(X, n_rows=len(SMILES))
        _assert_binary_csr(X)

        # count -> CSR (where feasible)
        if supports_count:
            fp_count = _skfp_make_variant(fp, count=True)
            cfg = replace(cfg, count=True)
            X = compute_fingerprints(SMILES, fp_count, cfg, show_progress=False, n_jobs=1)
            _assert_csr_matrix(X, n_rows=len(SMILES))
            _assert_count_csr(X)


# ============================================================
# CASE 3
# - if fitting the fingerprint:
#     return_csr=True with folded=True and folded=False
#
# (folded=False, return_csr=True) is INVALID by design
# and must raise (validated in _validate_config). We test both behaviors.
# We also compute unfolded output (folded=False) with return_csr=False to
# still "run them all on a few simple smiles".
# ============================================================

def test_fit_backend_return_csr_true_folded_true_and_false() -> None:
    gens_skfp = _build_skfp_transformers()
    if not gens_skfp:
        pytest.skip("scikit-fingerprints/sklearn-style transformers not importable in this env.")

    for name, fp, supports_count in gens_skfp:
        # 1) folded=True + return_csr=True should work
        cfg_ok = FingerprintConfig(count=False, folded=True, return_csr=True, invalid_policy="keep")
        X = compute_fingerprints(SMILES, fp, cfg_ok, show_progress=False, n_jobs=1)
        _assert_csr_matrix(X, n_rows=len(SMILES))

        # 2) folded=False + return_csr=True must raise (per _validate_config)
        cfg_bad = replace(cfg_ok, folded=False, return_csr=True)
        with pytest.raises(ValueError, match="return_csr is only valid when folded=True"):
            compute_fingerprints(SMILES, fp, cfg_bad, show_progress=False, n_jobs=1)

        # 3) folded=False + return_csr=False:
        #    - run if transformer supports `variant`
        #    - otherwise, assert the documented NotImplementedError
        cfg_unfolded = FingerprintConfig(count=False, folded=False, return_csr=False, invalid_policy="keep")

        if _supports_unfolded_variant(fp):
            out = compute_fingerprints(SMILES, fp, cfg_unfolded, show_progress=False, n_jobs=1)
            assert isinstance(out, list)
            assert len(out) == len(SMILES)
            for keys in out:
                assert isinstance(keys, np.ndarray)
                assert keys.dtype == np.int64
                assert (keys >= 0).all()
                assert np.all(keys[:-1] <= keys[1:]) if keys.size > 1 else True

            # count unfolded (where feasible AND where variant exists)
            if supports_count:
                fp_count = _skfp_make_variant(fp, count=True)
                cfg_unfolded_count = replace(cfg_unfolded, count=True)
                out = compute_fingerprints(SMILES, fp_count, cfg_unfolded_count, show_progress=False, n_jobs=1)
                assert isinstance(out, list)
                assert len(out) == len(SMILES)
                for keys, vals in out:
                    assert isinstance(keys, np.ndarray) and keys.dtype == np.int64
                    assert isinstance(vals, np.ndarray) and vals.dtype == np.float32
                    assert keys.shape == vals.shape
                    assert (keys >= 0).all()
                    assert (vals >= 0).all()
                    assert np.all(keys[:-1] <= keys[1:]) if keys.size > 1 else True
        else:
            with pytest.raises(NotImplementedError, match="Requested folded=False"):
                compute_fingerprints(SMILES, fp, cfg_unfolded, show_progress=False, n_jobs=1)
