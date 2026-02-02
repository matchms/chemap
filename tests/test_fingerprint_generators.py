from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple
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
    for key in ("count", "counts", "use_counts", "useCounts", "use_count"):
        if key in params:
            return key
    return None


def _build_skfp_transformers() -> List[Tuple[str, Any, bool]]:
    mod = _skfp_generators()
    out: List[Tuple[str, Any, bool]] = []

    for cls_name, cls in mod.items():
        try:
            base = cls()  # type: ignore[call-arg]
            params = base.get_params(deep=False)
            supports_count = _supports_count_param(params) is not None
            out.append((f"skfp_{cls_name}", base, supports_count))
        except Exception:
            continue

    return out


def _skfp_make_variant(fp: Any, *, count: bool) -> Any:
    params = fp.get_params(deep=False)
    count_key = _supports_count_param(params)
    if count_key is None:
        return fp
    params = dict(params)
    params[count_key] = bool(count)
    return fp.__class__(**params)


def _supports_unfolded_variant(fp: Any) -> bool:
    try:
        params = fp.get_params(deep=False)
    except Exception:
        return False
    return "variant" in params


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
    u = np.unique(X)
    assert set(u.tolist()).issubset({0.0, 1.0})


def _assert_count_dense(X: np.ndarray) -> None:
    assert (X >= 0).all()


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


# ============================================================
# Parametrized case builders
# ============================================================

def _case1_dense_cases():
    """
    CASE 1:
    - dense fingerprint (binary and where feasible count)
    Each run = separate pytest param case.
    """
    cases: List[pytest.ParamSpecArg] = []

    # RDKit: binary + count
    for name, gen in _rdkit_generators():
        cfg_bin = FingerprintConfig(count=False, folded=True, return_csr=False, invalid_policy="keep")
        cases.append(pytest.param(name, gen, cfg_bin, "dense-binary", id=f"{name}__dense__binary"))

        cfg_cnt = replace(cfg_bin, count=True)
        cases.append(pytest.param(name, gen, cfg_cnt, "dense-count", id=f"{name}__dense__count"))

    # skfp: binary always; count only if supported
    for name, fp, supports_count in _build_skfp_transformers():
        cfg_bin = FingerprintConfig(count=False, folded=True, return_csr=False, invalid_policy="keep")
        cases.append(pytest.param(name, fp, cfg_bin, "dense-binary", id=f"{name}__dense__binary"))

        if supports_count:
            fp_count = _skfp_make_variant(fp, count=True)
            cfg_cnt = replace(cfg_bin, count=True)
            cases.append(pytest.param(name, fp_count, cfg_cnt, "dense-count", id=f"{name}__dense__count"))

    return cases


def _case2_csr_cases():
    """
    CASE 2:
    - return_csr=True (binary and where feasible count)
    Each run = separate pytest param case.
    """
    cases: List[pytest.ParamSpecArg] = []

    # RDKit: binary + count
    for name, gen in _rdkit_generators():
        cfg_bin = FingerprintConfig(count=False, folded=True, return_csr=True, invalid_policy="keep")
        cases.append(pytest.param(name, gen, cfg_bin, "csr-binary", id=f"{name}__csr__binary"))

        cfg_cnt = replace(cfg_bin, count=True)
        cases.append(pytest.param(name, gen, cfg_cnt, "csr-count", id=f"{name}__csr__count"))

    # skfp: binary always; count only if supported
    for name, fp, supports_count in _build_skfp_transformers():
        cfg_bin = FingerprintConfig(count=False, folded=True, return_csr=True, invalid_policy="keep")
        cases.append(pytest.param(name, fp, cfg_bin, "csr-binary", id=f"{name}__csr__binary"))

        if supports_count:
            fp_count = _skfp_make_variant(fp, count=True)
            cfg_cnt = replace(cfg_bin, count=True)
            cases.append(pytest.param(name, fp_count, cfg_cnt, "csr-count", id=f"{name}__csr__count"))

    return cases


def _case3_fit_backend_cases():
    """
    CASE 3 (skfp only):
    - "return_csr=True with folded=True and folded=False"
      We split into separate runs:
        A) folded=True, return_csr=True works
        B) folded=False, return_csr=True raises ValueError
        C) folded=False, return_csr=False:
           - if supports variant: works (unfolded)
           - else: raises NotImplementedError
    """
    cases: List[pytest.ParamSpecArg] = []

    for name, fp, supports_count in _build_skfp_transformers():
        # A)
        cfg_ok = FingerprintConfig(count=False, folded=True, return_csr=True, invalid_policy="keep")
        cases.append(pytest.param(name, fp, cfg_ok, "folded_true_csr_ok", supports_count,
                                  id=f"{name}__case3__folded_true__csr_ok"))

        # B)
        cfg_bad = replace(cfg_ok, folded=False, return_csr=True)
        cases.append(pytest.param(name, fp, cfg_bad, "folded_false_csr_raises_valueerror", supports_count,
                                  id=f"{name}__case3__folded_false__csr_raises_valueerror"))

        # C)
        cfg_unfolded = FingerprintConfig(count=False, folded=False, return_csr=False, invalid_policy="keep")
        behavior = "unfolded_ok" if _supports_unfolded_variant(fp) else "unfolded_raises_notimplemented"
        cases.append(pytest.param(name, fp, cfg_unfolded, behavior, supports_count,
                                  id=f"{name}__case3__folded_false__unfolded_{behavior}"))

        # (Optional) If you also want unfolded-count as separate cases where feasible + variant exists:
        if supports_count and _supports_unfolded_variant(fp):
            fp_count = _skfp_make_variant(fp, count=True)
            cfg_unfolded_count = replace(cfg_unfolded, count=True)
            cases.append(pytest.param(name, fp_count, cfg_unfolded_count, "unfolded_count_ok", True,
                                      id=f"{name}__case3__folded_false__unfolded_count_ok"))

    return cases


# ============================================================
# CASE 1: dense (parametrized per run)
# ============================================================

@pytest.mark.parametrize("name, fpgen, cfg, mode", _case1_dense_cases())
def test_case1_dense_per_run(name: str, fpgen: Any, cfg: FingerprintConfig, mode: str) -> None:
    X = compute_fingerprints(SMILES, fpgen, cfg, show_progress=False, n_jobs=1)

    _assert_dense_matrix(X, n_rows=len(SMILES))
    if mode.endswith("binary"):
        _assert_binary_dense(X)
    else:
        _assert_count_dense(X)


# ============================================================
# CASE 2: csr (parametrized per run)
# ============================================================

@pytest.mark.parametrize("name, fpgen, cfg, mode", _case2_csr_cases())
def test_case2_csr_per_run(name: str, fpgen: Any, cfg: FingerprintConfig, mode: str) -> None:
    X = compute_fingerprints(SMILES, fpgen, cfg, show_progress=False, n_jobs=1)

    _assert_csr_matrix(X, n_rows=len(SMILES))
    if mode.endswith("binary"):
        _assert_binary_csr(X)
    else:
        _assert_count_csr(X)


# ============================================================
# CASE 3: fitting backend + folded True/False (parametrized)
# ============================================================

@pytest.mark.parametrize("name, fpgen, cfg, behavior, supports_count", _case3_fit_backend_cases())
def test_case3_fit_backend_per_run(
    name: str,
    fpgen: Any,
    cfg: FingerprintConfig,
    behavior: str,
    supports_count: bool,
) -> None:
    if behavior == "folded_true_csr_ok":
        X = compute_fingerprints(SMILES, fpgen, cfg, show_progress=False, n_jobs=1)
        _assert_csr_matrix(X, n_rows=len(SMILES))
        return

    if behavior == "folded_false_csr_raises_valueerror":
        with pytest.raises(ValueError, match="return_csr is only valid when folded=True"):
            compute_fingerprints(SMILES, fpgen, cfg, show_progress=False, n_jobs=1)
        return

    if behavior in ("unfolded_ok", "unfolded_count_ok"):
        out = compute_fingerprints(SMILES, fpgen, cfg, show_progress=False, n_jobs=1)
        assert isinstance(out, list)
        assert len(out) == len(SMILES)

        if cfg.count:
            for keys, vals in out:
                assert isinstance(keys, np.ndarray) and keys.dtype == np.int64
                assert isinstance(vals, np.ndarray) and vals.dtype == np.float32
                assert keys.shape == vals.shape
                assert (keys >= 0).all()
                assert (vals >= 0).all()
                assert np.all(keys[:-1] <= keys[1:]) if keys.size > 1 else True
        else:
            for keys in out:
                assert isinstance(keys, np.ndarray)
                assert keys.dtype == np.int64
                assert (keys >= 0).all()
                assert np.all(keys[:-1] <= keys[1:]) if keys.size > 1 else True
        return

    if behavior == "unfolded_raises_notimplemented":
        with pytest.raises(NotImplementedError, match="Requested folded=False"):
            compute_fingerprints(SMILES, fpgen, cfg, show_progress=False, n_jobs=1)
        return

    raise AssertionError(f"Unknown behavior: {behavior!r}")
