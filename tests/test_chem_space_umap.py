from __future__ import annotations
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, issparse
import chemap.plotting.chem_space_umap as csu
from chemap import FingerprintConfig


DATA_DIR = Path(__file__).parent / "data"
SMILES_CSV = DATA_DIR / "smiles.csv"


@pytest.fixture
def smiles_df() -> pd.DataFrame:
    return pd.read_csv(SMILES_CSV)


class DummyUMAP:
    """Small umap-learn stand-in that records constructor/input arguments."""

    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_input = None
        type(self).last_instance = self

    def fit_transform(self, X):
        self.fit_input = X
        n_rows = X.shape[0]
        return np.column_stack(
            (
                np.arange(n_rows, dtype=np.float32),
                -np.arange(n_rows, dtype=np.float32),
            )
        )


class DummyCuUMAP(DummyUMAP):
    pass


def _install_fake_umap(monkeypatch, cls=DummyUMAP) -> None:
    module = ModuleType("umap")
    module.UMAP = cls
    monkeypatch.setitem(sys.modules, "umap", module)


def _install_fake_cuml(monkeypatch, cls=DummyCuUMAP) -> None:
    cuml = ModuleType("cuml")
    manifold = ModuleType("cuml.manifold")
    umap = ModuleType("cuml.manifold.umap")
    umap.UMAP = cls
    manifold.umap = umap
    cuml.manifold = manifold

    monkeypatch.setitem(sys.modules, "cuml", cuml)
    monkeypatch.setitem(sys.modules, "cuml.manifold", manifold)
    monkeypatch.setitem(sys.modules, "cuml.manifold.umap", umap)


def _folded_config(*, count=True, invalid_policy="raise"):
    return FingerprintConfig(
        count=count,
        folded=True,
        invalid_policy=invalid_policy,
    )


def _unfolded_config(*, count=True, invalid_policy="raise"):
    return FingerprintConfig(
        count=count,
        folded=False,
        invalid_policy=invalid_policy,
    )


# ---------------------------------------------------------------------------
# Small validation / normalization helpers
# ---------------------------------------------------------------------------


def test_ensure_smiles_column_accepts_dataframe_with_requested_column():
    csu._ensure_smiles_column(pd.DataFrame({"structure": ["CCO"]}), "structure")


def test_ensure_smiles_column_rejects_non_dataframe():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        csu._ensure_smiles_column(["CCO"], "smiles")


def test_ensure_smiles_column_rejects_missing_column():
    with pytest.raises(KeyError, match="SMILES column"):
        csu._ensure_smiles_column(pd.DataFrame({"other": ["CCO"]}), "smiles")


def test_default_fpgen_is_an_rdkit_generator():
    pytest.importorskip("rdkit")
    fpgen = csu._default_fpgen()
    assert callable(getattr(fpgen, "GetFingerprint", None))
    assert callable(getattr(fpgen, "GetCountFingerprint", None))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("log", "log1p"),
        ("LOG", "log1p"),
        ("log1p", "log1p"),
        ("TFIDF", "tfidf"),
    ],
)
def test_normalize_scaling(value, expected):
    assert csu._normalize_scaling(value) == expected


def test_normalize_scaling_rejects_non_string():
    with pytest.raises(TypeError, match="scaling must be"):
        csu._normalize_scaling(1)


def test_normalize_scaling_rejects_unknown_string():
    with pytest.raises(ValueError, match="Unsupported scaling"):
        csu._normalize_scaling("sqrt")


def test_validate_fingerprint_config_accepts_compatible_cpu_config():
    csu._validate_fingerprint_config(
        _unfolded_config(count=True),
        require_folded=False,
        scaling="log1p",
    )


def test_validate_fingerprint_config_rejects_unfolded_gpu_config():
    with pytest.raises(ValueError, match="requires fingerprint_config.folded=True"):
        csu._validate_fingerprint_config(
            _unfolded_config(),
            require_folded=True,
            scaling=None,
        )


def test_validate_fingerprint_config_rejects_log_scaling_for_binary_fingerprints():
    with pytest.raises(ValueError, match="requires fingerprint_config.count=True"):
        csu._validate_fingerprint_config(
            _folded_config(count=False),
            require_folded=True,
            scaling="log1p",
        )


def test_validate_fingerprint_config_rejects_non_raise_invalid_policy():
    # SimpleNamespace is intentional: this helper should reject the semantic
    # state regardless of which invalid-policy strings FingerprintConfig itself
    # currently permits.
    config = SimpleNamespace(count=True, folded=True, invalid_policy="skip")
    with pytest.raises(ValueError, match="invalid_policy='raise'"):
        csu._validate_fingerprint_config(
            config,
            require_folded=True,
            scaling=None,
        )


@pytest.mark.parametrize(
    ("folded", "expected"),
    [
        (True, csu.tanimoto_distance_dense),
        (False, csu.tanimoto_distance_sparse),
    ],
)
def test_choose_cpu_metric_selects_tanimoto_implementation(folded, expected):
    config = SimpleNamespace(folded=folded)
    assert csu._choose_cpu_metric(config, "tanimoto") is expected


def test_choose_cpu_metric_returns_cosine_string():
    config = SimpleNamespace(folded=False)
    assert csu._choose_cpu_metric(config, "COSINE") == "cosine"


def test_choose_cpu_metric_rejects_non_string():
    with pytest.raises(TypeError, match="distance_function"):
        csu._choose_cpu_metric(SimpleNamespace(folded=True), None)


def test_choose_cpu_metric_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Choose 'tanimoto' or 'cosine'"):
        csu._choose_cpu_metric(SimpleNamespace(folded=True), "euclidean")


# ---------------------------------------------------------------------------
# Fingerprint scaling / representation helpers
# ---------------------------------------------------------------------------


def test_log1p_csr_converts_integer_data_to_float_and_scales_nonzeros():
    X = csr_matrix(np.array([[0, 1, 3], [7, 0, 0]], dtype=np.int32))
    result = csu._log1p_csr(X)

    assert issparse(result)
    assert result.dtype == np.float32
    np.testing.assert_allclose(
        result.toarray(),
        np.log1p(np.array([[0, 1, 3], [7, 0, 0]], dtype=np.float32)),
    )


def test_scale_dense_none_preserves_values():
    X = np.array([[0, 2], [3, 4]], dtype=np.int32)
    result = csu._scale_dense(X, None)
    np.testing.assert_array_equal(result, X)


def test_scale_dense_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="Expected a 2D fingerprint matrix"):
        csu._scale_dense(np.array([1, 2, 3]), None)


def test_scale_dense_log1p_returns_float32_values():
    X = np.array([[0, 1], [3, 7]], dtype=np.int32)
    result = csu._scale_dense(X, "log1p")

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.log1p(X.astype(np.float32)))


def test_scale_dense_tfidf_uses_chemap_idf(monkeypatch):
    X = np.array([[1, 0, 2], [0, 3, 4]], dtype=np.float32)
    seen = {}

    def fake_idf(document_frequency, n_documents):
        seen["document_frequency"] = np.asarray(document_frequency)
        seen["n_documents"] = n_documents
        return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(csu, "idf_normalized", fake_idf)
    result = csu._scale_dense(X, "tfidf")

    np.testing.assert_array_equal(seen["document_frequency"], [1, 1, 2])
    assert seen["n_documents"] == 2
    np.testing.assert_allclose(result, X * np.array([1.0, 2.0, 3.0]))


def test_prepare_cpu_fingerprints_folded_uses_dense_scaling(monkeypatch):
    expected = object()
    monkeypatch.setattr(csu, "_scale_dense", lambda fps, scaling: expected)

    result = csu._prepare_cpu_fingerprints(
        np.ones((2, 3)),
        SimpleNamespace(folded=True),
        "tfidf",
    )
    assert result is expected


def test_prepare_cpu_fingerprints_unfolded_tfidf_uses_conversion(monkeypatch):
    expected = csr_matrix(np.eye(2, dtype=np.float32))
    monkeypatch.setattr(
        csu,
        "fingerprints_to_tfidf",
        lambda fingerprints: SimpleNamespace(X=expected),
    )

    result = csu._prepare_cpu_fingerprints(
        [object(), object()],
        SimpleNamespace(folded=False),
        "tfidf",
    )
    assert result is expected


def test_prepare_cpu_fingerprints_unfolded_log1p_uses_csr(monkeypatch):
    raw = csr_matrix(np.array([[0, 2], [4, 0]], dtype=np.int32))
    monkeypatch.setattr(
        csu,
        "fingerprints_to_csr",
        lambda fingerprints: SimpleNamespace(X=raw),
    )

    result = csu._prepare_cpu_fingerprints(
        [object(), object()],
        SimpleNamespace(folded=False),
        "log1p",
    )
    np.testing.assert_allclose(
        result.toarray(),
        np.log1p(np.array([[0, 2], [4, 0]], dtype=np.float32)),
    )


def test_prepare_gpu_fingerprints_uses_float32_without_int8_overflow():
    # These values specifically catch the old int8 conversion, under which
    # 128, 255 and 1000 would wrap to different values.
    X = np.array([[0, 128, 255, 1000]], dtype=np.int32)
    result = csu._prepare_gpu_fingerprints(X, None)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, X.astype(np.float32))
    assert np.all(result >= 0)


def test_prepare_gpu_fingerprints_applies_log1p():
    X = np.array([[0, 1, 3]], dtype=np.int32)
    result = csu._prepare_gpu_fingerprints(X, "log1p")
    np.testing.assert_allclose(result, np.log1p(X.astype(np.float32)))


# ---------------------------------------------------------------------------
# Coordinate conversion and fingerprint-call contract
# ---------------------------------------------------------------------------


def test_to_numpy_coordinates_accepts_numpy():
    coords = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = csu._to_numpy_coordinates(coords, 2)
    np.testing.assert_array_equal(result, coords)


def test_to_numpy_coordinates_accepts_to_numpy_object():
    expected = np.array([[1.0, 2.0]])

    class CuDFLike:
        def to_numpy(self):
            return expected

    result = csu._to_numpy_coordinates(CuDFLike(), 1)
    np.testing.assert_array_equal(result, expected)


def test_to_numpy_coordinates_accepts_get_object():
    expected = np.array([[1.0, 2.0]])

    class CuPyLike:
        def get(self):
            return expected

    result = csu._to_numpy_coordinates(CuPyLike(), 1)
    np.testing.assert_array_equal(result, expected)


def test_to_numpy_coordinates_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"expected \(2, 2\)"):
        csu._to_numpy_coordinates(np.zeros((2, 3)), 2)


def test_to_numpy_coordinates_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="non-finite"):
        csu._to_numpy_coordinates(np.array([[0.0, np.nan]]), 1)


def test_compute_fingerprints_forwards_arguments_and_preserves_rows(monkeypatch):
    df = pd.DataFrame({"smiles": ["CCO", "CCC"]})
    fpgen = object()
    config = _folded_config()
    expected = np.ones((2, 4), dtype=np.int32)
    seen = {}

    def fake_compute(smiles, generator, *, config, show_progress):
        seen["smiles"] = list(smiles)
        seen["generator"] = generator
        seen["config"] = config
        seen["show_progress"] = show_progress
        return expected

    monkeypatch.setattr(csu, "compute_fingerprints", fake_compute)
    result = csu._compute_fingerprints(
        df,
        col_smiles="smiles",
        fpgen=fpgen,
        fingerprint_config=config,
        show_progress=False,
    )

    assert result is expected
    assert seen == {
        "smiles": ["CCO", "CCC"],
        "generator": fpgen,
        "config": config,
        "show_progress": False,
    }


def test_compute_fingerprints_rejects_changed_row_count(monkeypatch):
    monkeypatch.setattr(
        csu,
        "compute_fingerprints",
        lambda *args, **kwargs: np.ones((1, 4)),
    )

    with pytest.raises(ValueError, match="changed row count"):
        csu._compute_fingerprints(
            pd.DataFrame({"smiles": ["CCO", "CCC"]}),
            col_smiles="smiles",
            fpgen=object(),
            fingerprint_config=_folded_config(),
            show_progress=False,
        )


def test_compute_fingerprints_rejects_object_without_length(monkeypatch):
    monkeypatch.setattr(
        csu,
        "compute_fingerprints",
        lambda *args, **kwargs: object(),
    )

    with pytest.raises(ValueError, match="without a row dimension"):
        csu._compute_fingerprints(
            pd.DataFrame({"smiles": ["CCO"]}),
            col_smiles="smiles",
            fpgen=object(),
            fingerprint_config=_folded_config(),
            show_progress=False,
        )


# ---------------------------------------------------------------------------
# Public CPU API
# ---------------------------------------------------------------------------


def test_create_chem_space_umap_configures_cpu_umap_and_returns_copy(
    monkeypatch,
    smiles_df,
):
    _install_fake_umap(monkeypatch)
    fingerprints = np.arange(len(smiles_df) * 8, dtype=np.int32).reshape(
        len(smiles_df), 8
    )
    seen = {}

    def fake_compute(df, **kwargs):
        seen.update(kwargs)
        return fingerprints

    monkeypatch.setattr(csu, "_compute_fingerprints", fake_compute)
    original = smiles_df.copy(deep=True)
    config = _folded_config(count=True)
    fpgen = object()

    result = csu.create_chem_space_umap(
        smiles_df,
        fpgen=fpgen,
        fingerprint_config=config,
        show_progress=False,
        scaling="log",
        n_neighbors=7,
        min_dist=0.1,
        n_jobs=2,
        umap_random_state=42,
        distance_function="tanimoto",
    )

    pd.testing.assert_frame_equal(smiles_df, original)
    assert result is not smiles_df
    assert list(result["x"]) == list(np.arange(len(smiles_df), dtype=np.float32))
    assert list(result["y"]) == list(-np.arange(len(smiles_df), dtype=np.float32))

    model = DummyUMAP.last_instance
    assert model.kwargs == {
        "n_neighbors": 7,
        "n_components": 2,
        "min_dist": 0.1,
        "metric": csu.tanimoto_distance_dense,
        "random_state": 42,
        "n_jobs": 2,
        "init": "random",
    }
    assert model.fit_input.dtype == np.float32
    np.testing.assert_allclose(model.fit_input, np.log1p(fingerprints))

    assert seen["col_smiles"] == "smiles"
    assert seen["fpgen"] is fpgen
    assert seen["fingerprint_config"] is config
    assert seen["show_progress"] is False


def test_create_chem_space_umap_inplace_returns_same_dataframe(monkeypatch):
    _install_fake_umap(monkeypatch)
    df = pd.DataFrame({"smiles": ["CCO", "CCC"]})
    monkeypatch.setattr(
        csu,
        "_compute_fingerprints",
        lambda *args, **kwargs: np.ones((2, 4), dtype=np.float32),
    )

    result = csu.create_chem_space_umap(
        df,
        inplace=True,
        fingerprint_config=_folded_config(),
        show_progress=False,
    )

    assert result is df
    assert {"x", "y"}.issubset(df.columns)


def test_create_chem_space_umap_uses_custom_output_columns(monkeypatch):
    _install_fake_umap(monkeypatch)
    df = pd.DataFrame({"smiles": ["CCO"]})
    monkeypatch.setattr(
        csu,
        "_compute_fingerprints",
        lambda *args, **kwargs: np.ones((1, 4), dtype=np.float32),
    )

    result = csu.create_chem_space_umap(
        df,
        x_col="visual_x",
        y_col="visual_y",
        fingerprint_config=_folded_config(),
        show_progress=False,
    )
    assert "visual_x" in result.columns
    assert "visual_y" in result.columns
    assert "x" not in result.columns
    assert "y" not in result.columns


def test_create_chem_space_umap_reports_missing_umap_learn(monkeypatch):
    df = pd.DataFrame({"smiles": ["CCO"]})
    monkeypatch.setattr(
        csu,
        "_compute_fingerprints",
        lambda *args, **kwargs: np.ones((1, 4), dtype=np.float32),
    )
    monkeypatch.setitem(sys.modules, "umap", None)

    with pytest.raises(ModuleNotFoundError, match="umap-learn is required"):
        csu.create_chem_space_umap(
            df,
            fingerprint_config=_folded_config(),
            show_progress=False,
        )


def test_create_chem_space_umap_real_fingerprints_from_smiles_fixture(
    monkeypatch,
    smiles_df,
):
    """Smoke-test the real Chemap/RDKit fingerprint boundary, but mock UMAP."""
    pytest.importorskip("rdkit")
    _install_fake_umap(monkeypatch)

    result = csu.create_chem_space_umap(
        smiles_df,
        fingerprint_config=_folded_config(count=True),
        show_progress=False,
        distance_function="cosine",
        n_neighbors=3,
        umap_random_state=42,
    )

    model = DummyUMAP.last_instance
    assert model.fit_input.shape[0] == len(smiles_df)
    assert model.fit_input.ndim == 2
    assert model.fit_input.shape[1] == 4096
    assert np.all(np.asarray(model.fit_input) >= 0)
    assert result[["x", "y"]].notna().all().all()


# ---------------------------------------------------------------------------
# Public GPU API (cuML mocked; no CUDA required in unit tests)
# ---------------------------------------------------------------------------


def test_create_chem_space_umap_gpu_respects_caller_config_and_uses_float32(
    monkeypatch,
    smiles_df,
):
    _install_fake_cuml(monkeypatch)
    config = _folded_config(count=False)
    fpgen = object()
    fingerprints = np.tile(
        np.array([[0, 1, 128, 255]], dtype=np.int32),
        (len(smiles_df), 1),
    )
    seen = {}

    def fake_compute(df, **kwargs):
        seen.update(kwargs)
        return fingerprints

    monkeypatch.setattr(csu, "_compute_fingerprints", fake_compute)

    result = csu.create_chem_space_umap_gpu(
        smiles_df,
        fpgen=fpgen,
        fingerprint_config=config,
        show_progress=False,
        n_neighbors=9,
        min_dist=0.2,
        umap_random_state=123,
    )

    assert seen["fingerprint_config"] is config
    assert seen["fpgen"] is fpgen

    model = DummyCuUMAP.last_instance
    assert model.kwargs == {
        "n_neighbors": 9,
        "build_algo": "nn_descent",
        "build_kwds": {"nnd_graph_degree": 9},
        "metric": "cosine",
        "min_dist": 0.2,
        "n_components": 2,
        "random_state": 123,
    }
    assert model.fit_input.dtype == np.float32
    np.testing.assert_array_equal(model.fit_input, fingerprints.astype(np.float32))
    assert np.all(model.fit_input >= 0)
    assert result[["x", "y"]].shape == (len(smiles_df), 2)


def test_create_chem_space_umap_gpu_rejects_unfolded_config(monkeypatch):
    _install_fake_cuml(monkeypatch)

    with pytest.raises(ValueError, match="requires fingerprint_config.folded=True"):
        csu.create_chem_space_umap_gpu(
            pd.DataFrame({"smiles": ["CCO"]}),
            fingerprint_config=_unfolded_config(),
            show_progress=False,
        )


def test_create_chem_space_umap_gpu_rejects_log_scaling_for_binary_config(
    monkeypatch,
):
    _install_fake_cuml(monkeypatch)

    with pytest.raises(ValueError, match="requires fingerprint_config.count=True"):
        csu.create_chem_space_umap_gpu(
            pd.DataFrame({"smiles": ["CCO"]}),
            fingerprint_config=_folded_config(count=False),
            scaling="log1p",
            show_progress=False,
        )


def test_create_chem_space_umap_gpu_accepts_cupy_like_coordinates(monkeypatch):
    class CuPyLikeCoords:
        def __init__(self, array):
            self.array = array

        def get(self):
            return self.array

    class CuPyReturningUMAP(DummyCuUMAP):
        def fit_transform(self, X):
            self.fit_input = X
            n_rows = X.shape[0]
            return CuPyLikeCoords(np.zeros((n_rows, 2), dtype=np.float32))

    _install_fake_cuml(monkeypatch, CuPyReturningUMAP)
    monkeypatch.setattr(
        csu,
        "_compute_fingerprints",
        lambda *args, **kwargs: np.ones((2, 4), dtype=np.int32),
    )

    result = csu.create_chem_space_umap_gpu(
        pd.DataFrame({"smiles": ["CCO", "CCC"]}),
        fingerprint_config=_folded_config(),
        show_progress=False,
    )
    np.testing.assert_array_equal(result[["x", "y"]].to_numpy(), 0.0)


def test_create_chem_space_umap_gpu_reports_missing_cuml(monkeypatch):
    monkeypatch.setitem(sys.modules, "cuml", None)
    monkeypatch.delitem(sys.modules, "cuml.manifold", raising=False)
    monkeypatch.delitem(sys.modules, "cuml.manifold.umap", raising=False)

    with pytest.raises(ModuleNotFoundError, match="cuml is required"):
        csu.create_chem_space_umap_gpu(
            pd.DataFrame({"smiles": ["CCO"]}),
            fingerprint_config=_folded_config(),
            show_progress=False,
        )
