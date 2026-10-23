from __future__ import annotations
from typing import Any, Literal
import numpy as np
import pandas as pd
from chemap import FingerprintConfig, compute_fingerprints
from chemap.fingerprint_conversions import (
    fingerprints_to_csr,
    fingerprints_to_tfidf,
    idf_normalized,
)


Scaling = Literal["log", "log1p", "tfidf"] | None
DistanceFunction = Literal["tanimoto", "cosine"]


def _ensure_smiles_column(df: pd.DataFrame, col_smiles: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if col_smiles not in df.columns:
        raise KeyError(
            f"data is missing required SMILES column: {col_smiles!r}"
        )


def _default_fpgen():
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=9,
        fpSize=4096,
    )


def _normalize_scaling(scaling: Scaling) -> Literal["log1p", "tfidf"] | None:
    if scaling is None:
        return None
    if not isinstance(scaling, str):
        raise TypeError("scaling must be None, 'log'/'log1p', or 'tfidf'")

    value = scaling.lower()
    if value == "log":
        value = "log1p"
    if value not in {"log1p", "tfidf"}:
        raise ValueError(
            f"Unsupported scaling={scaling!r}. "
            "Choose from None, 'log'/'log1p', or 'tfidf'."
        )
    return value


def _validate_fingerprint_config(
    config: FingerprintConfig,
    *,
    require_folded: bool,
    scaling: Literal["log1p", "tfidf"] | None,
) -> None:
    """Validate UMAP-specific fingerprint requirements without rewriting config."""
    if require_folded and not config.folded:
        raise ValueError(
            "create_chem_space_umap_gpu() requires "
            "fingerprint_config.folded=True because cuML receives a dense, "
            "fixed-size fingerprint matrix."
        )

    if scaling == "log1p" and not config.count:
        raise ValueError(
            "scaling='log'/'log1p' requires fingerprint_config.count=True."
        )

    # UMAP coordinates must remain row-aligned with the input DataFrame.
    # So avoiding silently a fingerprint policy that may remove invalid structures.
    invalid_policy = getattr(config, "invalid_policy", "raise")
    if invalid_policy != "raise":
        raise ValueError(
            "Chemical-space UMAP requires fingerprint_config.invalid_policy="
            "'raise' so each input row maps to exactly one coordinate row."
        )


def _choose_cpu_metric(
    config: FingerprintConfig,
    distance_function: str,
) -> Any:
    """Return a metric suitable for umap-learn."""

    distance = distance_function.lower()

    if distance == "cosine":
        return "cosine"

    if distance != "tanimoto":
        raise ValueError(
            f"Unsupported distance_function={distance_function!r}. "
            "Currently only 'tanimoto' and 'cosine' are supported."
        )

    # Import CPU/Numba metrics only when Tanimoto is actually requested.
    from chemap.metrics import (
        tanimoto_distance_dense,
        tanimoto_distance_sparse,
    )

    if config.folded:
        return tanimoto_distance_dense

    return tanimoto_distance_sparse


def _log1p_csr(X):
    """Return a CSR matrix with log1p applied to its non-zero values."""
    X = X.astype(np.float32, copy=False)
    X.data = np.log1p(X.data)
    return X


def _scale_dense(
    fingerprints: Any,
    scaling: Literal["log1p", "tfidf"] | None,
) -> np.ndarray:
    """Return a dense numeric fingerprint matrix with optional scaling."""
    X = np.asarray(fingerprints)
    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2D fingerprint matrix; got shape {X.shape}."
        )

    if scaling is None:
        return X

    X = X.astype(np.float32, copy=False)
    if scaling == "log1p":
        return np.log1p(X)

    idf = idf_normalized(
        (X > 0).sum(axis=0),
        X.shape[0],
    )
    return X * np.asarray(idf, dtype=np.float32)


def _prepare_cpu_fingerprints(
    fingerprints: Any,
    config: FingerprintConfig,
    scaling: Literal["log1p", "tfidf"] | None,
):
    if config.folded:
        return _scale_dense(fingerprints, scaling)

    if scaling == "tfidf":
        return fingerprints_to_tfidf(fingerprints).X

    X = fingerprints_to_csr(fingerprints).X
    if scaling == "log1p":
        X = _log1p_csr(X)
    return X


def _prepare_gpu_fingerprints(
    fingerprints: Any,
    scaling: Literal["log1p", "tfidf"] | None,
) -> np.ndarray:
    """Return a cuML-safe dense float32 matrix without lossy integer casting."""
    X = _scale_dense(fingerprints, scaling)
    return np.asarray(X, dtype=np.float32)


def _to_numpy_coordinates(coords: Any, expected_rows: int) -> np.ndarray:
    """Convert NumPy/CuPy/cuDF-like coordinates to a validated NumPy array."""
    if hasattr(coords, "to_numpy"):
        coords = coords.to_numpy()
    elif hasattr(coords, "get"):
        coords = coords.get()

    coords_np = np.asarray(coords)
    expected_shape = (expected_rows, 2)
    if coords_np.shape != expected_shape:
        raise ValueError(
            f"UMAP returned coordinates with shape {coords_np.shape}; "
            f"expected {expected_shape}."
        )
    if not np.isfinite(coords_np).all():
        raise ValueError("UMAP returned non-finite coordinates.")
    return coords_np


def _compute_fingerprints(
    df: pd.DataFrame,
    *,
    col_smiles: str,
    fpgen: Any,
    fingerprint_config: FingerprintConfig,
    show_progress: bool,
):
    fingerprints = compute_fingerprints(
        df[col_smiles],
        fpgen,
        config=fingerprint_config,
        show_progress=show_progress,
    )

    try:
        n_rows = len(fingerprints)
    except TypeError as exc:
        raise ValueError(
            "compute_fingerprints() returned an object without a row dimension."
        ) from exc

    if n_rows != len(df):
        raise ValueError(
            "compute_fingerprints() changed row count: "
            f"returned {n_rows} fingerprint rows for {len(df)} input rows."
        )
    return fingerprints


def create_chem_space_umap(
    data: pd.DataFrame,
    *,
    col_smiles: str = "smiles",
    inplace: bool = False,
    x_col: str = "x",
    y_col: str = "y",
    # fingerprinting
    fpgen: Any | None = None,
    fingerprint_config: FingerprintConfig | None = None,
    show_progress: bool = True,
    scaling: Scaling = None,
    # UMAP (CPU / umap-learn)
    n_neighbors: int = 100,
    min_dist: float = 0.25,
    n_jobs: int = -1,
    umap_random_state: int | None = None,
    distance_function: DistanceFunction = "tanimoto",
) -> pd.DataFrame:
    """Compute molecular fingerprints and 2D UMAP coordinates on CPU.

    Fingerprint semantics are controlled explicitly by ``fingerprint_config``.
    The configuration is never modified internally.

    Defaults
    --------
    ``fpgen``
        Morgan generator with radius=9 and fpSize=4096.
    ``fingerprint_config``
        ``FingerprintConfig(count=True, folded=False, invalid_policy='raise')``.
    ``distance_function``
        ``'tanimoto'``. For Tanimoto, the dense or sparse Chemap metric is
        selected from ``fingerprint_config.folded``. ``'cosine'`` is also
        supported.
    ``scaling``
        ``None``. ``'log'``/``'log1p'`` applies log1p to count values and
        therefore requires ``count=True``. ``'tfidf'`` applies Chemap's TF-IDF
        scaling.

    Notes
    -----
    ``invalid_policy='raise'`` is required so input rows and coordinate rows
    remain aligned.
    """
    _ensure_smiles_column(data, col_smiles)
    scaling = _normalize_scaling(scaling)

    df = data if inplace else data.copy()
    fpgen = _default_fpgen() if fpgen is None else fpgen
    config = fingerprint_config or FingerprintConfig(
        count=True,
        folded=False,
        invalid_policy="raise",
    )
    _validate_fingerprint_config(
        config,
        require_folded=False,
        scaling=scaling,
    )

    fingerprints = _compute_fingerprints(
        df,
        col_smiles=col_smiles,
        fpgen=fpgen,
        fingerprint_config=config,
        show_progress=show_progress,
    )

    try:
        import umap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "umap-learn is required for create_chem_space_umap(). "
            "Install 'umap-learn'."
        ) from exc

    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        n_components=2,
        min_dist=float(min_dist),
        metric=_choose_cpu_metric(config, distance_function),
        random_state=umap_random_state,
        n_jobs=n_jobs,
        init="random",
    )

    X = _prepare_cpu_fingerprints(
        fingerprints,
        config,
        scaling,
    )
    coords = _to_numpy_coordinates(
        reducer.fit_transform(X),
        len(df),
    )

    df[x_col] = coords[:, 0]
    df[y_col] = coords[:, 1]
    return df


def create_chem_space_umap_gpu(
    data: pd.DataFrame,
    *,
    col_smiles: str = "smiles",
    inplace: bool = False,
    x_col: str = "x",
    y_col: str = "y",
    # fingerprinting
    fpgen: Any | None = None,
    fingerprint_config: FingerprintConfig | None = None,
    show_progress: bool = True,
    scaling: Scaling = None,
    # UMAP (GPU / cuML)
    n_neighbors: int = 100,
    min_dist: float = 0.25,
    umap_random_state: int | None = None,
) -> pd.DataFrame:
    """Compute molecular fingerprints and 2D UMAP coordinates with cuML.

    Fingerprint semantics are controlled explicitly by ``fingerprint_config``.
    A caller-supplied configuration is never rewritten.

    GPU requirements
    ----------------
    - cuML UMAP uses ``metric='cosine'``.
    - ``fingerprint_config.folded`` must be ``True`` so the input is a dense,
      fixed-size matrix.
    - ``invalid_policy='raise'`` is required to preserve row alignment.

    Defaults
    --------
    ``fpgen``
        Morgan generator with radius=9 and fpSize=4096.
    ``fingerprint_config``
        ``FingerprintConfig(count=True, folded=True, invalid_policy='raise')``.
    ``scaling``
        ``None``. ``'log'``/``'log1p'`` applies log1p to count values and
        requires ``count=True``. ``'tfidf'`` applies IDF scaling.

    Notes
    -----
    The dense GPU matrix is converted to ``float32``. Count fingerprints are
    deliberately not cast to int8 because counts above 127 would overflow and
    change cosine distances.
    """
    _ensure_smiles_column(data, col_smiles)
    scaling = _normalize_scaling(scaling)

    df = data if inplace else data.copy()

    try:
        from cuml.manifold.umap import UMAP as cuUMAP
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cuml is required for create_chem_space_umap_gpu(). "
            "Install RAPIDS cuML for your CUDA setup."
        ) from exc

    fpgen = _default_fpgen() if fpgen is None else fpgen
    config = fingerprint_config or FingerprintConfig(
        count=True,
        folded=True,
        invalid_policy="raise",
    )
    _validate_fingerprint_config(
        config,
        require_folded=True,
        scaling=scaling,
    )

    fingerprints = _compute_fingerprints(
        df,
        col_smiles=col_smiles,
        fpgen=fpgen,
        fingerprint_config=config,
        show_progress=show_progress,
    )
    X = _prepare_gpu_fingerprints(fingerprints, scaling)

    umap_model = cuUMAP(
        n_neighbors=int(n_neighbors),
        build_algo="nn_descent",
        build_kwds={"nnd_graph_degree": int(n_neighbors)},
        metric="cosine",
        min_dist=float(min_dist),
        n_components=2,
        random_state=umap_random_state,
    )

    coords = _to_numpy_coordinates(
        umap_model.fit_transform(X),
        len(df),
    )
    df[x_col] = coords[:, 0]
    df[y_col] = coords[:, 1]
    return df
