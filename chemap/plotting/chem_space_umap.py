from dataclasses import replace
from typing import Any, Optional
import numpy as np
import pandas as pd
from chemap import FingerprintConfig, compute_fingerprints
from chemap.fingerprint_conversions import fingerprints_to_csr
from chemap.metrics import (
    tanimoto_distance_dense,
    tanimoto_distance_sparse,
)


def _ensure_smiles_column(df: pd.DataFrame, col_smiles: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if col_smiles not in df.columns:
        raise KeyError(f"data is missing required SMILES column: {col_smiles!r}")


def _choose_cpu_metric(config: FingerprintConfig, distance_function: str) -> Any:
    """Return a metric callable/object suitable for umap-learn based on FingerprintConfig.
    
    - unfolded => tanimoto_distance_sparse
    - folded => tanimoto_distance_dense
    """
    if distance_function.lower() == "cosine":
        return "cosine"
    if distance_function.lower() != "tanimoto":
        raise ValueError(
            f"Unsupported distance_function={distance_function!r}. "
            "Currently only 'tanimoto' and 'cosine' is supported here."
        )

    if getattr(config, "folded", True):
        return tanimoto_distance_dense
    return tanimoto_distance_sparse


def _log1p_csr_inplace(X) -> Any:
    """Apply log1p to CSR data in-place (only affects non-zeros)."""
    X.data = np.log1p(X.data)
    return X


def create_chem_space_umap(
    data: pd.DataFrame,
    *,
    col_smiles: str = "smiles",
    inplace: bool = False,
    x_col: str = "x",
    y_col: str = "y",
    # fingerprinting
    fpgen: Optional[Any] = None,
    fingerprint_config: Optional[FingerprintConfig] = None,
    show_progress: bool = True,
    log_count: bool = False,
    # UMAP (CPU / umap-learn)
    n_neighbors: int = 100,
    min_dist: float = 0.25,
    n_jobs: int = -1,
    umap_random_state: Optional[int] = None,
    distance_function: str = "tanimoto",
) -> pd.DataFrame:
    """Compute fingerprints (CPU) and create 2D UMAP coordinates (CPU).

    Parameters
    ----------
    data:
        Input dataframe containing a SMILES column.
    col_smiles:
        Name of the SMILES column.
    inplace:
        If True, write x/y columns into `data` and return it. Else returns a copy.
    x_col, y_col:
        Output coordinate column names.
    fpgen:
        RDKit fingerprint generator. Defaults to Morgan radius=9, fpSize=4096.
    fingerprint_config:
        FingerprintConfig for chemap.compute_fingerprints. Defaults to:
            FingerprintConfig(count=True, folded=False, invalid_policy="raise")
    show_progress:
        Forwarded to compute_fingerprints.
    log_count:
        If True, apply np.log1p to counts (works for sparse CSR and dense arrays).
        (For binary fingerprints this is harmless)
    n_neighbors, min_dist, umap_random_state:
        Standard UMAP parameters.
    n_jobs:
        Passed to umap-learn UMAP for parallelism. Ignores random_state when n_jobs != 1.
        Default -1 uses all CPUs.
    distance_function:
        Currently only "tanimoto" is supported. Metric is chosen based on config.

    Returns
    -------
    DataFrame with added columns x_col and y_col.
    """
    _ensure_smiles_column(data, col_smiles)

    df = data if inplace else data.copy()

    if fpgen is None:
        from rdkit.Chem import rdFingerprintGenerator

        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=9, fpSize=4096)

    if fingerprint_config is None:
        fingerprint_config = FingerprintConfig(count=True, folded=False, invalid_policy="raise")

    # Compute fingerprints
    fingerprints = compute_fingerprints(
        df[col_smiles],
        fpgen,
        config=fingerprint_config,
        show_progress=show_progress,
    )

    # UMAP (CPU, supports sparse + custom metrics)
    try:
        import umap  # umap-learn
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "umap-learn is required for create_chem_space_umap(). Install 'umap-learn'."
        ) from e

    metric = _choose_cpu_metric(fingerprint_config, distance_function)
    
    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        n_components=2,
        min_dist=float(min_dist),
        metric=metric,
        random_state=umap_random_state,
        n_jobs=n_jobs,
        init="random",
    )

    if not fingerprint_config.folded:
        # Convert to CSR matrix
        fps_csr = fingerprints_to_csr(fingerprints).X
    
        if log_count:
            # Works well for count fingerprints ( for binary it's essentially unchanged).
            fps_csr = _log1p_csr_inplace(fps_csr)

        coords = reducer.fit_transform(fps_csr)
    else:
        coords = reducer.fit_transform(fingerprints)

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
    fpgen: Optional[Any] = None,
    fingerprint_config: Optional[FingerprintConfig] = None,
    show_progress: bool = True,
    log_count: bool = True,
    # UMAP (GPU / cuML)
    n_neighbors: int = 15,
    min_dist: float = 0.25,
) -> pd.DataFrame:
    """Compute fingerprints and create 2D UMAP coordinates using cuML (GPU).

    Notes
    -----
    - cuML UMAP here is fixed to metric="cosine"
    - This function enforces a default fingerprint config of:
        FingerprintConfig(count=True, folded=True, invalid_policy="raise")
      Possible to pass `fingerprint_config`, but it will be overridden to ensure folded=True.
    - Before feeding to cuML, fingerprints are converted to a dense array and cast to int8
      to reduce memory; log_count (if enabled) converts to float via np.log1p.

    Returns
    -------
    DataFrame with added columns x_col and y_col.
    """
    _ensure_smiles_column(data, col_smiles)

    df = data if inplace else data.copy()

    # Ensure cuML is available
    try:
        from cuml.manifold.umap import UMAP as cuUMAP
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "cuml is required for create_chem_space_umap_gpu(). "
            "Install RAPIDS cuML for your CUDA setup."
        ) from e

    if fpgen is None:
        from rdkit.Chem import rdFingerprintGenerator

        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=9, fpSize=4096)

    # Force the GPU defaults (folded=True)
    default_cfg = FingerprintConfig(count=True, folded=True, invalid_policy="raise")
    if fingerprint_config is None:
        fingerprint_config = default_cfg
    else:
        # Override to ensure folded=True and count=True
        fingerprint_config = replace(
            fingerprint_config,
            count=True,
            folded=True,
            invalid_policy=getattr(fingerprint_config, "invalid_policy", "raise"),
        )

    fingerprints = compute_fingerprints(
        df[col_smiles],
        fpgen,
        config=fingerprint_config,
        show_progress=show_progress,
    )

    # Convert to sparse array
    # fps_csr = fingerprints_to_csr(fingerprints).X

    # Reduce memory footprint (works well for count fingerprints)
    if not log_count:
        # stays integer-like
        fps = fingerprints.astype(np.int8, copy=False)
    else:
        # log1p returns float
        fps = np.log1p(fingerprints).astype(np.float32, copy=False)

    umap_model = cuUMAP(
        n_neighbors=int(n_neighbors),
        build_algo="nn_descent",
        build_kwds={"nnd_graph_degree": int(n_neighbors)},
        metric="cosine",
        min_dist=float(min_dist),
        n_components=2,
    )

    coords = umap_model.fit_transform(fps)

    # cuML may return cupy/cudf-backed arrays; np.asarray makes it safe for pandas columns.
    coords_np = np.asarray(coords)

    df[x_col] = coords_np[:, 0]
    df[y_col] = coords_np[:, 1]
    return df
