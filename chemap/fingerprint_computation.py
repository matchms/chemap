import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Union
import numpy as np
import scipy.sparse as sp
from rdkit import Chem
from tqdm import tqdm


# -----------------------------
# Public types and configuration
# -----------------------------

InvalidPolicy = Literal["drop", "keep", "raise"]
Scaling = Optional[Literal["log"]]

UnfoldedBinary = List[np.ndarray]  # list of int64 feature IDs per molecule
UnfoldedCount = List[Tuple[np.ndarray, np.ndarray]]  # list of (int64 feature IDs, float32 values)

FingerprintResult = Union[np.ndarray, sp.csr_matrix, UnfoldedBinary, UnfoldedCount]


@dataclass(frozen=True)
class FingerprintConfig:
    """
    Fingerprint computation settings.

    Core switches
    -------------
    count:
        If True, compute count fingerprints (counts/weights per feature).
        If False, compute binary fingerprints.

    folded:
        If True, return a fixed-length representation with feature dimension D.
        If False, return an **unfolded** representation (feature IDs, optionally with values).

        Note: Unfolded outputs are inherently sparse (lists of non-zero feature IDs), and are
        never returned as matrices.

    return_csr:
        Only relevant when folded=True.
        If True, return a SciPy CSR matrix (N, D) in float32 (memory efficient for very large N).
        If False, return a dense NumPy array (N, D) in float32.

    scaling:
        Optional scaling for count outputs:
          - None: no scaling
          - "log": apply log1p to counts

    folded_weights:
        Optional 1D float array applied elementwise to folded outputs (dense or CSR).

    unfolded_weights:
        Optional dict {feature_id: weight} applied to unfolded count outputs (folded=False + count=True only).
        Missing keys default to weight 1.0.

    invalid_policy:
        Handling of invalid/unparseable SMILES:
          - "drop": drop invalid molecules from output
          - "keep": keep alignment; returns all-zeros row (folded) or empty arrays (unfolded)
          - "raise": raise ValueError on first invalid SMILES
    """

    count: bool = True
    folded: bool = True
    return_csr: bool = False  # only applies when folded=True
    scaling: Scaling = None
    folded_weights: Optional[np.ndarray] = None
    unfolded_weights: Optional[Dict[int, float]] = None
    invalid_policy: InvalidPolicy = "keep"


class SklearnTransformer(Protocol):
    """Protocol for sklearn-like fingerprint transformers (including scikit-fingerprints)."""

    def transform(self, X: Sequence[str]) -> Any:
        ...

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        ...


# -----------------------------
# Public entry point
# -----------------------------

def compute_fingerprints(
    smiles: Sequence[str],
    fpgen: Any,
    config: FingerprintConfig = FingerprintConfig(),
    *,
    show_progress: bool = False,
) -> FingerprintResult:
    """
    Compute fingerprints for a sequence of SMILES.

    Backends
    --------
    - RDKit rdFingerprintGenerator generators (Morgan, RDKitFP, ...)
    - scikit-fingerprints / sklearn-style transformers with `.transform()`

    Returns
    -------
    If config.folded is True:
        - config.return_csr False: np.ndarray (N, D) float32
        - config.return_csr True : scipy.sparse.csr_matrix (N, D) float32

    If config.folded is False (unfolded):
        - config.count False: List[np.ndarray[int64]] (sorted feature IDs)
        - config.count True : List[Tuple[np.ndarray[int64], np.ndarray[float32]]] (sorted feature IDs + values)
    """
    _validate_config(config)
    _quick_smiles_check(smiles)

    if _looks_like_rdkit_fpgen(fpgen):
        return _compute_rdkit(smiles, fpgen, config, show_progress=show_progress)

    if _looks_like_sklearn_transformer(fpgen):
        return _compute_sklearn(smiles, fpgen, config, show_progress=show_progress)

    raise TypeError(
        "Unsupported fpgen. Expected an RDKit rdFingerprintGenerator-like object "
        "or an sklearn/scikit-fingerprints transformer exposing transform/get_params."
    )


# -----------------------------
# Validation & numeric utilities
# -----------------------------

def _validate_config(cfg: FingerprintConfig) -> None:
    if cfg.scaling not in (None, "log"):
        raise ValueError("config.scaling must be None or 'log'.")

    if not cfg.folded:
        # unfolded output
        if cfg.return_csr:
            raise ValueError("return_csr is only valid when folded=True (unfolded outputs are lists).")
        if cfg.folded_weights is not None:
            raise ValueError("folded_weights is only valid when folded=True.")
        if cfg.unfolded_weights is not None and cfg.count is False:
            raise ValueError("unfolded_weights is only valid when folded=False and count=True.")
        return

    # folded output
    if cfg.unfolded_weights is not None:
        raise ValueError("unfolded_weights is only valid when folded=False and count=True.")
    if cfg.folded_weights is not None:
        w = np.asarray(cfg.folded_weights)
        if w.ndim != 1:
            raise ValueError("folded_weights must be a 1D array.")


def _log1p_inplace_safe(x: np.ndarray) -> np.ndarray:
    return np.log1p(x).astype(np.float32, copy=False)


def _apply_folded_weights_dense(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32).ravel()
    if X.shape[1] != w.shape[0]:
        raise ValueError(f"folded_weights length {w.shape[0]} does not match feature dim {X.shape[1]}.")
    return (X * w[None, :]).astype(np.float32, copy=False)


def _apply_folded_weights_csr(X: sp.csr_matrix, weights: np.ndarray) -> sp.csr_matrix:
    w = np.asarray(weights, dtype=np.float32).ravel()
    if X.shape[1] != w.shape[0]:
        raise ValueError(f"folded_weights length {w.shape[0]} does not match feature dim {X.shape[1]}.")
    return X.multiply(w).astype(np.float32)


def _apply_unfolded_weights(keys: np.ndarray, vals: np.ndarray, weights: Dict[int, float]) -> np.ndarray:
    w = np.array([float(weights.get(int(k), 1.0)) for k in keys], dtype=np.float32)
    return (vals * w).astype(np.float32, copy=False)


def _postprocess_unfolded_vals(keys: np.ndarray, vals: np.ndarray, cfg: FingerprintConfig) -> np.ndarray:
    """
    Shared post-processing for unfolded *count* outputs:
    - optional log1p scaling
    - optional per-feature weighting
    """
    out = vals.astype(np.float32, copy=False)

    if cfg.scaling == "log":
        out = _log1p_inplace_safe(out)

    if cfg.unfolded_weights is not None:
        out = _apply_unfolded_weights(keys, out, cfg.unfolded_weights)

    return out


def _handle_invalid(policy: InvalidPolicy, s: str) -> None:
    if policy == "raise":
        raise ValueError(f"Invalid SMILES: {s}")


def _empty_unfolded_binary() -> np.ndarray:
    return np.array([], dtype=np.int64)


def _empty_unfolded_count() -> Tuple[np.ndarray, np.ndarray]:
    return np.array([], dtype=np.int64), np.array([], dtype=np.float32)


def _quick_smiles_check(smiles_lst: Sequence[str]) -> None:
    regexp = r"^([^J][0-9a-zA-Z@+\-\[\]\(\)\\\/%=#$,.~&!]*)$"
    for s in smiles_lst:
        if s is None:
            raise ValueError(f"Invalid SMILES: {s}")
        if not re.match(regexp, s):
            raise ValueError(f"Invalid SMILES: {s}")


# -----------------------------
# RDKit backend
# -----------------------------

def _looks_like_rdkit_fpgen(fpgen: Any) -> bool:
    return hasattr(fpgen, "GetFingerprintAsNumPy") and hasattr(fpgen, "GetSparseCountFingerprint")


def _mol_from_smiles_robust(smiles: str) -> Optional["Chem.Mol"]:
    """
    Parse SMILES into an RDKit Mol.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("MolFromSmiles returned None with default sanitization.")
    except Exception as e:
        print(f"Error processing SMILES {smiles} with default sanitization: {e}")
        print("Retrying with sanitize=False...")
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                | Chem.SanitizeFlags.SANITIZE_KEKULIZE
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                catchErrors=True,
            )
            if mol is None:
                raise ValueError("MolFromSmiles returned None even with sanitize=False.")
        except Exception as e2:
            print(f"Error processing SMILES {smiles} with sanitize=False: {e2}")
            return None
    return mol


def _infer_fp_size_folded(fpgen: Any, mol: "Chem.Mol", count: bool) -> int:
    """
    Infer folded vector length for RDKit generator from a molecule.
    """
    if count:
        v = fpgen.GetCountFingerprint(mol)
        return int(v.GetLength())
    bv = fpgen.GetFingerprint(mol)
    return int(bv.GetNumBits())


def _compute_rdkit(
    smiles: Sequence[str],
    fpgen: Any,
    cfg: FingerprintConfig,
    *,
    show_progress: bool,
) -> FingerprintResult:
    if not cfg.folded:
        return _rdkit_unfolded(smiles, fpgen, cfg, show_progress=show_progress)

    if cfg.return_csr:
        return _rdkit_folded_csr(smiles, fpgen, cfg, show_progress=show_progress)

    return _rdkit_folded_dense(smiles, fpgen, cfg, show_progress=show_progress)


def _rdkit_unfolded(
    smiles: Sequence[str],
    fpgen: Any,
    cfg: FingerprintConfig,
    *,
    show_progress: bool,
) -> FingerprintResult:
    """
    Unfolded output for RDKit: use fpgen.GetSparseCountFingerprint(mol) to obtain feature IDs.

    - count=False: List[np.ndarray[int64]] feature IDs
    - count=True : List[(keys:int64, vals:float32)] feature IDs + counts (optionally scaled/weighted)
    """
    if cfg.count:
        out: UnfoldedCount = []
        for s in tqdm(smiles, disable=(not show_progress)):
            mol = _mol_from_smiles_robust(s)
            if mol is None:
                _handle_invalid(cfg.invalid_policy, s)
                if cfg.invalid_policy == "keep":
                    out.append(_empty_unfolded_count())
                continue

            nz = fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements()
            keys = np.array(sorted(nz.keys()), dtype=np.int64)
            vals = np.array([float(nz[k]) for k in keys], dtype=np.float32)

            vals = _postprocess_unfolded_vals(keys, vals, cfg)
            out.append((keys, vals))

        return out

    out: UnfoldedBinary = []
    for s in tqdm(smiles, disable=(not show_progress)):
        mol = _mol_from_smiles_robust(s)
        if mol is None:
            _handle_invalid(cfg.invalid_policy, s)
            if cfg.invalid_policy == "keep":
                out.append(_empty_unfolded_binary())
            continue

        nz = fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements()
        keys = np.array(sorted(nz.keys()), dtype=np.int64)
        out.append(keys)

    return out


def _rdkit_folded_dense(
    smiles: Sequence[str],
    fpgen: Any,
    cfg: FingerprintConfig,
    *,
    show_progress: bool,
) -> np.ndarray:
    """
    Dense folded output (N, D) float32 for RDKit generators.
    """
    rows: List[np.ndarray] = []
    n_features: Optional[int] = None
    pending_invalid: List[int] = []  # indices in `rows` that need backfill after we learn D

    for s in tqdm(smiles, disable=(not show_progress)):
        mol = _mol_from_smiles_robust(s)
        if mol is None:
            _handle_invalid(cfg.invalid_policy, s)
            if cfg.invalid_policy == "keep":
                rows.append(np.array([], dtype=np.float32))
                pending_invalid.append(len(rows) - 1)
            continue

        if n_features is None:
            n_features = _infer_fp_size_folded(fpgen, mol, cfg.count)

        arr = fpgen.GetCountFingerprintAsNumPy(mol) if cfg.count else fpgen.GetFingerprintAsNumPy(mol)
        arr = arr.astype(np.float32, copy=False)

        if cfg.count and cfg.scaling == "log":
            arr = _log1p_inplace_safe(arr)

        rows.append(arr)

    if cfg.invalid_policy == "drop":
        X = np.stack(rows).astype(np.float32, copy=False) if rows else np.zeros((0, 0), dtype=np.float32)
    else:
        if n_features is None:
            X = np.zeros((len(smiles), 0), dtype=np.float32)
        else:
            for idx in pending_invalid:
                rows[idx] = np.zeros((n_features,), dtype=np.float32)
            X = np.stack(rows).astype(np.float32, copy=False)

    if cfg.folded_weights is not None and X.size > 0:
        X = _apply_folded_weights_dense(X, cfg.folded_weights)

    return X


def _rdkit_folded_csr(
    smiles: Sequence[str],
    fpgen: Any,
    cfg: FingerprintConfig,
    *,
    show_progress: bool,
) -> sp.csr_matrix:
    """
    Folded CSR output for RDKit generators.

    Builds CSR by collecting per-row index/value chunks and concatenating once at the end.

    Row semantics w.r.t invalid SMILES:
    - drop: row is omitted (output has fewer rows)
    - keep: row is kept as all-zeros (output aligned to input)
    - raise: raises ValueError
    """
    n_features: Optional[int] = None

    idx_chunks: List[np.ndarray] = []
    val_chunks: List[np.ndarray] = []
    row_lengths: List[int] = []

    w: Optional[np.ndarray] = None
    if cfg.folded_weights is not None:
        w = np.asarray(cfg.folded_weights, dtype=np.float32).ravel()

    for s in tqdm(smiles, disable=(not show_progress)):
        mol = _mol_from_smiles_robust(s)
        if mol is None:
            _handle_invalid(cfg.invalid_policy, s)

            if cfg.invalid_policy == "keep":
                row_lengths.append(0)

            # drop: omit row entirely
            continue

        if n_features is None:
            n_features = _infer_fp_size_folded(fpgen, mol, cfg.count)
            if w is not None and w.shape[0] != n_features:
                raise ValueError(f"folded_weights length {w.shape[0]} does not match feature dim {n_features}.")

        arr = fpgen.GetCountFingerprintAsNumPy(mol) if cfg.count else fpgen.GetFingerprintAsNumPy(mol)
        arr = arr.astype(np.float32, copy=False)

        if cfg.count and cfg.scaling == "log":
            arr = _log1p_inplace_safe(arr)

        nz = np.flatnonzero(arr)  # sorted indices
        if nz.size:
            vals = arr[nz]
            if w is not None:
                vals = vals * w[nz]

            idx_chunks.append(nz.astype(np.int32, copy=False))
            val_chunks.append(vals.astype(np.float32, copy=False))
            row_lengths.append(int(nz.size))
        else:
            row_lengths.append(0)

    n_rows = len(row_lengths)

    if n_features is None:
        # empty input OR all invalid; shape depends on whether rows were kept
        return sp.csr_matrix((n_rows, 0), dtype=np.float32)

    indptr = np.empty(n_rows + 1, dtype=np.int64)
    indptr[0] = 0
    if n_rows:
        indptr[1:] = np.cumsum(np.asarray(row_lengths, dtype=np.int64))
    else:
        indptr[1:] = 0

    if idx_chunks:
        indices = np.concatenate(idx_chunks, axis=0).astype(np.int32, copy=False)
        data = np.concatenate(val_chunks, axis=0).astype(np.float32, copy=False)
    else:
        indices = np.array([], dtype=np.int32)
        data = np.array([], dtype=np.float32)

    return sp.csr_matrix((data, indices, indptr), shape=(n_rows, int(n_features)), dtype=np.float32)


# -----------------------------
# sklearn / scikit-fingerprints backend
# -----------------------------

def _looks_like_sklearn_transformer(fpgen: Any) -> bool:
    return hasattr(fpgen, "transform") and hasattr(fpgen, "get_params")


def _clone_transformer_with_params(fpgen: SklearnTransformer, updates: Dict[str, Any]) -> SklearnTransformer:
    params = fpgen.get_params(deep=False)
    params.update(updates)
    return fpgen.__class__(**params)  # type: ignore[arg-type]


def _skfp_configure_output(
    fpgen: SklearnTransformer,
    cfg: FingerprintConfig,
    *,
    show_progress: bool,
) -> SklearnTransformer:
    """
    Configure scikit-fingerprints/sklearn transformer to match (folded, return_csr).

    - folded=True : use the transformer's folded output
    - folded=False: require variant='raw_bits' if supported
    - return_csr=True (only when folded=True): prefer transformer sparse CSR if supported
    """
    params = fpgen.get_params(deep=False)
    updates: Dict[str, Any] = {}

    if "verbose" in params:
        updates["verbose"] = 1 if show_progress else 0

    if not cfg.folded:
        if "variant" not in params:
            raise NotImplementedError(
                "Requested folded=False (unfolded), but this transformer does not expose a `variant` parameter "
                "for an unfolded feature space (e.g., variant='raw_bits')."
            )
        if params.get("variant") != "raw_bits":
            updates["variant"] = "raw_bits"

        # For unfolded conversion we can accept either dense or CSR outputs, so we do not force "sparse".
        return _clone_transformer_with_params(fpgen, updates) if updates else fpgen

    # folded=True
    if "variant" in params and params.get("variant") == "raw_bits":
        updates["variant"] = "folded"

    if "sparse" in params:
        desired = bool(cfg.return_csr)
        if params.get("sparse") != desired:
            updates["sparse"] = desired

    return _clone_transformer_with_params(fpgen, updates) if updates else fpgen


def _compute_sklearn(
    smiles: Sequence[str],
    fpgen: SklearnTransformer,
    cfg: FingerprintConfig,
    *,
    show_progress: bool = False,
) -> FingerprintResult:
    fp = _skfp_configure_output(fpgen, cfg, show_progress=show_progress)
    X = fp.transform(smiles)

    if not cfg.folded:
        # unfolded output
        if sp.issparse(X):
            return _csr_matrix_to_unfolded(X.tocsr().astype(np.float32), cfg)
        return _dense_matrix_to_unfolded(np.asarray(X, dtype=np.float32), cfg)

    # folded=True
    if cfg.return_csr:
        if sp.issparse(X):
            X_csr = X.tocsr().astype(np.float32)
        else:
            X_csr = sp.csr_matrix(np.asarray(X, dtype=np.float32), dtype=np.float32)

        if cfg.count and cfg.scaling == "log":
            X_csr = X_csr.copy()
            X_csr.data = np.log1p(X_csr.data).astype(np.float32, copy=False)

        if cfg.folded_weights is not None:
            X_csr = _apply_folded_weights_csr(X_csr, cfg.folded_weights)

        return X_csr

    # folded=True, return_csr=False => dense
    X_dense = np.asarray(X, dtype=np.float32)

    if cfg.count and cfg.scaling == "log":
        X_dense = _log1p_inplace_safe(X_dense)
    if cfg.folded_weights is not None and X_dense.size > 0:
        X_dense = _apply_folded_weights_dense(X_dense, cfg.folded_weights)

    return X_dense


def _dense_matrix_to_unfolded(X: np.ndarray, cfg: FingerprintConfig) -> FingerprintResult:
    """
    Convert a dense (N, D) matrix into unfolded feature IDs (and optional values).
    """
    if cfg.count:
        out: UnfoldedCount = []
        for i in range(X.shape[0]):
            row = X[i]
            keys = np.flatnonzero(row).astype(np.int64)
            vals = row[keys].astype(np.float32, copy=False)
            vals = _postprocess_unfolded_vals(keys, vals, cfg)
            out.append((keys, vals))
        return out

    out: UnfoldedBinary = []
    for i in range(X.shape[0]):
        out.append(np.flatnonzero(X[i]).astype(np.int64))
    return out


def _csr_matrix_to_unfolded(X: sp.csr_matrix, cfg: FingerprintConfig) -> FingerprintResult:
    """
    Convert a CSR (N, D) matrix into unfolded feature IDs (and optional values).
    """
    if cfg.count:
        out: UnfoldedCount = []
        for i in range(X.shape[0]):
            start, end = X.indptr[i], X.indptr[i + 1]
            keys = X.indices[start:end].astype(np.int64, copy=False)
            vals = X.data[start:end].astype(np.float32, copy=False)
            vals = _postprocess_unfolded_vals(keys, vals, cfg)
            out.append((keys, vals))
        return out

    out: UnfoldedBinary = []
    for i in range(X.shape[0]):
        start, end = X.indptr[i], X.indptr[i + 1]
        out.append(X.indices[start:end].astype(np.int64, copy=False))
    return out
