from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, Protocol
from tqdm import tqdm
import numpy as np
from rdkit import Chem


# -----------------------------
# Public types and configuration
# -----------------------------

InvalidPolicy = Literal["drop", "keep", "raise"]
Scaling = Optional[Literal["log"]]

RaggedBinary = List[np.ndarray]  # list of int64 keys per molecule
RaggedCount = List[Tuple[np.ndarray, np.ndarray]]  # list of (int64 keys, float32 values)
Dense = np.ndarray

FingerprintResult = Union[Dense, RaggedBinary, RaggedCount]


@dataclass(frozen=True)
class FingerprintConfig:
    """
    Fingerprint computation settings.

    Important terminology
    ---------------------
    sparse:
        Uses the *RDKit* interpretation of "sparse": an **unfolded / not-folded**
        representation returned as a **ragged** structure:
          - count=False: List[np.ndarray[int64]] (bit IDs)
          - count=True : List[Tuple[np.ndarray[int64], np.ndarray[float32]]] (bit IDs, counts)

        This is intentionally **NOT** SciPy sparse matrices (CSR). Even for scikit-fingerprints,
        we avoid CSR outputs because downstream similarity code expects either dense NumPy arrays
        or ragged sparse representations.

    Parameters
    ----------
    count:
        If True, compute count fingerprints (counts/weights per feature).
        If False, compute binary fingerprints.
    sparse:
        If True, return ragged sparse representation (unfolded / not-folded).
        If False, return dense folded vectors as a 2D NumPy array (N, D).
    scaling:
        Optional scaling for count outputs:
          - None: no scaling
          - "log": apply log1p to counts
    dense_weights:
        Optional 1D float array applied elementwise to dense folded outputs (sparse=False only).
    ragged_weights:
        Optional dict {bit_id: weight} applied to ragged count outputs (sparse=True + count=True only).
        Missing keys default to weight 1.0.
    invalid_policy:
        Handling of invalid/unparseable SMILES:
          - "drop": drop invalid molecules from output
          - "keep": keep alignment; returns all-zeros row (dense) or empty arrays (ragged)
          - "raise": raise ValueError on first invalid SMILES
    """

    count: bool = True
    sparse: bool = False  # RDKit-style "sparse" (ragged), NOT SciPy CSR
    scaling: Scaling = None
    dense_weights: Optional[np.ndarray] = None
    ragged_weights: Optional[Dict[int, float]] = None
    invalid_policy: InvalidPolicy = "drop"


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
    If config.sparse is False:
        Dense folded vectors as np.ndarray of shape (N, D), dtype float32.

    If config.sparse is True (RDKit-style sparse/unfolded):
        - config.count False: List[np.ndarray[int64]] (sorted bit IDs)
        - config.count True : List[Tuple[np.ndarray[int64], np.ndarray[float32]]] (sorted bit IDs + counts)

    Notes on scikit-fingerprints
    ----------------------------
    scikit-fingerprints supports returning SciPy CSR matrices via its own `sparse` parameter, but
    this function intentionally avoids CSR outputs and instead returns either dense NumPy arrays
    or ragged RDKit-style sparse representations. :contentReference[oaicite:3]{index=3}

    For config.sparse=True with scikit-fingerprints, we require the transformer to support an
    unfolded feature-space via a `variant="raw_bits"` parameter (e.g., PharmacophoreFingerprint). :contentReference[oaicite:4]{index=4}
    """
    _validate_config(config)

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

    if cfg.sparse is False:
        if cfg.ragged_weights is not None:
            raise ValueError("ragged_weights is only valid when sparse=True and count=True.")
        if cfg.dense_weights is not None:
            w = np.asarray(cfg.dense_weights)
            if w.ndim != 1:
                raise ValueError("dense_weights must be a 1D array.")
    else:
        # ragged output
        if cfg.dense_weights is not None:
            raise ValueError("dense_weights is only valid when sparse=False.")
        if cfg.ragged_weights is not None and cfg.count is False:
            raise ValueError("ragged_weights is only valid when sparse=True and count=True.")


def _log1p_inplace_safe(x: np.ndarray) -> np.ndarray:
    return np.log1p(x).astype(np.float32, copy=False)


def _apply_dense_weights(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32).ravel()
    if X.shape[1] != w.shape[0]:
        raise ValueError(f"dense_weights length {w.shape[0]} does not match feature dim {X.shape[1]}.")
    return (X * w[None, :]).astype(np.float32, copy=False)


def _apply_ragged_weights(keys: np.ndarray, vals: np.ndarray, weights: Dict[int, float]) -> np.ndarray:
    w = np.array([float(weights.get(int(k), 1.0)) for k in keys], dtype=np.float32)
    return (vals * w).astype(np.float32, copy=False)


def _handle_invalid(policy: InvalidPolicy, s: str) -> None:
    if policy == "raise":
        raise ValueError(f"Invalid SMILES: {s}")


def _empty_ragged_binary() -> np.ndarray:
    return np.array([], dtype=np.int64)


def _empty_ragged_count() -> Tuple[np.ndarray, np.ndarray]:
    return np.array([], dtype=np.int64), np.array([], dtype=np.float32)


# -----------------------------
# RDKit backend
# -----------------------------

def _looks_like_rdkit_fpgen(fpgen: Any) -> bool:
    return hasattr(fpgen, "GetFingerprintAsNumPy") and hasattr(fpgen, "GetSparseCountFingerprint")


def _mol_from_smiles_robust(smiles: str) -> Optional["Chem.Mol"]:
    """
    Parse SMILES into an RDKit Mol..
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
            # Regenerate computed properties like implicit valence and ring information
            mol.UpdatePropertyCache(strict=False)

            # Apply several sanitization rules (taken from http://rdkit.org/docs/Cookbook.html)
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                | Chem.SanitizeFlags.SANITIZE_KEKULIZE
                | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
                | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                catchErrors=True
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
    if cfg.sparse:
        return _rdkit_ragged(smiles, fpgen, cfg, show_progress=show_progress)
    return _rdkit_dense(smiles, fpgen, cfg, show_progress=show_progress)


def _rdkit_ragged(
    smiles: Sequence[str],
    fpgen: Any,
    cfg: FingerprintConfig,
    *,
    show_progress: bool,
) -> FingerprintResult:
    """
    RDKit-style sparse/unfolded output (ragged).

    Uses fpgen.GetSparseCountFingerprint(mol), which yields a dict-like mapping
    from feature IDs (potentially large ID space) to counts.
    """
    if cfg.count:
        out: RaggedCount = []
        for s in tqdm(smiles, disable=(not show_progress)):
            mol = _mol_from_smiles_robust(s)
            if mol is None:
                _handle_invalid(cfg.invalid_policy, s)
                if cfg.invalid_policy == "keep":
                    out.append(_empty_ragged_count())
                continue

            nz = fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements()
            keys = np.array(sorted(nz.keys()), dtype=np.int64)
            vals = np.array([float(nz[k]) for k in keys], dtype=np.float32)

            if cfg.scaling == "log":
                vals = _log1p_inplace_safe(vals)
            if cfg.ragged_weights is not None:
                vals = _apply_ragged_weights(keys, vals, cfg.ragged_weights)

            out.append((keys, vals))

        if cfg.invalid_policy == "drop":
            # already dropped by skipping appends
            pass
        return out

    # binary ragged: just the feature IDs where count > 0
    outb: RaggedBinary = []
    for s in tqdm(smiles, disable=(not show_progress)):
        mol = _mol_from_smiles_robust(s)
        if mol is None:
            _handle_invalid(cfg.invalid_policy, s)
            if cfg.invalid_policy == "keep":
                outb.append(_empty_ragged_binary())
            continue

        nz = fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements()
        keys = np.array(sorted(nz.keys()), dtype=np.int64)
        outb.append(keys)

    return outb


def _rdkit_dense(
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
        # no alignment guarantee; invalids are skipped
        X = np.stack(rows).astype(np.float32, copy=False) if rows else np.zeros((0, 0), dtype=np.float32)
    else:
        if n_features is None:
            # all invalid
            X = np.zeros((len(smiles), 0), dtype=np.float32)
        else:
            # backfill invalid rows with zeros now that we know D
            for idx in pending_invalid:
                rows[idx] = np.zeros((n_features,), dtype=np.float32)
            X = np.stack(rows).astype(np.float32, copy=False)

    if cfg.dense_weights is not None and X.size > 0:
        X = _apply_dense_weights(X, cfg.dense_weights)

    return X


# -----------------------------
# sklearn / scikit-fingerprints backend
# -----------------------------

def _looks_like_sklearn_transformer(fpgen: Any) -> bool:
    return hasattr(fpgen, "transform") and hasattr(fpgen, "get_params")


def _clone_transformer_with_params(fpgen: SklearnTransformer, updates: Dict[str, Any]) -> SklearnTransformer:
    """
    Rebuild transformer by copying current shallow params and overriding `updates`.
    """
    params = fpgen.get_params(deep=False)
    params.update(updates)
    return fpgen.__class__(**params)  # type: ignore[arg-type]


def _skfp_force_dense_output(
        fpgen: SklearnTransformer,
        show_progress: bool = True,
        ) -> SklearnTransformer:
    """
    scikit-fingerprints exposes `sparse` as: return dense NumPy vs SciPy CSR. :contentReference[oaicite:5]{index=5}
    We always force dense NumPy output.
    """
    params = fpgen.get_params(deep=False)
    if "sparse" in params and params.get("sparse") is not False:
        return _clone_transformer_with_params(
            fpgen, {
                "sparse": False,
                "verbose": 1 if show_progress else 0,
                })
    return _clone_transformer_with_params(
        fpgen, {
            "verbose": 1 if show_progress else 0,
            })


def _skfp_set_variant_raw_bits(fpgen: SklearnTransformer) -> SklearnTransformer:
    """
    If transformer supports `variant`, request raw/unfolded bits via variant='raw_bits'
    (e.g. PharmacophoreFingerprint supports 'raw_bits' and 'folded'). :contentReference[oaicite:6]{index=6}
    """
    params = fpgen.get_params(deep=False)
    if "variant" not in params:
        raise NotImplementedError(
            "Requested sparse=True (ragged/unfolded), but this transformer does not expose a `variant` "
            "parameter (no supported raw/unfolded feature space). For many skfp fingerprints (e.g. ECFP), "
            "use sparse=False (dense folded) instead."
        )
    if params.get("variant") != "raw_bits":
        return _clone_transformer_with_params(fpgen, {"variant": "raw_bits"})
    return fpgen


def _compute_sklearn(
        smiles: Sequence[str],
        fpgen: SklearnTransformer,
        cfg: FingerprintConfig,
        show_progress: bool = False,
        ) -> FingerprintResult:
    """
    Compute using sklearn/scikit-fingerprints transformers.

    - Always forces transformer param `sparse=False` to avoid CSR outputs.
    - If cfg.sparse=True (ragged), we require a raw/unfolded representation via `variant="raw_bits"`.
    """
    fp_dense = _skfp_force_dense_output(fpgen, show_progress)

    if cfg.sparse:
        fp_dense = _skfp_set_variant_raw_bits(fp_dense)

    X = fp_dense.transform(smiles)  # expected to be NumPy due to force_dense
    X = np.asarray(X, dtype=np.float32)

    if not cfg.sparse:
        # dense folded output
        if cfg.count and cfg.scaling == "log":
            X = _log1p_inplace_safe(X)
        if cfg.dense_weights is not None:
            X = _apply_dense_weights(X, cfg.dense_weights)
        return X

    # cfg.sparse=True: return ragged (unfolded/raw feature space)
    return _dense_matrix_to_ragged(X, cfg)


def _dense_matrix_to_ragged(X: np.ndarray, cfg: FingerprintConfig) -> FingerprintResult:
    """
    Convert a dense (N, D) matrix into ragged keys (and values).

    This is used for "raw_bits" style outputs where the feature space is not folded
    into a small fp_size, but we still get a finite D from the transformer.
    """
    if cfg.count:
        out: RaggedCount = []
        for i in range(X.shape[0]):
            row = X[i]
            keys = np.flatnonzero(row).astype(np.int64)  # sorted
            vals = row[keys].astype(np.float32, copy=False)

            if cfg.scaling == "log":
                vals = _log1p_inplace_safe(vals)
            if cfg.ragged_weights is not None:
                vals = _apply_ragged_weights(keys, vals, cfg.ragged_weights)

            out.append((keys, vals))
        return out

    outb: RaggedBinary = []
    for i in range(X.shape[0]):
        outb.append(np.flatnonzero(X[i]).astype(np.int64))
    return outb
