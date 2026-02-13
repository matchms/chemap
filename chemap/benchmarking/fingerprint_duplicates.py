import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np


# ---------------------------------------------------------------------------
# Encoding / decoding duplicates (list[list[int]]) <-> NPZ (indices, indptr)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DuplicatesNPZ:
    """CSR-like encoding of duplicate groups."""
    indices: np.ndarray  # shape (nnz,), int
    indptr: np.ndarray   # shape (n_groups+1,), int
    n_items: Optional[int] = None  # optional size of the original universe


def encode_duplicates(
    duplicates: Sequence[Sequence[int]],
    *,
    dtype: np.dtype = np.int32,
    n_items: Optional[int] = None,
) -> DuplicatesNPZ:
    """Encode list-of-lists duplicate groups into CSR-like arrays."""
    # Build indptr
    lengths = np.fromiter((len(g) for g in duplicates), dtype=np.int64)
    indptr = np.empty(lengths.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])

    # Build indices
    if indptr[-1] == 0:
        indices = np.array([], dtype=dtype)
    else:
        indices = np.concatenate([np.asarray(g, dtype=dtype) for g in duplicates], axis=0)

    # Validate non-negative
    if indices.size and np.any(indices < 0):
        raise ValueError("duplicates contains negative indices")

    # Optional validate upper bound
    if n_items is not None and indices.size and np.any(indices >= n_items):
        raise ValueError("duplicates contains indices outside [0, n_items)")

    return DuplicatesNPZ(
        indices=indices.astype(dtype, copy=False),
        indptr=indptr.astype(np.int64, copy=False),
        n_items=n_items
        )


def decode_duplicates(encoded: DuplicatesNPZ) -> List[List[int]]:
    """Decode CSR-like arrays back into list-of-lists."""
    indices = np.asarray(encoded.indices)
    indptr = np.asarray(encoded.indptr)

    if indptr.ndim != 1 or indices.ndim != 1:
        raise ValueError("indices and indptr must be 1D arrays")
    if indptr.size == 0 or indptr[0] != 0:
        raise ValueError("indptr must start with 0 and have length n_groups+1")
    if indptr[-1] != indices.size:
        raise ValueError("indptr[-1] must equal len(indices)")
    if np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("indptr must be non-decreasing")

    out: List[List[int]] = []
    for i in range(indptr.size - 1):
        start = int(indptr[i])
        end = int(indptr[i + 1])
        out.append(indices[start:end].astype(int, copy=False).tolist())
    return out


# ---------------------------------------------------------------------------
# File I/O (NPZ + optional JSON metadata)
# ---------------------------------------------------------------------------

def save_duplicates_npz(
    filepath: Union[str, Path],
    duplicates: Sequence[Sequence[int]],
    *,
    n_items: Optional[int] = None,
    dtype: np.dtype = np.int32,
    metadata: Optional[Mapping[str, Any]] = None,
    metadata_suffix: str = ".json",
) -> Path:
    """Save duplicates to a compressed NPZ file (+ optional JSON metadata sidecar)."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    enc = encode_duplicates(duplicates, dtype=dtype, n_items=n_items)
    np.savez_compressed(
        path,
        indices=enc.indices,
        indptr=enc.indptr,
        n_items=np.array([-1 if enc.n_items is None else int(enc.n_items)], dtype=np.int64),
    )

    if metadata is not None:
        meta_path = path.with_suffix(path.suffix + metadata_suffix)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(dict(metadata), f, indent=2, sort_keys=True)

    return path


def load_duplicates_npz(
    filepath: Union[str, Path],
    *,
    load_metadata: bool = False,
    metadata_suffix: str = ".json",
) -> Tuple[List[List[int]], Optional[Dict[str, Any]]]:
    """Load duplicates from NPZ (and optional JSON metadata if present)."""
    path = Path(filepath)
    with np.load(path, allow_pickle=False) as z:
        indices = z["indices"]
        indptr = z["indptr"]
        n_items_arr = z.get("n_items", None)

    n_items: Optional[int] = None
    if n_items_arr is not None:
        v = int(np.asarray(n_items_arr).reshape(-1)[0])
        n_items = None if v < 0 else v

    duplicates = decode_duplicates(DuplicatesNPZ(indices=indices, indptr=indptr, n_items=n_items))

    meta: Optional[Dict[str, Any]] = None
    if load_metadata:
        meta_path = path.with_suffix(path.suffix + metadata_suffix)
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

    return duplicates, meta


# ---------------------------------------------------------------------------
# Bulk loader for a folder of precomputed experiments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrecomputedDuplicates:
    """One precomputed experiment entry loaded from disk."""
    name: str
    path_npz: Path
    duplicates: List[List[int]]
    metadata: Optional[Dict[str, Any]] = None


def load_precomputed_duplicates_folder(
    folder: Union[str, Path],
    *,
    pattern: str = "*_duplicates.npz",
    name_from_filename: Optional[Any] = None,
    load_metadata: bool = False,
) -> List[PrecomputedDuplicates]:
    """Load all precomputed duplicate results from a folder.

    Parameters
    ----------
    folder:
        Directory containing files like "<experiment>_duplicates.npz".
    pattern:
        Glob pattern to select files.
    name_from_filename:
        Optional callable (Path -> str). If None, uses stem with trailing "_duplicates" removed.
    load_metadata:
        If True, also load "<file>.npz.json" sidecar metadata when present.

    Returns
    -------
    List[PrecomputedDuplicates], sorted by name.
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    files = sorted(folder_path.glob(pattern))
    out: List[PrecomputedDuplicates] = []

    def default_name(p: Path) -> str:
        stem = p.stem  # for "x_duplicates.npz" => "x_duplicates"
        return stem[:-11] if stem.endswith("_duplicates") else stem

    name_fn = name_from_filename or default_name

    for fp in files:
        duplicates, meta = load_duplicates_npz(fp, load_metadata=load_metadata)
        out.append(
            PrecomputedDuplicates(
                name=str(name_fn(fp)),
                path_npz=fp,
                duplicates=duplicates,
                metadata=meta,
            )
        )

    out.sort(key=lambda x: x.name.lower())
    return out
