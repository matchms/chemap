import hashlib
import re
from collections import defaultdict
from collections.abc import Sequence
from numbers import Integral
import numpy as np
from scipy.sparse import csr_array
from skfp.utils import ensure_smiles
from sklearn.utils._param_validation import Interval
from chemap.fingerprints import ChemapBaseFingerprint


class LingoFingerprint(ChemapBaseFingerprint):
    """
    Lingo fingerprint with chemap unfolded support.

    folded=True:
        behaves like scikit-fingerprints: fixed-size hashed vector (dense or CSR)
    folded=False:
        returns chemap unfolded formats with stable 64-bit feature IDs derived from SHA-1:
          - count=False: List[np.ndarray[int64]] (feature IDs)
          - count=True : List[Tuple[np.ndarray[int64], np.ndarray[float32]]] (IDs + counts)
    """

    _parameter_constraints: dict = {
        **ChemapBaseFingerprint._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "substring_length": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        fp_size: int = 4096,
        substring_length: int = 4,
        count: bool = False,
        sparse: bool = False,
        folded: bool = True,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            count=count,
            sparse=sparse,
            folded=folded,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.substring_length = substring_length

    # --------------------
    # Shared preprocessing
    # --------------------

    def smiles_to_dicts(self, X: Sequence[str]) -> list[dict[str, int]]:
        """
        Convert SMILES to dicts of substring counts (original Lingo raw features).
        """
        X = ensure_smiles(X)

        # same canonicalization as skfp
        X = [re.sub(r"[123456789]", "0", smi) for smi in X]
        X = [re.sub(r"Cl", "L", smi) for smi in X]
        X = [re.sub(r"Br", "R", smi) for smi in X]

        result: list[dict[str, int]] = []
        L = self.substring_length

        for smi in X:
            d: defaultdict[str, int] = defaultdict(int)
            # overlapping substrings
            for i in range(len(smi) - L + 1):
                d[smi[i : i + L]] += 1
            result.append(dict(d))

        return result

    # --------------------
    # Folded (matrix) path
    # --------------------

    def _calculate_fingerprint(self, X: Sequence[str]) -> np.ndarray | csr_array:
        """
        Called by BaseFingerprintTransformer when folded=True.
        """
        dicts = self.smiles_to_dicts(X)
        arr = self._dicts_to_folded_array(dicts)
        return csr_array(arr) if self.sparse else arr

    def _dicts_to_folded_array(self, dicts: list[dict[str, int]]) -> np.ndarray:
        """
        Hash and fold into [0..fp_size-1], identical to skfp folding rule.
        """
        dtype = np.uint32 if self.count else np.uint8
        out = np.zeros((len(dicts), self.fp_size), dtype=dtype)

        for i, d in enumerate(dicts):
            for token, c in d.items():
                digest = hashlib.sha1(token.encode("utf-8"), usedforsecurity=False).digest()
                hash_index = int.from_bytes(digest, byteorder="big") % self.fp_size

                if self.count:
                    out[i, hash_index] += c
                else:
                    out[i, hash_index] = 1

        return out

    # -----------------------
    # Unfolded (chemap) path
    # -----------------------

    def _calculate_unfolded(self, X_smiles: Sequence[str]):
        """
        Return chemap unfolded formats.

        Feature IDs are stable int64 derived from SHA-1 digest:
          id64 = int.from_bytes(digest[:8], "big")  (uint64, then viewed as int64 safely via np.uint64->np.int64 cast)
        """
        dicts = self.smiles_to_dicts(X_smiles)

        def token_to_id32(token: str) -> int:
            digest = hashlib.sha1(token.encode("utf-8"), usedforsecurity=False).digest()
            return int.from_bytes(digest[:4], byteorder="big", signed=False)

        if self.count:
            def one(d: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
                if not d:
                    return (np.array([], dtype=np.int64), np.array([], dtype=np.float32))

                agg: dict[int, float] = {}
                for token, c in d.items():
                    fid = token_to_id32(token)
                    agg[fid] = agg.get(fid, 0.0) + float(c)

                keys = np.array(sorted(agg.keys()), dtype=np.int64)
                vals = np.array([agg[int(k)] for k in keys], dtype=np.float32)
                return keys, vals

            return self._parallel_map(one, dicts)

        def one_bin(d: dict[str, int]) -> np.ndarray:
            if not d:
                return np.array([], dtype=np.int64)

            ids = np.fromiter((token_to_id32(t) for t in d.keys()), dtype=np.int64)
            # np.unique sorts ascending
            return np.unique(ids).astype(np.int64, copy=False)

        return self._parallel_map(one_bin, dicts)

