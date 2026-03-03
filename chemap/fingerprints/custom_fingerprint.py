import numpy as np
from collections.abc import Sequence
from typing import Any, Callable, Dict, Optional

from chemap.fingerprints import ChemapBaseFingerprint
from chemap.types import UnfoldedBinary, UnfoldedCount


class CustomFingerprint(ChemapBaseFingerprint):
    def __init__(
            self,
            fp_func: Callable[[Any], np.ndarray],
            n_features_out: int,
            fp_func_kwargs: Optional[Dict[str, Any]] = None,
            count: bool = False,
            sparse: bool = False,
            folded: bool = True,
            n_jobs: int | None = None,
            batch_size: int | None = None,
            verbose: int | dict = 0,
    ):
        super().__init__(
            n_features_out=n_features_out,
            count=count,
            sparse=sparse,
            folded=folded,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.fp_func = fp_func
        self.fp_func_kwargs = fp_func_kwargs or {}

    def _calculate_fingerprint(self, X: Sequence[Any]):
        def one(item):
            if item is None:
                return np.array([], dtype=np.float32)
            return self.fp_func(item, **self.fp_func_kwargs)

        results = self._parallel_map(one, X)

        # Infer dimension
        D = next((r.shape[0] for r in results if r.size), self.n_features_out)

        rows = []
        for r in results:
            if r.size == 0:
                rows.append(np.zeros(D, dtype=np.float32))
            else:
                rows.append(r.astype(np.float32, copy=False))

        return np.vstack(rows)

    def _calculate_unfolded(self, X: Sequence[Any]) -> UnfoldedBinary | UnfoldedCount:

        def one(item):
            if item is None:
                return np.array([], dtype=np.int64) if not self.count else (np.array([], dtype=np.int64),
                                                                            np.array([], dtype=np.float32))

            fp = self.fp_func(item, **self.fp_func_kwargs)

            if self.count:
                keys = np.flatnonzero(fp).astype(np.int64)
                vals = fp[keys].astype(np.float32)
                return keys, vals
            else:
                return np.flatnonzero(fp).astype(np.int64)

        return self._parallel_map(one, X)
