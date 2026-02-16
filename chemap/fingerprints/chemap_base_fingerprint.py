from collections.abc import Sequence
from typing import Any
import numpy as np
from joblib import Parallel, delayed
from rdkit.Chem import Mol
from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_smiles
from chemap.types import UnfoldedBinary, UnfoldedCount


class ChemapBaseFingerprint(BaseFingerprintTransformer):
    """
    Extension of scikit-fingerprints BaseFingerprintTransformer that adds `folded`.

    - folded=True: behaves like scikit-fingerprints (returns dense ndarray or sparse csr_array)
    - folded=False: returns chemap unfolded formats (lists of feature IDs / (IDs, values))

    Important: this class intentionally subclasses scikit-fingerprints' base class
    to preserve their behavior (validation, parallelization patterns, etc.) where possible.
    """

    def __init__(
        self,
        *,
        n_features_out: int,
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
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.folded = folded

    def transform(self, X: Sequence[str | Mol], copy: bool = False) -> Any:
        """
        If folded=True: defer to BaseFingerprintTransformer.transform (matrix output).
        If folded=False: return chemap unfolded formats.
        """
        if self.folded:
            return super().transform(X, copy=copy)

        # unfolded route: we accept SMILES or Mol, but Lingo-like methods want SMILES
        smiles = ensure_smiles(X)
        return self._calculate_unfolded(smiles)

    # ---- hooks for subclasses ----

    def _calculate_unfolded(self, X_smiles: Sequence[str]) -> UnfoldedBinary | UnfoldedCount:
        """
        Subclasses must implement when folded=False.
        Must return chemap unfolded formats:
          - count=False: List[np.ndarray[int64]]
          - count=True : List[Tuple[np.ndarray[int64], np.ndarray[float32]]]
        """
        raise NotImplementedError

    # ---- helpers ----

    def _parallel_map(self, fn, items):
        n_jobs = self.n_jobs if self.n_jobs is not None else 1

        if n_jobs == 1:
            return [fn(x) for x in items]

        batch_size = self.batch_size if self.batch_size is not None else "auto"
        return Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(fn)(x) for x in items
        )
