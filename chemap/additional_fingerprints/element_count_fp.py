from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin


class ElementCountFingerprint(BaseEstimator, TransformerMixin):
    """
    sklearn-fingerprints-style element-count "fingerprint". This is not what can be consider a real fingerprint,
    but it can be useful as a simple baseline or for certain applications (for instance in concatenation with
    other representations).

    Input:  Sequence[rdkit.Chem.Mol]
    Output: (N, D) float32 dense or CSR float32 (if sparse=True)

    Notes
    -----
    - Feature order is defined by `elements`. If provided, D is fixed.
    - include_hs:
        * "implicit": counts implicit Hs via atom.GetTotalNumHs()
        * "explicit": counts only explicit H atoms present in the graph
        * "none": do not count H at all
    - unknown_policy:
        * "ignore": drop elements not in `elements`
        * "other": accumulate into an "Other" feature (appended if not in `elements`)
        * "error": raise if an unknown element is encountered
    """

    def __init__(
        self,
        *,
        elements: Optional[Sequence[str]] = None,
        include_hs: str = "implicit",     # "implicit" | "explicit" | "none"
        unknown_policy: str = "other",    # "ignore" | "other" | "error"
        sparse: bool = False,
        n_jobs: int = 1,
        verbose: int = 0,
    ):
        """Initialize the ElementCountFingerprint transformer.
        
        Parameters
        ----------
        elements : Optional[Sequence[str]], optional
            List of element symbols to include as features. If None, the vocabulary will be inferred from the data.
        include_hs : str, optional
            How to handle hydrogen atoms. Options are "implicit", "explicit", or "none".
        unknown_policy : str, optional
            Policy for handling elements not in the `elements` list. Options are "ignore", "other", or "error".
        sparse : bool, optional
            Whether to return the output as a sparse matrix (CSR) or a dense NumPy array.
        n_jobs : int, optional
            Number of parallel jobs to run for the transformation. Default is 1 (no parallelism).
        verbose : int, optional
            Verbosity level for parallel processing. Default is 0 (no verbosity).
        """
        self.elements = list(elements) if elements is not None else None
        self.include_hs = include_hs
        self.unknown_policy = unknown_policy
        self.sparse = sparse
        self.variant = "folded"  # unfolded does not really make sense for element counts, only for API consistency.
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: Sequence[Any], y: Any = None) -> "ElementCountFingerprint":
        # If elements not given, infer a stable vocabulary from X (plus H depending on include_hs)
        if self.elements is None:
            vocab = set()
            for mol in X:
                if mol is None:
                    continue
                for atom in mol.GetAtoms():
                    vocab.add(atom.GetSymbol())
                if self.include_hs == "implicit":
                    vocab.add("H")
                elif self.include_hs == "explicit":
                    # Only adds H if explicit H atoms exist
                    # Default is that RDKit mols do not have explicit Hs, so this won't add H unless they are present
                    vocab.add("H")

            elems = sorted(vocab)
            if self.unknown_policy == "other" and "Other" not in elems:
                elems.append("Other")
            self.elements_ = elems
        else:
            elems = list(self.elements)
            if self.unknown_policy == "other" and "Other" not in elems:
                elems.append("Other")
            self.elements_ = elems

        self._elem2idx_: Dict[str, int] = {e: i for i, e in enumerate(self.elements_)}
        self.n_features_in_ = len(self.elements_)
        return self

    def transform(self, X: Sequence[Any]):
        """
        Transform a sequence of RDKit Mol objects into element count fingerprints.
        
        Parameters
        ----------
        X : Sequence[Any]
            A sequence of RDKit Mol objects to transform.
        """
        if not hasattr(self, "_elem2idx_"):
            # sklearn convention: allow transform without explicit fit
            self.fit(X)

        D = self.n_features_in_

        def fp_row(mol) -> np.ndarray:
            if mol is None:
                return np.zeros((D,), dtype=np.float32)

            out = np.zeros((D,), dtype=np.float32)
            other_idx = self._elem2idx_.get("Other", None)

            # Count non-H atoms only (H is handled separately below)
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                if sym == "H":
                    continue  # avoid double-counting in explicit/implicit modes

                idx = self._elem2idx_.get(sym, None)
                if idx is not None:
                    out[idx] += 1.0
                else:
                    if self.unknown_policy == "ignore":
                        pass
                    elif self.unknown_policy == "other" and other_idx is not None:
                        out[other_idx] += 1.0
                    else:
                        raise ValueError(f"Unknown element '{sym}' not in elements vocabulary.")

            # Count H depending on mode
            if self.include_hs == "implicit":
                h_idx = self._elem2idx_.get("H", None)
                if h_idx is not None:
                    h_count = 0
                    for atom in mol.GetAtoms():
                        # Note: for explicit-H molecules (after AddHs), GetTotalNumHs() on heavy atoms is typically 0,
                        # but we don't rely on that in explicit mode anyway.
                        h_count += int(atom.GetTotalNumHs())
                    out[h_idx] += float(h_count)

            elif self.include_hs == "explicit":
                h_idx = self._elem2idx_.get("H", None)
                if h_idx is not None:
                    h_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "H")
                    out[h_idx] += float(h_count)

            return out

        rows: List[np.ndarray] = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(fp_row)(mol) for mol in X
        )

        X_dense = np.vstack(rows).astype(np.float32, copy=False) if rows else np.zeros((0, D), dtype=np.float32)

        if not self.sparse:
            return X_dense

        return sp.csr_matrix(X_dense, dtype=np.float32)
