"""
chemap-compatible MAP4 FP generator (in parts based on Luca Cappelletti's implementation of MAP4:
https://github.com/LucaCappelletti94/map4/blob/master/map4/map4.py
Which is based on the original MAP4 implementation 
`Alice Capecchi, Daniel Probst, Jean-Louis Reymond
        "One molecular fingerprint to rule them all: drugs, biomolecules, and the metabolome"
        J Cheminform 12, 43 (2020)
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00445-4>`_

There are a few particular aspects about this implementation tough:
- Folded output:
    * binary (uint8) uses MHFP-style MinHash folding (chemap.fingerprints.mhfp)
    * count  (float32) accumulates true shingle multiplicities into folded bins (not a MinHash signature,
    so different from the original implementation!)
- Unfolded output:
    * count=True  -> true counts per raw feature id
    * count=False -> keys only (chemap will read keys from GetSparseCountFingerprint)
    * feature ids are SHA1 by default, unless minhash_for_unfolded=True
"""

import itertools
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha1
from typing import Dict, List, Optional, Set
import numpy as np
from rdkit.Chem import Mol, MolToSmiles, PathToSubmol
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN, GetDistanceMatrix
from chemap.fingerprints.mhfp import MHFPEncoderLite


# -----------------------------
# Minimal RDKit-like return types
# -----------------------------

@dataclass(frozen=True)
class _SparseCountFingerprint:
    """RDKit SparseIntVect-like shim for chemap."""
    nz: Dict[int, int]
    def GetNonzeroElements(self) -> Dict[int, int]:
        return self.nz


@dataclass(frozen=True)
class _BitFingerprint:
    """RDKit ExplicitBitVect-like shim for chemap size inference."""
    n_bits: int
    def GetNumBits(self) -> int:
        return self.n_bits


@dataclass(frozen=True)
class _CountFingerprint:
    """RDKit IntSparseIntVect-like shim for chemap size inference."""
    length: int
    def GetLength(self) -> int:
        return self.length


# -----------------------------
# MAP4 shingling core
# -----------------------------

class _MAP4Shingler:
    """
    Generates MAP4 shingles as bytes:
      - envs for radii 1..R
      - for each atom pair (i<j) and each radius index k in [0..R-1]:
          shingle = f"{smaller_env}|{dist}|{larger_env}"
        where smaller/larger chosen by length comparison (ties go to env_b as larger)
      - optional include_duplicated_shingles "suffix trick" is available, but for chemap counts
        we SHOULD NOT use it (we want true multiplicities).
    """

    def __init__(
        self,
        radius: int = 2,
        *,
        include_duplicated_shingles: bool = False,
        max_dist: Optional[int] = None,
        dist_binning: Optional[np.ndarray] = None,
    ):
        if radius <= 0:
            raise ValueError("radius must be > 0.")
        self.radius = int(radius)
        self.include_duplicated_shingles = bool(include_duplicated_shingles)
        self.max_dist = max_dist
        self.dist_binning = dist_binning

    def shingles_unique(self, mol: Mol) -> Set[bytes]:
        return set(self._all_pairs(mol, self._get_atom_envs(mol)))

    def shingles_with_counts_true(self, mol: Mol) -> Dict[bytes, int]:
        """
        True multiplicities (counts) WITHOUT suffix trick, regardless of include_duplicated_shingles.
        """
        counts: Dict[bytes, int] = defaultdict(int)
        for sh in self._all_pairs(mol, self._get_atom_envs(mol), force_no_suffix=True):
            counts[sh] += 1
        return dict(counts)

    def _convert_dist(self, dist: float) -> int:
        if self.dist_binning is None:
            return int(dist)
        return int(np.digitize(dist, self.dist_binning, right=True))

    def _get_atom_envs(self, mol: Mol) -> Dict[int, List[Optional[str]]]:
        atoms_env: Dict[int, List[Optional[str]]] = {}
        for atom in mol.GetAtoms():
            atom_identifier = atom.GetIdx()
            for r in range(1, self.radius + 1):
                atoms_env.setdefault(atom_identifier, []).append(
                    self._find_env(mol, atom_identifier, r)
                )
        return atoms_env

    @staticmethod
    def _find_env(mol: Mol, atom_identifier: int, radius: int) -> Optional[str]:
        atom_identifiers_within_radius: List[int] = FindAtomEnvironmentOfRadiusN(
            mol=mol, radius=radius, rootedAtAtom=atom_identifier
        )
        atom_map: Dict[int, int] = {}
        sub_molecule: Mol = PathToSubmol(mol, atom_identifiers_within_radius, atomMap=atom_map)

        if atom_identifier not in atom_map:
            return None

        return MolToSmiles(
            sub_molecule,
            rootedAtAtom=atom_map[atom_identifier],
            canonical=True,
            isomericSmiles=False,
        )

    def _all_pairs(
        self,
        mol: Mol,
        atoms_env: Dict[int, List[Optional[str]]],
        *,
        force_no_suffix: bool = False,
    ) -> List[bytes]:
        """
        Return shingles as bytes. If include_duplicated_shingles is enabled and not forced off,
        suffix trick is applied to make duplicates unique (MAP4C-style behavior).
        """
        out: List[bytes] = []
        dm = GetDistanceMatrix(mol)
        n = mol.GetNumAtoms()
        shingle_dict: Dict[str, int] = defaultdict(int)

        for i, j in itertools.combinations(range(n), 2):
            dist_val = float(dm[i][j])
            if self.max_dist is not None and dist_val > self.max_dist:
                continue
            dist = str(self._convert_dist(dist_val))

            for k in range(self.radius):
                env_a = atoms_env[i][k] or ""
                env_b = atoms_env[j][k] or ""

                # compare by length, not lexicographic
                if len(env_a) > len(env_b):
                    larger_env, smaller_env = env_a, env_b
                else:
                    larger_env, smaller_env = env_b, env_a

                shingle = f"{smaller_env}|{dist}|{larger_env}"

                if self.include_duplicated_shingles and not force_no_suffix:
                    shingle_dict[shingle] += 1
                    shingle = f"{shingle}|{shingle_dict[shingle]}"

                out.append(shingle.encode("utf-8"))

        return out


# -----------------------------
# MAP4 fpgen for chemap
# -----------------------------

class MAP4FPGen:
    """
    chemap-compatible MAP4 fingerprint generator.

    Folded outputs (fixed length):
      - GetFingerprintAsNumPy: uint8[D] binary
          computed by minhash signature (MHFPEncoderLite) folded to bits (mod D)
      - GetCountFingerprintAsNumPy: float32[D] counts
          computed by hashing each shingle (token hash32) -> bin (mod D) and summing true counts

    Unfolded outputs (raw feature ids):
      - GetSparseCountFingerprint returns {feature_id: count}
          feature_id:
            * sha1 truncation (default) OR
            * token-hash32 (sha1 first 4 bytes) if minhash_for_unfolded=True

    Parameters
    ----------
    folded:
        Whether folded functions are meaningful (chemap controls this, but we keep for safety).
    minhash_for_unfolded:
        If True, unfolded uses MHFP-style token hash32 rather than sha1 truncation.
    unfolded_bits:
        32 or 64 (only used when minhash_for_unfolded=False).
    include_duplicated_shingles:
        For MAP4C-like behavior in *set shingles* (folded binary). For true counts we ignore suffix.
    """

    def __init__(
        self,
        dimensions: int = 1024,
        radius: int = 2,
        *,
        seed: int = 75434278,
        folded: bool = True,
        # counts/dup behavior
        include_duplicated_shingles: bool = False,
        # unfolded hashing behavior
        minhash_for_unfolded: bool = False,
        unfolded_bits: int = 32,  # 32 or 64, only if minhash_for_unfolded=False
        # optional distance handling
        max_dist: Optional[int] = None,
        dist_binning: Optional[np.ndarray] = None,
    ):
        self.dimensions = int(dimensions)
        self.radius = int(radius)
        self.seed = int(seed)
        self.folded = bool(folded)

        self.include_duplicated_shingles = bool(include_duplicated_shingles)
        self.minhash_for_unfolded = bool(minhash_for_unfolded)
        self.unfolded_bits = int(unfolded_bits)

        if self.dimensions <= 0:
            raise ValueError("dimensions must be > 0.")
        if self.radius <= 0:
            raise ValueError("radius must be > 0.")
        if self.unfolded_bits not in (32, 64):
            raise ValueError("unfolded_bits must be 32 or 64.")

        self._shingler = _MAP4Shingler(
            radius=self.radius,
            include_duplicated_shingles=self.include_duplicated_shingles,
            max_dist=max_dist,
            dist_binning=dist_binning,
        )

        # Folded uses MHFPEncoderLite
        self._mhfp = MHFPEncoderLite(
            n_permutations=self.dimensions,
            seed=self.seed,
        )

    # --------- chemap size inference ---------

    def GetFingerprint(self, mol: Mol) -> _BitFingerprint:
        return _BitFingerprint(self.dimensions)

    def GetCountFingerprint(self, mol: Mol) -> _CountFingerprint:
        return _CountFingerprint(self.dimensions)

    # --------- unfolded API ---------

    def GetSparseCountFingerprint(self, mol: Mol) -> _SparseCountFingerprint:
        """
        Returns {feature_id: count} for unfolded outputs.

        - count=True in chemap: keys+values used
        - count=False in chemap: keys used, values ignored
        """
        counts = self._shingler.shingles_with_counts_true(mol)
        if not counts:
            return _SparseCountFingerprint({})

        nz: Dict[int, int] = defaultdict(int)

        if self.minhash_for_unfolded:
            # MHFP token-hash domain: sha1 first 4 bytes (little endian)
            for sh, c in counts.items():
                fid32 = int.from_bytes(sha1(sh).digest()[:4], "little", signed=False)
                nz[int(fid32)] += int(c)
        else:
            for sh, c in counts.items():
                fid = int(self._sha1_to_int(sh, bits=self.unfolded_bits))
                nz[fid] += int(c)

        return _SparseCountFingerprint(dict(nz))

    # --------- folded API ---------

    def GetFingerprintAsNumPy(self, mol: Mol) -> np.ndarray:
        """
        Folded binary vector uint8[D], matching original MAP4Calculator folded path:

            folded = fold(hash(set(shingles)), D)

        i.e. hash each unique shingle token -> set bit at (hash % D).
        """
        if not self.folded:
            return np.zeros(self.dimensions, dtype=np.uint8)

        shingles = self._shingler.shingles_unique(mol)  # set[bytes]
        if not shingles:
            return np.zeros(self.dimensions, dtype=np.uint8)

        # Per-shingle 32-bit hash (matches the common MAP4/scikit-fingerprints style: sha1/sha256 truncated)
        hashed = np.fromiter(
            (int.from_bytes(sha1(sh).digest()[:4], "little", signed=False) for sh in shingles),
            dtype=np.uint32,
            count=len(shingles),
        )

        fp = np.zeros(self.dimensions, dtype=np.uint8)
        fp[(hashed % np.uint32(self.dimensions)).astype(np.int64, copy=False)] = 1
        return fp

    def GetCountFingerprintAsNumPy(self, mol: Mol) -> np.ndarray:
        """
        Folded counts float32[D] using TRUE multiplicities.

        This is *not* a MinHash (classic MAP4 is set-based). We instead provide a stable
        count-fold baseline:
          bin = token_hash32(shingle) % D
          fp[bin] += count
        """
        if not self.folded:
            return np.zeros(self.dimensions, dtype=np.float32)

        counts = self._shingler.shingles_with_counts_true(mol)
        if not counts:
            return np.zeros(self.dimensions, dtype=np.float32)

        fp = np.zeros(self.dimensions, dtype=np.float32)
        for sh, c in counts.items():
            h32 = int.from_bytes(sha1(sh).digest()[:4], "little", signed=False)
            fp[h32 % self.dimensions] += float(c)
        return fp

    # -----------------------------
    # Hash utilities
    # -----------------------------

    @staticmethod
    def _sha1_to_int(data: bytes, *, bits: int = 64) -> np.uint64:
        d = sha1(data).digest()
        if bits == 32:
            return np.uint64(int.from_bytes(d[:4], byteorder="little", signed=False))
        return np.uint64(int.from_bytes(d[:8], byteorder="little", signed=False))
