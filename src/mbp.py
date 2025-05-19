import itertools
from typing import List, Set, Dict, Iterable, Optional, Tuple
from collections import defaultdict, Counter
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from rdkit.Chem import MolToSmiles, Mol
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetDistanceMatrix

from mhfp.encoder import MHFPEncoder


class MBP:
    """Morgan Bit-Pair (MBP) fingerprint calculator.
    
    This is a re‐implemented MAP4 fingerprint calculator that uses Morgan fingerprint bits
    (for radii 0 to self.radius) as the basic atomic feature rather than submolecule
    SMILES. Shingles are constructed as:
    
        morgan_fp_bit1|distance|morgan_fp_bit2
        
    The ordering is canonicalized by sorting the two bits.
    """

    def __init__(
        self,
        dimensions: int = 2048,
        radius: int = 2,
        include_duplicated_shingles: bool = False,
        includeChirality: bool = False,
        max_dist: int = None,
        dist_binning = None,
        seed: int = 75434278,
    ):
        """
        Parameters
        ----------
        dimensions : int, default=2048
            The number of dimensions for the fingerprint.
        radius : int, default=2
            The maximum radius for the Morgan bit calculation.
            (Morgan bits are collected for radii 0,1,...,radius.)
        include_duplicated_shingles : bool, default=False
            If True, a counter is employed to make each shingle unique,
            thereby accounting for duplicate occurrences.
        seed : int, default=75434278
            The seed for the MinHash encoder.
        """
        self.dimensions: int = dimensions
        self.radius: int = radius
        self.include_duplicated_shingles: bool = include_duplicated_shingles
        self.encoder: MHFPEncoder = MHFPEncoder(dimensions, seed=seed)
        self.fpgen = AllChem.GetMorganGenerator(radius=radius, includeChirality=includeChirality)
        self.max_dist = max_dist
        self.dist_binning = dist_binning

    def calculate(self, mol: Mol, count: bool = False) -> np.ndarray:
        """
        Calculate the folded MAP4 fingerprint for the molecule.
        
        Parameters
        ----------
        mol : Mol
            RDKit molecule.
        count : bool, default=False
            If True, returns a fixed-length fingerprint that contains the counts
            (0, 1, 2, ...) of each respective bit.
        
        Returns
        -------
        np.ndarray
            The folded fingerprint.
        """
        atom_env_pairs = self._calculate(mol, count)
        if count:
            return self.fold_count(atom_env_pairs)
        else:
            return self._fold(atom_env_pairs)

    def calculate_sparse(self, mol: Mol, count: bool = False) -> np.ndarray:
        """
        Calculate the sparse MAP4neo fingerprint for the molecule.
        
        Parameters
        ----------
        mol : Mol
            RDKit molecule.
        count : bool, default=False
            Whether to return the count variant (i.e. a tuple of (bits, counts)).
        """
        if count:
            atom_env_pairs = self._calculate(mol, count)
            bits_hashed = np.sort(self.encoder.hash([s.encode("utf-8") for s in list(atom_env_pairs.keys())]))
            counts = np.array(list(atom_env_pairs.values()))
            order = np.argsort(bits_hashed)
            return bits_hashed[order], counts[order]
        else:
            atom_env_pairs: Set[str] = self._calculate(mol, count)
            return np.sort(self.encoder.hash(atom_env_pairs))

    def calculate_many(
        self,
        mols: Iterable[Mol],
        count: bool = False,
        number_of_threads: Optional[int] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Calculate fingerprints for many molecules using parallel processing.

        Parameters
        ----------
        mols : Iterable[Mol]
            An iterable of RDKit molecules.
        count : bool, default=False
            Whether to return the count variant (i.e. a tuple of (bits, counts)).
        number_of_threads : Optional[int], default=None
            Number of threads. If None, uses the number of CPUs.
        verbose : bool, default=False
            Whether to show a progress bar.

        Returns
        -------
        np.ndarray
            Array of fingerprints.
        """
        with ThreadPool(number_of_threads) as pool:
            fingerprints: np.ndarray = np.empty(
                (len(mols), self.dimensions), dtype=np.uint8
            )
            for i, fingerprint in enumerate(
                pool.imap(lambda m: self.calculate(m, count=count), mols)
            ):
                fingerprints[i] = fingerprint
            pool.close()
            pool.join()
        return fingerprints

    def calculate_many_sparse(
        self,
        mols: Iterable[Mol],
        number_of_threads: Optional[int] = None,
        count: bool = False,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Calculate sparse fingerprints for many molecules using parallel processing.

        Parameters
        ----------
        mols : Iterable[Mol]
            An iterable of RDKit molecules.
        number_of_threads : Optional[int], default=None
            Number of threads. If None, uses the number of CPUs.
        count : bool, default=False
            Whether to return the count variant (i.e. a tuple of (bits, counts)).
        verbose : bool, default=False
            Whether to show a progress bar.

        Returns
        -------
        np.ndarray
            An array (dtype=object) of sparse fingerprints.
            If count==False, each element is a sorted numpy array of hashed bits.
            If count==True, each element is a tuple (bits, counts), where both are sorted.
        """
        results = []
        # Use ThreadPool to avoid pickling issues
        with ThreadPool(number_of_threads) as pool:
            if verbose:
                from tqdm.auto import tqdm
                iterator = tqdm(
                    pool.imap(lambda m: self.calculate_sparse(m, count=count), mols),
                    total=len(mols),
                    desc="Calculating sparse fingerprints",
                )
            else:
                iterator = pool.imap(lambda m: self.calculate_sparse(m, count=count), mols)
            for fingerprint in iterator:
                results.append(fingerprint)
            pool.close()
            pool.join()
        return results

    def _calculate(self, mol: Mol, count: bool = False) -> Set[str]:
        """
        For a given molecule, return the set (or dict if count=True) of shingles.
        Shingles are built by pairing the Morgan fingerprint bits (for each radius 0...radius)
        from each atom with every other atom, together with the distance between them.
        """
        atoms_bits: Dict[int, List[Optional[str]]] = self._get_atom_bits(mol)
        return self._all_pairs(mol, atoms_bits, count=count)

    def _fold(self, pairs: Set[str]) -> np.ndarray:
        """
        Folds the fingerprint using the MinHash encoder.
        
        Parameters
        ----------
        pairs : Set[str]
            The set of shingle bytes.
        
        Returns
        -------
        np.ndarray
            The folded fingerprint.
        """
        fp_hash = self.encoder.hash(pairs)
        return self.encoder.fold(fp_hash, self.dimensions)

    def fold_count(self, shingle_counts: Dict[str, int]) -> np.ndarray:
        """
        Folds the fingerprint using the MinHash encoder and counts.
        
        Parameters
        ----------
        shingle_counts : Dict[str, int]
            A dictionary mapping each shingle string to its occurrence count.
        
        Returns
        -------
        np.ndarray
            A fixed-length fingerprint vector where each index holds the sum of counts
            for all shingles that fold into that bin.
        """
        # Convert shingle strings to bytes and compute their hash values.
        shingle_bytes = [s.encode("utf-8") for s in shingle_counts.keys()]
        hashed = np.array(self.encoder.hash(shingle_bytes))
        counts = np.array(list(shingle_counts.values()), dtype=np.int32)
        # Initialize fingerprint vector.
        fingerprint = np.zeros(self.dimensions, dtype=np.int32)
        # Fold hashed values into fixed dimensions using modulo arithmetic.
        indices = np.mod(hashed, self.dimensions)
        np.add.at(fingerprint, indices, counts)
        return fingerprint

    def _convert_dist(self, dist):
        if self.dist_binning is None:
            return dist
        dist = np.digitize(dist, self.dist_binning, right=True)
        return dist
    
    def _get_atom_bits(self, mol: Mol) -> Dict[int, List[Optional[str]]]:
        """
        Compute the Morgan fingerprint bits for each atom in the molecule.
        
        For each atom, a list is created holding the bit corresponding to
        each radius from 0 to self.radius (inclusive). This uses RDKit's
        GetMorganFingerprint with a bitInfo dictionary.
        
        Parameters
        ----------
        mol : Mol
            The RDKit molecule.
        
        Returns
        -------
        Dict[int, List[Optional[str]]]
            A dictionary mapping atom indices to lists (length=self.radius+1)
            of Morgan bit strings. If a bit is not found for a given radius,
            the slot remains None.
        """
        atoms_bits: Dict[int, List[Optional[str]]] = {
            atom.GetIdx(): [None] * (self.radius + 1) for atom in mol.GetAtoms()
        }
        bitInfo: Dict[int, List[Tuple[int, int]]] = {}

        ao = AllChem.AdditionalOutput()
        ao.CollectBitInfoMap()
        _ = self.fpgen.GetSparseCountFingerprint(mol, additionalOutput=ao)
        bitInfo = ao.GetBitInfoMap()

        for bit, atom_rad_list in bitInfo.items():
            for atom_idx, rad in atom_rad_list:
                if rad <= self.radius:
                    atoms_bits[atom_idx][rad] = str(bit)
        return atoms_bits

    def _all_pairs(
        self, mol: Mol, atoms_bits: Dict[int, List[Optional[str]]],
        count: bool = False
    ) -> Set[str]:
        """
        Build the set (or dict if count=True) of shingle strings from pairs of atoms.
        
        For every pair of atoms (idx1, idx2), and for every radius r in 0...self.radius,
        the two Morgan bits (one for each atom) are retrieved. The shingle is then defined as:
        
            canonical_bit1 | distance | canonical_bit2
        
        where canonical_bit1 and canonical_bit2 are the two bits sorted lexicographically.
        
        Parameters
        ----------
        mol : Mol
            The molecule.
        atoms_bits : Dict[int, List[Optional[str]]]
            Dictionary of Morgan bits for each atom.
        count : bool, default=False
            Whether to count duplicates.
        
        Returns
        -------
        Set[str] or Dict[str, int]
            The set of shingle bytes if count is False, or a dictionary mapping shingle
            strings to their counts if count is True.
        """
        if count:
            atom_pairs = {}
        else:
            atom_pairs: Set[str] = set()
        distance_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        shingle_dict = defaultdict(int)
        for idx1, idx2 in itertools.combinations(range(num_atoms), 2):
            dist = distance_matrix[idx1][idx2]
            if self.max_dist and dist > self.max_dist:
                continue
            dist = str(int(self._convert_dist(dist)))
            
            for r in range(self.radius + 1):
                bit_a = atoms_bits.get(idx1, [None] * (self.radius + 1))[r]
                bit_b = atoms_bits.get(idx2, [None] * (self.radius + 1))[r]
                if bit_a is None:
                    bit_a = ""
                if bit_b is None:
                    bit_b = ""
                sorted_bits = sorted([bit_a, bit_b])
                shingle: str = f"{sorted_bits[0]}|{dist}|{sorted_bits[1]}"
                if self.include_duplicated_shingles:
                    shingle_dict[shingle] += 1
                    shingle = f"{shingle}|{shingle_dict[shingle]}"
                if count:
                    if shingle in atom_pairs:
                        atom_pairs[shingle] += 1
                    else:
                        atom_pairs[shingle] = 1
                else:
                    atom_pairs.add(shingle.encode("utf-8"))
        return atom_pairs
