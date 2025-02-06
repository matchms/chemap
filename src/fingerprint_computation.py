import numpy as np
import numba
from rdkit import Chem
from tqdm import tqdm


class FingerprintGenerator:
    def __init__(self, fpgen):
        self.fpgen = fpgen

    def fingerprint_from_smiles(self, smiles, count=False):
        """Compute fingerprint from SMILES using the generator attribute.
        
        Parameters:
        smiles (str): The SMILES string of the molecule.
        count (bool): If True, returns the count fingerprint, else the regular fingerprint.

        Returns:
        np.array: The fingerprint as a NumPy array, or None if there's an error.
        """
        mol = get_mol_from_smiles(smiles)
        try:
            if count:
                return self.fpgen.GetCountFingerprintAsNumPy(mol)
            return self.fpgen.GetFingerprintAsNumPy(mol)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None


class SparseFingerprintGenerator:
    def __init__(self, fpgen):
        self.fpgen = fpgen

    def fingerprint_from_smiles(
            self, smiles: str,
            count: bool = False,
            bit_scaling: str = None,
            bit_weighing: dict = None
            ):
        """Compute sparse fingerprint from SMILES using the generator attribute.
        
        Parameters:
        smiles: 
            The SMILES string of the molecule.
        count: 
            If True, returns the count fingerprint, else the regular fingerprint.
        bit_scaling:
            Optional. Default is None in which case the counts will not be scaled.
            Can be set to 'log' for logarithmic scaling of counts (`log-count = log(1 + count)`).
        bit_weighing:
            Optional. When a dictionary of shape {bit: value} is given, the respective bits will be multiplied
            by the respective given value. Fingerprint bits not in this dictionary will be multiplied
            by one, so it is generally advisable to use normalized bit weights.

        Returns:
        dict: A dictionary where keys are bit indices and values are counts (for count fingerprints)
              or a list of indices for regular sparse fingerprints.
        """
        if (bit_weighing is not None) and not count:
            raise NotImplementedError("Weighing is currently only implemented for count vectors.")
        
        mol = get_mol_from_smiles(smiles)

        # Now generate the fingerprint.
        try:
            # If count=True, return a prepared sparse vector.
            if count:
                fp_dict = self.fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements()
                return prepare_sparse_vector(fp_dict, bit_scaling, bit_weighing)
            # Otherwise, return the sorted indices as a numpy array.
            else:
                fp_elements = self.fpgen.GetSparseCountFingerprint(mol).GetNonzeroElements()
                return np.array(sorted(fp_elements.keys()), dtype=np.int64)
        except Exception as e:
            print(f"Error generating fingerprint for SMILES {smiles}: {e}")
            return None


def get_mol_from_smiles(smiles):
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
            Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE\
                                |Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION\
                                |Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                                catchErrors=True)
            if mol is None:
                raise ValueError("MolFromSmiles returned None even with sanitize=False.")
        except Exception as e2:
            print(f"Error processing SMILES {smiles} with sanitize=False: {e2}")
            return None
    return mol


def prepare_sparse_vector(
        sparse_fp_dict: dict,
        bit_scaling: str = None,
        bit_weighing: dict = None
        ):
    """Convert dictionaries to sorted arrays.
    """
    def scaling(value, bit_scaling):
        if bit_scaling is None:
            return value
        if bit_scaling.lower() == "log":
            return np.log(1 + value)

    keys = np.array(sorted(sparse_fp_dict.keys()), dtype=np.int64)
    if (bit_weighing is None) and (bit_scaling is None):
        values = np.array([sparse_fp_dict[k] for k in keys], dtype=np.int32)
    elif bit_weighing is None:
        values = np.array([scaling(sparse_fp_dict[k], bit_scaling) for k in keys], dtype=np.float32)
    else:
        values = np.array([scaling(sparse_fp_dict[k], bit_scaling) * bit_weighing.get(k, 1) for k in keys], dtype=np.float32)
    return keys, values


def compute_fingerprints(
        compounds,
        fpgen,
        count=True,
        sparse=True,
        bit_scaling=None,
        bit_weighing=None,
        ):
    if sparse:
        fp_generator = SparseFingerprintGenerator(fpgen)
    else:
        fp_generator = FingerprintGenerator(fpgen)
    
    fingerprints = []
    for inchikey, row in tqdm(compounds.iterrows(), total=len(compounds)):
        if sparse:
            fp = fp_generator.fingerprint_from_smiles(row.smiles, count, bit_scaling, bit_weighing)
        else:
            fp = fp_generator.fingerprint_from_smiles(row.smiles, count)
        if fp is None:
            print(f"Missing fingerprint for {inchikey}: {row.smiles}")
        else:
            fingerprints.append(fp)
    return fingerprints


def fingerprint_from_smiles_wrapper(smiles, fpgen, count=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if count:
            return fpgen.GetCountFingerprintAsNumPy(mol)
        return fpgen.GetFingerprintAsNumPy(mol)
    except:
        return None


@numba.njit
def count_fingerprint_keys(fingerprints, max_keys: int = 10**7):
    """
    Count the occurrences of keys across all sparse fingerprints.

    Parameters:
    fingerprints (list of tuples): 
        Each tuple contains two numpy arrays: (keys, values) for a fingerprint.
    max_keys:
        Maximum number of unique bits that can be counted.

    Returns:
        A tuple of 3 Numpy arrays (unique_keys, counts, first_instances).
    """

    unique_keys = np.zeros(max_keys, dtype=np.int64)
    counts = np.zeros(max_keys, dtype=np.int32)
    first_instances = np.zeros(max_keys, dtype=np.int16)  # Store first fingerprint where the respective bit occurred (for later analysis)
    num_unique = 0
    reached_max_keys = False

    for idx, (keys, _) in enumerate(fingerprints):
        for key in keys:
            # Check if the key is already in unique_keys
            found = False
            for i in range(num_unique):
                if unique_keys[i] == key:
                    counts[i] += 1
                    found = True
                    break
            # If the key is new, add it
            if not found:
                if (num_unique >= max_keys):
                    if not reached_max_keys:
                        print(f"Maximum number of keys was reached at fingerprint number {idx}.")
                        print("Consider raising the max_keys argument.")
                        reached_max_keys = True
                    continue
                unique_keys[num_unique] = key
                counts[num_unique] = 1
                first_instances[num_unique] = idx
                num_unique += 1

    # Trim arrays to the actual size and sort by key
    bit_order = np.argsort(unique_keys[:num_unique])
    return unique_keys[bit_order], counts[bit_order], first_instances[bit_order]


### ------------------------
### Bit Scaling and Weighing
### ------------------------

def compute_idf(vector_array):
    """Compute inverse document frequency (IDF).duplicates
    """
    N = vector_array.shape[0]
    return np.log(N / (vector_array > 0).sum(axis=0))
