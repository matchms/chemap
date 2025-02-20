import numpy as np
import numba
from numba import types, typed
from rdkit import Chem
from tqdm import tqdm


class FingerprintGenerator:
    def __init__(self, fpgen):
        self.fpgen = fpgen

    def fingerprint_from_smiles(self, smiles, count=False, bit_weighing=None):
        """Compute fingerprint from SMILES using the generator attribute.
        
        Parameters:
        smiles (str): The SMILES string of the molecule.
        count (bool): If True, returns the count fingerprint, else the regular fingerprint.

        Returns:
        np.array: The fingerprint as a NumPy array, or None if there's an error.
        """
        if (bit_weighing is not None) and not count:
            raise NotImplementedError("Weighing is currently only implemented for count vectors.")

        mol = get_mol_from_smiles(smiles)
        try:
            if count:
                return self.fpgen.GetCountFingerprintAsNumPy(mol)
            fp = self.fpgen.GetFingerprintAsNumPy(mol)
            if bit_weighing is None:
                return fp
            elif bit_scaling.lower() == "log":
                return np.log(1 + fp)
            else:
                raise ValueError("Expected bit_scaling to be 'log' or 'None'.")
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
    elif bit_scaling is None:
        values = np.array([sparse_fp_dict[k] * bit_weighing.get(k, 1) for k in keys], dtype=np.float32)
    else:
        values = np.array([scaling(sparse_fp_dict[k], bit_scaling) * bit_weighing.get(k, 1) for k in keys], dtype=np.float32)
    return keys, values


def compute_fingerprints_from_smiles(
        smiles_lst,
        fpgen,
        count=True,
        sparse=True,
        bit_scaling=None,
        bit_weighing=None,
        progress_bar=False,
        ):
    if sparse:
        fp_generator = SparseFingerprintGenerator(fpgen)
    else:
        fp_generator = FingerprintGenerator(fpgen)
    
    fingerprints = []
    for i, smiles in tqdm(enumerate(smiles_lst), total=len(smiles_lst), disable=(not progress_bar)):
        if sparse:
            fp = fp_generator.fingerprint_from_smiles(smiles, count, bit_scaling, bit_weighing)
        else:
            fp = fp_generator.fingerprint_from_smiles(smiles, count, bit_weighing)
        if fp is None:
            print(f"Missing fingerprint for element {i}: {smiles}")
        else:
            fingerprints.append(fp)
    if sparse:
        return fingerprints
    return np.stack(fingerprints)


@numba.njit
def count_fingerprint_keys(fingerprints):
    """
    Count the occurrences of keys across all sparse fingerprints using two dictionaries
    (one for counts, one for the first fingerprint index) for fast lookup.
    
    Parameters:
        fingerprints (list of bits).
    
    Returns:
        A tuple of 3 Numpy arrays (unique_keys, counts, first_instances) where:
            - unique_keys: Sorted unique bit keys.
            - counts: The number of occurrences of each key.
            - first_instances: The first fingerprint index where each key occurred.
    """
    # Create dictionaries with key type int64 and value type int32.
    counts = typed.Dict.empty(key_type=types.int64, value_type=types.int32)
    first_instance = typed.Dict.empty(key_type=types.int64, value_type=types.int32)
    
    # Loop over each fingerprint.
    for i, fp_bits in enumerate(fingerprints):
        for bit in fp_bits:
            if bit in counts:
                counts[bit] += 1
            else:
                counts[bit] = 1
                first_instance[bit] = i
    
    # Allocate arrays to hold the results.
    n = len(counts)
    unique_keys = np.empty(n, dtype=np.int64)
    count_arr   = np.empty(n, dtype=np.int32)
    first_arr   = np.empty(n, dtype=np.int32)
    
    # Transfer dictionary contents to arrays.
    idx = 0
    for key in counts:
        unique_keys[idx] = key
        count_arr[idx] = counts[key]
        first_arr[idx] = first_instance[key]
        idx += 1
    
    # Sort by key.
    order = np.argsort(unique_keys)
    return unique_keys[order], count_arr[order], first_arr[order]


### ------------------------
### Bit Scaling and Weighing
### ------------------------

def compute_idf(vector_array):
    """Compute inverse document frequency (IDF).duplicates
    """
    N = vector_array.shape[0]
    return np.log(N / (vector_array > 0).sum(axis=0))
