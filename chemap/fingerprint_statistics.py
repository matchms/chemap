import numba
import numpy as np
from numba import typed, types


def unfolded_fingerprint_bit_statistics(fingerprints):
    """
    Count the occurrences of bits across all sparse (indices-only) fingerprints
    using two dictionaries (one for counts, one for first index) for fast lookup.
    
    Parameters
    ----------
    fingerprints : iterable of 1D arrays of int64 bit indices.
    
    Returns
    -------
    unique_keys : int64 array
    counts : int32 array
    first_instances : int32 array
    """
    tl = typed.List.empty_list(types.int64[:])
    for fp_bits in fingerprints:
        tl.append(np.asarray(fp_bits, dtype=np.int64))
    return _unfolded_fingerprint_bit_statistics(tl)


@numba.njit
def _unfolded_fingerprint_bit_statistics(
        fingerprints
        ):
    counts = typed.Dict.empty(key_type=types.int64, value_type=types.int32)
    first_instance = typed.Dict.empty(key_type=types.int64, value_type=types.int32)
    
    for i, fp_bits in enumerate(fingerprints):
        for bit in fp_bits:
            if bit in counts:
                counts[bit] += 1
            else:
                counts[bit] = 1
                first_instance[bit] = i
    
    n = len(counts)
    unique_keys = np.empty(n, dtype=np.int64)
    count_arr   = np.empty(n, dtype=np.int32)
    first_arr   = np.empty(n, dtype=np.int32)
    
    idx = 0
    for key in counts:
        unique_keys[idx] = key
        count_arr[idx] = counts[key]
        first_arr[idx] = first_instance[key]
        idx += 1
    
    order = np.argsort(unique_keys)
    return unique_keys[order], count_arr[order], first_arr[order]


def unfolded_count_fingerprint_bit_statistics(
        fingerprints: list[tuple[np.ndarray, np.ndarray]]
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count the occurrences of bits across all unfolded count fingerprints.

    Parameters
    ----------
    fingerprints:
        List of sparse fingerprints (tuple of two arrays: keys and counts).
    """
    return unfolded_fingerprint_bit_statistics(
        [fp[0] for fp in fingerprints]  # only keys
    )
