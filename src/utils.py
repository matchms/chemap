import os
import hashlib
import numpy as np
import urllib.request
from pathlib import Path
from scipy.stats import rankdata


def row_to_hash(row: np.ndarray) -> str:
    """
    Converts a NumPy array row to a SHA-256 hash string.

    Parameters
    ----------
    row : np.ndarray
        A 1D NumPy array representing a fingerprint or count vector.
    """
    row_bytes = row.tobytes()
    return hashlib.sha256(row_bytes).hexdigest()


def find_duplicates_with_hashing(arr: list) -> list:
    """
    Finds duplicate entries in an array-like structure using hashing.

    Supports arrays of NumPy arrays or tuples representing sparse fingerprints
    (e.g., (bits, counts)). Converts each row to a hash and groups identical hashes.

    Parameters
    ----------
    arr : list
        A list of 1D NumPy arrays or tuples of NumPy arrays (e.g., sparse count fingerprints).

    Returns
    -------
    list
        A list of lists. Each inner list contains indices of duplicate rows.
    """
    hash_dict = {}
    duplicates = []

    for idx, row in enumerate(arr):
        if isinstance(row, tuple):  # Sparse count fingerprint: (bits, counts)
            row_hash = row_to_hash(row[0]) + row_to_hash(row[1])
        else:
            row_hash = row_to_hash(row)

        hash_dict.setdefault(row_hash, []).append(idx)

    for indices in hash_dict.values():
        if len(indices) > 1:
            duplicates.append(indices)

    return duplicates


def percentile_scores(similarities: np.ndarray) -> np.ndarray:
    """
    Converts the upper-triangular part of a similarity matrix into percentile scores (0–100).

    This transformation helps standardize similarity values by ranking them
    and then scaling ranks to a percentile scale. The matrix is assumed to be symmetric.

    Parameters
    ----------
    similarities : np.ndarray
        2D symmetric similarity matrix of shape (N, N).

    Returns
    -------
    np.ndarray
        A new symmetric matrix of the same shape where upper-triangle entries
        are replaced by percentile scores. The diagonal remains unchanged.
    """
    assert similarities.shape[0] == similarities.shape[1], "Expected similarities to be symmetric matrix"

    iu1 = np.triu_indices(similarities.shape[0], k=1)
    arr = similarities[iu1]

    ranks = rankdata(arr, method="average")
    percentiles = (ranks - 1) / (len(ranks) - 1) * 100 if len(ranks) > 1 else np.zeros_like(ranks)

    percentile_matrix = np.zeros_like(similarities, dtype=float)
    percentile_matrix[iu1] = percentiles

    return percentile_matrix + percentile_matrix.T


def remove_diagonal(matrix: np.ndarray) -> np.ndarray:
    """
    Removes the diagonal entries from a square matrix.

    Common use case: Remove self-comparisons in a similarity matrix.

    Parameters
    ----------
    matrix : np.ndarray
        A square matrix of shape (N, N).
    """
    nr_of_rows, nr_of_cols = matrix.shape
    if nr_of_rows != nr_of_cols:
        raise ValueError("Expected square matrix for self-comparison removal.")

    diagonal_mask = np.eye(nr_of_rows, dtype=bool)
    return matrix[~diagonal_mask].reshape(nr_of_rows, nr_of_cols - 1)


def download_dataset(url: str, output_dir: str = None) -> None:
    """
    Downloads a file from a URL to a local directory.

    Parameters
    ----------
    url : str
        The URL to download the file from.
    output_dir : str, optional
        Directory to save the downloaded file. Defaults to the parent of the current working directory.
    """
    if output_dir is None:
        output_dir = Path(os.getcwd()).parents[0]

    os.makedirs(output_dir, exist_ok=True)

    fn = Path(url).name
    destination = os.path.join(output_dir, fn)
    urllib.request.urlretrieve(url, destination)
    print(f"File {fn} was downloaded successfully to {output_dir}.")
