import os
import numpy as np
import urllib.request
from pathlib import Path
from scipy.stats import rankdata


def percentile_scores(similarities: np.ndarray) -> np.ndarray:
    """
    Computes percentile ranks (0..100) for the unique upper-triangular (k=1) entries
    of a symmetrical similarity matrix using scipy's rankdata with average ranks.
    The results are then mirrored to the lower triangle so that the returned matrix
    is again symmetric.
    
    The diagonal is left as-is (default: 0.0), but you can adjust it if needed.

    Parameters
    ----------
    similarities : np.ndarray
        2D symmetric similarity matrix, shape (N, N).

    Returns
    -------
    np.ndarray
        A new matrix of the same shape, where the upper-triangular entries (and their
        mirrored lower-triangular counterparts) have been replaced by percentile ranks.
        The diagonal is untouched (default = 0).
    """
    # Step 1: Extract the upper-triangular entries (excluding diagonal)
    iu1 = np.triu_indices(similarities.shape[0], k=1)
    arr = similarities[iu1]

    # Step 2: Rank them using 'average' method (duplicates get same rank)
    ranks = rankdata(arr, method="average")  # Ranks in [1..len(arr)]
    
    # Step 3: Convert ranks to percentile range [0..100]
    percentiles = (ranks - 1) / (len(ranks) - 1) * 100 if len(ranks) > 1 else np.zeros_like(ranks)

    # Step 4: Create a new matrix for output, same shape and dtype
    percentile_matrix = np.zeros_like(similarities, dtype=float)
    
    # Step 5: Place the percentile scores back into the upper triangle
    percentile_matrix[iu1] = percentiles

    # Step 6: Mirror to the lower triangle
    percentile_matrix = percentile_matrix + percentile_matrix.T

    return percentile_matrix


def remove_diagonal(matrix):
    """Removes the diagonal from a matrix.

    Meant for removing matches of spectra against itself.
    """
    # Get the number of rows and columns
    nr_of_rows, nr_of_cols = matrix.shape
    if nr_of_rows != nr_of_cols:
        raise ValueError("Expected predictions against itself")

    # Create a mask for the diagonal elements
    diagonal_mask = np.eye(nr_of_rows, dtype=bool)

    # Use the mask to remove the diagonal elements
    matrix_without_diagonal = matrix[~diagonal_mask].reshape(nr_of_rows, nr_of_cols - 1)
    return matrix_without_diagonal

def download_dataset(url: str, output_dir: str = None):
    if output_dir is None:
        output_dir = Path(os.getcwd()).parents[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fn = Path(url).name

    urllib.request.urlretrieve(url, os.path.join(output_dir, fn))
    print(f"File {fn} was downloaded successfully.")