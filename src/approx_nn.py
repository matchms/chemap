"""
Module: approx_nn
-----------------

This module defines a function for computing approximate nearest neighbors
from a list of SMILES strings. It uses two different fingerprints:
  - A dense (1024-bit) fingerprint for dimensionality reduction via PCA.
  - A sparse (4096-bit) fingerprint for a refined nearest neighbor search
    based on a Ruzicka similarity.

The general steps are:
    1. Compute fingerprints from SMILES using RDKit's Morgan generator.
    2. Scale and reduce the dense fingerprints with PCA.
    3. Build an approximate NN graph on the PCA vectors.
    4. Refine the neighbor search using a Ruzicka-based candidate search.
"""

import time
from typing import List, Tuple, Any

import numpy as np
import numba
from numba import prange
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from rdkit.Chem import rdFingerprintGenerator
from pynndescent import NNDescent

from fingerprint_computation import compute_fingerprints_from_smiles


def compound_nearest_neighbors(
    smiles: List[str], k_pca: int = 500, k_morgan: int = 100
) -> Tuple[Any, Any]:
    """
    Compute approximate nearest neighbors for a list of SMILES strings.

    This function computes two sets of molecular fingerprints using RDKit’s Morgan
    generator (one dense and one sparse), reduces the dimensionality of the dense
    fingerprints via PCA, builds an approximate nearest neighbor graph using NNDescent,
    and finally refines the search with a Ruzicka-based candidate search.

    Parameters:
        smiles (List[str]): List of SMILES strings representing molecules.
        k_pca (int): Number of neighbors used for the initial approximate NN search
                     (using PCA vectors). Must be larger than k_morgan. Default: 500.
        k_morgan (int): Number of neighbors used for the refined Ruzicka-based search.
                        Default: 100.

    Returns:
        Tuple[Any, Any]: A tuple (order, scores) where:
            - order: An array-like structure of neighbor indices.
            - scores: An array-like structure of similarity scores corresponding to the neighbors.
    """
    # Validate input: ensure that the PCA search uses more neighbors than the refined search.
    assert k_pca > k_morgan, "Expected k_pca to be larger than k_morgan"

    t_start = time.time()
    print(">" * 20, "Compute fingerprints")
    # Compute dense fingerprints (1024 bits) for PCA.
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    fingerprints_morgan3_count_1024 = compute_fingerprints_from_smiles(
        smiles, fpgen, count=True, sparse=False, progress_bar=True,
    )

    # Compute sparse fingerprints for refined neighbor search.
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    fingerprints_morgan3_count_sparse = compute_fingerprints_from_smiles(
        smiles, fpgen, count=True, sparse=True, progress_bar=True,
    )
    print(f"Took: {(time.time() - t_start):.4f} s.")

    order, scores = compute_approx_nearest_neighbors(fingerprints_morgan3_count_1024, fingerprints_morgan3_count_sparse, k_pca, k_morgan)
    return order, scores


def compute_approx_nearest_neighbors(
    fingerprints_coarse, fingerprints_fine, k_pca: int = 500, k_morgan: int = 100
) -> Tuple[Any, Any]:

    t_start = time.time()
    print(">" * 20, "Compute PCA vectors")
    pca = PCA(n_components=100)
    scaler = StandardScaler()
    pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca)])
    pca_vectors = pipe.fit_transform(fingerprints_coarse)
    print(f"Took: {(time.time() - t_start):.4f} s.")

    t_start = time.time()
    print(">" * 20, f"Build NN-graph ({k_pca} neighbors)")
    ann_graph = NNDescent(pca_vectors, metric="cosine", n_neighbors=k_pca)
    print(f"Took: {(time.time() - t_start):.4f} s.")

    t_start = time.time()
    print(">" * 20, f"Build Ruzicka based NN-graph ({k_morgan} neighbors)")
    order, scores = ruzicka_candidate_search(
        fingerprints_fine, fingerprints_fine,
        ann_graph.neighbor_graph[0],
        k_morgan,
    )
    print(f"Took: {(time.time() - t_start):.4f} s.")

    return order, scores


@numba.jit(nopython=True)
def ruzicka_candidate_search(
        references: list, queries: list,
        knn_indices_approx: list,
        k
        ) -> np.ndarray:
    """Search all candidates...

    Parameters
    ----------
    references:
        List of sparse fingerprints (tuple of two arrays: keys and counts).
    queries
        List of sparse fingerprints (tuple of two arrays: keys and counts).
    """
    size = len(queries)
    candidate_idx = np.zeros((size, k))#, dtyoe=np.int32)
    candidate_scores = np.zeros((size, k), dtype=np.float64)
    for i, knn_indices in enumerate(knn_indices_approx):
        
        order, scores = ruzicka_similarity_query_search(
            [references[i] for i in knn_indices],
            queries[i], k
        )
        candidate_idx[i, :] = knn_indices[order]
        candidate_scores[i, :] = scores
    return candidate_idx, candidate_scores


import numba
from numba import prange
from metrics import ruzicka_similarity_sparse_numba


@numba.jit(nopython=True, fastmath=True, parallel=True)
def ruzicka_similarity_query_search(
    candidates: list, query, k) -> np.ndarray:
    """Returns matrix of Ruzicka similarity between all-vs-all vectors of references and queries.

    Parameters
    ----------
    references:
        List of sparse fingerprints (tuple of two arrays: keys and counts).
    queries
        Sparse fingerprint (tuple of two arrays: keys and counts).

    Returns
    -------
    scores:
        Matrix of all-vs-all similarity scores. scores[i, j] will contain the score
        between the vectors references[i, :] and queries[j, :].
    """
    size1 = len(candidates)
    distances = np.zeros(size1)
    for i in prange(size1):
        distances[i] = 1 - ruzicka_similarity_sparse_numba(
            candidates[i][0], candidates[i][1],
            query[0], query[1])
    order = np.argsort(distances)[:k]
    return order, distances[order]
