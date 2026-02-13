from typing import List
import numpy as np


# ------------------------------------------------------
# Duplicate fingerprint statistics
# ------------------------------------------------------

def compute_duplicate_max_mass_differences(
        duplicates, masses: np.ndarray):
    """
    Compute all maximum mass differences between duplicates.
    """
    max_diffs: List[float] = []
    for group in duplicates:
        idx = np.asarray(group, dtype=int)
        group_masses = masses[idx]
        mass_diffs = compute_compound_max_mass_differences(group_masses)
        max_diffs.extend(mass_diffs)
    return np.asarray(max_diffs, dtype=float)



def compute_compound_max_mass_differences(masses):
    all_max_diffs = []
    min_mass = masses.min()
    max_mass = masses.max()
    for mass in masses:
        max_diff = max(mass - min_mass, max_mass - mass)
        all_max_diffs.append(max_diff)

    return np.array(all_max_diffs)
