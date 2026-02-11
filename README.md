
<img src="./materials/chemap_logo_green_pink.png" width="400">

![GitHub License](https://img.shields.io/github/license/matchms/chemap?color=#00B050)
[![PyPI](https://img.shields.io/pypi/v/chemap?color=#00B050)](https://pypi.org/project/chemap/)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/matchms/chemap/CI_build_and_matrix_tests.yml?color=#00B050)
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)
    
# chemap - Mapping chemical space
Library for computing molecular fingerprint based similarities as well as dimensionality reduction based chemical space visualizations.

## Installation
`chemap` can be installed using pip.
```bash
pip install chemap
```
Or, to include UMAP computation abilities on either CPU or GPU chose one of the following option:
- CPU version: ```pip install "chemap[cpu]"```
- GPU version (CUDA 12): ```pip install "chemap[gpu-cu12]"```
- GPU version (CUDA 13): ```pip install "chemap[gpu-cu13]"```

## Fingerprint computations
Fingerprints can be computed using generators from `RDKit` or `scikit-fingerprints`. 
This includes popular fingerprint types such as:

### Path-based and circular fingerprints
- RDKit fingerprints
- Morgan fingerprints

### Predefined substructure fingerprints
- MACCS fingerprints
- PubChem fingerprints
- Klekota-Roth fingerprints

### Topological distance based fingerprints
- Atom pair fingerprints
- MAP4 fingerprints



Here a code example:

```python
import numpy as np
import scipy.sparse as sp
from rdkit.Chem import rdFingerprintGenerator
from skfp.fingerprints import MAPFingerprint, AtomPairFingerprint

from chemap import compute_fingerprints, DatasetLoader, FingerprintConfig


ds_loader = DatasetLoader()
smiles = ds_loader.load("tests/data/smiles.csv")

# ----------------------------
# RDKit: Morgan (folded, dense)
# ----------------------------
morgan = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
X_morgan = compute_fingerprints(
    smiles,
    morgan,
    config=FingerprintConfig(
        count=False,
        folded=True,
        return_csr=False,   # dense numpy
        invalid_policy="raise",
    ),
)
print("RDKit Morgan:", X_morgan.shape, X_morgan.dtype)

# -----------------------------------
# RDKit: RDKitFP (folded, CSR sparse)
# -----------------------------------
rdkitfp = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=4096)
X_rdkitfp_csr = compute_fingerprints(
    smiles,
    rdkitfp,
    config=FingerprintConfig(
        count=False,
        folded=True,
        return_csr=True,    # SciPy CSR
        invalid_policy="raise",
    ),
)
assert sp.issparse(X_rdkitfp_csr)
print("RDKit RDKitFP (CSR):", X_rdkitfp_csr.shape, X_rdkitfp_csr.dtype, "nnz=", X_rdkitfp_csr.nnz)

# --------------------------------------------------
# scikit-fingerprints: MAPFingerprint (folded, dense)
# --------------------------------------------------
# MAPFingerprint is a MinHash-like fingerprint (different from MAP4 lib).
map_fp = MAPFingerprint(fp_size=4096, count=False, sparse=False)
X_map = compute_fingerprints(
    smiles,
    map_fp,
    config=FingerprintConfig(
        count=False,
        folded=True,
        return_csr=False,
        invalid_policy="raise",
    ),
)
print("skfp MAPFingerprint:", X_map.shape, X_map.dtype)

# ----------------------------------------------------
# scikit-fingerprints: AtomPairFingerprint (folded, CSR)
# ----------------------------------------------------
atom_pair = AtomPairFingerprint(fp_size=4096, count=False, sparse=False, use_3D=False)
X_ap_csr = compute_fingerprints(
    smiles,
    atom_pair,
    config=FingerprintConfig(
        count=False,
        folded=True,
        return_csr=True,
        invalid_policy="raise",
    ),
)
assert sp.issparse(X_ap_csr)
print("skfp AtomPair (CSR):", X_ap_csr.shape, X_ap_csr.dtype, "nnz=", X_ap_csr.nnz)

# (Optional) convert CSR -> dense if you need a NumPy array downstream:
X_ap = X_ap_csr.toarray().astype(np.float32, copy=False)


```
