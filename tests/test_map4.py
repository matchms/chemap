import numpy as np
import pytest
from rdkit import Chem
from chemap.additional_fingerprints.map4 import MAP4FPGen, _MAP4Shingler


def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit failed to parse SMILES: {smiles}"
    return mol


@pytest.fixture(scope="module")
def mol_single_atom():
    # 1 atom -> no (i<j) pairs -> no shingles
    return mol_from_smiles("C")


@pytest.fixture(scope="module")
def mol_ethanol():
    return mol_from_smiles("CCO")


@pytest.fixture(scope="module")
def mol_benzene():
    return mol_from_smiles("c1ccccc1")


# -----------------------------
# Constructor / validation
# -----------------------------

def test_ctor_validates_dimensions_radius_unfolded_bits():
    with pytest.raises(ValueError, match="dimensions must be > 0"):
        MAP4FPGen(dimensions=0)
    with pytest.raises(ValueError, match="radius must be > 0"):
        MAP4FPGen(radius=0)
    with pytest.raises(ValueError, match="unfolded_bits must be 32 or 64"):
        MAP4FPGen(unfolded_bits=16)


# -----------------------------
# Size inference shims
# -----------------------------

def test_size_inference_shims(mol_ethanol):
    gen = MAP4FPGen(dimensions=256, radius=2)
    bitfp = gen.GetFingerprint(mol_ethanol)
    cntfp = gen.GetCountFingerprint(mol_ethanol)
    assert bitfp.GetNumBits() == 256
    assert cntfp.GetLength() == 256


# -----------------------------
# Folded binary
# -----------------------------

def test_folded_binary_dtype_shape_and_values(mol_ethanol):
    gen = MAP4FPGen(dimensions=512, radius=2)
    fp = gen.GetFingerprintAsNumPy(mol_ethanol)

    assert fp.shape == (512,)
    assert fp.dtype == np.uint8
    assert fp.min() >= 0
    assert fp.max() <= 1


def test_folded_binary_is_all_zeros_if_folded_disabled(mol_ethanol):
    gen = MAP4FPGen(dimensions=128, radius=2, folded=False)
    fp = gen.GetFingerprintAsNumPy(mol_ethanol)
    assert fp.shape == (128,)
    assert fp.dtype == np.uint8
    assert int(fp.sum()) == 0


def test_folded_binary_is_all_zeros_for_single_atom(mol_single_atom):
    gen = MAP4FPGen(dimensions=128, radius=2)
    fp = gen.GetFingerprintAsNumPy(mol_single_atom)
    assert int(fp.sum()) == 0


def test_folded_binary_deterministic(mol_benzene):
    gen1 = MAP4FPGen(dimensions=512, radius=2, seed=123)
    gen2 = MAP4FPGen(dimensions=512, radius=2, seed=999)  # seed should NOT affect hash/fold path
    fp1a = gen1.GetFingerprintAsNumPy(mol_benzene)
    fp1b = gen1.GetFingerprintAsNumPy(mol_benzene)
    fp2 = gen2.GetFingerprintAsNumPy(mol_benzene)

    # Same generator repeated => identical
    assert np.array_equal(fp1a, fp1b)
    # Seed currently not used in hash/fold path (by design of this implementation)
    assert np.array_equal(fp1a, fp2)


def test_folded_binary_matches_reference_hash_fold(mol_ethanol):
    """
    The docstring says folded binary matches:
        folded = fold(hash(set(shingles)), D)

    Here we recompute that definition directly from the shingler and compare.
    """
    gen = MAP4FPGen(dimensions=256, radius=2)
    mol = mol_ethanol

    fp = gen.GetFingerprintAsNumPy(mol)

    shingles = gen._shingler.shingles_unique(mol)  # set[bytes]
    if not shingles:
        assert int(fp.sum()) == 0
        return

    hashed = np.fromiter(
        (int.from_bytes(__import__("hashlib").sha1(sh).digest()[:4], "little", signed=False) for sh in shingles),
        dtype=np.uint32,
        count=len(shingles),
    )
    ref = np.zeros(gen.dimensions, dtype=np.uint8)
    ref[(hashed % np.uint32(gen.dimensions)).astype(np.int64, copy=False)] = 1

    assert np.array_equal(fp, ref)


# -----------------------------
# Folded count
# -----------------------------

def test_folded_count_dtype_shape(mol_ethanol):
    gen = MAP4FPGen(dimensions=512, radius=2)
    fp = gen.GetCountFingerprintAsNumPy(mol_ethanol)

    assert fp.shape == (512,)
    assert fp.dtype == np.float32
    assert np.all(fp >= 0.0)


def test_folded_count_is_all_zeros_if_folded_disabled(mol_ethanol):
    gen = MAP4FPGen(dimensions=128, radius=2, folded=False)
    fp = gen.GetCountFingerprintAsNumPy(mol_ethanol)
    assert fp.dtype == np.float32
    assert float(fp.sum()) == 0.0


def test_folded_count_is_all_zeros_for_single_atom(mol_single_atom):
    gen = MAP4FPGen(dimensions=128, radius=2)
    fp = gen.GetCountFingerprintAsNumPy(mol_single_atom)
    assert float(fp.sum()) == 0.0


def test_folded_count_sums_to_total_true_shingle_multiplicity(mol_benzene):
    """
    Folded counts are feature-hashed counts:
        fp[hash32(sh) % D] += count(sh)
    so the sum of fp equals sum of true counts.
    """
    gen = MAP4FPGen(dimensions=256, radius=2)
    counts = gen._shingler.shingles_with_counts_true(mol_benzene)
    fp = gen.GetCountFingerprintAsNumPy(mol_benzene)

    assert fp.dtype == np.float32
    assert float(fp.sum()) == float(sum(counts.values()))


def test_folded_count_matches_reference_count_hashing(mol_ethanol):
    """
    Recompute count-folding directly and compare elementwise.
    """
    gen = MAP4FPGen(dimensions=128, radius=2)
    mol = mol_ethanol

    counts = gen._shingler.shingles_with_counts_true(mol)
    fp = gen.GetCountFingerprintAsNumPy(mol)

    ref = np.zeros(gen.dimensions, dtype=np.float32)
    for sh, c in counts.items():
        h32 = int.from_bytes(__import__("hashlib").sha1(sh).digest()[:4], "little", signed=False)
        ref[h32 % gen.dimensions] += float(c)

    assert np.array_equal(fp, ref)


# -----------------------------
# Unfolded sparse counts
# -----------------------------

def test_sparse_counts_empty_for_single_atom(mol_single_atom):
    gen = MAP4FPGen(dimensions=128, radius=2)
    s = gen.GetSparseCountFingerprint(mol_single_atom).GetNonzeroElements()
    assert s == {}


def test_sparse_counts_are_int_keys_and_int_values(mol_ethanol):
    gen = MAP4FPGen(dimensions=128, radius=2)
    s = gen.GetSparseCountFingerprint(mol_ethanol).GetNonzeroElements()
    assert isinstance(s, dict)
    for k, v in s.items():
        assert isinstance(k, int)
        assert isinstance(v, int)
        assert v > 0


@pytest.mark.parametrize("unfolded_bits", [32, 64])
def test_sparse_counts_sha1_id_range_matches_bits(mol_ethanol, unfolded_bits):
    gen = MAP4FPGen(dimensions=128, radius=2, minhash_for_unfolded=False, unfolded_bits=unfolded_bits)
    s = gen.GetSparseCountFingerprint(mol_ethanol).GetNonzeroElements()
    if unfolded_bits == 32:
        assert all(0 <= k < 2**32 for k in s.keys())
    else:
        assert all(0 <= k < 2**64 for k in s.keys())


def test_sparse_counts_minhash_domain_ids_are_32bit(mol_ethanol):
    gen = MAP4FPGen(dimensions=128, radius=2, minhash_for_unfolded=True)
    s = gen.GetSparseCountFingerprint(mol_ethanol).GetNonzeroElements()
    assert all(0 <= k < 2**32 for k in s.keys())


def test_sparse_counts_sum_matches_true_total_counts(mol_benzene):
    """
    Unfolded sparse counts store true multiplicities (possibly with collisions if using 32-bit IDs,
    but sum should still match total multiplicity because we add counts).
    """
    gen = MAP4FPGen(dimensions=128, radius=2, minhash_for_unfolded=True)
    counts_true = gen._shingler.shingles_with_counts_true(mol_benzene)
    s = gen.GetSparseCountFingerprint(mol_benzene).GetNonzeroElements()
    assert sum(s.values()) == sum(counts_true.values())


# -----------------------------
# Shingler-specific behavior
# -----------------------------

def test_counts_true_ignores_suffix_trick_even_if_enabled(mol_benzene):
    """
    The shingler promises true multiplicities WITHOUT suffix trick regardless of
    include_duplicated_shingles.
    """
    sh1 = _MAP4Shingler(radius=2, include_duplicated_shingles=False)
    sh2 = _MAP4Shingler(radius=2, include_duplicated_shingles=True)

    c1 = sh1.shingles_with_counts_true(mol_benzene)
    c2 = sh2.shingles_with_counts_true(mol_benzene)

    assert c1 == c2


def test_max_dist_filters_shingles(mol_benzene):
    """
    With max_dist=0, all atom pairs have dist>0 (except i==j which isn't used), so no shingles.
    """
    sh = _MAP4Shingler(radius=2, max_dist=0)
    u = sh.shingles_unique(mol_benzene)
    c = sh.shingles_with_counts_true(mol_benzene)
    assert u == set()
    assert c == {}


def test_dist_binning_changes_distance_encoding():
    """
    Deterministic test: choose a molecule with topological distances > 2
    and use a binning that collapses distances so encodings must change.
    """
    # Linear chain: longest topological distance = 4 (between terminal atoms)
    mol = Chem.MolFromSmiles("CCCCC")
    assert mol is not None

    sh_plain = _MAP4Shingler(radius=2, dist_binning=None)

    # Collapse distances into 2 bins:
    # right=True => digitize returns:
    #   dist <= 1 -> bin 1
    #   dist  > 1 -> bin 2
    # so distances 2,3,4 all become "2" (string), which differs from plain "2","3","4".
    sh_binned = _MAP4Shingler(radius=2, dist_binning=np.array([1], dtype=np.int32))

    u_plain = sh_plain.shingles_unique(mol)
    u_binned = sh_binned.shingles_unique(mol)

    assert u_plain != u_binned

    # Extra sanity: binned distances should contain fewer distinct distance tokens.
    def extract_dists(shs):
        dists = set()
        for b in shs:
            s = b.decode("utf-8")
            parts = s.split("|")
            assert len(parts) >= 3
            dists.add(parts[1])
        return dists

    assert len(extract_dists(u_binned)) <= len(extract_dists(u_plain))
