from pathlib import Path
import numpy as np
import pytest
from chemap.benchmarking.fingerprint_duplicates import (
    DuplicatesNPZ,
    PrecomputedDuplicates,
    decode_duplicates,
    encode_duplicates,
    load_duplicates_npz,
    load_precomputed_duplicates_folder,
    save_duplicates_npz,
)


# ---------------------------------------------------------------------
# encode/decode
# ---------------------------------------------------------------------

def test_encode_decode_roundtrip_preserves_groups():
    duplicates = [[1, 5, 7], [0, 2], [9]]
    enc = encode_duplicates(duplicates, dtype=np.int32, n_items=10)

    assert isinstance(enc, DuplicatesNPZ)
    assert enc.indices.dtype == np.int32
    assert enc.indptr.dtype == np.int64
    assert enc.n_items == 10

    decoded = decode_duplicates(enc)
    assert decoded == duplicates


def test_encode_duplicates_empty_groups_and_empty_input():
    # empty input
    enc = encode_duplicates([], n_items=0)
    assert enc.indices.size == 0
    assert enc.indptr.tolist() == [0]
    assert decode_duplicates(enc) == []

    # includes empty groups
    duplicates = [[], [1, 2], []]
    enc2 = encode_duplicates(duplicates, n_items=3)
    dec2 = decode_duplicates(enc2)
    assert dec2 == duplicates


def test_encode_duplicates_raises_on_negative_indices():
    with pytest.raises(ValueError, match="negative"):
        _ = encode_duplicates([[0, -1, 2]])


def test_encode_duplicates_raises_on_out_of_range_if_n_items_given():
    with pytest.raises(ValueError, match="outside"):
        _ = encode_duplicates([[0, 5]], n_items=5)  # 5 is out of range


def test_decode_duplicates_validates_shapes_and_constraints():
    # indices not 1D
    with pytest.raises(ValueError, match="1D"):
        _ = decode_duplicates(DuplicatesNPZ(indices=np.zeros((2, 2)), indptr=np.array([0, 1])))

    # indptr not starting with 0
    with pytest.raises(ValueError, match="start with 0"):
        _ = decode_duplicates(DuplicatesNPZ(indices=np.array([1, 2]), indptr=np.array([1, 2, 2])))

    # indptr[-1] != len(indices)
    with pytest.raises(ValueError, match=r"indptr\[-1\]"):
        _ = decode_duplicates(DuplicatesNPZ(indices=np.array([1, 2]), indptr=np.array([0, 1])))

    # decreasing indptr (ensure indptr[-1] == len(indices) so we reach the non-decreasing check)
    with pytest.raises(ValueError, match="non-decreasing"):
        _ = decode_duplicates(DuplicatesNPZ(indices=np.array([1]), indptr=np.array([0, 2, 1])))


# ---------------------------------------------------------------------
# save/load
# ---------------------------------------------------------------------

def test_save_and_load_npz_roundtrip_without_metadata(tmp_path: Path):
    duplicates = [[3, 1], [0], [2, 2, 4]]
    fp = tmp_path / "exp_duplicates.npz"

    saved = save_duplicates_npz(fp, duplicates, n_items=10)
    assert saved.exists()

    loaded_dups, meta = load_duplicates_npz(fp, load_metadata=False)
    assert loaded_dups == duplicates
    assert meta is None


def test_save_and_load_npz_roundtrip_with_metadata(tmp_path: Path):
    duplicates = [[1, 2], [5]]
    meta_in = {"experiment": "expA", "param": 123, "nested": {"x": 1}}

    fp = tmp_path / "expA_duplicates.npz"
    save_duplicates_npz(fp, duplicates, n_items=None, metadata=meta_in)

    # Sidecar should exist at "<file>.npz.json"
    meta_path = fp.with_suffix(fp.suffix + ".json")
    assert meta_path.exists()

    loaded_dups, meta_out = load_duplicates_npz(fp, load_metadata=True)
    assert loaded_dups == duplicates
    assert meta_out == meta_in


def test_load_duplicates_npz_returns_none_metadata_if_missing(tmp_path: Path):
    duplicates = [[0, 1]]
    fp = tmp_path / "x_duplicates.npz"
    save_duplicates_npz(fp, duplicates, n_items=2, metadata=None)

    loaded_dups, meta_out = load_duplicates_npz(fp, load_metadata=True)
    assert loaded_dups == duplicates
    assert meta_out is None


def test_save_duplicates_npz_creates_parent_dirs(tmp_path: Path):
    duplicates = [[1]]
    fp = tmp_path / "nested" / "dir" / "exp_duplicates.npz"
    assert not fp.parent.exists()

    save_duplicates_npz(fp, duplicates, n_items=2)
    assert fp.exists()
    assert fp.parent.exists()


# ---------------------------------------------------------------------
# bulk loader
# ---------------------------------------------------------------------

def test_load_precomputed_duplicates_folder_raises_if_folder_missing(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        _ = load_precomputed_duplicates_folder(missing)


def test_load_precomputed_duplicates_folder_loads_all_and_sorts(tmp_path: Path):
    # Create three experiments with different names; ensure sorting by name.lower()
    d_a = [[1, 2]]
    d_b = [[3]]
    d_c = [[0, 9, 8]]

    # Note: filenames determine names by default: stem minus "_duplicates"
    save_duplicates_npz(tmp_path / "b_duplicates.npz", d_b, metadata={"m": "b"})
    save_duplicates_npz(tmp_path / "A_duplicates.npz", d_a, metadata={"m": "a"})
    save_duplicates_npz(tmp_path / "c_duplicates.npz", d_c, metadata=None)

    loaded = load_precomputed_duplicates_folder(tmp_path, load_metadata=True)
    assert all(isinstance(x, PrecomputedDuplicates) for x in loaded)

    names = [x.name for x in loaded]
    assert names == ["A", "b", "c"]  # sorted case-insensitive

    # Check content
    by_name = {x.name: x for x in loaded}
    assert by_name["A"].duplicates == d_a
    assert by_name["b"].duplicates == d_b
    assert by_name["c"].duplicates == d_c

    assert by_name["A"].metadata == {"m": "a"}
    assert by_name["b"].metadata == {"m": "b"}
    assert by_name["c"].metadata is None


def test_load_precomputed_duplicates_folder_custom_pattern(tmp_path: Path):
    save_duplicates_npz(tmp_path / "x_duplicates.npz", [[1]])
    save_duplicates_npz(tmp_path / "y_something_else.npz", [[2]])

    loaded = load_precomputed_duplicates_folder(tmp_path, pattern="*_duplicates.npz")
    assert [x.name for x in loaded] == ["x"]


def test_load_precomputed_duplicates_folder_custom_name_fn(tmp_path: Path):
    fp = tmp_path / "myexp_duplicates.npz"
    save_duplicates_npz(fp, [[1, 2, 3]])

    def name_fn(p: Path) -> str:
        return f"NAME::{p.stem}"

    loaded = load_precomputed_duplicates_folder(tmp_path, name_from_filename=name_fn)
    assert loaded[0].name == "NAME::myexp_duplicates"
    assert loaded[0].path_npz == fp
    assert loaded[0].duplicates == [[1, 2, 3]]
