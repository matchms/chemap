import struct
from hashlib import sha1
import numpy as np
import pytest
from chemap.additional_fingerprints.mhfp import MHFPEncoderLite


def _ref_token_hash32(token: bytes) -> np.uint32:
    # exact reference: struct.unpack("<I", sha1(t).digest()[:4])[0]
    return np.uint32(struct.unpack("<I", sha1(token).digest()[:4])[0])


def test_token_hash32_matches_reference_for_known_values():
    # deterministic across platforms
    tokens = [b"", b"a", b"abc", b"MAP4|1|X", b"\xff\x00\x01"]
    for t in tokens:
        got = MHFPEncoderLite._token_hash32(t)  # pylint: disable=protected-access
        exp = _ref_token_hash32(t)
        assert int(got) == int(exp)


def test_hash_returns_uint32_array_and_matches_token_hash():
    tokens = [b"a", b"b", b"c", b"a"]  # duplicates are allowed in hash()
    out = MHFPEncoderLite.hash(tokens)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint32
    assert out.shape == (len(tokens),)
    # elementwise match
    for i, t in enumerate(tokens):
        assert int(out[i]) == int(_ref_token_hash32(t))


def test_fold_sets_bits_at_mod_indices_and_is_binary_uint8():
    hv = np.array([0, 1, 2, 3, 1025, 2049], dtype=np.uint64)
    fp = MHFPEncoderLite.fold(hv, length=1024)
    assert fp.dtype == np.uint8
    assert fp.shape == (1024,)
    assert fp.sum() == 4  # indices {0,1,2,3} after mod
    assert fp[0] == 1 and fp[1] == 1 and fp[2] == 1 and fp[3] == 1


def test_fold_empty_returns_all_zeros():
    fp = MHFPEncoderLite.fold([], length=128)
    assert fp.shape == (128,)
    assert fp.dtype == np.uint8
    assert fp.sum() == 0


def test_constructor_validates_n_permutations():
    with pytest.raises(ValueError):
        _ = MHFPEncoderLite(n_permutations=0, seed=0)
    with pytest.raises(ValueError):
        _ = MHFPEncoderLite(n_permutations=-5, seed=0)


def test_fold_validates_length():
    with pytest.raises(ValueError):
        _ = MHFPEncoderLite.fold([1, 2, 3], length=0)
    with pytest.raises(ValueError):
        _ = MHFPEncoderLite.fold([1, 2, 3], length=-1)
