import struct
from hashlib import sha1
from typing import Iterable, Sequence, Union
import numpy as np


BytesLike = Union[bytes, bytearray, memoryview]


class MHFPEncoderLite:
    """
    Compatibility-focused reimplementation of the original mhfp.encoder.MHFPEncoder.
    (Original is from the Reymond group: https://github.com/reymond-group/mhfp)

    Notes
    -----
    - The original uses:
        prime    = 2^61 - 1
        max_hash = 2^32 - 1
      and outputs uint32 signatures.
    - Token hash is:
        struct.unpack("<I", sha1(token).digest()[:4])[0]
    """

    prime: int = (1 << 61) - 1
    max_hash: int = (1 << 32) - 1

    def __init__(self, n_permutations: int = 2048, seed: int = 42):
        if n_permutations <= 0:
            raise ValueError("n_permutations must be > 0.")
        self.n_permutations = int(n_permutations)
        self.seed = int(seed)

        # Match original: generate uint32 a,b with uniqueness constraints
        rand = np.random.RandomState(self.seed)

        a = np.zeros(self.n_permutations, dtype=np.uint32)
        b = np.zeros(self.n_permutations, dtype=np.uint32)

        # Original code used `while a in self.permutations_a` checks (O(n)),
        # but that behavior means "no duplicates". We'll enforce the same.
        used_a = set()
        used_b = set()

        for i in range(self.n_permutations):
            ai = int(rand.randint(1, MHFPEncoderLite.max_hash, dtype=np.uint32))
            bi = int(rand.randint(0, MHFPEncoderLite.max_hash, dtype=np.uint32))

            while ai in used_a:
                ai = int(rand.randint(1, MHFPEncoderLite.max_hash, dtype=np.uint32))
            while bi in used_b:
                bi = int(rand.randint(0, MHFPEncoderLite.max_hash, dtype=np.uint32))

            used_a.add(ai)
            used_b.add(bi)
            a[i] = np.uint32(ai)
            b[i] = np.uint32(bi)

        # Match original: reshape to column vectors (n_perm, 1)
        self._a = a.reshape((self.n_permutations, 1)).astype(np.uint64, copy=False)
        self._b = b.reshape((self.n_permutations, 1)).astype(np.uint64, copy=False)

    # -----------------------------
    # Token hashing (exact)
    # -----------------------------

    @staticmethod
    def _token_hash32(token: BytesLike) -> np.uint32:
        # EXACT original semantics: struct.unpack("<I", sha1(t).digest()[:4])[0]
        return np.uint32(struct.unpack("<I", sha1(bytes(token)).digest()[:4])[0])

    # -----------------------------
    # Original helper API: hash / fold / merge / distance
    # -----------------------------

    @staticmethod
    def hash(tokens: Iterable[BytesLike]) -> np.ndarray:
        """
        For compatibility with original MHFPEncoder.hash(shingling):
        returns per-token uint32 hash values (NOT minhash signature).
        """
        return np.fromiter(
            (MHFPEncoderLite._token_hash32(t) for t in tokens),
            dtype=np.uint32,
        )

    @staticmethod
    def fold(hash_values: Sequence[int], length: int = 2048) -> np.ndarray:
        """
        Compatibility with original fold(): binary uint8 vector with bits set at hash % length.
        """
        length = int(length)
        if length <= 0:
            raise ValueError("length must be > 0.")
        folded = np.zeros(length, dtype=np.uint8)
        if len(hash_values) == 0:
            return folded
        hv = np.asarray(hash_values, dtype=np.uint64)
        folded[(hv % np.uint64(length)).astype(np.int64, copy=False)] = 1
        return folded
