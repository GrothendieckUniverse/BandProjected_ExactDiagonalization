"""Pure-NumPy bitmask Fock basis, meet-in-the-middle sector enumeration, and O(1) lookup."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import numpy as np

MAX_ORBITALS_UINT64 = 63


def popcount_u64(arr: np.ndarray) -> np.ndarray:
    """Pure-NumPy vectorized population count on uint64 arrays."""
    x = np.asarray(arr, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.int64)
    y = x.copy()
    y -= (y >> np.uint64(1)) & np.uint64(0x5555555555555555)
    y = (y & np.uint64(0x3333333333333333)) + ((y >> np.uint64(2)) & np.uint64(0x3333333333333333))
    y = (y + (y >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((y * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def parity_sign_u64(arr: np.ndarray) -> np.ndarray:
    """Fast vectorized fermion sign (-1)^{popcount(arr)} without int64 widening when supported."""
    x = np.asarray(arr, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.where((np.bitwise_count(x) & np.uint8(1)) == 0, 1.0, -1.0)
    return 1.0 - 2.0 * (popcount_u64(x) & 1)


def make_between_masks(n_orbitals: int) -> np.ndarray:
    """Bitmasks of bits strictly between ``i`` and ``j`` for branch-free JW parity."""
    masks = np.zeros((n_orbitals, n_orbitals), dtype=np.uint64)
    for i in range(n_orbitals):
        for j in range(i + 2, n_orbitals):
            masks[i, j] = np.uint64(((1 << (j - i - 1)) - 1) << (i + 1))
            masks[j, i] = masks[i, j]
    return masks


def make_below_masks(n_orbitals: int) -> np.ndarray:
    """Bitmasks of bits strictly below ``i``."""
    return np.array([(1 << i) - 1 for i in range(n_orbitals)], dtype=np.uint64)


class SectorLookup:
    """Hash-free upper-bit prefix-bucketed index lookup into a sorted uint64 basis."""

    def __init__(self, sorted_masks: np.ndarray, n_orbitals: int) -> None:
        self.masks = np.asarray(sorted_masks, dtype=np.uint64)
        self.n_orbitals = int(n_orbitals)
        d = len(self.masks)
        self.size = d
        if d <= 256 or n_orbitals == 0:
            self.prefix_bits = 0
            self.shift = n_orbitals
            self.starts = np.array([0, d], dtype=np.int64)
            self.max_bucket = d
            return
        # Aim for <= 1-2 items per bucket while bounding lookup table <= 2^18 entries
        b = min(18, n_orbitals, max(8, int(math.ceil(math.log2(max(d, 2)))) + 1))
        self.prefix_bits = b
        self.shift = n_orbitals - b
        n_buckets = 1 << b
        prefixes = (self.masks >> np.uint64(self.shift)).astype(np.int64)
        counts = np.bincount(prefixes, minlength=n_buckets)
        starts = np.zeros(n_buckets + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        self.starts = starts
        self.max_bucket = int(counts.max()) if d else 0

    def find(self, queries: np.ndarray) -> np.ndarray:
        """Return exact indices in ``self.masks`` for sorted/unsorted ``queries`` (-1 if absent)."""
        q = np.asarray(queries, dtype=np.uint64)
        n = len(q)
        if n == 0 or self.size == 0:
            return np.full(n, -1, dtype=np.int64)
        if self.prefix_bits > 0 and self.max_bucket <= 1:
            p = (q >> np.uint64(self.shift)).astype(np.int64)
            lo = self.starts[p]
            hi = self.starts[p + 1]
            safe = np.minimum(lo, self.size - 1)
            hit = (lo < hi) & (self.masks[safe] == q)
            return np.where(hit, safe, -1)
        idx = np.searchsorted(self.masks, q)
        safe = np.minimum(idx, self.size - 1)
        hit = (idx < self.size) & (self.masks[safe] == q)
        return np.where(hit, safe, -1)

    def find_valid(self, queries: np.ndarray) -> np.ndarray:
        """Fast lookup assuming every query is guaranteed to exist in ``self.masks``."""
        q = np.asarray(queries, dtype=np.uint64)
        if len(q) == 0:
            return np.empty(0, dtype=np.int64)
        if self.prefix_bits > 0 and self.max_bucket <= 1:
            return self.starts[(q >> np.uint64(self.shift)).astype(np.int64)]
        return np.searchsorted(self.masks, q)


@dataclass
class SectorBasis:
    """Fixed particle-number, fixed total-momentum Fock basis stored as sorted uint64 masks."""

    n_orbitals: int
    n_particles: int
    masks: np.ndarray
    sector: tuple[int, int]
    geometry_shape: tuple[int, int]
    lookup: SectorLookup

    def __len__(self) -> int:
        return len(self.masks)

    def configurations(self) -> np.ndarray:
        """Decode bitmasks to sorted occupied-orbital matrix of shape (D, N_particles)."""
        d = len(self.masks)
        configs = np.empty((d, self.n_particles), dtype=np.int16)
        rem = self.masks.copy()
        for col in range(self.n_particles):
            lsb = rem & (~rem + np.uint64(1))
            # bit_length via log2 or loop? Wait: use bitwise table or bit_length!
            # Pure numpy exact trailing zeros using De Bruijn or float64 frexp/bit count:
            # Count trailing zeros via popcount((lsb - 1))!
            configs[:, col] = popcount_u64(lsb - np.uint64(1))
            rem ^= lsb
        return configs


def _half_bitmasks(n_bits: int, max_pop: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate (masks, popcounts) for ``n_bits`` bits up to popcount ``max_pop``."""
    if n_bits <= 20:
        total = 1 << n_bits
        m = np.arange(total, dtype=np.uint64)
        pop = popcount_u64(m)
        if max_pop < n_bits:
            keep = pop <= max_pop
            return m[keep], pop[keep]
        return m, pop
    # Popcount-bounded exact enumeration when n_bits > 20
    masks_list = [np.array([0], dtype=np.uint64)]
    pops_list = [np.array([0], dtype=np.int64)]
    for k in range(1, min(max_pop, n_bits) + 1):
        comb = np.asarray(list(combinations(range(n_bits), k)), dtype=np.uint64)
        m_k = np.bitwise_or.reduce(np.uint64(1) << comb, axis=1) if comb.size else np.empty(0, dtype=np.uint64)
        masks_list.append(np.sort(m_k))
        pops_list.append(np.full(len(m_k), k, dtype=np.int64))
    return np.concatenate(masks_list), np.concatenate(pops_list)


class MomentumSectorCatalog:
    """Meet-in-the-middle momentum-sector Fock basis generator.

    Splits ``N_orb`` orbitals into left (low bits ``0..n_L-1``) and right
    (high bits ``n_L..N_orb-1``) halves, precomputing half-basis buckets
    grouped by ``(popcount, momentum_code)``. Any ``(N_e, K)`` sector is then
    assembled directly without ever enumerating the full ``C(N_orb, N_e)`` space.
    """

    def __init__(
        self,
        n_orbitals: int,
        geometry_shape: tuple[int, int],
        orbital_momenta: np.ndarray,
        max_particles: int | None = None,
    ) -> None:
        if not 0 < n_orbitals <= MAX_ORBITALS_UINT64:
            raise ValueError(f"Requires 1 <= n_orbitals <= {MAX_ORBITALS_UINT64}, got {n_orbitals}")
        self.n_orbitals = int(n_orbitals)
        self.shape = (int(geometry_shape[0]), int(geometry_shape[1]))
        self.n_k = self.shape[0] * self.shape[1]
        self.orbital_momenta = np.asarray(orbital_momenta, dtype=np.int64) % self.shape
        if self.orbital_momenta.shape != (self.n_orbitals, 2):
            raise ValueError("orbital_momenta shape mismatch")

        self.n_left = self.n_orbitals // 2
        self.n_right = self.n_orbitals - self.n_left
        max_pop = self.n_orbitals if max_particles is None else int(max_particles)

        self._left_buckets = self._build_half_buckets(0, self.n_left, min(max_pop, self.n_left))
        self._right_buckets = self._build_half_buckets(
            self.n_left, self.n_right, min(max_pop, self.n_right)
        )
        self._sector_cache: dict[tuple[int, int, int], SectorBasis] = {}

    def _build_half_buckets(
        self, offset: int, n_bits: int, max_pop: int
    ) -> dict[tuple[int, int], np.ndarray]:
        masks, pops = _half_bitmasks(n_bits, max_pop)
        mom_sub = self.orbital_momenta[offset : offset + n_bits]
        # Vectorized momentum accumulation across bits
        k1 = np.zeros(len(masks), dtype=np.int64)
        k2 = np.zeros(len(masks), dtype=np.int64)
        for bit in range(n_bits):
            active = ((masks >> np.uint64(bit)) & np.uint64(1)) != 0
            k1[active] += mom_sub[bit, 0]
            k2[active] += mom_sub[bit, 1]
        k1 %= self.shape[0]
        k2 %= self.shape[1]
        k_code = k1 * self.shape[1] + k2

        # Group by (pop, k_code) via single lexsort while keeping masks sorted within each bucket
        key = pops * self.n_k + k_code
        order = np.argsort(key, kind="mergesort")
        s_key = key[order]
        s_masks = masks[order]
        uniq_keys, starts, counts = np.unique(s_key, return_index=True, return_counts=True)
        buckets: dict[tuple[int, int], np.ndarray] = {}
        for uk, st, ct in zip(uniq_keys, starts, counts, strict=True):
            p = int(uk // self.n_k)
            kc = int(uk % self.n_k)
            sub = s_masks[st : st + ct]
            # mergesort kept ascending mask order inside each bucket
            sub = np.sort(sub)
            sub.flags.writeable = False
            buckets[(p, kc)] = sub
        return buckets

    def basis(self, n_particles: int, sector: tuple[int, int]) -> SectorBasis:
        """Build or return cached SectorBasis for ``(n_particles, sector)``."""
        n_p = int(n_particles)
        s1, s2 = int(sector[0]) % self.shape[0], int(sector[1]) % self.shape[1]
        cache_key = (n_p, s1, s2)
        if cache_key in self._sector_cache:
            return self._sector_cache[cache_key]

        if n_p < 0 or n_p > self.n_orbitals:
            raise ValueError(f"Invalid particle number {n_p} for {self.n_orbitals} orbitals")

        n1, n2 = self.shape
        shift = np.uint64(self.n_left)
        parts = []
        for n_l in range(max(0, n_p - self.n_right), min(n_p, self.n_left) + 1):
            n_r = n_p - n_l
            for k1_l in range(n1):
                k1_r = (s1 - k1_l) % n1
                for k2_l in range(n2):
                    k2_r = (s2 - k2_l) % n2
                    left = self._left_buckets.get((n_l, k1_l * n2 + k2_l))
                    if left is None or len(left) == 0:
                        continue
                    right = self._right_buckets.get((n_r, k1_r * n2 + k2_r))
                    if right is None or len(right) == 0:
                        continue
                    joined = (right[:, None] << shift) | left[None, :]
                    parts.append(joined.ravel())

        if parts:
            masks = np.sort(np.concatenate(parts))
        else:
            masks = np.empty(0, dtype=np.uint64)
        masks.flags.writeable = False
        sb = SectorBasis(
            n_orbitals=self.n_orbitals,
            n_particles=n_p,
            masks=masks,
            sector=(s1, s2),
            geometry_shape=self.shape,
            lookup=SectorLookup(masks, self.n_orbitals),
        )
        self._sector_cache[cache_key] = sb
        return sb
