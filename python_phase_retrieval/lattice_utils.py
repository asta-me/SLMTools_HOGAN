from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


ArrayLike = np.ndarray
Lattice = Tuple[np.ndarray, ...]


def natrange(n: int) -> np.ndarray:
    """Natural 1D lattice that is self-dual under the DFT convention used here."""
    return np.arange(-np.floor(n / 2), np.floor((n - 1) / 2) + 1, dtype=float) / np.sqrt(n)


def natlat(shape: Sequence[int]) -> Lattice:
    """Natural N-D lattice tuple for a given array shape."""
    return tuple(natrange(int(n)) for n in shape)


def lattice_step(x: np.ndarray) -> float:
    if x.size < 2:
        return 1.0
    return float(x[1] - x[0])


def to_dim(v: np.ndarray, d: int, n: int) -> np.ndarray:
    """Reshape v to broadcast only along axis d in an n-dimensional tensor."""
    shape = [1] * n
    shape[d] = v.shape[0]
    return v.reshape(shape)


def lattice_displacement(L: Lattice) -> np.ndarray:
    return np.array([l[0] + np.floor(l.size / 2) * lattice_step(l) for l in L], dtype=float)


def dual_shift_lattice(L: Lattice, flambda: float = 1.0) -> Lattice:
    out = []
    for l in L:
        n = l.size
        dx = lattice_step(l)
        idx = np.arange(-np.floor(n / 2), np.floor((n - 1) / 2) + 1, dtype=float)
        out.append(idx * flambda / (n * dx))
    return tuple(out)


def r2(L: Lattice) -> np.ndarray:
    n = len(L)
    return sum(to_dim(L[i] ** 2, i, n) for i in range(n))


def ldot(v: Sequence[float], L: Lattice) -> np.ndarray:
    n = len(L)
    if len(v) != n:
        raise ValueError("Vector length must match lattice dimension.")
    return sum(float(v[i]) * to_dim(L[i], i, n) for i in range(n))


def sft(v: np.ndarray) -> np.ndarray:
    """Shifted forward FFT, matching the Julia sft convention."""
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(v)))


def isft(v: np.ndarray) -> np.ndarray:
    """Shifted inverse FFT, matching the Julia isft convention."""
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(v)))


def phasor(x: np.ndarray) -> np.ndarray:
    """Unit-modulus complex phasor from a complex or real phase-like input."""
    if np.iscomplexobj(x):
        return np.exp(1j * np.angle(x))
    return np.exp(1j * x)


def dual_phase(L: Lattice, flambda: float = 1.0, dL: Lattice | None = None) -> np.ndarray:
    if dL is None:
        dL = dual_shift_lattice(L, flambda)
    b = lattice_displacement(L)
    n = len(L)
    return sum(b[i] * to_dim(dL[i], i, n) for i in range(n)) / flambda


def normalize_distribution(u: np.ndarray) -> np.ndarray:
    up = np.abs(u)
    s = np.sum(up)
    if s <= 0:
        raise ValueError("Distribution sum must be positive.")
    return up / s
