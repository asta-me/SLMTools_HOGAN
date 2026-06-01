#%%
"""
SLMTools_HOGAN - lattice_utils.py

Update date: 2026-06-01

What changed today:
1. Migrated from a literal Julia-style translation to a more Python-friendly API.
2. Kept numerical behavior and indexing conventions aligned with previous code.
3. Added clearer function names (for example: reciprocal_lattice, shifted_fftn).
4. Kept legacy names as backward-compatible wrappers with DeprecationWarning.

Important note about history/recovery:
- The original code can always be restored from Git/GitHub history.
- Use commit history around this date to retrieve the previous version.
- Prefer restoring by commit hash (most reliable), not by date alone.
"""

from __future__ import annotations

from typing import Sequence, Tuple
import warnings

import numpy as np


ArrayLike = np.ndarray
Lattice = Tuple[np.ndarray, ...]


def _deprecated(old_name: str, new_name: str) -> None:
    warnings.warn(
        f"{old_name}() is deprecated and will be removed in a future release; "
        f"use {new_name}() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

# Array from -n/2 to n/2-1, scaled by 1/sqrt(n) 
# so that dx = 1/sqrt(n) is equal to du = 1/(n*dx) = 1/sqrt(n)
def centered_self_dual_axis(n: int) -> np.ndarray:
    """Centered 1D axis with self-dual scaling for this DFT convention."""
    return np.arange(-np.floor(n / 2), np.floor((n - 1) / 2) + 1, dtype=float) / np.sqrt(n)


def centered_self_dual_lattice(shape: Sequence[int]) -> Lattice:
    """Build an N-D lattice as a tuple of centered 1D self-dual axes."""
    return tuple(centered_self_dual_axis(int(n)) for n in shape)


def axis_step(axis: np.ndarray) -> float:
    """Return uniform spacing of a 1D axis; fallback to 1.0 for degenerate size."""
    if axis.size < 2:
        return 1.0
    return float(axis[1] - axis[0])


def broadcast_axis(v: np.ndarray, axis: int, ndim: int) -> np.ndarray:
    """Reshape a 1D vector so it broadcasts only along one axis in an N-D tensor."""
    shape = [1] * ndim              # [1, 1, 1, ...  1]
    shape[axis] = v.shape[0]        # [1, 1, n, 1, ... 1]
    return np.reshape(v, shape)     # [[..[v]..]]


def centered_lattice_offset(lattice: Lattice) -> np.ndarray:
    """Offset of each axis from the centered indexing convention used here."""
    return np.array(
        [axis[0] + np.floor(axis.size / 2) * axis_step(axis) for axis in lattice],
        dtype=float,
    )


def reciprocal_lattice(lattice: Lattice, flambda: float = 1.0) -> Lattice:
    """Reciprocal lattice matching the centered FFT indexing convention."""
    out = []
    for axis in lattice:
        n = axis.size
        dx = axis_step(axis)
        freq = np.fft.fftshift(np.fft.fftfreq(n, d=dx / flambda))
        out.append(freq.astype(float, copy=False))
    return tuple(out)


def squared_radius(lattice: Lattice) -> np.ndarray:
    """Return sum of squared coordinates across all lattice dimensions."""
    ndim = len(lattice) # n of dimensions
    # For each dimension i, compute lattice[i]**2 and reshape it to broadcast along that dimension, then sum
    return sum(broadcast_axis(lattice[i] ** 2, i, ndim) for i in range(ndim))


def lattice_dot(vector: Sequence[float], lattice: Lattice) -> np.ndarray:
    """Broadcasted dot product between a vector and an N-D lattice tuple."""
    ndim = len(lattice)         # n of dimensions
    if len(vector) != ndim:     
        raise ValueError("Vector length must match lattice dimension.")
    return sum(float(vector[i]) * broadcast_axis(lattice[i], i, ndim) for i in range(ndim))


def shifted_fftn(v: np.ndarray) -> np.ndarray:
    """Centered forward FFT: fftshift(fftn(ifftshift(v)))."""
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(v)))


def shifted_ifftn(v: np.ndarray) -> np.ndarray:
    """Centered inverse FFT: fftshift(ifftn(ifftshift(v)))."""
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(v)))


def unit_phasor(x: np.ndarray) -> np.ndarray:
    """Unit-modulus complex phasor from real phase or complex field input."""
    if np.iscomplexobj(x):
        return np.exp(1j * np.angle(x))
    return np.exp(1j * x)


def reciprocal_phase_ramp(
    lattice: Lattice,
    flambda: float = 1.0,
    reciprocal: Lattice | None = None,
) -> np.ndarray:
    """Linear phase ramp on the reciprocal lattice induced by lattice centering offset.
    Shift theorem : f(x+deltax) <-> F(u) * exp(2pi i (u * deltax)) 
    u = reciprocal lattice coordinate;  deltax = offset of the spatial lattice;
    u*deltax is the reciprocal phase ramp 
    frequency to space scaling is flambda
    """
    if reciprocal is None:
        reciprocal = reciprocal_lattice(lattice, flambda)
    offset = centered_lattice_offset(lattice) # Measure the offset
    ndim = len(lattice)                       # N of dimensions
    return sum(offset[i] * broadcast_axis(reciprocal[i], i, ndim) for i in range(ndim)) / flambda


def normalize_magnitude_distribution(u: np.ndarray) -> np.ndarray:
    """Normalize abs(u) so that its sum is exactly 1."""
    up = np.abs(u)
    s = np.sum(up)
    if s <= 0:
        raise ValueError("Distribution sum must be positive.")
    return up / s


# Backward-compatible deprecated aliases.
def natrange(n: int) -> np.ndarray:
    _deprecated("natrange", "centered_self_dual_axis")
    return centered_self_dual_axis(n)


def natlat(shape: Sequence[int]) -> Lattice:
    _deprecated("natlat", "centered_self_dual_lattice")
    return centered_self_dual_lattice(shape)


def lattice_step(x: np.ndarray) -> float:
    _deprecated("lattice_step", "axis_step")
    return axis_step(x)


def to_dim(v: np.ndarray, d: int, n: int) -> np.ndarray:
    _deprecated("to_dim", "broadcast_axis")
    return broadcast_axis(v, d, n)


def lattice_displacement(L: Lattice) -> np.ndarray:
    _deprecated("lattice_displacement", "centered_lattice_offset")
    return centered_lattice_offset(L)


def dual_shift_lattice(L: Lattice, flambda: float = 1.0) -> Lattice:
    _deprecated("dual_shift_lattice", "reciprocal_lattice")
    return reciprocal_lattice(L, flambda)


def r2(L: Lattice) -> np.ndarray:
    _deprecated("r2", "squared_radius")
    return squared_radius(L)


def ldot(v: Sequence[float], L: Lattice) -> np.ndarray:
    _deprecated("ldot", "lattice_dot")
    return lattice_dot(v, L)


def sft(v: np.ndarray) -> np.ndarray:
    _deprecated("sft", "shifted_fftn")
    return shifted_fftn(v)


def isft(v: np.ndarray) -> np.ndarray:
    _deprecated("isft", "shifted_ifftn")
    return shifted_ifftn(v)


def phasor(x: np.ndarray) -> np.ndarray:
    _deprecated("phasor", "unit_phasor")
    return unit_phasor(x)


def dual_phase(L: Lattice, flambda: float = 1.0, dL: Lattice | None = None) -> np.ndarray:
    _deprecated("dual_phase", "reciprocal_phase_ramp")
    return reciprocal_phase_ramp(L, flambda, reciprocal=dL)


def normalize_distribution(u: np.ndarray) -> np.ndarray:
    _deprecated("normalize_distribution", "normalize_magnitude_distribution")
    return normalize_magnitude_distribution(u)
