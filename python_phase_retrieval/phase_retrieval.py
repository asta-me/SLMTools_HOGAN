from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
try:
    import cupy as cp
except Exception:
    cp = None

# Support both package import and direct script execution.
if __package__ is None or __package__ == "":
    # Allow direct execution: python python_phase_retrieval/phase_retrieval.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from python_phase_retrieval.lattice_utils import Lattice, dual_phase, dual_shift_lattice, isft, ldot, phasor, r2, sft
else:
    from .lattice_utils import Lattice, dual_phase, dual_shift_lattice, isft, ldot, phasor, r2, sft


@dataclass
class PDGSLogEntry:
    """Per-iteration diagnostics captured during PDGS optimization."""
    # Iteration index (1-based in the main loop).
    iteration: int
    # Data-consistency metric across all diversity images.
    self_consistency_error: float
    # Spread between per-image updates and the averaged update.
    mean_update_norm: float
    # Wall-clock time from start of optimization.
    elapsed_ms: float
    # Optional phase RMSE against known ground truth (if provided).
    ground_truth_phase_rmse: float | None = None


def _to_numpy(a):
    """Convert arrays to NumPy, moving from GPU to CPU when needed."""
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def _phasor_xp(x, xp):
    """Return unit-magnitude complex field that keeps phase of x."""
    return xp.exp(1j * xp.angle(x))


def _phase_rmse(est: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Compute wrapped phase RMSE between estimated and reference phases."""
    diff = np.angle(np.exp(1j * (est - ref)))
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(np.mean(diff**2)))


def _align_global_phase(est: np.ndarray, ref: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Remove global complex phase ambiguity by best matching est to ref."""
    if weights is None:
        c = np.mean(est * np.conj(ref))
    else:
        w = np.asarray(weights, dtype=float)
        c = np.sum(w * est * np.conj(ref)) / np.sum(w)
    return est * np.exp(-1j * np.angle(c))


def one_shot(
    img_intensity: np.ndarray,
    alpha: float,
    beta: Sequence[float],
    L: Lattice,
    flambda: float = 1.0,
) -> np.ndarray:
    """
    Build an analytic single-pass field estimate from one measured intensity image.

    This is the Python analogue of Julia `oneShot` and applies the same lattice/
    propagation phase factors before inverse-transforming to the object plane.
    """
    n = len(L)
    if len(beta) != n:
        raise ValueError("beta length must match lattice dimension")

    # Build shifted dual lattice and center offset vector.
    dL = dual_shift_lattice(L, flambda)
    xc = np.asarray(beta, dtype=float) * flambda

    # Phase terms defined by the lattice model.
    div_phase = (alpha / 2.0) * r2(dL) + ldot(beta, dL)
    dual_div_phase = -(r2(L) - 2.0 * ldot(xc, L) + float(np.sum(xc**2))) / (2.0 * alpha * flambda**2)

    # Intensity -> modulus, then inverse SFT back to complex field estimate.
    mod = np.sqrt(np.clip(np.asarray(img_intensity, dtype=float), 0.0, None))
    return isft(mod * np.exp(1j * dual_div_phase)) * np.exp(-1j * div_phase)


def pdgs_iter(
    guess,
    phis,
    mods,
    xp,
) -> Tuple[object, Tuple[object, ...]]:
    """Run one PDGS update: one projection per diversity image, then average."""
    # guess, phis, mods are all in ifftshift-space: use plain FFT (no extra shifts).
    # This matches Julia's plan_fft / plan_ifft used inside pdgsIter.
    updates: List[object] = []
    for phi_i, mod_i in zip(phis, mods):
        # Forward propagate with diversity phase, impose measured modulus,
        # and backpropagate to object space.
        upd = xp.fft.ifftn(mod_i * _phasor_xp(xp.fft.fftn(guess * phi_i), xp)) * xp.conj(phi_i)
        updates.append(upd)
    new_guess = sum(updates) / len(updates)
    return new_guess, tuple(updates)


def pdgs(
    imgs_modulus: Sequence[np.ndarray],
    div_phases: Sequence[np.ndarray],
    nit: int,
    beam_guess: np.ndarray,
    L: Lattice,
    flambda: float = 1.0,
    verbose: bool = False,
    progress_every: int | None = None,
    use_gpu: bool = False,
) -> np.ndarray:
    """Convenience wrapper returning only final PDGS estimate (no logs)."""
    beam, _ = pdgs_log(
        imgs_modulus=imgs_modulus,
        div_phases=div_phases,
        nit=nit,
        beam_guess=beam_guess,
        L=L,
        flambda=flambda,
        every=nit + 1,
        verbose=verbose,
        progress_every=progress_every,
        use_gpu=use_gpu,
    )
    return beam


def pdgs_log(
    imgs_modulus: Sequence[np.ndarray],
    div_phases: Sequence[np.ndarray],
    nit: int,
    beam_guess: np.ndarray,
    L: Lattice,
    flambda: float = 1.0,
    every: int = 1,
    phase_truth: np.ndarray | None = None,
    phase_rmse_mask: np.ndarray | None = None,
    verbose: bool = False,
    progress_every: int | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, List[PDGSLogEntry]]:
    """
    Run PDGS iterations and collect convergence diagnostics.

    Inputs are expected in the same coordinate/shift convention used by Julia code.
    If `use_gpu=True` and CuPy is available, computations run on GPU.
    """
    # Basic shape and consistency checks to fail early with clear messages.
    if len(imgs_modulus) == 0:
        raise ValueError("imgs_modulus cannot be empty")
    if len(imgs_modulus) != len(div_phases):
        raise ValueError("imgs_modulus and div_phases must have same length")

    shape = imgs_modulus[0].shape
    if any(img.shape != shape for img in imgs_modulus):
        raise ValueError("All images must have the same shape")
    if any(phi.shape != shape for phi in div_phases):
        raise ValueError("All diversity phases must match image shape")
    if beam_guess.shape != shape:
        raise ValueError("beam_guess shape mismatch")

    # Choose backend (NumPy CPU vs CuPy GPU).
    gpu_enabled = bool(use_gpu and cp is not None)
    if use_gpu and cp is None:
        print("[PDGS] CuPy not available; falling back to NumPy CPU.")
    xp = cp if gpu_enabled else np

    # Use lighter precision on GPU for speed/memory, full precision on CPU.
    guess_dtype = xp.complex64 if gpu_enabled else np.complex128
    real_dtype = xp.float32 if gpu_enabled else np.float64

    # Keep internal arrays in ifftshift-space to match FFT convention used below.
    guess = xp.fft.ifftshift(xp.asarray(beam_guess, dtype=guess_dtype))

    # Precompute static per-image factors.
    dphase = dual_phase(L, flambda)
    phis = tuple(xp.fft.ifftshift(xp.exp(1j * xp.asarray(phi + dphase, dtype=real_dtype))) for phi in div_phases)
    mods = tuple(xp.fft.ifftshift(xp.asarray(m, dtype=real_dtype)) for m in imgs_modulus)

    logs: List[PDGSLogEntry] = []
    t0 = perf_counter()
    if progress_every is None:
        progress_every = max(1, nit // 20)

    for j in range(1, nit + 1):
        new_guess, updates = pdgs_iter(guess, phis, mods, xp)
        # RMS step size between consecutive guesses.
        mean_step_norm = float(_to_numpy(xp.sqrt(xp.mean(xp.abs(new_guess - guess) ** 2))))

        if (j - 1) % every == 0:
            # Internal consistency among per-image updates.
            mean_update_norm = float(
                _to_numpy(xp.sqrt(sum(xp.sum(xp.abs(u - new_guess) ** 2) for u in updates)) / len(updates))
            )
            # Same ifftshift-space convention as pdgs_iter: use plain fftn.
            per_img_err = xp.stack(
                [xp.mean((xp.abs(xp.fft.fftn(new_guess * phis[i])) - mods[i]) ** 2) for i in range(len(mods))]
            )
            self_consistency_error = float(
                _to_numpy(xp.mean(per_img_err))
            )

            gt_rmse = None
            if phase_truth is not None:
                # Align global phase before computing phase error against truth.
                est_centered = _to_numpy(xp.fft.fftshift(new_guess))
                est_aligned = _align_global_phase(est_centered, np.exp(1j * phase_truth))
                gt_rmse = _phase_rmse(np.angle(est_aligned), phase_truth, mask=phase_rmse_mask)

            logs.append(
                PDGSLogEntry(
                    iteration=j,
                    self_consistency_error=self_consistency_error,
                    mean_update_norm=mean_update_norm,
                    elapsed_ms=1000.0 * (perf_counter() - t0),
                    ground_truth_phase_rmse=gt_rmse,
                )
            )

        if verbose and (j == 1 or j == nit or (j % progress_every == 0)):
            elapsed_s = perf_counter() - t0
            eta_s = elapsed_s / j * (nit - j)
            latest_sc = logs[-1].self_consistency_error if logs else float("nan")
            latest_gt = logs[-1].ground_truth_phase_rmse if logs else None
            payload = {
                "iteration": j,
                "nit": nit,
                "elapsed_s": elapsed_s,
                "eta_s": eta_s,
                "mean_step_norm": mean_step_norm,
                "self_consistency_error": latest_sc,
                "ground_truth_phase_rmse": latest_gt,
            }
            if progress_callback is not None:
                # Let caller handle progress reporting (GUI, logger, etc.).
                progress_callback(payload)
            else:
                gt_text = "nan" if latest_gt is None else f"{latest_gt:.3e}"
                print(
                    f"[PDGS] iter {j:>6}/{nit} | elapsed {elapsed_s:>7.1f}s | "
                    f"eta {eta_s:>7.1f}s | step {mean_step_norm:.3e} | "
                    f"sc {latest_sc:.3e} | gt {gt_text}"
                )

        guess = new_guess

    return _to_numpy(xp.fft.fftshift(guess)), logs
