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

if __package__ is None or __package__ == "":
    # Allow direct execution: python python_phase_retrieval/phase_retrieval.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from python_phase_retrieval.lattice_utils import Lattice, dual_phase, dual_shift_lattice, isft, ldot, phasor, r2, sft
else:
    from .lattice_utils import Lattice, dual_phase, dual_shift_lattice, isft, ldot, phasor, r2, sft


@dataclass
class PDGSLogEntry:
    iteration: int
    self_consistency_error: float
    mean_update_norm: float
    elapsed_ms: float
    ground_truth_phase_rmse: float | None = None


def _to_numpy(a):
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def _phasor_xp(x, xp):
    return xp.exp(1j * xp.angle(x))


def _phase_rmse(est: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.angle(np.exp(1j * (est - ref)))
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(np.mean(diff**2)))


def _align_global_phase(est: np.ndarray, ref: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        c = np.mean(est * np.conj(ref))
    else:
        w = np.asarray(weights, dtype=float)
        c = np.sum(w * est * np.conj(ref)) / np.sum(w)
    return est * np.exp(-1j * np.angle(c))


def one_shot(img_intensity, alpha, beta, L, flambda):
    dL = dual_shift_lattice(L, flambda)
    
    # 1. Converti beta (rad/m) in u0 (cicli/m), poi ricava lo shift fisico xc
    u0 = np.asarray(beta, dtype=float) / (2.0 * np.pi)
    xc = u0 * flambda

    # 2. Calcola div_phase sulla griglia SLM (L)
    div_phase = (alpha / 2.0) * r2(L) + ldot(beta, L)

    # 3. Calcola dual_div_phase sulla griglia Fotocamera (dL) con scala (2*pi)^2
    X_minus_xc_sq = r2(dL) - 2.0 * ldot(xc, dL) + float(np.sum(xc**2))
    dual_div_phase = - (4.0 * np.pi**2) * X_minus_xc_sq / (2.0 * alpha * flambda**2)

    mod = np.sqrt(np.clip(np.asarray(img_intensity, dtype=float), 0.0, None))
    
    # ISFT trasporta il campo dalla Fotocamera (dL) all'SLM (L)
    return isft(mod * np.exp(1j * dual_div_phase)) * np.exp(-1j * div_phase)


def pdgs_iter(guess, phis, mods, xp, return_updates=False):
    new_guess = xp.zeros_like(guess)
    updates = [] if return_updates else None
    
    for phi_i, mod_i in zip(phis, mods):
        upd = xp.fft.ifftn(mod_i * _phasor_xp(xp.fft.fftn(guess * phi_i), xp)) * xp.conj(phi_i)
        new_guess += upd
        if return_updates:
            updates.append(upd)
            
    new_guess /= len(phis)
    return new_guess, updates


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
    """PDGS with convergence logging similar to Julia pdgsLog plus richer metrics."""
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

    gpu_enabled = bool(use_gpu and cp is not None)
    if use_gpu and cp is None:
        print("[PDGS] CuPy not available; falling back to NumPy CPU.")
    xp = cp if gpu_enabled else np

    guess_dtype = xp.complex64 if gpu_enabled else np.complex128
    real_dtype = xp.float32 if gpu_enabled else np.float64

    guess = xp.fft.ifftshift(xp.asarray(beam_guess, dtype=guess_dtype))

    dphase = dual_phase(L, flambda)
    phis = tuple(xp.fft.ifftshift(xp.exp(1j * xp.asarray(phi + 2.0 * np.pi * dphase, dtype=real_dtype))) for phi in div_phases)
    mods = tuple(xp.fft.ifftshift(xp.asarray(m, dtype=real_dtype)) for m in imgs_modulus)

    logs: List[PDGSLogEntry] = []
    t0 = perf_counter()
    if progress_every is None:
        progress_every = max(1, nit // 20)

    for j in range(1, nit + 1):
        should_log = (j - 1) % every == 0
        new_guess, updates = pdgs_iter(guess, phis, mods, xp, return_updates=should_log)
        mean_step_norm = float(_to_numpy(xp.sqrt(xp.mean(xp.abs(new_guess - guess) ** 2))))

        if should_log:
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
