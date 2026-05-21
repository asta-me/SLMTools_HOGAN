# ===============================================================
# Phase Retrieval Demo (Interactive, no file output)
#
# What this script does:
# 1) Builds a synthetic beam with known phase aberration.
# 2) Generates multiple diversity images in the Fourier plane.
# 3) Runs one-shot initialization.
# 4) Refines with PDGS (CPU/GPU selectable).
# 5) Shows convergence and phase diagnostic plots.
#
# Notes:
# - Parameters "*_at_waist" are in radians evaluated at beam waist radius w0.
# - beta is the linear phase term (tilt/blaze), useful when modeling off-axis order.
# - Zernike-like modes are normalized on the inscribed pupil radius.
# - Phase is kept continuous on the whole grid (no zeroing outside the pupil).
# - This script only visualizes results (no CSV/PNG saving).
# ===============================================================
#%% Imports
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from python_phase_retrieval.lattice_utils import ldot, r2, sft
from python_phase_retrieval.phase_retrieval import one_shot, pdgs_log

#%% Helper functions

def wrap_phase(x: np.ndarray) -> np.ndarray:
    # Wrap any real phase to the principal interval [-pi, pi].
    return np.angle(np.exp(1j * x))

def phase_rmse(est: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> float:
    # Compute wrapped phase RMSE, optionally restricted to a mask.
    diff = np.angle(np.exp(1j * (est - ref)))
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(np.mean(diff**2)))

def align_global_phase(
    est_complex: np.ndarray,
    ref_phase: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    # Remove global piston by matching the estimate's mean complex phase to the reference.
    ref_complex = np.exp(1j * ref_phase)
    if weights is None:
        c = np.mean(est_complex * np.conj(ref_complex))
    else:
        c = np.sum(weights * est_complex * np.conj(ref_complex)) / np.sum(weights)
    return est_complex * np.exp(-1j * np.angle(c))

def zernike_low_order_mix(x: np.ndarray, y: np.ndarray, r_ref: float, coeffs: dict[str, float]) -> np.ndarray:
    """Return a small low-order Zernike-like phase mix evaluated on a Cartesian grid.

    Coefficients are in radians at rho=1, where rho = sqrt(x^2+y^2)/r_ref.
    Here r_ref should be the inscribed pupil radius. The phase is still evaluated
    on the entire grid to keep continuity outside the pupil.
    """
    # Build rho/theta coordinates and sum selected low-order real Zernike-like modes.
    rho = np.sqrt(x**2 + y**2) / r_ref
    theta = np.arctan2(y, x)

    modes = {
        # Noll-like low orders (real-valued forms).
        "tilt_x": 2.0 * rho * np.cos(theta),
        "tilt_y": 2.0 * rho * np.sin(theta),
        "defocus": np.sqrt(3.0) * (2.0 * rho**2 - 1.0),
        "astig_0": np.sqrt(6.0) * rho**2 * np.cos(2.0 * theta),
        "astig_45": np.sqrt(6.0) * rho**2 * np.sin(2.0 * theta),
        "coma_x": np.sqrt(8.0) * (3.0 * rho**3 - 2.0 * rho) * np.cos(theta),
        "coma_y": np.sqrt(8.0) * (3.0 * rho**3 - 2.0 * rho) * np.sin(theta),
        "trefoil_x": np.sqrt(8.0) * rho**3 * np.cos(3.0 * theta),
        "trefoil_y": np.sqrt(8.0) * rho**3 * np.sin(3.0 * theta),
    }

    out = np.zeros((x.shape[0], y.shape[1]), dtype=float)
    for name, value in coeffs.items():
        if name not in modes:
            raise ValueError(f"Unknown Zernike mode: {name}")
        out += float(value) * modes[name]
    return out

#%% Dataset parameters
# Experiment/solver controls.
n = 1080                       # Grid size (n x n)
nit = 300                      # Number of iterations
beta = (0.0, 0.0)              # Linear phase ramp [rad/m] in SLM plane
progress_every = 25   # Print progress every N iterations
use_gpu = True        # Set False to force NumPy CPU
amp_threshold = 0.15    # Threshold for phase RMSE metric (only where amplitude is relevant)
show_progress = True

# Physical setup (SI units, SLM-plane coordinates).
lambda_m = 780e-9              # Wavelength [m]
f_m = 0.300                    # Lens focal length [m]
flambda = f_m * lambda_m       # Wavelength × focal-length product [m^2]
slm_pixel_pitch_m = 8.0e-6     # SLM pixel pitch [m]
beam_sigma_on_slm_m = 0.9e-3   # Gaussian sigma on SLM [m]



# All phase values below are defined on the SLM plane.
# diversity_phase_at_slm_radius[i] = 0.5 * alpha_i * r_ref^2
# where r_ref is the inscribed SLM reference radius.
# Example low-order phase content in rad at rho=1 (rho = r/r_ref).
# Keep values small to emulate mild but realistic aberrations.
zernike_coeffs = {
    "tilt_x": 0.0,
    "tilt_y": 0.0,
    "defocus": 0.00,
    "astig_0": 0.05,
    "astig_45": -0.0,
    "coma_x": 0.01,
    "coma_y": 0.0,
    "trefoil_x": 0.01,
    "trefoil_y": 0.0,
}
diversity_phase_at_slm_radius = [1.0, 1.6, 2.4]  # [rad] at r = SLM reference radius

#%% Build synthetic dataset
# Build a Gaussian amplitude and known astigmatic phase on the SLM lattice.
coords_1d = (np.arange(n, dtype=float) - n // 2) * slm_pixel_pitch_m
L = (coords_1d, coords_1d)
rr = r2(L)           # r^2 array on the lattice
x = L[0].reshape(-1, 1)
y = L[1].reshape(1, -1)

# Inscribed reference radius on the SLM grid.
slm_ref_radius_m = float(min(np.max(np.abs(L[0])), np.max(np.abs(L[1]))))
amp_true = np.exp(-rr / (2.0 * beam_sigma_on_slm_m**2))

# Convert diversity phase from [rad at SLM reference radius] to alpha [rad/m^2].
alphas = [2.0 * d / slm_ref_radius_m**2 for d in diversity_phase_at_slm_radius]

# True phase: mixture of low-order Zernike-like modes normalized on SLM reference radius.
# Phase is evaluated on the whole grid (not just pupil)
aberr = zernike_low_order_mix(x, y, r_ref=slm_ref_radius_m, coeffs=zernike_coeffs)
phase_true = wrap_phase(aberr)
beam_true = amp_true * np.exp(1j * phase_true)
rho_pupil = np.sqrt(x**2 + y**2) / slm_ref_radius_m
inside_pupil = rho_pupil <= 1.0
power_total = float(np.sum(amp_true**2))
power_inside = float(np.sum((amp_true**2)[inside_pupil]))
power_inside_frac = power_inside / power_total if power_total > 0 else float("nan")

# Diversity images: |FT(beam * exp(i * alpha/2 * r^2))|^2  (camera / Fourier plane)
div_phases = []
imgs_intensity = []
imgs_modulus = []
for a, d in zip(alphas, diversity_phase_at_slm_radius):
    div = 0.5 * a * rr + ldot(beta, L)
    div_phases.append(div)
    far_field = sft(beam_true * np.exp(1j * div))
    inten = np.abs(far_field) ** 2
    imgs_intensity.append(inten)
    imgs_modulus.append(np.sqrt(inten))

print(f"Dataset: {len(alphas)} diversity images, grid {n}×{n}")
print(f"SLM size = {n}×{n}, pixel pitch = {slm_pixel_pitch_m*1e6:.2f} um")
print(f"lambda = {lambda_m*1e9:.1f} nm, f = {f_m*1e3:.1f} mm, flambda = {flambda:.3e} m^2")
print(f"SLM reference radius = {slm_ref_radius_m*1e3:.3f} mm, beam sigma = {beam_sigma_on_slm_m*1e3:.3f} mm")
print(f"Beam power inside inscribed pupil: {100.0 * power_inside_frac:.2f}%")
print(f"# zernike modes = {len(zernike_coeffs)}")
print(f"Zernike coeffs [rad @ rho=1]: {zernike_coeffs}")
print(f"Diversity phase [rad @ SLM ref radius]: {diversity_phase_at_slm_radius}  →  alphas [rad/m^2]: {[f'{a:.3e}' for a in alphas]}")
print(f"Beam peak: {amp_true.max():.3f},  phase range: [{phase_true.min():.2f}, {phase_true.max():.2f}] rad")


#%% One-Shot initialisation
# One-shot gives a fast initial guess before iterative refinement.

beam_init = one_shot(
    imgs_intensity[-1],
    alpha=alphas[-1],
    beta=beta,
    L=L,
    flambda=flambda,
)

beam_init_aligned = align_global_phase(beam_init, phase_true, weights=amp_true)
ph_init_err = phase_rmse(np.angle(beam_init_aligned), phase_true, mask=amp_true > amp_threshold)
print(f"One-Shot initial phase RMSE: {ph_init_err:.4f} rad")

#%% PDGS iterative refinement
# Main solver: enforces modulus constraints from all diversity images.

beam_est, logs = pdgs_log(
    imgs_modulus=imgs_modulus,
    div_phases=div_phases,
    nit=nit,
    beam_guess=beam_init,
    L=L,
    flambda=flambda,
    every=1,
    phase_truth=phase_true,
    phase_rmse_mask=amp_true > amp_threshold,
    verbose=show_progress,
    progress_every=progress_every,
    use_gpu=use_gpu,
)

print(f"PDGS finished: {len(logs)} logged iterations")

#%% Metrics
# Compare reconstructed amplitude/phase to synthetic ground truth.
beam_est_aligned = align_global_phase(beam_est, phase_true, weights=amp_true) # Remove global piston before comparison.
phase_est = np.angle(beam_est_aligned)
amp_est = np.abs(beam_est_aligned)

# Root mean square error on amplitude
amp_err = float(np.sqrt(np.mean((amp_est - amp_true) ** 2)))
# Root mean square error on phase (only where amplitude is relevant)
ph_err = phase_rmse(phase_est, phase_true, mask=amp_true > amp_threshold)

print(f"Final amplitude RMSE    : {amp_err:.6f}")
print(f"Final phase RMSE (masked): {ph_err:.6f} rad  (one-shot was {ph_init_err:.4f} rad)")


#%% Convergence plot
# Visual-only convergence diagnostics.

iters = np.array([x.iteration for x in logs], dtype=int)
e_sc  = np.array([x.self_consistency_error for x in logs], dtype=float)
e_upd = np.array([x.mean_update_norm for x in logs], dtype=float)
e_gt  = np.array([x.ground_truth_phase_rmse for x in logs], dtype=float)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(iters, e_sc,  label="self_consistency_error")
axes[0].plot(iters, e_upd, label="mean_update_norm")
if np.any(np.isfinite(e_gt)):
    axes[0].plot(iters, e_gt, label="gt_phase_rmse [rad]")
axes[0].set_xlabel("iteration")
axes[0].set_ylabel("error")
axes[0].set_title("Convergence — linear scale")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

eps = 1e-14
axes[1].semilogy(iters, np.maximum(e_sc, eps),  label="self_consistency_error")
axes[1].semilogy(iters, np.maximum(e_upd, eps), label="mean_update_norm")
if np.any(np.isfinite(e_gt)):
    axes[1].semilogy(iters, np.maximum(np.nan_to_num(e_gt, nan=np.inf), eps), label="gt_phase_rmse [rad]")
axes[1].set_xlabel("iteration")
axes[1].set_ylabel("error")
axes[1].set_title("Convergence — log scale")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
plt.show()


#%% Phase diagnostics plot
# Visual-only phase comparison.

fig2, ax = plt.subplots(1, 3, figsize=(13, 4))

im0 = ax[0].imshow(phase_true, cmap="twilight", origin="lower")
ax[0].set_title("True phase")
plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

im1 = ax[1].imshow(phase_est, cmap="twilight", origin="lower")
ax[1].set_title("Estimated phase")
plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

phase_diff = np.angle(np.exp(1j * (phase_est - phase_true)))
im2 = ax[2].imshow(phase_diff, cmap="coolwarm", origin="lower", vmin=-np.pi, vmax=np.pi)
ax[2].set_title("Wrapped phase error")
plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

for a in ax:
    a.set_xticks([])
    a.set_yticks([])

fig2.suptitle(
    f"Phase RMSE: init = {ph_init_err:.4f} rad  →  final = {ph_err:.4f} rad",
    fontsize=11,
)
fig2.tight_layout()
plt.show()
