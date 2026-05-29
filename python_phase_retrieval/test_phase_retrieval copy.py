# ===============================================================
# Beta-Diversity Convergence Test (Interactive, no file output)
#
# Goal:
# Compare convergence when using beta diversity (different angular tilts)
# versus a single-angle setup, keeping the same total number of
# phase-diversity images.
#
# In practice, this test checks whether spreading beta over multiple
# quadrants improves conditioning and convergence speed/quality.
#
# Current observation:
# At parity of dataset size, beta diversity appears to converge better.
#
# Pipeline summary:
# 1) Build a synthetic beam with known phase profile.
# 2) Generate diversity images in the Fourier plane.
# 3) Run one-shot initialization.
# 4) Refine with PDGS (CPU/GPU selectable).
# 5) Plot convergence and phase diagnostics.
#
# Note: this script is for interactive visualization only
# (no CSV/PNG export).
# ===============================================================
#%% Imports
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# Include parent directory into sys.path, allows for the next import
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from python_phase_retrieval.phase_retrieval import one_shot, pdgs_log

#%% Helper functions

def wrap_phase(x: np.ndarray) -> np.ndarray:
    # Wrap any real phase to the principal interval [-pi, pi].
    return np.angle(np.exp(1j * x))


def normalize_field_energy(field: np.ndarray) -> np.ndarray:
    """Normalize a complex field so that sum(|field|^2) = 1."""
    u = np.asarray(field)
    energy = float(np.sum(np.abs(u) ** 2))
    if energy <= 0.0:
        raise ValueError("Field energy must be positive for normalization")
    return u / np.sqrt(energy)

def fourier_dir(v: np.ndarray) -> np.ndarray:
    # Shifted FFT
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(v)))

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
# Physical setup (SI units, SLM-plane coordinates).
lambda_m = 532e-9              # Wavelength [m]
f_m = 0.100                    # Lens focal length [m]
flambda = f_m * lambda_m       # Wavelength × focal-length product [m^2]
slm_pixel_pitch_m = 8.0e-6     # SLM pixel pitch [m]
beam_sigma_on_slm_m = 0.9e-3   # Gaussian sigma on SLM [m]

# Experiment/solver controls.
n = 1080                       # Grid size (n x n)
nit = 500                     # Number of iterations
progress_every = 100           # Print progress every N iterations
use_gpu = True                 # Set False to force NumPy CPU
amp_threshold = 0.15           # Relative threshold on normalized intensity for phase RMSE mask
show_progress = True          # Print iteration logs 

# Phase-diversity parameters
beta_diversity = True  # <-- IL TUO TOGGLE (True per 4 quadranti, False per 1 quadrante)

# Linear term : phi_lin = 2*pi*(u*x + v*y). 
u_nyquist_cpm = 1.0 / (2.0 * slm_pixel_pitch_m)         
shift_x = 0.5 * u_nyquist_cpm
shift_y = 0.5 * u_nyquist_cpm

if beta_diversity:
    # MODO 1 (Tilt-Diversity): 4 curvature x 4 angolazioni = 16 immagini totali
    alphas = np.linspace(20, 80, 4) * 1e6  
    betas = [
        np.array([ 2.0 * np.pi * shift_x,  2.0 * np.pi * shift_y], dtype=float), # Q1 (++)
        np.array([ 2.0 * np.pi * shift_x, -2.0 * np.pi * shift_y], dtype=float), # Q4 (+-)
        np.array([-2.0 * np.pi * shift_x,  2.0 * np.pi * shift_y], dtype=float), # Q2 (-+)
        np.array([-2.0 * np.pi * shift_x, -2.0 * np.pi * shift_y], dtype=float)  # Q3 (--)
    ]
else:
    # MODO 2 (Quadrante Singolo): 16 curvature x 1 angolazione = 16 immagini totali
    alphas = np.linspace(20, 80, 16) * 1e6  
    betas = [
        np.array([ 2.0 * np.pi * shift_x,  2.0 * np.pi * shift_y], dtype=float)  # Solo Q1 (++)
    ]


#%% Build SLM ground truth beam

# Gaussian ampl
coords_1d = (np.arange(n, dtype=float) - n // 2) * slm_pixel_pitch_m
L = (coords_1d, coords_1d)
# Explicit 2D coordinate grids (same indexing convention as previous broadcast form).
x, y = np.meshgrid(coords_1d, coords_1d, indexing="ij")
rr = x**2 + y**2
amp_true = np.exp(-rr / (2.0 * beam_sigma_on_slm_m**2))

# Rectangular Ampl
# rect_height_frac = 0.30
# rect_width_frac = 0.40
# h_box = int(round(n * rect_height_frac))
# w_box = int(round(n * rect_width_frac))
# y0 = (n - h_box) // 2
# x0 = (n - w_box) // 2
# amp_true = np.zeros((n, n), dtype=float)
# amp_true[y0:y0 + h_box, x0:x0 + w_box] = 1.0

# True phase: mixture of low-order Zernike-like modes normalized on SLM reference radius.
# Phase is evaluated on the whole grid (not just pupil)

# Phase to be recovered
zernike_coeffs = {
    "tilt_x": 0.3,      "tilt_y": 0.3,
    "defocus": 0.40,
    "astig_0": 0.2,     "astig_45": -0.0,
    "coma_x": 0.2,      "coma_y": 0.0,
    "trefoil_x": 0.01,  "trefoil_y": 0.0,
}
# Inscribed reference radius on the SLM grid.
slm_ref_radius_m = float(min(np.max(np.abs(L[0])), np.max(np.abs(L[1]))))
aberr = zernike_low_order_mix(x, y, r_ref=slm_ref_radius_m, coeffs=zernike_coeffs)
phase_true = wrap_phase(aberr)

#Complex field at the SLM plane (ground truth to be recovered)
beam_true = amp_true * np.exp(1j * phase_true)

# Phase-metric evaluation mask: threshold relative to max intensity.
intensity_true = amp_true**2
phase_eval_mask = intensity_true > (amp_threshold * float(np.max(intensity_true)))
if not np.any(phase_eval_mask):
    raise ValueError("Phase evaluation mask is empty; lower amp_threshold")
# Enforce unit energy on the input field: sum(|beam_true|^2) = 1.
beam_true = normalize_field_energy(beam_true)
amp_true = np.abs(beam_true)


#%% Generate diversity dataset (imgs_intensity, imgs_modulus, div_phases)
# Diversity images: |FT(beam * exp(i * alpha/2 * r^2 + i * beta * r))|^2
div_phases = []
imgs_intensity = []
imgs_modulus = []

# Dobbiamo tenere traccia della coppia esatta (alpha, beta) usata per ogni immagine,
# ci servirà per dare le giuste coordinate all'inizializzazione one-shot.
used_params = []

for alpha in alphas:
    for beta in betas:
        # Build Diversity Phase
        div = (alpha / 2.0) * rr + beta[0] * x + beta[1] * y
        div_phases.append(div)
        used_params.append((alpha, beta))
        
        # Measure diversity image
        far_field = fourier_dir(beam_true * np.exp(1j * div))
        inten = np.abs(far_field) ** 2
        inten_sum = float(np.sum(inten))
        if inten_sum <= 0.0:
            raise ValueError("Diversity intensity energy must be positive")
        inten = inten / inten_sum           # Normalize each diversity image to sum = 1
        imgs_intensity.append(inten)        # Diversity image
        imgs_modulus.append(np.sqrt(inten)) # Diversity Modulus

if beta_diversity:
    print(f"Dataset: {len(imgs_modulus)} diversity images ({len(alphas)} alphas x {len(betas)} quadranti)")
else:
    print(f"Dataset: {len(imgs_modulus)} diversity images ({len(alphas)} alphas x {len(betas)} quadrante)")

print(f"SLM size = {n}×{n}, pixel pitch = {slm_pixel_pitch_m*1e6:.2f} um")
print(f"lambda = {lambda_m*1e9:.1f} nm, f = {f_m*1e3:.1f} mm")
print(f"alphas [rad/m^2]: {[f'{a:.3e}' for a in alphas]}")
print(f"beta [rad/m]: beta_x={beta[0]:.3e}, beta_y={beta[1]:.3e}")

#%% Inspect dataset: imgs_modulus
#Verify the diversity images look reasonable 
n_preview = min(3, len(imgs_modulus))
fig_mod, ax_mod = plt.subplots(1, n_preview, figsize=(4 * n_preview, 4))
if n_preview == 1:
    ax_mod = [ax_mod]

for i in range(n_preview):
    ax_mod[i].imshow(imgs_intensity[i], cmap="gray", origin="lower")
    ax_mod[i].set_title(f"imgs_intensity[{i}]")
    ax_mod[i].set_xticks([])
    ax_mod[i].set_yticks([])

fig_mod.tight_layout()
plt.show()


#%% One-Shot initialisation
# One-shot gives an initialization for iterative refinement.
# To be run on the last image (strongest diversity) for best conditioning.
last_alpha, last_beta = used_params[-1]

beam_init = one_shot(
    imgs_intensity[-1],
    alpha=last_alpha,
    beta=last_beta,
    L=L,
    flambda=flambda,
)

# Normalise energy
beam_init = normalize_field_energy(beam_init)
# Align global phase of the initial guess to the true phase for a fair RMSE evaluation (remove piston).
beam_init_aligned = align_global_phase(beam_init, phase_true, weights=amp_true)
# 0-th iteration error
ph_init_err = phase_rmse(np.angle(beam_init_aligned), phase_true, mask=phase_eval_mask)
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
    phase_rmse_mask=phase_eval_mask,
    verbose=show_progress,
    progress_every=progress_every,
    use_gpu=use_gpu,
)

print(f"PDGS finished: {len(logs)} logged iterations")

#%% Metrics
# Align global phase of the initial guess to the true phase for a fair RMSE evaluation (remove piston).
beam_est_aligned = align_global_phase(beam_est, phase_true, weights=amp_true) 
#Normalise Energy of the final estimate (should be close to 1 but just in case).
beam_est_aligned = normalize_field_energy(beam_est_aligned)

phase_est = np.angle(beam_est_aligned) # Estimated phase (wrapped to [-pi, pi])
amp_est = np.abs(beam_est_aligned)     # Estimated amplitude 

# Root mean square error on amplitude
amp_err = float(np.sqrt(np.mean((amp_est - amp_true) ** 2)))
# Root mean square error on phase (only where amplitude is relevant)
ph_err = phase_rmse(phase_est, phase_true, mask=phase_eval_mask)

print(f"Final amplitude RMSE    : {amp_err:.6f}")
print(f"Final phase RMSE (masked): {ph_err:.6f} rad  (one-shot was {ph_init_err:.4f} rad)")


#%% Convergence metrics (separate linear plots)
# Plot each metric on its own axis because their characteristic scales differ.

iters = np.array([x.iteration for x in logs], dtype=int)
e_sc  = np.array([x.self_consistency_error for x in logs], dtype=float)
e_upd = np.array([x.mean_update_norm for x in logs], dtype=float)
e_gt  = np.array([x.ground_truth_phase_rmse for x in logs], dtype=float)
has_gt = np.any(np.isfinite(e_gt))
nrows = 3 if has_gt else 2
fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.2 * nrows), sharex=True)
if nrows == 2:
    ax_sc, ax_upd = axes
    ax_gt = None
else:
    ax_sc, ax_upd, ax_gt = axes

ax_sc.plot(iters, e_sc, color="tab:blue")
ax_sc.set_ylabel("MSE")
ax_sc.set_title("Self-consistency error")
ax_sc.grid(True, alpha=0.3)

ax_upd.plot(iters, e_upd, color="tab:orange")
ax_upd.set_ylabel("RMS")
ax_upd.set_title("Mean update norm")
ax_upd.grid(True, alpha=0.3)

if has_gt and ax_gt is not None:
    ax_gt.plot(iters, e_gt, color="tab:green")
    ax_gt.set_ylabel("rad")
    ax_gt.set_title("Ground-truth phase RMSE")
    ax_gt.grid(True, alpha=0.3)
    ax_gt.set_xlabel("iteration")
else:
    ax_upd.set_xlabel("iteration")

fig.tight_layout()
plt.show()

#%% Phase diagnostics plot
# Visual-only phase comparison.

fig2, ax = plt.subplots(1, 3, figsize=(13, 4))

im0 = ax[0].imshow(phase_true, cmap="twilight", origin="lower", vmin=-np.pi, vmax=np.pi)
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

#%% Amplitude diagnostics plot
# Visual comparison between true and reconstructed amplitude.
fig3, ax3 = plt.subplots(1, 3, figsize=(13, 4))

im_a0 = ax3[0].imshow(amp_true, cmap="gray", origin="lower")
ax3[0].set_title("True amplitude")
plt.colorbar(im_a0, ax=ax3[0], fraction=0.046, pad=0.04)

im_a1 = ax3[1].imshow(amp_est, cmap="gray", origin="lower")
ax3[1].set_title("Estimated amplitude")
plt.colorbar(im_a1, ax=ax3[1], fraction=0.046, pad=0.04)

amp_diff = amp_est - amp_true
vmax = np.max(np.abs(amp_diff)) if np.max(np.abs(amp_diff)) > 0 else 1.0
im_a2 = ax3[2].imshow(amp_diff, cmap="coolwarm", origin="lower", vmin=-vmax, vmax=vmax)
ax3[2].set_title("Amplitude error (est - true)")
plt.colorbar(im_a2, ax=ax3[2], fraction=0.046, pad=0.04)

for a in ax3:
    a.set_xticks([])
    a.set_yticks([])

fig3.suptitle(f"Amplitude RMSE: {amp_err:.6f}", fontsize=11)
fig3.tight_layout()
plt.show()