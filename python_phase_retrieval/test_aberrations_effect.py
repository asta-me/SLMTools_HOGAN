#!/usr/bin/env python3
# =============================================================================
#  test_aberrations_effect.py
#  ---------------------------------------------------------------------------
#  GS hologram synthesis and reconstruction comparison using simulated beams.
#
#  Workflow:
#   1) Run GS with uniform input amplitude.
#   2) Run GS with Gaussian input amplitude.
#   3) For each GS phase, reconstruct with:
#      - uniform amplitude
#      - Gaussian amplitude
#      - Gaussian amplitude + Zernike aberration
#
#  Dependencies: numpy, matplotlib, scikit-image, cupy (GPU accelerated)
# =============================================================================

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize

try:
    import cupy as cp
    gpu_available = True
except ImportError:
    gpu_available = False
    print("Warning: CuPy not available, will use NumPy (slower)")

# ─────────────────────────────────────────────────────────────────────────────
# User parameters  –  edit here
# ─────────────────────────────────────────────────────────────────────────────

# Target image to load (grayscale or colour; will be converted and resized).
# Same loading convention as 04_test_correction_gs.
input_image_path = r"C:\Users\astam\Desktop\Target_Imgs\Siemens_star_2160_centered.tif"
invert_target    = True    # Set True if the TIFF appears contrast-inverted

res              = 1080    # SLM side length in pixels (aperture) – FULL RESOLUTION
m                = 1080    # Signal region (2× aperture for GS constraints)
gs_iters         = 300     # Gerchberg-Saxton iterations for hologram synthesis
random_seed      = 42      # Reproducible GS initialisation
mixing_parameter = 0.5     # SR/NR energy mixing (from reference 04_test_correction_gs.py)

# ── Gaussian beam (Cases 2 & 3) ──────────────────────────────────────────────
gaussian_sigma_frac = 0.35   # σ as fraction of aperture half-width
                              # (normalised ±1 grid)  →  e.g. 0.35 ≈ 70% fill

# ── Zernike aberration coefficients (Case 3) ─────────────────────────────────
# Each entry:  "<name>": <coefficient in radians (RMS)>
# Polynomials follow the Noll/Born&Wolf convention on the unit disk.
# Set a coefficient to 0.0 to exclude that term.
zernike_coeffs = {
    "defocus"   : 1.5,    # Z4  :  2ρ²−1
    "astig_0"   : 0.8,    # Z5  :  ρ²·cos 2θ    (0° astigmatism)
    "astig_45"  : 0.4,    # Z6  :  ρ²·sin 2θ   (45° astigmatism)
    "coma_x"    : 0.6,    # Z7  :  ρ(3ρ²−2)·cos θ
    "coma_y"    : 0.3,    # Z8  :  ρ(3ρ²−2)·sin θ
    "spherical" : 0.5,    # Z11 :  6ρ⁴−6ρ²+1
}

# ── Display ───────────────────────────────────────────────────────────────────
colormap = "inferno"

# ── PSF oversampling / visualization ─────────────────────────────────────────
# Large zero-padding gives finer sampling in the focal plane.
psf_pad_factor = 6       # output grid = (pad_factor * res)^2
psf_crop_size = 300      # central crop shown for PSF visualization
psf_log_compression = 120.0  # higher -> more central-lobe emphasis in log view

# ─────────────────────────────────────────────────────────────────────────────
# Helper: image loading and normalization
# ─────────────────────────────────────────────────────────────────────────────

def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert image to single channel float32."""
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        return np.mean(arr.astype(np.float32), axis=2)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def _normalize01(arr: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Normalize to [0,1] and optionally clamp a minimum floor."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - float(np.min(arr))
    arr = arr / (float(np.max(arr)) + 1e-12)
    if floor > 0.0:
        arr = np.maximum(arr, floor)
    return arr


def load_target(path: str | Path, res: int, invert: bool = False) -> np.ndarray:
    """
    Load an image, convert to grayscale float32, resize to (res, res),
    and normalise to [0, 1].  If *invert* is True the contrast is flipped
    (useful when the TIFF encodes a dark target on a bright background).
    """
    img = _to_grayscale(imread(str(path)))
    if invert:
        img = np.max(img) - img
    img = resize(img, (res, res), order=1, preserve_range=True, anti_aliasing=True)
    img = _normalize01(img, floor=1e-3)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Helper: GS hologram synthesis with GPU acceleration (CuPy) + SR/NR constraints
# ─────────────────────────────────────────────────────────────────────────────

def sinc_interpol(image):
    """2x sinc-like interpolation via FFT zero-padding (GPU-compatible)."""
    h, w = image.shape
    if gpu_available:
        spec = cp.fft.fftshift(cp.fft.fft2(image))
        spec_up = cp.zeros((2 * h, 2 * w), dtype=cp.complex128)
        spec_up[h // 2:h // 2 + h, w // 2:w // 2 + w] = spec
        up = cp.fft.ifft2(cp.fft.ifftshift(spec_up))
        up = cp.maximum(cp.real(up), 0.0)
        in_energy = cp.sum(image)
        out_energy = cp.sum(up)
        if out_energy > 0:
            up *= in_energy / out_energy
        return up.astype(cp.float32)
    else:
        spec = np.fft.fftshift(np.fft.fft2(image))
        spec_up = np.zeros((2 * h, 2 * w), dtype=np.complex128)
        spec_up[h // 2:h // 2 + h, w // 2:w // 2 + w] = spec
        up = np.fft.ifft2(np.fft.ifftshift(spec_up))
        up = np.maximum(np.real(up), 0.0)
        in_energy = np.sum(image)
        out_energy = np.sum(up)
        if out_energy > 0:
            up *= in_energy / out_energy
        return up.astype(np.float32)


def run_gs_gpu(
    input_amp_slm,
    target_intensity_slm,
    n_iters: int,
    m_signal: int,
    mixing: float,
):
    """
    Original AP/GS loop style with SR/NR amplitude constraints (GPU-accelerated).
    Uses CuPy if available, falls back to NumPy.
    
    Returns (phase_slm, rmse_hist, recon_intensity_padded)
    """
    if gpu_available:
        return _run_gs_cupy(input_amp_slm, target_intensity_slm, n_iters, m_signal, mixing)
    else:
        return _run_gs_numpy(input_amp_slm, target_intensity_slm, n_iters, m_signal, mixing)


def _run_gs_cupy(input_amp_slm, target_intensity_slm, n_iters, m_signal, mixing):
    """GS with SR/NR constraints using CuPy (GPU)."""
    n = input_amp_slm.shape[0]
    nn = 2 * n
    
    # Convert to CuPy
    target_cp = cp.asarray(target_intensity_slm, dtype=cp.float32)
    input_amp_cp = cp.asarray(input_amp_slm, dtype=cp.float32)
    
    # Target is moved to 2N computational grid using FFT zero-padding
    target_work = sinc_interpol(target_cp)
    target_amp_work = cp.sqrt(target_work)
    
    active_rows = slice(n // 2, n // 2 + n)
    active_cols = slice(n // 2, n // 2 + n)
    
    # SR/NR masks: enforce target in SR, keep NR free with controlled energy
    bandlim_in = cp.zeros((nn, nn), dtype=cp.float32)
    sr_r0, sr_r1 = (nn - m_signal) // 2, (nn + m_signal) // 2
    sr_c0, sr_c1 = (nn - m_signal) // 2, (nn + m_signal) // 2
    bandlim_in[sr_r0:sr_r1, sr_c0:sr_c1] = 1.0
    bandlim_ou = 1.0 - bandlim_in
    
    sr_idx = cp.where(bandlim_in > 0)
    target_sr = target_work[sr_idx]
    
    # Embed measured input amplitude in active SLM window
    incident = cp.zeros((nn, nn), dtype=cp.float32)
    incident[active_rows, active_cols] = input_amp_cp
    
    E = cp.sum(target_intensity_slm)
    El = mixing * E
    
    phi = cp.exp(1j * 2 * cp.pi * cp.random.rand(nn, nn))
    amp = cp.random.rand(nn, nn).astype(cp.float32)
    
    rmse_hist = []
    E2_k = None
    
    for i in range(n_iters):
        # Forward step with SR/NR amplitude constraints in target plane
        amp = bandlim_in * target_amp_work + bandlim_ou * amp
        E1 = amp * phi
        E2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(E1)))
        
        # SLM-plane amplitude constraint from measured input beam
        E2_ave = cp.sqrt((E + El) * incident**2 / (cp.sum(incident**2) + 1e-12))
        E2_k = E2_ave * cp.exp(1j * cp.angle(E2))
        es = cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(E2_k)))
        
        amp_curr = cp.abs(es)
        amp_in = bandlim_in * amp_curr
        amp_ou = bandlim_ou * amp_curr
        
        norm_in = cp.sqrt(E)
        norm_ou = cp.sqrt(El)
        amp = (
            norm_in * (amp_in / (cp.sqrt(cp.sum(amp_in**2)) + 1e-12))
            + norm_ou * (amp_ou / (cp.sqrt(cp.sum(amp_ou**2)) + 1e-12))
        )
        
        # Convergence metric on SR support
        I_full = amp**2
        I_full = E * I_full / (cp.sum(I_full) + 1e-12)
        rmse = cp.sqrt(cp.mean((I_full[sr_idx] - target_sr) ** 2))
        rmse_hist.append(float(rmse.get()))
        
        phi = cp.exp(1j * cp.angle(es))
        
        if (i + 1) % 50 == 0:
            print(f"  GS iter {i + 1:4d}/{n_iters}  RMSE={rmse_hist[-1]:.5f}")
    
    if E2_k is None:
        raise RuntimeError("GS loop did not run")
    
    phase_slm = cp.mod(cp.angle(E2_k), 2.0 * cp.pi)
    phase_active = phase_slm[active_rows, active_cols]
    
    hologram = incident * cp.exp(1j * cp.angle(E2_k))
    rec = cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(hologram)))
    recon_intensity = cp.abs(rec) ** 2
    recon_full = recon_intensity / (cp.max(recon_intensity) + 1e-12)
    
    return phase_active, rmse_hist, recon_full


def _run_gs_numpy(input_amp_slm, target_intensity_slm, n_iters, m_signal, mixing):
    """GS with SR/NR constraints using NumPy (fallback)."""
    n = input_amp_slm.shape[0]
    nn = 2 * n
    
    # Match CuPy path: 2x interpolation via FFT zero-padding
    target_work = sinc_interpol(target_intensity_slm)
    target_amp_work = np.sqrt(target_work)
    
    active_rows = slice(n // 2, n // 2 + n)
    active_cols = slice(n // 2, n // 2 + n)
    
    # SR/NR masks
    bandlim_in = np.zeros((nn, nn), dtype=np.float32)
    sr_r0, sr_r1 = (nn - m_signal) // 2, (nn + m_signal) // 2
    sr_c0, sr_c1 = (nn - m_signal) // 2, (nn + m_signal) // 2
    bandlim_in[sr_r0:sr_r1, sr_c0:sr_c1] = 1.0
    bandlim_ou = 1.0 - bandlim_in
    sr_idx = np.where(bandlim_in > 0)
    target_sr = target_work[sr_idx]
    
    # Embed input amplitude
    incident = np.zeros((nn, nn), dtype=np.float32)
    incident[active_rows, active_cols] = input_amp_slm
    
    E = np.sum(target_intensity_slm)
    El = mixing * E
    
    phi = np.exp(1j * 2 * np.pi * np.random.rand(nn, nn))
    amp = np.random.rand(nn, nn).astype(np.float32)
    
    rmse_hist = []
    
    for i in range(n_iters):
        # Forward step
        amp = bandlim_in * target_amp_work + bandlim_ou * amp
        E1 = amp * phi
        E2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E1)))
        
        # SLM-plane constraint
        E2_ave = np.sqrt((E + El) * incident**2 / (np.sum(incident**2) + 1e-12))
        E2_k = E2_ave * np.exp(1j * np.angle(E2))
        es = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E2_k)))
        
        amp_curr = np.abs(es)
        amp_in = bandlim_in * amp_curr
        amp_ou = bandlim_ou * amp_curr
        
        norm_in = np.sqrt(E)
        norm_ou = np.sqrt(El)
        amp = (
            norm_in * (amp_in / (np.sqrt(np.sum(amp_in**2)) + 1e-12))
            + norm_ou * (amp_ou / (np.sqrt(np.sum(amp_ou**2)) + 1e-12))
        )
        
        I_full = amp**2
        I_full = E * I_full / (np.sum(I_full) + 1e-12)
        rmse = np.sqrt(np.mean((I_full[sr_idx] - target_sr) ** 2))
        rmse_hist.append(float(rmse))
        
        phi = np.exp(1j * np.angle(es))
        
        if (i + 1) % 50 == 0:
            print(f"  GS iter {i + 1:4d}/{n_iters}  RMSE={rmse_hist[-1]:.5f}")
    
    phase_slm = np.mod(np.angle(E2_k), 2.0 * np.pi)
    phase_active = phase_slm[active_rows, active_cols]
    
    hologram = incident * np.exp(1j * np.angle(E2_k))
    rec = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(hologram)))
    recon_intensity = np.abs(rec) ** 2
    recon_full = recon_intensity / (np.max(recon_intensity) + 1e-12)
    
    return phase_active, rmse_hist, recon_full


# ─────────────────────────────────────────────────────────────────────────────
# Helper: pupil amplitude and wavefront functions
# ─────────────────────────────────────────────────────────────────────────────

def circular_pupil_mask(res: int) -> np.ndarray:
    """Binary circular pupil mask on a normalized [-1,1] grid."""
    y, x = np.mgrid[-1.0:1.0:res * 1j, -1.0:1.0:res * 1j]
    return (x**2 + y**2 <= 1.0).astype(np.float32)


def gaussian_amplitude(res: int, sigma_frac: float) -> np.ndarray:
    """
    Circularly symmetric Gaussian amplitude over the SLM aperture.

    *sigma_frac* is the standard deviation expressed as a fraction of the
    aperture half-width in the normalised ±1 coordinate system.
    The peak is normalised to 1.
    """
    y, x  = np.mgrid[-1.0:1.0:res * 1j, -1.0:1.0:res * 1j]
    sigma = sigma_frac * 2.0    # half-width of the normalised grid = 1
    return np.exp(-(x**2 + y**2) / (2.0 * sigma**2)).astype(np.float32)


def zernike_wavefront(res: int, coeffs: dict) -> np.ndarray:
    """
    Build a wavefront map as a weighted sum of low-order Zernike polynomials.

    Polynomials are evaluated on the unit disk (ρ ≤ 1); the wavefront is
    zero outside the disk.  Coefficient values are in radians.

    Supported keys in *coeffs*
    ──────────────────────────
        "defocus"   → Z4  :  2ρ²−1
        "astig_0"   → Z5  :  ρ²·cos 2θ      (0°  astigmatism)
        "astig_45"  → Z6  :  ρ²·sin 2θ     (45°  astigmatism)
        "coma_x"    → Z7  :  ρ(3ρ²−2)·cos θ
        "coma_y"    → Z8  :  ρ(3ρ²−2)·sin θ
        "spherical" → Z11 :  6ρ⁴−6ρ²+1

    Returns a float32 array of shape (res, res) in radians.
    """
    y, x  = np.mgrid[-1.0:1.0:res * 1j, -1.0:1.0:res * 1j]
    rho   = np.hypot(x, y)
    theta = np.arctan2(y, x)
    mask  = rho <= 1.0    # unit-disk aperture

    polys = {
        "defocus"   : lambda: 2.0 * rho**2 - 1.0,
        "astig_0"   : lambda: rho**2 * np.cos(2.0 * theta),
        "astig_45"  : lambda: rho**2 * np.sin(2.0 * theta),
        "coma_x"    : lambda: rho * (3.0 * rho**2 - 2.0) * np.cos(theta),
        "coma_y"    : lambda: rho * (3.0 * rho**2 - 2.0) * np.sin(theta),
        "spherical" : lambda: 6.0 * rho**4 - 6.0 * rho**2 + 1.0,
    }

    wavefront = np.zeros((res, res), dtype=np.float64)
    for name, value in coeffs.items():
        if value == 0.0:
            continue
        if name not in polys:
            raise ValueError(
                f"Unknown Zernike term '{name}'. Valid keys: {list(polys.keys())}"
            )
        wavefront += value * polys[name]()

    wavefront[~mask] = 0.0    # zero outside the pupil disk
    return wavefront.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: PSF and reconstruction computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_psf(
    amp: np.ndarray,
    wavefront: np.ndarray | None = None,
    pad_factor: int = 2,
) -> np.ndarray:
    """
    Compute the output-plane intensity (PSF) for a given pupil field.

    PSF = |FFT( A · exp(i·W) )|²

    The pupil is zero-padded by *pad_factor* before the FFT.
    No hologram phase is included – this is the bare diffraction spot of
    the illumination field itself.

    Parameters
    ----------
    amp       : pupil amplitude, shape (res, res).
    wavefront : wavefront error in radians, shape (res, res), or None (flat).

    Returns
    -------
    psf : output-plane |FFT|², shape (pad_factor·res, pad_factor·res), float32.
    """
    res = amp.shape[0]
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1")
    nn  = pad_factor * res
    s   = (nn - res) // 2

    w     = wavefront if wavefront is not None else np.zeros_like(amp)
    field = amp * np.exp(1j * w.astype(np.float64))

    padded = np.zeros((nn, nn), dtype=np.complex128)
    padded[s:s + res, s:s + res] = field

    out = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    return (np.abs(out)**2).astype(np.float32)


def center_crop(arr: np.ndarray, crop_size: int) -> np.ndarray:
    """Return square central crop of size crop_size x crop_size."""
    h, w = arr.shape
    size = min(crop_size, h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return arr[y0:y0 + size, x0:x0 + size]


def normalize_log_view(intensity: np.ndarray, compression: float) -> np.ndarray:
    """Log-compress intensity for visibility of PSF features."""
    norm = intensity / (float(np.max(intensity)) + 1e-12)
    return np.log1p(compression * norm) / np.log1p(compression)


def center_line_profile(intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return central horizontal line profile with centered pixel coordinates."""
    h, w = intensity.shape
    cy = h // 2
    line = intensity[cy, :].astype(np.float64)
    line_norm = line / (float(np.max(line)) + 1e-12)
    x = np.arange(w, dtype=np.int32) - (w // 2)
    return x, line_norm


def reconstruct_hologram(
    slm_phase: np.ndarray,
    amp: np.ndarray,
    extra_wavefront: np.ndarray | None = None,
) -> np.ndarray:
    """
    Simulate the holographic reconstruction:  |FFT( amp · exp(i·(phase+W)) )|².
    Uses the same 2× zero-padding as compute_psf.
    This is shown for reference only – it includes the hologram phase.
    """
    if extra_wavefront is None:
        phase_eff = slm_phase
    else:
        phase_eff = slm_phase + extra_wavefront
    return compute_psf(amp, wavefront=phase_eff)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: display utilities
# ─────────────────────────────────────────────────────────────────────────────

def plot_image(
    intensity : np.ndarray,
    title     : str,
    cmap      : str   = "inferno",
    fig_size  : tuple = (6, 5),
) -> None:
    """Display a single image in its own figure window."""
    zoomed_norm = intensity / (intensity.max() + 1e-12)

    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(
        zoomed_norm,
        cmap          = cmap,
        vmin          = 0,
        vmax          = 1,
        interpolation = "nearest",
        origin        = "upper",
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()


def plot_amplitude_phase(amplitude: np.ndarray, phase: np.ndarray, title_prefix: str = "") -> None:
    """Display amplitude and phase side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Amplitude
    amp_norm = amplitude / (amplitude.max() + 1e-12)
    im1 = axes[0].imshow(amp_norm, cmap="gray", origin="upper", vmin=0, vmax=1)
    axes[0].set_title(f"{title_prefix} Amplitude", fontsize=11)
    axes[0].axis("off")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Phase
    phase_mod = np.mod(phase, 2.0 * np.pi)
    im2 = axes[1].imshow(phase_mod, cmap="hsv", origin="upper", vmin=0, vmax=2*np.pi)
    axes[1].set_title(f"{title_prefix} Phase", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label="Phase [rad]")
    
    fig.suptitle(f"{title_prefix} Amplitude & Phase", fontsize=12)
    fig.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.perf_counter()
    
    rng = np.random.default_rng(random_seed)
    if gpu_available:
        cp.random.seed(random_seed)

    # ── 1. Load target image at FULL resolution ──────────────────────────────
    print(f"Loading target from:\n  {input_image_path}")
    target = load_target(input_image_path, res, invert=invert_target)
    print(f"  → resized to {target.shape},  range [{target.min():.3f}, {target.max():.3f}]")

    # ── 2. Build simulated amplitude maps on circular pupil ─────────────────
    pupil_mask = circular_pupil_mask(res)
    amp_uniform = pupil_mask.astype(np.float32)
    amp_gaussian = gaussian_amplitude(res, gaussian_sigma_frac) * pupil_mask

    # ── 3. Build wavefront map for aberrated reconstructions ────────────────

    # Case 3 – Zernike wavefront error (sum of low-order terms)
    aberration = zernike_wavefront(res, zernike_coeffs) * pupil_mask
    pv_rad     = float(aberration.max() - aberration.min())   # peak-to-valley
    rms_rad    = float(aberration.std())                      # RMS across full array

    print(f"\nZernike wavefront:  P-V = {pv_rad:.2f} rad,  RMS = {rms_rad:.2f} rad")
    active_terms = {k: v for k, v in zernike_coeffs.items() if v != 0.0}
    for name, coeff in active_terms.items():
        print(f"  {name:<12s}  {coeff:+.3f} rad")

    # ── 4. Initial overview: amplitudes and aberration used ─────────────────
    fig0, axes0 = plt.subplots(1, 3, figsize=(14, 4.5))

    axes0[0].imshow(amp_uniform, cmap="gray", origin="upper", vmin=0, vmax=1)
    axes0[0].set_title("Amplitude\nUniform circular pupil", fontsize=10)
    axes0[0].axis("off")

    axes0[1].imshow(amp_gaussian, cmap="gray", origin="upper", vmin=0, vmax=1)
    axes0[1].set_title(
        f"Amplitude\nGaussian on circular pupil (sigma={gaussian_sigma_frac:.0%})",
        fontsize=10,
    )
    axes0[1].axis("off")

    im0 = axes0[2].imshow(aberration, cmap="RdBu_r", origin="upper")
    axes0[2].set_title(
        f"Aberration map\nPV={pv_rad:.2f} rad, RMS={rms_rad:.2f} rad",
        fontsize=10,
    )
    axes0[2].axis("off")
    fig0.colorbar(im0, ax=axes0[2], fraction=0.046, pad=0.04, label="Phase [rad]")

    fig0.suptitle("Input amplitudes and aberration used", fontsize=11)
    fig0.tight_layout()

    # ── 5. Highlighted PSFs with high oversampling ──────────────────────────
    print(
        f"\nComputing oversampled PSFs with pad_factor={psf_pad_factor} and crop_size={psf_crop_size}"
    )
    psf_specs = [
        (amp_uniform, None, "PSF: Uniform circular"),
        (amp_gaussian, None, "PSF: Gaussian circular"),
        (amp_gaussian, aberration, "PSF: Gaussian circular + Aberration"),
    ]

    # Compute one PSF at a time to keep memory bounded even with large padding.
    psf_log_cases = []
    psf_line_cases = []
    for amp_case, wf_case, label in psf_specs:
        psf_full = compute_psf(amp_case, wavefront=wf_case, pad_factor=psf_pad_factor)
        psf_crop = center_crop(psf_full, psf_crop_size)
        psf_log_cases.append((normalize_log_view(psf_crop, psf_log_compression), label))
        x_pix, line = center_line_profile(psf_crop)
        psf_line_cases.append((x_pix, line, label))

    fig_psf, axes_psf = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (psf_show, label) in zip(axes_psf, psf_log_cases):
        ax.imshow(
            psf_show,
            cmap=colormap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            origin="upper",
        )
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    fig_psf.suptitle(
        f"Highlighted PSFs (zero-padding x{psf_pad_factor}, center crop {psf_crop_size} px)",
        fontsize=11,
    )
    fig_psf.tight_layout()

    # 1D linear PSF profiles (central horizontal cut)
    fig_psf_1d, ax_psf_1d = plt.subplots(figsize=(8, 5))
    line_colors = ["tab:blue", "tab:orange", "tab:green"]
    for (x_pix, line, label), color in zip(psf_line_cases, line_colors):
        ax_psf_1d.plot(x_pix, line, linewidth=1.8, color=color, label=label)

    ax_psf_1d.set_title("PSF 1D central profiles (linear scale)", fontsize=11)
    ax_psf_1d.set_xlabel("Pixel offset from center", fontsize=10)
    ax_psf_1d.set_ylabel("Normalized intensity", fontsize=10)
    ax_psf_1d.set_ylim(0.0, 1.02)
    ax_psf_1d.grid(True, alpha=0.25)
    ax_psf_1d.legend(loc="upper right", fontsize=9)
    fig_psf_1d.tight_layout()

    # ── 6. Run GS twice: once on uniform, once on gaussian ───────────────────
    print(f"\nRunning GS hologram synthesis with SR/NR constraints:")
    print(f"  Resolution: {res}×{res}")
    print(f"  Signal region (M): {m}×{m}")
    print(f"  Iterations: {gs_iters}")
    print(f"  Mixing parameter: {mixing_parameter}")

    target_gs = target.astype(np.float32)

    gs_configs = [
        ("uniform", amp_uniform),
        ("gaussian", amp_gaussian),
    ]

    gs_results = {}
    for mode_name, input_amp_gs in gs_configs:
        print(f"\nGS input mode: {mode_name}")
        print(f"  Amplitude range: [{input_amp_gs.min():.3f}, {input_amp_gs.max():.3f}]")
        phase_gs, rmse_hist, _ = run_gs_gpu(
            input_amp_gs,
            target_gs,
            gs_iters,
            m,
            mixing_parameter,
        )

        if gpu_available:
            phase_gs = cp.asnumpy(phase_gs) if isinstance(phase_gs, cp.ndarray) else phase_gs

        gs_results[mode_name] = {
            "phase": phase_gs,
            "rmse_hist": rmse_hist,
        }
        print(f"  Final RMSE ({mode_name}): {rmse_hist[-1]:.6f}")

    # ── 7. Compute reconstructions for each GS phase ────────────────────────
    recon_by_mode = {}
    for mode_name, result in gs_results.items():
        phase_gs = result["phase"]
        recon_by_mode[mode_name] = [
            (reconstruct_hologram(phase_gs, amp_uniform), "Reconstruction\n+ Uniform circular Amp"),
            (reconstruct_hologram(phase_gs, amp_gaussian), "Reconstruction\n+ Gaussian circular Amp"),
            (
                reconstruct_hologram(phase_gs, amp_gaussian, extra_wavefront=aberration),
                "Reconstruction\n+ Gaussian circular Amp + Aberration",
            ),
        ]

    # ── 8. Required output figures: exactly two 3-image reconstruction panels ──
    for mode_name, recon_cases in recon_by_mode.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, (intensity, label) in zip(axes, recon_cases):
            img_norm = intensity / (intensity.max() + 1e-12)
            ax.imshow(
                img_norm,
                cmap=colormap,
                vmin=0,
                vmax=1,
                interpolation="nearest",
                origin="upper",
            )
            ax.set_title(label, fontsize=10)
            ax.axis("off")

        fig.suptitle(
            f"GS run on {mode_name} input amplitude — full 2x-frame reconstructions",
            fontsize=11,
        )
        fig.tight_layout()

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    plt.show()
