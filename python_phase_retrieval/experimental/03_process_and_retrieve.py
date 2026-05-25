"""Manual Fourier-plane masking and PDGS retrieval."""
#%%
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise ImportError("Pillow is required to load images. Install with: pip install Pillow") from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("Matplotlib is required for manual ROI selection. Install with: pip install matplotlib") from exc

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from python_phase_retrieval.phase_retrieval import one_shot, pdgs_log


#%% Configuration
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "output"

patterns_dir = OUTPUT_DIR / "patterns_bmp"
captures_dir = OUTPUT_DIR / "captures_tif"
results_dir = OUTPUT_DIR / "results"

# Calibration capture name created by 01_generate_phases.py.
calibration_pattern_stem = "calib_rs_frame"

# SLM dimensions used by PDGS.
slm_height = 1080
slm_width = 1080
pixel_pitch = 8e-6

# Physical setup used by one_shot/PDGS model.
wavelength_m = 532e-9
focal_length_m = 100e-3

# PDGS settings.
nit = 100
flambda = wavelength_m * focal_length_m
use_gpu = False
verbose = True

# Retrieval visualization settings.
save_retrieved_plot = True
show_retrieved_plot = True
show_modulus_preview = True
modulus_preview_max_images = 6
save_one_shot_debug = True

# ROI selection behavior.
force_manual_selection = False
roi_config_filename = "fourier_roi_config.json"

# Experimental preprocessing: remove constant camera pedestal not present in simulation.
subtract_background = True
background_percentile = 5.0

# FFT/alignment controls for experimental captures.
# If True, recenter selected first-order patch before resize and set linear term to 0.
center_signal_order = True

# Optional extra shift on camera data before PDGS (for FFT convention debugging).
# Allowed values: "none", "fftshift", "ifftshift".
input_fft_shift_mode = "none"


#%% Helpers
_tag_re = re.compile(r"^([pm])(\d+(?:p\d+)?)e(\d+)$")


def _load_gray_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Convert color to grayscale while preserving dynamic range.
        return np.mean(arr.astype(np.float64), axis=2)
    raise ValueError(f"Unsupported image shape for {path}: {arr.shape}")


def _save_gray_image(array: np.ndarray, path: Path) -> None:
    arr = np.asarray(array, dtype=np.float64)
    vmax = arr.max()
    if vmax > 0:
        arr = np.clip(arr / vmax * 255.0, 0, 255)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _parse_tag(tag: str) -> float:
    m = _tag_re.match(tag)
    if m is None:
        raise ValueError(f"Invalid tag format: {tag}")

    sign, number, exponent = m.groups()
    value = float(number.replace("p", ".")) * (10 ** int(exponent))
    if sign == "m":
        value = -value
    return value


def _parse_pattern_params(stem: str) -> tuple[float, float, float]:
    # Expected: a_<alpha_tag>_x_<linx_tag>_y_<liny_tag>
    parts = stem.split("_")
    if len(parts) != 6 or parts[0] != "a" or parts[2] != "x" or parts[4] != "y":
        raise ValueError(f"Unexpected pattern filename stem: {stem}")

    alpha = _parse_tag(parts[1])
    linx = _parse_tag(parts[3])
    liny = _parse_tag(parts[5])
    return alpha, linx, liny


def _is_target_frame_pattern(stem: str) -> bool:
    return stem.endswith("_target_frame")


def _is_calibration_pattern(stem: str) -> bool:
    return stem == calibration_pattern_stem


def _pattern_sort_key(path: Path) -> tuple[float, float, float, str]:
    # Numeric sort by physical params; fallback to stem for deterministic tie-break.
    alpha, linx, liny = _parse_pattern_params(path.stem)
    return (alpha, linx, liny, path.stem)


def _build_phase_pdgs(
    y_m: np.ndarray,
    x_m: np.ndarray,
    alpha_rad_per_m2: float,
    beta_row: float,
    beta_col: float,
) -> np.ndarray:
    # div_phase uses physical SLM coordinates in meters.
    quadratic = alpha_rad_per_m2 * (y_m**2 + x_m**2)
    linear = beta_row * y_m + beta_col * x_m
    return quadratic + linear


def _clip_rect_to_shape(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x, y, rw, rh = rect
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    return (x, y, rw, rh)


def _manual_pick_rect(img: np.ndarray, title: str) -> tuple[int, int, int, int]:
    # Contrast enhancement and gamma correction for better edge visibility
    img_normalized = img.astype(np.float64)
    img_normalized = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min() + 1e-8)
    gamma = 0.5
    img_corrected = np.power(img_normalized, gamma)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_corrected, cmap="gray")
    ax.set_title(title + "\nClick TOP-LEFT then BOTTOM-RIGHT")
    pts = plt.ginput(2, timeout=-1)
    plt.close(fig)

    if len(pts) != 2:
        raise RuntimeError(f"Selection cancelled for: {title}")

    (x1, y1), (x2, y2) = pts
    x0 = int(round(min(x1, x2)))
    y0 = int(round(min(y1, y2)))
    x1 = int(round(max(x1, x2)))
    y1 = int(round(max(y1, y2)))
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _crop_rect(img: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    return img[y:y + h, x:x + w]


def _apply_rect_zero(mask: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    mask[y:y + h, x:x + w] = 0.0


def _resize_nearest(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    row_idx = np.linspace(0, src_h - 1, target_h).astype(int)
    col_idx = np.linspace(0, src_w - 1, target_w).astype(int)
    return img[row_idx][:, col_idx]


def _recenter_by_mask(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    weights = np.asarray(img, dtype=np.float64) * np.asarray(mask, dtype=np.float64)
    ws = float(np.sum(weights))

    if ws > 0.0:
        yy, xx = np.indices(img.shape, dtype=np.float64)
        cy = float(np.sum(yy * weights) / ws)
        cx = float(np.sum(xx * weights) / ws)
    else:
        idx = np.argwhere(mask > 0.0)
        if idx.size == 0:
            return img, (0, 0)
        cy, cx = np.mean(idx, axis=0)

    target_y = (img.shape[0] - 1) / 2.0
    target_x = (img.shape[1] - 1) / 2.0
    dy = int(round(target_y - cy))
    dx = int(round(target_x - cx))

    shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
    return shifted, (dy, dx)


def _apply_input_fft_shift(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return img
    if mode == "fftshift":
        return np.fft.fftshift(img)
    if mode == "ifftshift":
        return np.fft.ifftshift(img)
    raise ValueError(f"Unsupported input_fft_shift_mode: {mode}")


def _frame_to_modulus(frame: np.ndarray) -> np.ndarray:
    intensity = frame.astype(np.float64)
    modulus = np.sqrt(np.clip(intensity, 0.0, None))
    mx = modulus.max()
    if mx > 0:
        modulus = modulus / mx
    return modulus


def _save_retrieved_amplitude_phase_plot(
    beam_estimate: np.ndarray,
    out_path: Path,
    show_plot: bool,
) -> None:
    amplitude = np.abs(beam_estimate)
    phase = np.angle(beam_estimate)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im_amp = axes[0].imshow(amplitude, cmap="gray")
    axes[0].set_title("Retrieved Amplitude")
    axes[0].axis("off")
    fig.colorbar(im_amp, ax=axes[0], fraction=0.046, pad=0.04)

    im_phase = axes[1].imshow(phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Retrieved Phase [rad]")
    axes[1].axis("off")
    fig.colorbar(im_phase, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _edge_energy_fraction(amplitude: np.ndarray, border_fraction: float = 0.08) -> float:
    a = np.asarray(amplitude, dtype=np.float64)
    h, w = a.shape
    bh = max(1, int(round(h * border_fraction)))
    bw = max(1, int(round(w * border_fraction)))

    edge = np.zeros_like(a, dtype=bool)
    edge[:bh, :] = True
    edge[-bh:, :] = True
    edge[:, :bw] = True
    edge[:, -bw:] = True

    total = float(np.sum(a**2))
    if total <= 0.0:
        return 0.0
    return float(np.sum((a[edge]) ** 2) / total)


def _save_amplitude_triplet(beam: np.ndarray, out_path: Path) -> dict[str, float]:
    amp = np.abs(beam)
    amp_fft = np.abs(np.fft.fftshift(beam))
    amp_ifft = np.abs(np.fft.ifftshift(beam))

    metrics = {
        "edge_energy_fraction_native": _edge_energy_fraction(amp),
        "edge_energy_fraction_fftshift": _edge_energy_fraction(amp_fft),
        "edge_energy_fraction_ifftshift": _edge_energy_fraction(amp_ifft),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        (amp, "|beam_guess| native"),
        (amp_fft, "|beam_guess| fftshift"),
        (amp_ifft, "|beam_guess| ifftshift"),
    ]

    for ax, (img, title) in zip(axes, panels):
        im = ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return metrics


def _load_or_select_roi_config(calib_img: np.ndarray, config_path: Path) -> dict:
    if config_path.exists() and not force_manual_selection:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"Loaded ROI config from: {config_path}")
        return cfg

    print("Manual ROI selection started")
    fov_rect = _manual_pick_rect(
        calib_img,
        "Select OUTER corners of calibration frame (defines full Fourier FOV)",
    )
    fov_rect = _clip_rect_to_shape(fov_rect, calib_img.shape)

    fov_img = _crop_rect(calib_img, fov_rect)

    signal_rect = _manual_pick_rect(fov_img, "Select SIGNAL ROI (e.g. top-left 1st order)")
    signal_rect = _clip_rect_to_shape(signal_rect, fov_img.shape)

    zero_h_rect = _manual_pick_rect(fov_img, "Select ZERO-ORDER HORIZONTAL rectangle")
    zero_h_rect = _clip_rect_to_shape(zero_h_rect, fov_img.shape)

    zero_v_rect = _manual_pick_rect(fov_img, "Select ZERO-ORDER VERTICAL rectangle")
    zero_v_rect = _clip_rect_to_shape(zero_v_rect, fov_img.shape)

    cfg = {
        "fov_rect": list(fov_rect),
        "signal_rect_in_fov": list(signal_rect),
        "zero_h_rect_in_fov": list(zero_h_rect),
        "zero_v_rect_in_fov": list(zero_v_rect),
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved ROI config to: {config_path}")
    return cfg


#%% Resolve paths and files
patterns_dir = patterns_dir.resolve()
captures_dir = captures_dir.resolve()
results_dir = results_dir.resolve()
results_dir.mkdir(parents=True, exist_ok=True)

all_pattern_files = list(patterns_dir.glob("*.bmp"))
if not all_pattern_files:
    raise FileNotFoundError(f"No pattern BMP files found in {patterns_dir}")

pattern_files = [
    p
    for p in all_pattern_files
    if not _is_target_frame_pattern(p.stem) and not _is_calibration_pattern(p.stem)
]
pattern_files.sort(key=_pattern_sort_key)

if not pattern_files:
    raise RuntimeError("No non-calibration patterns found after filtering target-frame files.")

calib_capture = captures_dir / f"{calibration_pattern_stem}.tif"
if not calib_capture.exists():
    raise FileNotFoundError(
        f"Calibration capture not found: {calib_capture}. "
        "Generate/acquire calibration pattern first."
    )


#%% Manual ROI selection or reuse saved config
calib_img = _load_gray_image(calib_capture)
roi_config_path = results_dir / roi_config_filename
roi_cfg = _load_or_select_roi_config(calib_img, roi_config_path)

fov_rect = tuple(int(v) for v in roi_cfg["fov_rect"])
signal_rect = tuple(int(v) for v in roi_cfg["signal_rect_in_fov"])
zero_h_rect = tuple(int(v) for v in roi_cfg["zero_h_rect_in_fov"])
zero_v_rect = tuple(int(v) for v in roi_cfg["zero_v_rect_in_fov"])


#%% Build diversity phases and processed modulus stack
y_m_1d = (np.arange(slm_height, dtype=float) - (slm_height - 1) / 2.0) * pixel_pitch
x_m_1d = (np.arange(slm_width, dtype=float) - (slm_width - 1) / 2.0) * pixel_pitch
L = (y_m_1d, x_m_1d)
y_m = L[0].reshape(-1, 1)
x_m = L[1].reshape(1, -1)

imgs_modulus: list[np.ndarray] = []
div_phases: list[np.ndarray] = []
imgs_intensity: list[np.ndarray] = []
pattern_params: list[tuple[float, float, float]] = []
pattern_params_physical: list[tuple[float, float, float]] = []
saved_debug = False

for pattern_path in pattern_files:
    capture_path = captures_dir / f"{pattern_path.stem}.tif"
    if not capture_path.exists():
        raise FileNotFoundError(f"Missing capture for pattern {pattern_path.name}: {capture_path}")

    alpha_phys, linx_phys, liny_phys = _parse_pattern_params(pattern_path.stem)
    # ldot(beta, L) follows (row, col) order => (y, x) in physical SI units.
    # If signal is recentered in Fourier plane, linear carrier is removed.
    if center_signal_order:
        beta_row = 0.0
        beta_col = 0.0
    else:
        beta_row = 2.0 * np.pi * float(liny_phys)
        beta_col = 2.0 * np.pi * float(linx_phys)
    div_phase = _build_phase_pdgs(
        y_m=y_m,
        x_m=x_m,
        alpha_rad_per_m2=alpha_phys,
        beta_row=beta_row,
        beta_col=beta_col,
    )

    raw_img = _load_gray_image(capture_path)
    fov_img = _crop_rect(raw_img, fov_rect)

    # Keep only selected signal ROI first.
    mask = np.zeros(fov_img.shape, dtype=np.float64)
    sx, sy, sw, sh = signal_rect
    mask[sy:sy + sh, sx:sx + sw] = 1.0

    # Then force zero in the two zero-order rectangles.
    _apply_rect_zero(mask, zero_h_rect)
    _apply_rect_zero(mask, zero_v_rect)

    masked_fov = fov_img.astype(np.float64) * mask

    if subtract_background:
        signal_pixels = masked_fov[mask > 0.0]
        if signal_pixels.size > 0:
            bg_level = float(np.percentile(signal_pixels, background_percentile))
            masked_fov = np.clip(masked_fov - bg_level * mask, 0.0, None)

    if center_signal_order:
        masked_fov, recenter_shift = _recenter_by_mask(masked_fov, mask)
    else:
        recenter_shift = (0, 0)

    resized_img = _resize_nearest(masked_fov, slm_height, slm_width)
    resized_img = _apply_input_fft_shift(resized_img, input_fft_shift_mode)

    imgs_intensity.append(np.clip(resized_img.astype(np.float64), 0.0, None))
    imgs_modulus.append(_frame_to_modulus(resized_img))
    div_phases.append(div_phase)
    pattern_params.append((alpha_phys, beta_row, beta_col))
    pattern_params_physical.append((alpha_phys, linx_phys, liny_phys))

    if not saved_debug:
        _save_gray_image(raw_img, results_dir / "debug_raw_capture.png")
        _save_gray_image(fov_img, results_dir / "debug_fov_crop.png")
        _save_gray_image(mask, results_dir / "debug_combined_mask.png")
        _save_gray_image(masked_fov, results_dir / "debug_masked_fov.png")
        _save_gray_image(resized_img, results_dir / "debug_resized_for_pdgs.png")
        np.save(results_dir / "debug_recenter_shift_dy_dx.npy", np.array(recenter_shift, dtype=int))
        saved_debug = True

if not imgs_modulus:
    raise RuntimeError("No non-calibration patterns found for PDGS.")


#%% Inspect generated dataset (imgs_modulus) before PDGS
print(f"Generated dataset: {len(imgs_modulus)} frames")
print(f"Each modulus frame shape: {imgs_modulus[0].shape}")

if show_modulus_preview:
    n_show = min(len(imgs_modulus), modulus_preview_max_images)
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    if n_show == 1:
        axes = [axes]

    for idx in range(n_show):
        alpha_millions = pattern_params_physical[idx][0] / 1e6
        axes[idx].imshow(imgs_modulus[idx], cmap="gray")
        axes[idx].set_title(f"|u| #{idx + 1}\nalpha={alpha_millions:g}e6")
        axes[idx].axis("off")

    fig.tight_layout()
    plt.show()


#%% Run PDGS
# Match simulation setup: initialize from one-shot using the last diversity image.
alpha_guess, beta_row_guess, beta_col_guess = pattern_params[-1]
beam_guess = one_shot(
    img_intensity=imgs_intensity[-1],
    # one_shot uses (alpha/2) * r^2 internally, so pass 2*alpha of the
    # diversity phase to keep consistency with div_phase construction.
    alpha=2.0 * alpha_guess,
    beta=(beta_row_guess, beta_col_guess),
    L=L,
    flambda=flambda,
)

one_shot_metrics: dict[str, float] = {}
if save_one_shot_debug:
    one_shot_metrics = _save_amplitude_triplet(
        beam=beam_guess,
        out_path=results_dir / "one_shot_amplitude_debug.png",
    )
    print(
        "One-shot edge energy fractions "
        f"(native/fftshift/ifftshift): "
        f"{one_shot_metrics['edge_energy_fraction_native']:.4f} / "
        f"{one_shot_metrics['edge_energy_fraction_fftshift']:.4f} / "
        f"{one_shot_metrics['edge_energy_fraction_ifftshift']:.4f}"
    )

beam_est, logs = pdgs_log(
    imgs_modulus=imgs_modulus,
    div_phases=div_phases,
    nit=nit,
    beam_guess=beam_guess,
    L=L,
    flambda=flambda,
    verbose=verbose,
    use_gpu=use_gpu,
)


#%% Save outputs
np.save(results_dir / "beam_estimate_complex.npy", beam_est)
np.save(results_dir / "phase_estimate_rad.npy", np.angle(beam_est))

if save_retrieved_plot:
    _save_retrieved_amplitude_phase_plot(
        beam_estimate=beam_est,
        out_path=results_dir / "retrieved_amplitude_phase.png",
        show_plot=show_retrieved_plot,
    )

with (results_dir / "pdgs_logs.json").open("w", encoding="utf-8") as f:
    json.dump([entry.__dict__ for entry in logs], f, indent=2)

run_summary = {
    "patterns_dir": str(patterns_dir),
    "captures_dir": str(captures_dir),
    "results_dir": str(results_dir),
    "calibration_pattern_stem": calibration_pattern_stem,
    "slm_height": slm_height,
    "slm_width": slm_width,
    "pixel_pitch": pixel_pitch,
    "nit": nit,
    "flambda": flambda,
    "use_gpu": use_gpu,
    "beam_guess_method": "one_shot",
    "beam_guess_alpha": float(2.0 * alpha_guess),
    "beam_guess_alpha_source_diversity": float(alpha_guess),
    "beam_guess_beta": [float(beta_row_guess), float(beta_col_guess)],
    "beam_guess_params_units": "SI",
    "wavelength_m": wavelength_m,
    "focal_length_m": focal_length_m,
    "save_one_shot_debug": save_one_shot_debug,
    "one_shot_metrics": one_shot_metrics,
    "beam_guess_source_pattern_params_physical": {
        "alpha_rad_per_m2": float(pattern_params_physical[-1][0]),
        "lin_x_cpm": float(pattern_params_physical[-1][1]),
        "lin_y_cpm": float(pattern_params_physical[-1][2]),
    },
    "center_signal_order": center_signal_order,
    "input_fft_shift_mode": input_fft_shift_mode,
    "subtract_background": subtract_background,
    "background_percentile": background_percentile,
    "roi_config_path": str(roi_config_path),
    "processed_pattern_stems": [p.stem for p in pattern_files],
    "processed_pattern_params_physical": [
        {
            "alpha_rad_per_m2": float(alpha),
            "lin_x_cpm": float(linx),
            "lin_y_cpm": float(liny),
        }
        for alpha, linx, liny in pattern_params_physical
    ],
}

with (results_dir / "processing_run_summary.json").open("w", encoding="utf-8") as f:
    json.dump(run_summary, f, indent=2)

print(f"PDGS completed. Results in: {results_dir}")
