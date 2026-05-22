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

from python_phase_retrieval.lattice_utils import natlat
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

# PDGS settings.
nit = 100
flambda = 1.0
use_gpu = False
verbose = True

# Retrieval visualization settings.
save_retrieved_plot = True
show_retrieved_plot = True
show_modulus_preview = True
modulus_preview_max_images = 6

# ROI selection behavior.
force_manual_selection = False
roi_config_filename = "fourier_roi_config.json"

# Experimental preprocessing: remove constant camera pedestal not present in simulation.
subtract_background = True
background_percentile = 5.0


#%% Helpers
_tag_re = re.compile(r"^([pm])(\d+(?:p\d+)?)e(\d+)$")


def _load_gray_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


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


def _to_pdgs_params(
    alpha_rad_per_m2: float,
    lin_x_cpm: float,
    lin_y_cpm: float,
    height: int,
    width: int,
    pixel_pitch_m: float,
) -> tuple[float, float, float]:
    """Convert physical pattern params to PDGS-native lattice params.

    PDGS in this script runs on natlat + flambda=1, so alpha/beta must be in
    those normalized coordinates, not in SI units from filename tags.
    """
    n_mean = 0.5 * (height + width)
    alpha_nat = float(alpha_rad_per_m2) * (pixel_pitch_m**2) * n_mean

    # beta components are in (row, col) axis order for ldot(beta, L).
    beta_row = 2.0 * np.pi * float(lin_y_cpm) * pixel_pitch_m * np.sqrt(height)
    beta_col = 2.0 * np.pi * float(lin_x_cpm) * pixel_pitch_m * np.sqrt(width)
    return alpha_nat, beta_row, beta_col


def _build_phase_pdgs(
    y_nat: np.ndarray,
    x_nat: np.ndarray,
    alpha_nat: float,
    beta_row: float,
    beta_col: float,
) -> np.ndarray:
    quadratic = 0.5 * alpha_nat * (y_nat**2 + x_nat**2)
    linear = beta_row * y_nat + beta_col * x_nat
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
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img, cmap="gray")
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


def _load_or_select_roi_config(calib_img: np.ndarray, config_path: Path) -> dict:
    if config_path.exists() and not force_manual_selection:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"Loaded ROI config from: {config_path}")
        return cfg

    print("Manual ROI selection started")
    fov_rect = _manual_pick_rect(calib_img, "Select Fourier FOV rectangle")
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

pattern_files = sorted(patterns_dir.glob("*.bmp"))
if not pattern_files:
    raise FileNotFoundError(f"No pattern BMP files found in {patterns_dir}")

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
L = natlat((slm_height, slm_width))
y_nat = L[0].reshape(-1, 1)
x_nat = L[1].reshape(1, -1)

imgs_modulus: list[np.ndarray] = []
div_phases: list[np.ndarray] = []
imgs_intensity: list[np.ndarray] = []
pattern_params: list[tuple[float, float, float]] = []
pattern_params_physical: list[tuple[float, float, float]] = []
saved_debug = False

for pattern_path in pattern_files:
    # Skip calibration pattern for PDGS stack.
    if pattern_path.stem == calibration_pattern_stem:
        continue

    capture_path = captures_dir / f"{pattern_path.stem}.tif"
    if not capture_path.exists():
        raise FileNotFoundError(f"Missing capture for pattern {pattern_path.name}: {capture_path}")

    alpha_phys, linx_phys, liny_phys = _parse_pattern_params(pattern_path.stem)
    alpha_nat, beta_row, beta_col = _to_pdgs_params(
        alpha_rad_per_m2=alpha_phys,
        lin_x_cpm=linx_phys,
        lin_y_cpm=liny_phys,
        height=slm_height,
        width=slm_width,
        pixel_pitch_m=pixel_pitch,
    )
    div_phase = _build_phase_pdgs(
        y_nat=y_nat,
        x_nat=x_nat,
        alpha_nat=alpha_nat,
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

    resized_img = _resize_nearest(masked_fov, slm_height, slm_width)

    imgs_intensity.append(np.clip(resized_img.astype(np.float64), 0.0, None))
    imgs_modulus.append(_frame_to_modulus(resized_img))
    div_phases.append(div_phase)
    pattern_params.append((alpha_nat, beta_row, beta_col))
    pattern_params_physical.append((alpha_phys, linx_phys, liny_phys))

    if not saved_debug:
        _save_gray_image(raw_img, results_dir / "debug_raw_capture.png")
        _save_gray_image(fov_img, results_dir / "debug_fov_crop.png")
        _save_gray_image(mask, results_dir / "debug_combined_mask.png")
        _save_gray_image(masked_fov, results_dir / "debug_masked_fov.png")
        _save_gray_image(resized_img, results_dir / "debug_resized_for_pdgs.png")
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
        axes[idx].imshow(imgs_modulus[idx], cmap="gray")
        axes[idx].set_title(f"|u| #{idx + 1}")
        axes[idx].axis("off")

    fig.tight_layout()
    plt.show()


#%% Run PDGS
# Match simulation setup: initialize from one-shot using the last diversity image.
alpha_guess, beta_row_guess, beta_col_guess = pattern_params[-1]
beam_guess = one_shot(
    img_intensity=imgs_intensity[-1],
    alpha=alpha_guess,
    beta=(beta_row_guess, beta_col_guess),
    L=L,
    flambda=flambda,
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
    "beam_guess_alpha": float(alpha_guess),
    "beam_guess_beta": [float(beta_row_guess), float(beta_col_guess)],
    "beam_guess_params_units": "natlat+flambda1",
    "beam_guess_source_pattern_params_physical": {
        "alpha_rad_per_m2": float(pattern_params_physical[-1][0]),
        "lin_x_cpm": float(pattern_params_physical[-1][1]),
        "lin_y_cpm": float(pattern_params_physical[-1][2]),
    },
    "subtract_background": subtract_background,
    "background_percentile": background_percentile,
    "roi_config_path": str(roi_config_path),
}

with (results_dir / "processing_run_summary.json").open("w", encoding="utf-8") as f:
    json.dump(run_summary, f, indent=2)

print(f"PDGS completed. Results in: {results_dir}")
