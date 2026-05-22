"""Generate phase masks from an alpha list and export BMP files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise ImportError("Pillow is required for BMP export. Install with: pip install Pillow") from exc


#%% Configuration
EXPERIMENT_DIR = Path(__file__).resolve().parent
output_dir = EXPERIMENT_DIR / "output" / "patterns_bmp"
slm_height = 1080
slm_width = 1080

# Generate a calibration hologram used to map Fourier plane on camera.
generate_calibration_pattern = True
calibration_name = "calib_rs_frame"

# Physical parameters
wavelength_m = 532e-9
pixel_pitch = 8*1e-6

# One phase is generated for each alpha in this list.
# Units: rad / m^2, because phase term is alpha * (x^2 + y^2).
# alphas = np.array([0.0, 4.5, 6.0, 12]) * 1e6
alphas = np.linspace(6, 15, 4) * 1e6

# Calibration frame parameters (target amplitude in Fourier plane).
calib_frame_inset_px = 140
calib_frame_thickness_px = 20
calib_random_seed = 0


# Linear spatial frequencies u,v in cycles/m.
# Linear phase term is: 2*pi*(u*x + v*y).
u_nyquist = 1.0 / (2.0 * pixel_pitch)
lin_x_cpm = 0.5 * u_nyquist
lin_y_cpm = 0.5 * u_nyquist


#%% Notes
# If you want a target close to the center of one Fourier quadrant,
# choose lin_x_cpm around 0.5 * u_nyquist (same idea for Y).
# wavelength_m is included here for physical bookkeeping and for future
# mappings where alpha is derived from optical distances.


#%% Helpers
def _make_physical_grid(height: int, width: int, pixel_pitch: float) -> tuple[np.ndarray, np.ndarray]:
    y = (np.arange(height) - (height - 1) / 2.0) * pixel_pitch
    x = (np.arange(width) - (width - 1) / 2.0) * pixel_pitch
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def _phase_to_uint8_mod_2pi(phase_rad: np.ndarray) -> np.ndarray:
    wrapped = np.mod(phase_rad, 2.0 * np.pi)
    scaled = wrapped * (255.0 / (2.0 * np.pi))
    return np.round(scaled).astype(np.uint8)

def _build_phase(
    xx_m: np.ndarray,
    yy_m: np.ndarray,
    alpha_rad_per_m2: float,
    lin_x_cpm: float,
    lin_y_cpm: float,
) -> np.ndarray:
    quadratic = alpha_rad_per_m2 * (xx_m**2 + yy_m**2)
    linear = 2.0 * np.pi * (lin_x_cpm * xx_m + lin_y_cpm * yy_m)
    return quadratic + linear


def _build_calibration_phase_rs(
    height: int,
    width: int,
    frame_inset_px: int,
    frame_thickness_px: int,
    random_seed: int,
) -> np.ndarray:
    # Target amplitude in Fourier plane: rectangular frame.
    amp_target = np.zeros((height, width), dtype=np.float64)
    y0 = frame_inset_px
    x0 = frame_inset_px
    y1 = height - frame_inset_px
    x1 = width - frame_inset_px

    if y1 <= y0 or x1 <= x0:
        raise ValueError("Invalid calibration frame inset for selected SLM size")

    t = frame_thickness_px
    amp_target[y0:y0 + t, x0:x1] = 1.0
    amp_target[y1 - t:y1, x0:x1] = 1.0
    amp_target[y0:y1, x0:x0 + t] = 1.0
    amp_target[y0:y1, x1 - t:x1] = 1.0

    # Random superposition (single-shot): assign random phase in Fourier plane,
    # inverse transform, and keep only phase on SLM plane.
    rng = np.random.default_rng(random_seed)
    random_phase = rng.uniform(0.0, 2.0 * np.pi, size=(height, width))
    fourier_field = amp_target * np.exp(1j * random_phase)
    slm_field = np.fft.ifft2(np.fft.ifftshift(fourier_field))
    return np.angle(slm_field)


def _sortable_tag(value: float, digits: int = 3, exp_width: int = 2) -> str:
    if value == 0:
        return f"p{'0' * digits}e{'0' * exp_width}"

    prefix = "p" if value > 0 else "m"
    abs_value = abs(value)
    exponent = int(np.floor(np.log10(abs_value)))
    mantissa = abs_value / (10**exponent)

    scaled = int(round(mantissa * 10 ** (digits - 1)))
    if scaled >= 10**digits:
        scaled //= 10
        exponent += 1

    if exponent < 0:
        raise ValueError(f"Negative exponent not supported in file tag for value {value}")

    return f"{prefix}{scaled:0{digits}d}e{exponent:0{exp_width}d}"


def _alpha_tag_e06(value: float) -> str:
    scaled = value / 1e6
    prefix = "p" if scaled >= 0 else "m"
    abs_scaled = abs(scaled)

    if np.isclose(abs_scaled, round(abs_scaled)):
        number = f"{int(round(abs_scaled)):03d}"
    else:
        number = f"{abs_scaled:06.3f}".replace(".", "p").rstrip("0")
        if number.endswith("p"):
            number = number[:-1]

    return f"{prefix}{number}e06"


#%% Generate and save
output_dir.mkdir(parents=True, exist_ok=True)

if generate_calibration_pattern:
    calib_phase = _build_calibration_phase_rs(
        slm_height,
        slm_width,
        frame_inset_px=calib_frame_inset_px,
        frame_thickness_px=calib_frame_thickness_px,
        random_seed=calib_random_seed,
    )
    calib_uint8 = _phase_to_uint8_mod_2pi(calib_phase)
    Image.fromarray(calib_uint8, mode="L").save(output_dir / f"{calibration_name}.bmp")
    print(f"Generated calibration pattern: {calibration_name}.bmp")

xx_m, yy_m = _make_physical_grid(slm_height, slm_width, pixel_pitch)
linx_tag = _sortable_tag(lin_x_cpm)
liny_tag = _sortable_tag(lin_y_cpm)

for alpha in np.sort(alphas):
    phase = _build_phase(
        xx_m,
        yy_m,
        alpha_rad_per_m2=float(alpha),
        lin_x_cpm=lin_x_cpm,
        lin_y_cpm=lin_y_cpm,
    )
    phase_uint8 = _phase_to_uint8_mod_2pi(phase)

    alpha_tag = _alpha_tag_e06(float(alpha))
    bmp_name = f"a_{alpha_tag}_x_{linx_tag}_y_{liny_tag}.bmp"
    Image.fromarray(phase_uint8, mode="L").save(output_dir / bmp_name)

print(f"Generated {len(alphas)} patterns in: {output_dir}")
