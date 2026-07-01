"""
Generate phase diversity masks 
from a list of curvatures (alpha) and shifts (beta) 
and export the phase pattern as a BMP files."""
#%% Imports
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

#%% Configuration
measurement_label = "20260701_test_01"

# Physical parameters
slm_height = 1080            # SLM resolution (height in pixels)
slm_width =  1080            # SLM resolution (width in pixels)
wavelength_m = 520e-9        # Wavelength in meters 
pixel_pitch = 8*1e-6         # SLM pixel pitch in meters
magn = 100/150               # There might be a 4f magnification
pixel_pitch *= magn


# Calibration frame parameters
generate_calibration_pattern = True
calibration_name = "calib_rs_frame"
calibration_target_tif_name = "FOV_calib_frame_2.tif"
calibration_target_tif_path = Path(r"C:\Users\astam\Desktop\Target_Imgs\FOV_calib_frame_3.tif")

# Phase diversity parameters
# Phase curvatures in rad / m^2 (alpha *x^2))
alphas = np.concatenate((np.linspace(-40, -10, 5), np.linspace(10, 40, 5))) * 1e6
alphas = np.concatenate((np.linspace(-100, -10, 5), np.linspace(10, 100, 5))) * 1e6
alphas = np.concatenate((np.linspace(-100, -50, 5), np.linspace(50, 100, 5))) * 1e6
# Linear phase term is: 2*pi*(u*x + v*y).
u_nyquist = 1.0 / (2.0 * pixel_pitch)   # Max shift
lin_x_cpm = 0.5* u_nyquist              # Linear shift in x
lin_y_cpm = 0.5 * u_nyquist             # Linear shift in y

#%% Define output paths
# Take the path of the folder containing of this script
experiment_directory = Path(__file__).resolve().parent
# Define dataset directories and measure log path
dataset_dir = experiment_directory / "dataset" / measurement_label
patterns_dir = dataset_dir / "01_patterns_bmp"
measure_log_path = dataset_dir / "measure_log.json"
# Generate a calibration hologram frame used to map Fourier plane on camera.

#%% Helpers
def _make_physical_grid(height: int, width: int, pixel_pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a physical grid for the SLM.
    Why the "-1" term? see (N=4):
    with (n - N/2): ([-2,-1,0,1],dx) is best for fft centering and symmetries
    with (n - (N-1)/2): ([-1.5,-0.5,0.5,1.5],dx) centers the grid on the center of the slm."""
    y = (np.arange(height) - (height - 1) / 2.0) * pixel_pitch
    x = (np.arange(width) - (width - 1) / 2.0) * pixel_pitch
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def _phase_to_uint8_mod_2pi(phase_rad: np.ndarray) -> np.ndarray:
    """Wrap phase to [0, 2pi) and scale to [0, 255] for uint8 representation."""
    wrapped = np.mod(phase_rad, 2.0 * np.pi)
    scaled = wrapped * (255.0 / (2.0 * np.pi))
    return np.round(scaled).astype(np.uint8)

def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1], returning zeros for constant inputs."""
    x = np.asarray(arr, dtype=np.float64)
    vmin = float(np.min(x))
    vmax = float(np.max(x))
    if np.isclose(vmax, vmin):
        return np.zeros_like(x, dtype=np.float64)
    return (x - vmin) / (vmax - vmin)

def _load_gray_image(path: Path) -> np.ndarray:
    """Load image as grayscale float64 array."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3:
        return np.mean(arr.astype(np.float64), axis=2)
    raise ValueError(f"Unsupported image shape for {path}: {arr.shape}")

def _amplitude_to_uint8(amplitude: np.ndarray) -> np.ndarray:
    """Rescale amplitude in [0,255] for uint8 representation."""
    amp = np.asarray(amplitude, dtype=np.float64)
    amp_min = np.min(amp)
    amp_max = np.max(amp)
    scaled = (amp - amp_min) / (amp_max - amp_min)
    return np.round(scaled * 255.0).astype(np.uint8)

def _build_phase(
    xx_m: np.ndarray,
    yy_m: np.ndarray,
    alpha_rad_per_m2: float,
    lin_x_cpm: float,
    lin_y_cpm: float,
) -> np.ndarray:
    """Build a quadratic + linear phase pattern from given parameters."""
    quadratic = (alpha_rad_per_m2 / 2.0) * (xx_m**2 + yy_m**2)
    linear = 2.0 * np.pi * (lin_x_cpm * xx_m + lin_y_cpm * yy_m)
    return quadratic + linear

def generate_target_frame(
    height: int,
    width: int,
) -> np.ndarray:
    """Return calibration target amplitude (rectangular frame)."""
    frame_thickness_px = 24
    amp_target = np.zeros((height, width), dtype=np.float64)
    y1 = height
    x1 = width

    t = frame_thickness_px
    amp_target[0:t, 0:x1] = 1.0
    amp_target[y1 - t:y1, 0:x1] = 1.0
    amp_target[0:y1, 0:t] = 1.0
    amp_target[0:y1, x1 - t:x1] = 1.0

    return amp_target

def rs_simple(target_amplitude: np.ndarray) -> np.ndarray:
    """Return SLM phase from a target amplitude using simple random superposition."""
    amp_target = np.asarray(target_amplitude, dtype=np.float64)
    random_phase = 2.0 * np.pi * np.random.rand(*amp_target.shape)
    fourier_field = amp_target * np.exp(1j * random_phase)
    slm_field = np.fft.ifft2(np.fft.ifftshift(fourier_field))
    return np.angle(slm_field)

#Helpers just for the naming
def _sortable_tag(value: float, digits: int = 3, exp_width: int = 2) -> str:
    """Convert a float value to a sortable string tag."""
    if value == 0:
        return f"p0p{'0' * (digits - 1)}e{'0' * exp_width}"

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

    # Keep lexical sortability while making mantissa explicit (e.g. p3p12e04).
    scaled_str = f"{scaled:0{digits}d}"
    mantissa_tag = f"{scaled_str[0]}p{scaled_str[1:]}"
    return f"{prefix}{mantissa_tag}e{exponent:0{exp_width}d}"

def _alpha_tag_e06(value: float) -> str:
    """Convert a float value to a sortable string tag with e06 exponent."""
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


#%% Generate and save patterns

# Calibration Frame
patterns_dir.mkdir(parents=True, exist_ok=True) # Create output directory if it doesn't exist

if generate_calibration_pattern:
    # 1) Load calibration target from TIFF and map to [0, 1] amplitude.
    if not calibration_target_tif_path.exists():
        raise FileNotFoundError(
            f"Calibration target TIFF not found: {calibration_target_tif_path}"
        )

    calib_target = _load_gray_image(calibration_target_tif_path)
    if calib_target.shape != (slm_height, slm_width):
        raise ValueError(
            "Calibration target shape mismatch: "
            f"{calib_target.shape} vs expected {(slm_height, slm_width)}"
        )
    calib_target = _normalize_01(calib_target)

    # 2) Build the RS hologram phase from that target.
    calib_phase = rs_simple(target_amplitude=calib_target)
    #Convert to uint8
    calib_uint8 = _phase_to_uint8_mod_2pi(calib_phase)
    #Save as bmp
    Image.fromarray(calib_uint8, mode="L").save(patterns_dir / f"{calibration_name}.bmp")
    
    print(
        f"Generated calibration pattern: {calibration_name}.bmp "
        f"from {calibration_target_tif_path.name}"
    )


# Phase diversity imgs
xx_m, yy_m = _make_physical_grid(slm_height, slm_width, pixel_pitch)
linx_tag = _sortable_tag(lin_x_cpm)
liny_tag = _sortable_tag(lin_y_cpm)

for alpha in np.sort(alphas):
    # Build the phase pattern
    phase = _build_phase(
        xx_m,
        yy_m,
        alpha_rad_per_m2=float(alpha),
        lin_x_cpm=lin_x_cpm,
        lin_y_cpm=lin_y_cpm,
    )
    # Convert to uint8
    phase_uint8 = _phase_to_uint8_mod_2pi(phase)
    # Save as bmp with a name encoding the parameters
    alpha_tag = _alpha_tag_e06(float(alpha))
    bmp_name = f"a_{alpha_tag}_x_{linx_tag}_y_{liny_tag}.bmp"
    Image.fromarray(phase_uint8, mode="L").save(patterns_dir / bmp_name)

print(f"Generated {len(alphas)} patterns in: {patterns_dir}")


#%% Write measurement log (initialization)
measure_log = {
    "measurement_label": measurement_label,
    "dataset_dir": str(dataset_dir),
    "created_by": "01_generate_phases.py",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "generation": {
        "slm_height": int(slm_height),
        "slm_width": int(slm_width),
        "wavelength_m": float(wavelength_m),
        "pixel_pitch": float(pixel_pitch),
        "generate_calibration_pattern": bool(generate_calibration_pattern),
        "calibration_name": calibration_name,
        "calibration_target_tif_name": calibration_target_tif_name,
        "calibration_target_tif_path": str(calibration_target_tif_path),
        "alphas_rad_per_m2": [float(a) for a in np.asarray(alphas).tolist()],
        "lin_x_cpm": float(lin_x_cpm),
        "lin_y_cpm": float(lin_y_cpm),
    },
}

with measure_log_path.open("w", encoding="utf-8") as f:
    json.dump(measure_log, f, indent=2)

print(f"Initialized measure log: {measure_log_path}")
