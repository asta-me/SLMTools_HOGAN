"""Display BMP phases on SLM display with pygame and acquire one camera image per phase."""
#%% Imports
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
import re

import numpy as np

from flir_camera_functions import acquire_image, close_camera, open_camera, set_exposure_us
from pygame_functions import close_pygame, display_bmp_hologram, init_pygame

from PIL import Image


#%% Helpers
_tag_re = re.compile(r"^([pm])(\d+(?:p\d+)?)e(\d+)$")


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


def _pattern_sort_key(path: Path) -> tuple[float, float, float, str]:
    # Numeric sort by physical params; fallback to stem for deterministic tie-break.
    alpha, linx, liny = _parse_pattern_params(path.stem)
    return (alpha, linx, liny, path.stem)


def _save_capture_image(frame: np.ndarray, out_path: Path) -> None:
    if frame.ndim != 2:
        raise ValueError("Camera frame must be a 2D grayscale array")
    # Keep camera dynamic range: save TIFF using the native array dtype.
    arr = np.asarray(frame)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)

    Image.fromarray(arr).save(out_path)


#%% Configuration
# SLM monitor index for pygame.
screen_index = 1
# Wait after displaying a pattern before capture.
wait_ms = 200.0
# Camera options for FLIR acquisition.
camera_index = 0                # Default to first detected camera
exposure_us = 1300             # Exposure time in microseconds
timeout_ms = 2000               # Timeout for camera acquisition in milliseconds
convert_to = "native"           # Conversion mode for acquired images

# Measurement dataset label.
measurement_label = "20260701_test_01"

# Folder with BMP patterns created by 01_generate_phases.py
experiment_directory = Path(__file__).resolve().parent
dataset_dir = experiment_directory / "dataset" / measurement_label
patterns_dir = dataset_dir / "01_patterns_bmp"

# Folder where acquired TIFF images will be saved.
captures_dir = dataset_dir / "02_captures_tif"

# Measurement log created by 01_generate_phases.py.
measure_log_path = dataset_dir / "measure_log.json"

# Calibration capture name without file extension (stem) created by 01_generate_phases.py.
calibration_pattern_stem = "calib_rs_frame"

#%% Prepare file lists and output folder

#Create Captures Directory if it doesn't exist.
captures_dir.mkdir(parents=True, exist_ok=True)

# Get all BMP files paths in the patterns directory
all_bmp_files = list(patterns_dir.glob("*.bmp"))
calibration_bmp = None          # Will hold the calibration pattern if found
diversity_bmps: list[Path] = [] # Will hold the diversity patterns paths, sorted by parameters

# Verify if there's any bmp
if not all_bmp_files:
    raise FileNotFoundError(f"No BMP files found in {patterns_dir}")
# Check them all
for bmp in all_bmp_files:
    stem = bmp.stem # Name without extension
    if stem == calibration_pattern_stem:
        calibration_bmp = bmp
    else: # Otherwise it's a diversity pattern.
        diversity_bmps.append(bmp)

# Sort diversity patterns by their encoded physical parameters (alpha, lin_x, lin_y).
diversity_bmps.sort(key=_pattern_sort_key)
# Final list of BMPs to display: calibration pattern first (if exists), then sorted diversity patterns.
bmp_files = ([calibration_bmp] if calibration_bmp is not None else []) + diversity_bmps

#Check
if not bmp_files:
    raise RuntimeError("No displayable BMP patterns found.")

#%% Run display + acquisition loop
camera = None
window = None
try:
    # Open SLM display window on the selected monitor.
    window = init_pygame(screen_index)

    # Open FLIR camera with automatic pixel format selection.
    camera = open_camera(camera_index=camera_index, pixel_format="auto")

    # Set the camera exposure once before starting the acquisition loop.
    set_exposure_us(camera, exposure_us)

    for idx, bmp_path in enumerate(bmp_files):
        # Show current phase pattern on the SLM screen.
        display_bmp_hologram(str(bmp_path), window)

        # Give SLM time to settle before taking the image.
        time.sleep(wait_ms / 1000.0)

        # Acquire one camera frame without forcing an 8-bit conversion.
        frame = acquire_image(camera, timeout_ms=timeout_ms, convert_to=convert_to)

        # Save the capture with the same stem as the displayed pattern.
        out_name = f"{bmp_path.stem}.tif"
        out_path = captures_dir / out_name

        # Save in TIFF format so the acquisition output stays in a dedicated folder.
        _save_capture_image(frame, out_path)

        print(f"[{idx + 1}/{len(bmp_files)}] saved: {out_path.name}")

finally:
    # Always release hardware resources cleanly.
    if camera is not None:
        close_camera(camera)
    if window is not None:
        close_pygame()

print(f"Completed acquisition of {len(bmp_files)} TIFF frames into: {captures_dir}")


#%% Update measurement log (acquisition)
if not measure_log_path.exists():
    raise FileNotFoundError(
        f"Measurement log not found: {measure_log_path}. "
        "Run 01_generate_phases.py first to initialize the measurement dataset."
    )

with measure_log_path.open("r", encoding="utf-8") as f:
    measure_log = json.load(f)

measure_log["measurement_label"] = measurement_label
measure_log["dataset_dir"] = str(dataset_dir)
measure_log["acquisition"] = {
    "screen_index": int(screen_index),
    "camera_index": int(camera_index),
    "exposure_us": float(exposure_us),
    "timeout_ms": int(timeout_ms),
    "wait_ms": float(wait_ms),
    "convert_to": convert_to,
    "calibration_pattern_stem": calibration_pattern_stem,
    "num_patterns": int(len(bmp_files)),
    "num_captures": int(len(bmp_files)),
    "updated_at": datetime.now().isoformat(timespec="seconds"),
}

with measure_log_path.open("w", encoding="utf-8") as f:
    json.dump(measure_log, f, indent=2)

print(f"Updated measure log: {measure_log_path}")
