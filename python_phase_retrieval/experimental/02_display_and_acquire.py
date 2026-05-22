"""Display BMP phases on SLM and acquire one camera image per phase."""
#%%
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from flir_camera_functions import acquire_image, close_camera, open_camera, set_exposure_us
from pygame_functions import close_pygame, display_bmp_hologram, init_pygame

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise ImportError("Pillow is required for image export. Install with: pip install Pillow") from exc


def _save_capture_image(frame: np.ndarray, out_path: Path) -> None:
    if frame.ndim != 2:
        raise ValueError("Camera frame must be a 2D grayscale array")

    if frame.dtype == np.uint8:
        arr = frame
    elif frame.dtype == np.uint16:
        arr = (frame / 257.0).astype(np.uint8)
    else:
        frame_float = frame.astype(np.float64)
        vmax = frame_float.max()
        if vmax <= 0:
            arr = np.zeros_like(frame_float, dtype=np.uint8)
        else:
            arr = np.clip(frame_float / vmax * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(arr, mode="L").save(out_path)


def _warn_if_saturated(frame: np.ndarray, frame_name: str) -> None:
    if not np.issubdtype(frame.dtype, np.integer):
        return

    max_value = np.iinfo(frame.dtype).max
    saturated_pixels = int(np.count_nonzero(frame >= max_value))
    if saturated_pixels == 0:
        return

    saturation_fraction = 100.0 * saturated_pixels / frame.size
    print(
        f"WARNING: saturation detected in {frame_name}")    


#%% Configuration
# Folder with BMP patterns created by 01_generate_phases.py
EXPERIMENT_DIR = Path(__file__).resolve().parent
patterns_dir = EXPERIMENT_DIR / "output" / "patterns_bmp"

# Folder where acquired TIFF images will be saved.
captures_dir = EXPERIMENT_DIR / "output" / "captures_tif"

# SLM monitor index for pygame.
screen_index = 1

# Wait after displaying a pattern before capture.
settle_ms = 120.0

# Camera options for FLIR acquisition.
camera_index = 0
exposure_us = 50.0
timeout_ms = 2000
convert_to = "mono8"


#%% Prepare file lists and output folder
patterns_dir = patterns_dir.resolve()
captures_dir = captures_dir.resolve()
captures_dir.mkdir(parents=True, exist_ok=True)

bmp_files = sorted(patterns_dir.glob("*.bmp"))
if not bmp_files:
    raise FileNotFoundError(f"No BMP files found in {patterns_dir}")


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
        time.sleep(settle_ms / 1000.0)

        # Acquire one camera frame and convert it to mono8.
        frame = acquire_image(camera, timeout_ms=timeout_ms, convert_to=convert_to)

        # Warn if the camera is saturating on the current exposure.
        _warn_if_saturated(frame, bmp_path.stem)

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
