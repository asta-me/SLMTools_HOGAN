import os
from dataclasses import dataclass

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
from PIL import Image
import PySpin


@dataclass
class FLIRCamera:
    system: PySpin.System
    cam_list: PySpin.CameraList
    cam: PySpin.CameraPtr
    processor: PySpin.ImageProcessor
    pixel_format: str


def _clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def _set_enum_node(node, entry_name):
    if not PySpin.IsReadable(node) or not PySpin.IsWritable(node):
        raise RuntimeError(f'Node {node.GetName()} is not writable.')

    entry = node.GetEntryByName(entry_name)
    if not PySpin.IsReadable(entry):
        raise RuntimeError(f'Entry {entry_name} is not available for node {node.GetName()}.')

    node.SetIntValue(entry.GetValue())


def _try_set_pixel_format(cam, pixel_format_names):
    if cam.PixelFormat.GetAccessMode() != PySpin.RW:
        current_value = None
        if PySpin.IsReadable(cam.PixelFormat):
            try:
                current_value = cam.PixelFormat.ToString()
            except Exception:
                current_value = None

        if current_value:
            return current_value

        return 'native'

    for name in pixel_format_names:
        enum_name = f'PixelFormat_{name}'
        if not hasattr(PySpin, enum_name):
            continue

        enum_value = getattr(PySpin, enum_name)
        entry = cam.PixelFormat.GetEntry(enum_value)
        if PySpin.IsReadable(entry):
            cam.PixelFormat.SetValue(enum_value)
            return name

    raise RuntimeError(
        f'None of the requested pixel formats are available: {pixel_format_names}'
    )


def open_camera(camera_index=0, pixel_format='auto'):
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError('No FLIR camera detected.')

    if camera_index < 0 or camera_index >= cam_list.GetSize():
        cam_list.Clear()
        system.ReleaseInstance()
        raise IndexError(f'Camera index {camera_index} out of range.')

    cam = cam_list[camera_index]
    cam.Init()

    _set_enum_node(cam.AcquisitionMode, 'Continuous')

    if pixel_format == 'auto':
        configured_pixel_format = _try_set_pixel_format(
            cam,
            ('Mono8', 'BayerRG8', 'BayerGB8', 'RGB8'),
        )
    else:
        configured_pixel_format = _try_set_pixel_format(cam, (pixel_format,))

    processor = PySpin.ImageProcessor()
    processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

    return FLIRCamera(
        system=system,
        cam_list=cam_list,
        cam=cam,
        processor=processor,
        pixel_format=configured_pixel_format,
    )


def set_exposure_us(camera, exposure_us):
    camera.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    exposure_node = camera.cam.ExposureTime
    if exposure_node.GetAccessMode() != PySpin.RW:
        raise RuntimeError('ExposureTime is not writable.')

    exposure_node.SetValue(_clamp(exposure_us, exposure_node.GetMin(), exposure_node.GetMax()))


def set_gain_db(camera, gain_db):
    camera.cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    gain_node = camera.cam.Gain
    if gain_node.GetAccessMode() != PySpin.RW:
        raise RuntimeError('Gain is not writable.')

    gain_node.SetValue(_clamp(gain_db, gain_node.GetMin(), gain_node.GetMax()))


def set_gamma(camera, enabled=False, gamma_value=None):
    gamma_enable = camera.cam.GammaEnable
    if gamma_enable.GetAccessMode() != PySpin.RW:
        return False

    gamma_enable.SetValue(enabled)
    if enabled and gamma_value is not None and camera.cam.Gamma.GetAccessMode() == PySpin.RW:
        gamma_node = camera.cam.Gamma
        gamma_node.SetValue(_clamp(gamma_value, gamma_node.GetMin(), gamma_node.GetMax()))

    return True


def _convert_image(image_result, processor, convert_to):
    if convert_to is None or convert_to == 'native':
        return image_result.GetNDArray().copy()

    conversion_map = {
        'mono8': PySpin.PixelFormat_Mono8,
        'rgb8': PySpin.PixelFormat_RGB8,
    }

    key = convert_to.lower()
    if key not in conversion_map:
        raise ValueError(f'Unsupported conversion target: {convert_to}')

    converted = processor.Convert(image_result, conversion_map[key])
    return converted.GetNDArray().copy()


def acquire_image(camera, timeout_ms=1000, convert_to='mono8'):
    camera.cam.BeginAcquisition()
    try:
        image_result = camera.cam.GetNextImage(timeout_ms)
        try:
            if image_result.IsIncomplete():
                raise RuntimeError(
                    f'Image incomplete with status {image_result.GetImageStatus()}.'
                )

            return _convert_image(image_result, camera.processor, convert_to)
        finally:
            image_result.Release()
    finally:
        camera.cam.EndAcquisition()


def acquire_average_image(camera, count, timeout_ms=1000, convert_to='mono8'):
    if count <= 0:
        raise ValueError('count must be greater than zero.')

    accumulator = None
    camera.cam.BeginAcquisition()
    try:
        for _ in range(count):
            image_result = camera.cam.GetNextImage(timeout_ms)
            try:
                if image_result.IsIncomplete():
                    raise RuntimeError(
                        f'Image incomplete with status {image_result.GetImageStatus()}.'
                    )

                frame = _convert_image(image_result, camera.processor, convert_to).astype(np.float32)
                if accumulator is None:
                    accumulator = frame
                else:
                    accumulator += frame
            finally:
                image_result.Release()
    finally:
        camera.cam.EndAcquisition()

    return accumulator / float(count)


def save_image(image_array, filename):
    array = np.asarray(image_array)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        image = Image.fromarray(array)
    elif array.ndim == 3 and array.shape[2] == 3:
        image = Image.fromarray(array, 'RGB')
    else:
        raise ValueError(f'Unsupported image shape: {array.shape}')

    image.save(filename)


def close_camera(camera):
    try:
        camera.cam.DeInit()
    finally:
        del camera.cam
        camera.cam_list.Clear()
        camera.system.ReleaseInstance()


def example_usage():
    camera = None
    try:
        camera = open_camera(pixel_format='auto')
        print(f'Camera opened with pixel format: {camera.pixel_format}')

        set_exposure_us(camera, 5000.0)
        set_gain_db(camera, 0.0)
        set_gamma(camera, enabled=False)

        image = acquire_image(camera, convert_to='mono8')
        averaged_image = acquire_average_image(camera, count=5, convert_to='mono8')

        print(f'Single frame shape: {image.shape}')
        print(f'Averaged frame shape: {averaged_image.shape}')

        save_image(image, 'flir_single_frame.png')
        save_image(averaged_image, 'flir_average_frame.png')
        print('Saved flir_single_frame.png and flir_average_frame.png')
    finally:
        if camera is not None:
            close_camera(camera)


if __name__ == '__main__':
    example_usage()