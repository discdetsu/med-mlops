from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from cxr_core.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    apply_window,
    is_dicom,
    load_image,
    min_max,
    preprocess,
)

DIGITAL_XRAY_FOR_PRESENTATION = "1.2.840.10008.5.1.4.1.1.1.1"


def write_dicom(path: Path, pixels: npt.NDArray[np.uint16], photometric: str, **tags: Any) -> Path:
    """Write a minimal uncompressed single-frame 16-bit DICOM file."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DIGITAL_XRAY_FOR_PRESENTATION
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "DX"
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixels.astype("<u2").tobytes()
    for key, value in tags.items():
        setattr(ds, key, value)
    ds.save_as(path, enforce_file_format=True)
    return path


@pytest.fixture
def gradient() -> npt.NDArray[np.uint16]:
    """16x16 horizontal ramp 0..1500, so left is dark and right is bright in MONOCHROME2."""
    return np.tile(np.linspace(0, 1500, 16), (16, 1)).astype(np.uint16)


class TestWindow:
    def test_linear_window_follows_dicom_formula(self) -> None:
        # center 50, width 101: x <= -0.5 → 0, x > 99.5 → 1, linear in between
        values = np.array([-10.0, -0.5, 49.5, 99.5, 200.0], dtype=np.float32)
        out = apply_window(values, center=50, width=101)
        np.testing.assert_allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-6)

    def test_width_one_is_a_step(self) -> None:
        values = np.array([9.0, 9.5, 10.0], dtype=np.float32)
        np.testing.assert_array_equal(apply_window(values, center=10, width=1), [0, 0, 1])

    def test_rejects_width_below_one(self) -> None:
        with pytest.raises(ValueError):
            apply_window(np.zeros(3, dtype=np.float32), center=0, width=0.5)

    def test_min_max_constant_image_is_zero(self) -> None:
        assert not min_max(np.full((4, 4), 7.0, dtype=np.float32)).any()


class TestDicom:
    def test_monochrome1_is_inverted(
        self, tmp_path: Path, gradient: npt.NDArray[np.uint16]
    ) -> None:
        m2 = load_image(write_dicom(tmp_path / "m2.dcm", gradient, "MONOCHROME2"))
        m1 = load_image(write_dicom(tmp_path / "m1.dcm", gradient, "MONOCHROME1"))

        np.testing.assert_allclose(m1, 1.0 - m2, atol=1e-6)
        # Highest stored value: white in MONOCHROME2, black in MONOCHROME1.
        assert m2[0, -1] == pytest.approx(1.0)
        assert m1[0, -1] == pytest.approx(0.0)

    def test_monochrome1_inverts_after_windowing(
        self, tmp_path: Path, gradient: npt.NDArray[np.uint16]
    ) -> None:
        path = write_dicom(
            tmp_path / "m1.dcm", gradient, "MONOCHROME1", WindowCenter=500, WindowWidth=400
        )
        expected = 1.0 - apply_window(gradient.astype(np.float32), 500, 400)
        np.testing.assert_allclose(load_image(path), expected, atol=1e-6)

    def test_rescale_then_window_from_tags(
        self, tmp_path: Path, gradient: npt.NDArray[np.uint16]
    ) -> None:
        path = write_dicom(
            tmp_path / "w.dcm",
            gradient,
            "MONOCHROME2",
            RescaleSlope=2,
            RescaleIntercept=-100,
            WindowCenter=1000,
            WindowWidth=800,
        )
        expected = apply_window(gradient.astype(np.float32) * 2 - 100, 1000, 800)
        np.testing.assert_allclose(load_image(path), expected, atol=1e-6)

    def test_multi_valued_window_uses_first_pair(
        self, tmp_path: Path, gradient: npt.NDArray[np.uint16]
    ) -> None:
        path = write_dicom(
            tmp_path / "multi.dcm",
            gradient,
            "MONOCHROME2",
            WindowCenter=[600, 50],
            WindowWidth=[300, 10],
        )
        expected = apply_window(gradient.astype(np.float32), 600, 300)
        np.testing.assert_allclose(load_image(path), expected, atol=1e-6)

    def test_without_window_uses_full_range(
        self, tmp_path: Path, gradient: npt.NDArray[np.uint16]
    ) -> None:
        image = load_image(write_dicom(tmp_path / "plain.dcm", gradient, "MONOCHROME2"))
        assert image.dtype == np.float32
        assert image.min() == 0.0 and image.max() == pytest.approx(1.0)

    def test_detected_by_magic_not_extension(
        self, tmp_path: Path, gradient: npt.NDArray[np.uint16]
    ) -> None:
        path = write_dicom(tmp_path / "scan.png", gradient, "MONOCHROME2")
        assert is_dicom(path)

    def test_rejects_colour(self, tmp_path: Path, gradient: npt.NDArray[np.uint16]) -> None:
        path = write_dicom(tmp_path / "rgb.dcm", gradient, "PALETTE COLOR")
        with pytest.raises(ValueError, match="PhotometricInterpretation"):
            load_image(path)

    @pytest.mark.parametrize("function", ["SIGMOID", "LINEAR_EXACT"])
    def test_rejects_non_linear_voi_function(
        self, tmp_path: Path, gradient: npt.NDArray[np.uint16], function: str
    ) -> None:
        path = write_dicom(
            tmp_path / "voi.dcm",
            gradient,
            "MONOCHROME2",
            WindowCenter=500,
            WindowWidth=400,
            VOILUTFunction=function,
        )
        with pytest.raises(ValueError, match="VOILUTFunction"):
            load_image(path)

    def test_explicit_linear_voi_function_is_windowed(
        self, tmp_path: Path, gradient: npt.NDArray[np.uint16]
    ) -> None:
        path = write_dicom(
            tmp_path / "voi.dcm",
            gradient,
            "MONOCHROME2",
            WindowCenter=500,
            WindowWidth=400,
            VOILUTFunction="LINEAR",
        )
        expected = apply_window(gradient.astype(np.float32), 500, 400)
        np.testing.assert_allclose(load_image(path), expected, atol=1e-6)

    def test_rejects_voi_lut_sequence(
        self, tmp_path: Path, gradient: npt.NDArray[np.uint16]
    ) -> None:
        item = Dataset()
        item.LUTDescriptor = [4, 0, 16]
        item.LUTData = np.array([0, 100, 200, 300], dtype="<u2").tobytes()
        path = write_dicom(tmp_path / "lut.dcm", gradient, "MONOCHROME2", VOILUTSequence=[item])
        with pytest.raises(ValueError, match="VOI LUT Sequence"):
            load_image(path)


class TestPng:
    def test_8bit(self, tmp_path: Path) -> None:
        pixels = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / "a.png")
        image = load_image(tmp_path / "a.png")
        assert not is_dicom(tmp_path / "a.png")
        np.testing.assert_allclose(image, pixels / 255.0, atol=1e-6)

    def test_16bit(self, tmp_path: Path) -> None:
        pixels = np.array([[0, 1000], [65535, 32768]], dtype=np.uint16)
        Image.fromarray(pixels).save(tmp_path / "b.png")
        np.testing.assert_allclose(load_image(tmp_path / "b.png"), pixels / 65535.0, atol=1e-6)

    def test_rgb_is_converted_to_gray(self, tmp_path: Path) -> None:
        pixels = np.full((4, 4, 3), 200, dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / "c.png")
        image = load_image(tmp_path / "c.png")
        assert image.shape == (4, 4)
        np.testing.assert_allclose(image, 200 / 255.0, atol=1e-6)


class TestPreprocess:
    def test_shape_dtype_and_normalization(self) -> None:
        out = preprocess(np.full((300, 300), 0.5, dtype=np.float32), size=128)
        assert out.shape == (1, 3, 128, 128)
        assert out.dtype == np.float32
        expected = (0.5 - IMAGENET_MEAN) / IMAGENET_STD
        np.testing.assert_allclose(out[0, :, 64, 64], expected, atol=1e-5)

    def test_non_square_input_is_resized_to_square(self) -> None:
        assert preprocess(np.zeros((200, 100), dtype=np.float32), size=64).shape == (1, 3, 64, 64)

    def test_orientation_is_preserved(self) -> None:
        image = np.zeros((256, 256), dtype=np.float32)
        image[:, 128:] = 1.0  # right half bright
        out = preprocess(image, size=32)[0, 0]
        assert out[:, 24:].mean() > out[:, :8].mean()

    def test_is_deterministic(self) -> None:
        image = np.random.default_rng(0).random((100, 100), dtype=np.float32)
        np.testing.assert_array_equal(preprocess(image, 64), preprocess(image, 64))

    @pytest.mark.parametrize("bad", [np.zeros((2, 4, 4)), np.array([[np.nan, 0.0]])])
    def test_rejects_bad_input(self, bad: npt.NDArray[np.float32]) -> None:
        with pytest.raises(ValueError):
            preprocess(bad.astype(np.float32), size=8)
