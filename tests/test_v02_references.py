"""Independent numerical and boundary tests for the v0.2 teaching references."""
import cv2
import numpy as np
import pytest

from opencv_tutorial.references import (
    adaptive_mean_manual, equalize_hist_manual, histogram_manual,
    median_manual, morphology_manual,
)
from opencv_tutorial.evaluation import compare_images, mask_scores, psnr, ssim


def test_median_opencv_parity_including_boundaries():
    rng = np.random.default_rng(42)
    for shape in ((1, 1), (2, 3), (13, 17), (31, 31)):
        src = rng.integers(0, 256, size=shape, dtype=np.uint8)
        for size in (3, 5):
            np.testing.assert_array_equal(median_manual(src, size), cv2.medianBlur(src, size))


@pytest.mark.parametrize("op,cvop", [
    ("erode", cv2.MORPH_ERODE), ("dilate", cv2.MORPH_DILATE),
    ("open", cv2.MORPH_OPEN), ("close", cv2.MORPH_CLOSE),
])
def test_morphology_rectangular_parity(op, cvop):
    src = np.random.default_rng(55).integers(0, 256, (39, 46), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    expected = cv2.morphologyEx(src, cvop, kernel)
    np.testing.assert_array_equal(morphology_manual(src, 5, op), expected)


def test_adaptive_mean_reference_near_opencv():
    src = np.random.default_rng(8).integers(0, 256, (31, 27), dtype=np.uint8)
    ref = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 5, 3)
    predicted = adaptive_mean_manual(src, 5, 3)
    # OpenCV rounds box-filtered local mean to uint8 before thresholding.
    assert np.mean(ref != predicted) < .02


def test_histogram_and_equalization():
    src = np.random.default_rng(2).integers(0, 256, (63, 57), dtype=np.uint8)
    assert histogram_manual(src).sum() == src.size
    assert np.max(np.abs(equalize_hist_manual(src).astype(int) -
                         cv2.equalizeHist(src).astype(int))) <= 1
    constant = np.full((17, 15), 27, np.uint8)
    np.testing.assert_array_equal(equalize_hist_manual(constant), constant)


def test_quality_reference_and_mask_score():
    src = np.zeros((30, 40), dtype=np.uint8)
    src[10:20, 10:20] = 255
    assert psnr(src, src) == float("inf")
    assert ssim(src, src) == pytest.approx(1, abs=1e-12)
    assert compare_images(src, src)["identical"] is True
    shifted = np.roll(src, 1, axis=0)
    assert psnr(src, shifted) < float("inf")
    assert mask_scores(src, src) == {"iou": 1, "dice": 1}
    assert mask_scores(src, shifted)["iou"] < 1
    assert mask_scores(np.zeros_like(src), np.zeros_like(src))["iou"] == 1
    with pytest.raises(ValueError):
        ssim(src[:4], src[:4])
    with pytest.raises(ValueError):
        compare_images(src, src[:, :-1])


def test_reference_resource_limits():
    with pytest.raises(ValueError):
        median_manual(np.zeros((800, 800), np.uint8), 7)
    with pytest.raises(ValueError):
        morphology_manual(np.zeros((20, 20), np.uint8), 4)
    with pytest.raises(ValueError):
        adaptive_mean_manual(np.zeros((20, 20), np.uint8), 3, float("nan"))
