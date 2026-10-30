import cv2
import numpy as np
import pytest

from opencv_tutorial.algorithms import (
    binary_threshold,
    correlate2d,
    gaussian_kernel,
    gaussian_manual,
    make_sample,
    otsu_manual,
    sobel_manual,
)
from opencv_tutorial.geometry import detect_document, order_corners, warp_document
from opencv_tutorial.pipeline import ALGORITHMS, run_pipeline


def test_correlation_matches_opencv():
    image = np.random.default_rng(12).integers(0, 256, size=(31, 43), dtype=np.uint8)
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    got = correlate2d(image, kernel)
    ref = cv2.filter2D(image, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT_101)
    np.testing.assert_allclose(got, ref, atol=1e-8)


def test_gaussian_kernel_and_reference():
    kernel = gaussian_kernel(5, 1.2)
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1])
    assert kernel.sum() == pytest.approx(1)
    image = np.random.default_rng(44).integers(0, 256, size=(60, 70), dtype=np.uint8)
    manual = gaussian_manual(image, 5, 1.2)
    reference = cv2.GaussianBlur(image, (5, 5), 1.2)
    assert np.mean(abs(manual.astype(float) - reference)) < 1.0


def test_sobel_gradient_reference():
    image = np.random.default_rng(6).integers(0, 256, size=(50, 64), dtype=np.uint8)
    gx, gy, magnitude = sobel_manual(image)
    np.testing.assert_allclose(gx, cv2.Sobel(image, cv2.CV_64F, 1, 0))
    np.testing.assert_allclose(gy, cv2.Sobel(image, cv2.CV_64F, 0, 1))
    np.testing.assert_allclose(magnitude, np.hypot(gx, gy))


def test_otsu_matches_opencv():
    rng = np.random.default_rng(22)
    image = np.concatenate([rng.normal(50, 9, 4000), rng.normal(195, 12, 4000)])
    image = np.clip(image, 0, 255).astype(np.uint8).reshape(80, 100)
    threshold = otsu_manual(image)
    cv_threshold, ref = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert threshold == int(cv_threshold)
    np.testing.assert_array_equal(binary_threshold(image, threshold), ref)


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_all_pipeline_algorithms(name):
    image = make_sample(224, 160)
    result = run_pipeline(image, name)
    assert result.image.shape[:2] == image.shape[:2]
    assert result.image.dtype == np.uint8
    assert "grayscale" in result.stages
    assert result.metrics["elapsed_ms"] >= 0
    assert result.metrics["algorithm"] == name


def test_parameter_validation():
    image = make_sample(128, 128)
    with pytest.raises(ValueError):
        run_pipeline(image, "gaussian", kernel=4)
    with pytest.raises(ValueError):
        run_pipeline(image, "unknown")
    with pytest.raises(ValueError):
        gaussian_kernel(2, -1)


def test_warp_document_corners():
    image = np.zeros((180, 260, 3), dtype=np.uint8)
    cv2.rectangle(image, (45, 30), (210, 150), (240, 240, 240), -1)
    corners = detect_document(image)
    assert corners is not None
    result = warp_document(image, corners)
    assert result.shape[0] >= 110 and result.shape[1] >= 150
    assert result.mean() > 200
    assert order_corners(np.array([[210, 150], [45, 30], [45, 150], [210, 30]])).shape == (4, 2)
