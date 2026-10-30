"""Local-only OpenCV DNN inference and tiny dependency-free ONNX test fixture.

The fixture emits ONNX protobuf wire format for a fixed GlobalAveragePool -> Flatten
model. It is for testing I/O and backend parity, never an accuracy demonstration.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("Expected nonnegative integer")
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _int(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def _bytes(field: int, raw: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(raw)) + raw


def _s(field: int, value: str) -> bytes:
    return _bytes(field, value.encode("utf-8"))


def _tensor_value(name: str, dims: tuple[int, ...]) -> bytes:
    shape = b"".join(_bytes(1, _int(1, d)) for d in dims)
    ttype = _int(1, 1) + _bytes(2, shape)  # float32, tensor shape
    return _s(1, name) + _bytes(2, _bytes(1, ttype))


def make_test_onnx(path: Path) -> str:
    """Create tiny 1x3x16x16 -> 1x3 channel mean ONNX (opset 11, no onnx dependency)."""
    a = _s(1, "input") + _s(2, "pooled") + _s(3, "avg") + _s(4, "GlobalAveragePool")
    b = _s(1, "pooled") + _s(2, "output") + _s(3, "flatten") + _s(4, "Flatten")
    graph = (_bytes(1, a) + _bytes(1, b) + _s(2, "test_average")
             + _bytes(11, _tensor_value("input", (1, 3, 16, 16)))
             + _bytes(12, _tensor_value("output", (1, 3))))
    model = (_int(1, 7) + _s(2, "opencv-tutorial") + _bytes(7, graph)
             + _bytes(8, _int(2, 11)))  # ONNX domain '' -> opset 11
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model)
    return sha256(model).hexdigest()


def load_net(model_path: Path, *, expected_sha256: str | None = None,
             backend: str = "cpu") -> cv2.dnn_Net:
    path = Path(model_path)
    if not path.is_file() or path.suffix.lower() != ".onnx" or not 0 < path.stat().st_size <= 64 * 1024**2:
        raise ValueError("Expected existing local .onnx model of size <=64 MiB")
    digest = sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError("Model SHA256 mismatch")
    if backend not in ("cpu", "cuda"):
        raise ValueError("Backend must be cpu or cuda")
    if backend == "cuda" and (not hasattr(cv2, "cuda") or
                              cv2.cuda.getCudaEnabledDeviceCount() < 1):
        raise RuntimeError("CUDA backend unavailable; refusing silent CPU fallback")
    try:
        net = cv2.dnn.readNetFromONNX(str(path))
        if backend == "cuda":
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    except cv2.error as exc:
        raise ValueError(f"Cannot load ONNX model: {str(exc)[:220]}") from exc


def run_onnx(model_path: Path, image: np.ndarray, *, input_width: int = 16,
             input_height: int = 16, scale: float = 1 / 255,
             swap_rb: bool = True, backend: str = "cpu",
             expected_sha256: str | None = None) -> dict:
    """Generic raw tensor inference; output is NOT a class probability by default."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
        raise ValueError("Input must be nonempty uint8 BGR image")
    if not (1 <= input_width <= 2048 and 1 <= input_height <= 2048) or not np.isfinite(scale):
        raise ValueError("Invalid model input dimensions or scale")
    net = load_net(model_path, backend=backend, expected_sha256=expected_sha256)
    blob = cv2.dnn.blobFromImage(image, scalefactor=scale,
                                size=(input_width, input_height),
                                mean=(0, 0, 0), swapRB=swap_rb, crop=False)
    try:
        net.setInput(blob)
        start = perf_counter()
        output = net.forward()
        latency_ms = (perf_counter() - start) * 1000
    except cv2.error as exc:
        raise ValueError(f"ONNX execution failed: {str(exc)[:220]}") from exc
    if not np.all(np.isfinite(output)):
        raise ValueError("Model returned nonfinite output")
    return {"backend": backend, "model_sha256": sha256(Path(model_path).read_bytes()).hexdigest(),
            "input_shape": list(blob.shape), "output_shape": list(output.shape),
            "raw_output": output.flatten().astype(float).tolist()[:128],
            "truncated": output.size > 128,
            "latency_ms": float(latency_ms),
            "note": "Raw model outputs; not calibrated probabilities or labels"}


def benchmark_onnx(model_path: Path, *, rounds: int = 15,
                   warmups: int = 3, backend: str = "cpu", seed: int = 17,
                   image_size: int = 16, expected_sha256: str | None = None) -> dict:
    if not 2 <= rounds <= 500 or not 0 <= warmups <= 50:
        raise ValueError("Invalid benchmark iteration count")
    if not 1 <= image_size <= 2048:
        raise ValueError("Invalid image size")
    net = load_net(model_path, backend=backend, expected_sha256=expected_sha256)
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (image_size, image_size, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (image_size, image_size), swapRB=True)
    timings = []
    try:
        for i in range(warmups + rounds):
            net.setInput(blob)
            start = perf_counter()
            out = net.forward()
            elapsed = (perf_counter() - start) * 1000
            if i >= warmups:
                timings.append(elapsed)
    except cv2.error as exc:
        raise ValueError(f"Inference benchmark failed: {str(exc)[:220]}") from exc
    return {"backend": backend, "rounds": rounds, "warmups": warmups, "seed": seed,
            "tensor_shape": list(blob.shape), "output_shape": list(out.shape),
            "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
            "mean_ms": float(np.mean(timings)), "min_ms": float(np.min(timings)),
            "max_ms": float(np.max(timings)),
            "scope": "forward only; excludes image decode, blob creation and model loading"}
