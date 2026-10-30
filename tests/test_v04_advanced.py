"""v0.4: numeric geometry, negative cases, local ONNX, HTTP + CLI regression tests."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

from opencv_tutorial.onnx_lab import benchmark_onnx, load_net, make_test_onnx, run_onnx
from opencv_tutorial.pose3d import (
    integrate_unit_poses,
    pnp_fixture_report,
    relative_pose,
    solve_pnp_ransac,
    synthetic_correspondences,
    vo_fixture_report,
)
from opencv_tutorial.server import LabHandler, advanced_payload, encode_image
from opencv_tutorial.stereo import stereo_depth, stereo_fixture_report, synthetic_stereo


def test_stereo_recovers_shift_in_metric_units():
    left, right = synthetic_stereo(seed=11, disparity_px=8)
    result = stereo_depth(left, right, fx_px=420., baseline_m=.12)
    interior = result.valid[35:-35, 90:-35]
    assert interior.mean() > .9
    assert abs(float(np.median(result.disparity[35:-35, 90:-35][interior])) - 8) < .3
    assert abs(float(np.median(result.depth_m[35:-35, 90:-35][interior])) - 6.3) < .3
    assert np.isnan(result.depth_m[~result.valid]).all()
    assert np.all(result.visualization[~result.valid] == 0)
    assert result.stats["rectification"] == "assumed; not verified"


@pytest.mark.parametrize("fx,baseline", [(0,.12), (-1,.12), (np.nan,.12),
                                         (400,0), (400,np.inf)])
def test_stereo_rejects_invalid_calibration(fx, baseline):
    a, b = synthetic_stereo()
    with pytest.raises(ValueError):
        stereo_depth(a, b, fx_px=fx, baseline_m=baseline)


def test_stereo_rejects_nonrectified_inputs_and_invalid_search():
    a, b = synthetic_stereo()
    for first, second, kw in [(a, b[:100], {}), (a.astype(np.float32), b, {}),
                              (a, b, {"num_disparities": 31}),
                              (a[:, :50], b[:, :50], {})]:
        with pytest.raises(ValueError):
            stereo_depth(first, second, fx_px=400., baseline_m=.1, **kw)
    # No geometric rectification verification is claimed; image-shape checks alone cannot prove it.


def test_pnp_resists_outliers_and_has_metric_pose():
    report = pnp_fixture_report(17)
    assert report["stats"]["inliers"] >= 220
    assert report["stats"]["reprojection_rmse_px"] < 1
    assert report["rotation_error_deg"] < .2
    assert report["translation_l2_error_world_units"] < .01


def test_pnp_rejection_bad_camera_and_correspondences():
    xyz, _, uv, K, _, _ = synthetic_correspondences()
    with pytest.raises(ValueError):
        solve_pnp_ransac(xyz, uv[:-1], K)
    with pytest.raises(ValueError):
        solve_pnp_ransac(xyz, uv, np.eye(4))
    with pytest.raises(ValueError):
        solve_pnp_ransac(xyz, uv, K, reprojection_px=-3)


def test_essential_relative_vo_measures_direction_not_scale():
    xyz, a, b, K, R, t = synthetic_correspondences()
    _ = xyz
    pose = relative_pose(a, b, K)
    assert len(pose.inliers) > 170
    assert np.linalg.norm(pose.translation) == pytest.approx(1., abs=.00001)
    assert pose.stats["translation_scale"] == "unknown (unit direction only)"
    assert np.dot(pose.translation, t) / np.linalg.norm(t) > .98
    assert np.linalg.norm(pose.rotation - R) < .005
    traj = integrate_unit_poses([pose, pose])
    assert len(traj) == 3
    assert np.linalg.norm(traj[1]) == pytest.approx(1, abs=.00001)


def test_pose_fixture_deterministic():
    assert stereo_fixture_report(17) == stereo_fixture_report(17)
    assert pnp_fixture_report(17) == pnp_fixture_report(17)
    assert vo_fixture_report(17) == vo_fixture_report(17)
    assert vo_fixture_report(17)["translation_direction_cosine"] > .98


def test_relative_pose_rejects_correspondence_mismatch():
    _, a, b, K, _, _ = synthetic_correspondences()
    with pytest.raises(ValueError):
        relative_pose(a, b[:12], K)
    with pytest.raises(ValueError):
        relative_pose(a, b, K, threshold_px=0)


def test_local_onnx_fixture_io_and_parity(tmp_path: Path):
    path = tmp_path / "mean.onnx"
    digest = make_test_onnx(path)
    image = np.zeros((16, 16, 3), np.uint8)
    image[:] = (30, 60, 90)
    result = run_onnx(path, image, expected_sha256=digest)
    assert result["input_shape"] == [1, 3, 16, 16]
    assert result["output_shape"] == [1, 3]
    assert result["raw_output"] == pytest.approx([90/255, 60/255, 30/255], abs=1e-5)
    assert result["note"].startswith("Raw model outputs")
    assert digest == make_test_onnx(path)


def test_local_onnx_integrity_and_missing_rejection(tmp_path: Path):
    path = tmp_path / "fixed.onnx"
    digest = make_test_onnx(path)
    with pytest.raises(ValueError, match="SHA256"):
        load_net(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="local"):
        load_net(tmp_path / "missing.onnx")
    with pytest.raises(ValueError):
        load_net(path, backend="notreal")
    assert len(digest) == 64


def test_dnn_benchmark_warmup_and_backend(tmp_path: Path):
    model = tmp_path / "fixture.onnx"
    make_test_onnx(model)
    data = benchmark_onnx(model, rounds=6, warmups=2)
    assert data["rounds"] == 6 and data["warmups"] == 2
    assert 0 <= data["median_ms"] <= data["p95_ms"] <= data["max_ms"]
    assert "excludes image decode" in data["scope"]
    with pytest.raises(ValueError):
        benchmark_onnx(model, rounds=0)
    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        with pytest.raises(RuntimeError, match="unavailable"):
            load_net(model, backend="cuda")


def test_advanced_payload_stereo_and_rejection():
    a, b = synthetic_stereo()
    data = advanced_payload({"first": encode_image(a), "second": encode_image(b),
                             "fx_px": 420, "baseline_m": .12}, "stereo")
    assert data["metrics"]["median_depth_m"] == pytest.approx(6.3, abs=.2)
    assert data["image"].startswith("data:image/png;base64,")
    with pytest.raises(ValueError):
        advanced_payload({"first": encode_image(a), "second": encode_image(b),
                          "fx_px": -1}, "stereo")
    with pytest.raises(ValueError):
        advanced_payload({}, "undefined")


@pytest.fixture
def api_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), LabHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def _post(url, data):
    req = Request(url, json.dumps(data).encode(),
                  {"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def test_http_stereo_and_fixture_and_v04_static(api_url):
    with urlopen(api_url + "/api/advanced/sample") as response:
        sample = json.load(response)
    assert sample["fixture"]["expected_depth_m"] == 6.3
    with urlopen(api_url + "/api/advanced/geometry-demo") as response:
        geometry = json.load(response)
    assert geometry["pnp"]["stats"]["inliers"] >= 200
    assert geometry["vo"]["stats"]["translation_scale"].startswith("unknown")
    with urlopen(api_url + "/advanced.js") as response:
        assert b"advancedMode" in response.read()
    data = _post(api_url + "/api/advanced/stereo",
                 {"first": sample["stereo"][0], "second": sample["stereo"][1],
                  **sample["calibration"]})
    assert abs(data["metrics"]["median_disparity_px"] - 8) < .5
    with pytest.raises(HTTPError) as exc:
        _post(api_url + "/api/advanced/stereo", {"first": sample["stereo"][0],
                                                 "second": sample["stereo"][1],
                                                 "fx_px": -1})
    assert exc.value.code == 400


def test_cli_advanced_and_onnx_bench(tmp_path: Path):
    report = tmp_path / "advanced.json"
    cmd = [sys.executable, "-m", "opencv_tutorial"]
    res = subprocess.run(cmd + ["evaluate-advanced", "--output", str(report)],
                         capture_output=True, text=True, check=True, timeout=40)
    assert json.loads(res.stdout)["pnp"]["stats"]["inliers"] >= 200
    assert report.exists()
    model = tmp_path / "toy.onnx"
    res = subprocess.run(cmd + ["onnx-fixture", "--output", str(model)],
                         capture_output=True, text=True, check=True, timeout=20)
    digest = json.loads(res.stdout)["sha256"]
    assert digest == make_test_onnx(model)
    bench = subprocess.run(cmd + ["bench-dnn", "--model", str(model),
                                  "--sha256", digest, "--rounds", "4"],
                           capture_output=True, text=True, check=True, timeout=25)
    assert json.loads(bench.stdout)["rounds"] == 4
    left, right = synthetic_stereo()
    for name, image in (("left.png", left), ("right.png", right)):
        assert cv2.imwrite(str(tmp_path / name), image)
    target = tmp_path / "stereo.png"
    run = subprocess.run(cmd + ["stereo", "--left", str(tmp_path / "left.png"),
                                "--right", str(tmp_path / "right.png"),
                                "--fx", "420", "--baseline", ".12",
                                "--output", str(target)],
                         capture_output=True, text=True, check=True, timeout=25)
    assert target.is_file() and json.loads(run.stdout)["valid_pixels"] > 30000
