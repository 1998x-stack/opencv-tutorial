"""Independent numeric, rejection, association, and HTTP/CLI regression tests."""
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

from opencv_tutorial.intermediate import (benchmark_synthetic, synthetic_panorama_pair,
                                          synthetic_tracking_frames, synthetic_translation,
                                          track_synthetic)
from opencv_tutorial.optical_flow import flow_endpoint_error, sparse_lk
from opencv_tutorial.panorama import stitch_pair
from opencv_tutorial.server import LabHandler, encode_image, intermediate_payload
from opencv_tutorial.tracking import CentroidTracker


def test_panorama_recovers_known_translation():
    first, second = synthetic_panorama_pair(17)
    result = stitch_pair(first, second)
    assert result.stats["inliers"] >= 90
    assert result.stats["inlier_ratio"] > .8
    assert abs(float(result.homography[0, 2]) + 230) < 2
    assert abs(float(result.homography[1, 2])) < 2
    assert 640 <= result.image.shape[1] <= 660
    assert result.stats["overlap_pixels"] > 45_000
    assert result.image.shape[:2] == result.overlap.shape
    assert np.count_nonzero(result.overlap) > 0


def test_panorama_preserves_true_black_overlap_pixels():
    a, b = synthetic_panorama_pair(17)
    a[120:160, 250:280] = 0  # true black is valid, not a missing pixel mask
    result = stitch_pair(a, b)
    assert result.stats["overlap_pixels"] > 45_000
    assert result.image.dtype == np.uint8


def test_panorama_rejects_blank_and_size_explosion():
    blank = np.full((180, 220, 3), 127, dtype=np.uint8)
    with pytest.raises(ValueError, match="Insufficient"):
        stitch_pair(blank, blank)
    a, b = synthetic_panorama_pair()
    with pytest.raises(ValueError, match="canvas limit"):
        stitch_pair(a, b, max_canvas_pixels=4096)
    with pytest.raises(ValueError):
        stitch_pair(a[:, :, 0], b)


def test_flow_recovers_known_translation_with_outlier_filter():
    first, second = synthetic_translation(19, dx=7, dy=4)
    flow = sparse_lk(first, second)
    error = flow_endpoint_error(flow, 7, 4)
    assert flow.stats["detected"] >= 100
    assert flow.stats["tracked"] >= 40
    assert abs(flow.stats["median_dx"] - 7) < .35
    assert abs(flow.stats["median_dy"] - 4) < .35
    assert error["median_epe"] < .5
    assert np.all(flow.forward_backward_error <= 1.5)
    assert flow.annotated.shape == second.shape


def test_flow_empty_weak_texture_and_bad_pair():
    blank = np.zeros((128, 160, 3), dtype=np.uint8)
    flow = sparse_lk(blank, blank)
    assert flow.stats["tracked"] == 0
    assert flow_endpoint_error(flow, 1, 2)["mean_epe"] is None
    with pytest.raises(ValueError, match="dimensions"):
        sparse_lk(blank, blank[:100])
    with pytest.raises(ValueError, match="Invalid optical"):
        sparse_lk(blank, blank, fb_max_error=0)


def test_tracker_ids_and_ttl_and_gate():
    tracker = CentroidTracker(max_distance=20, max_missed=1)
    t1 = tracker.update([(10, 20, 10, 10), (150, 20, 10, 10)])
    assert [t.id for t in t1] == [1, 2]
    t2 = tracker.update([(13, 21, 10, 10), (146, 20, 10, 10)])
    assert [t.id for t in t2] == [1, 2]
    t3 = tracker.update([(17, 24, 10, 10)])
    assert [(t.id, t.missed) for t in t3] == [(1, 0), (2, 1)]
    t4 = tracker.update([(17, 24, 10, 10), (300, 30, 10, 10)])
    assert [t.id for t in t4] == [1, 3]
    assert tracker.next_id == 4
    assert len(tracker.tracks[1].path) == 4


def test_tracker_duplicates_and_invalid_inputs():
    tracker = CentroidTracker(max_distance=50, max_missed=0)
    assert len(tracker.update([(10, 10, 10, 10)])) == 1
    assert len(tracker.update([(11, 10, 10, 10), (12, 10, 10, 10)])) == 2
    assert len({t.id for t in tracker.tracks.values()}) == 2
    with pytest.raises(ValueError):
        tracker.update([(1, 2, -3, 4)])
    with pytest.raises(ValueError):
        CentroidTracker(max_distance=float('nan'))


def test_synthetic_evaluation_is_deterministic():
    one = benchmark_synthetic(17)
    two = benchmark_synthetic(17)
    assert one == two
    assert one["flow_error"]["median_epe"] < .5
    assert one["panorama"]["inliers"] > 90
    assert len(synthetic_tracking_frames(4)) == 4
    frames, steps = track_synthetic(5)
    assert len(frames) == len(steps) == 5
    assert [s["tracks"][0]["id"] for s in steps] == [1] * 5


def test_intermediate_payload_encodes_real_result():
    a, b = synthetic_panorama_pair()
    out = intermediate_payload({"first": encode_image(a), "second": encode_image(b)},
                               "panorama")
    assert out["metrics"]["inliers"] > 90
    assert out["image"].startswith("data:image/png;base64,")
    a, b = synthetic_translation()
    flow = intermediate_payload({"first": encode_image(a), "second": encode_image(b)},
                                "flow")
    assert flow["metrics"]["tracked"] > 20
    with pytest.raises(ValueError):
        intermediate_payload({}, "unsupported")


@pytest.fixture
def server_url():
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
    request = Request(url, json.dumps(data).encode(),
                      {"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def test_http_new_endpoints_and_rejection(server_url):
    with urlopen(server_url + "/api/intermediate/sample", timeout=5) as response:
        pairs = json.load(response)
    assert len(pairs["panorama"]) == len(pairs["flow"]) == 2
    out = _post(server_url + "/api/intermediate/panorama",
                dict(zip(["first", "second"], pairs["panorama"])))
    assert out["metrics"]["inliers"] > 90
    flow = _post(server_url + "/api/intermediate/flow",
                 dict(zip(["first", "second"], pairs["flow"])))
    assert flow["metrics"]["tracked"] > 20
    with urlopen(server_url + "/api/intermediate/tracking-demo", timeout=10) as response:
        tracks = json.load(response)
    assert len(tracks["frames"]) == len(tracks["events"]) == 8
    with urlopen(server_url + "/intermediate.js", timeout=5) as response:
        assert b"sparse" not in response.read()  # execution is done in Python backend
    blank = np.full((180, 220, 3), 127, dtype=np.uint8)
    with pytest.raises(HTTPError) as error:
        _post(server_url + "/api/intermediate/panorama",
              {"first": encode_image(blank), "second": encode_image(blank)})
    assert error.value.code == 400


def test_cli_end_to_end(tmp_path: Path):
    a, b = synthetic_panorama_pair()
    f1, f2 = synthetic_translation()
    for name, image in [("a.png", a), ("b.png", b), ("f1.png", f1), ("f2.png", f2)]:
        assert cv2.imwrite(str(tmp_path / name), image)
    for command, first, second in [("panorama", "a.png", "b.png"),
                                   ("flow", "f1.png", "f2.png")]:
        output = tmp_path / (command + ".png")
        result = subprocess.run([sys.executable, "-m", "opencv_tutorial", command,
                                 "--first", str(tmp_path / first),
                                 "--second", str(tmp_path / second),
                                 "--output", str(output)],
                                capture_output=True, text=True, timeout=15, check=True)
        assert output.exists()
        assert json.loads(result.stdout)
    report = tmp_path / "v03-report.json"
    subprocess.run([sys.executable, "-m", "opencv_tutorial", "evaluate-intermediate",
                    "--output", str(report)], capture_output=True, check=True, timeout=15)
    assert json.loads(report.read_text())["flow_error"]["valid"]
