import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pytest

from opencv_tutorial.algorithms import make_sample
from opencv_tutorial.curriculum import get_curriculum
from opencv_tutorial.motion import MotionDetector
from opencv_tutorial.server import LabHandler, decode_image, encode_image, process_payload


def test_curriculum_covers_24_weeks():
    course = get_curriculum()
    assert len(course) == 24
    assert [w["id"] for w in course] == [f"W{n:02d}" for n in range(1, 25)]
    assert {w["level"] for w in course} == {"Beginner", "Intermediate", "Advanced"}


def test_image_roundtrip_and_api_payload():
    original = make_sample(160, 128)
    encoded = encode_image(original)
    np.testing.assert_array_equal(decode_image(encoded), original)
    result = process_payload({"image": encoded, "algorithm": "sobel", "kernel": 5})
    assert {"original", "grayscale", "result"}.issubset(result["images"])
    assert result["metrics"]["width"] == 160
    with pytest.raises(ValueError):
        decode_image("data:image/png;base64,not-valid")


def test_motion_detector_suppresses_stationary_background():
    detector = MotionDetector(history=20, min_area=30)
    background = np.zeros((100, 120, 3), dtype=np.uint8)
    for _ in range(25):
        result = detector.step(background)
    assert np.count_nonzero(result.mask) == 0
    active = background.copy()
    cv2.rectangle(active, (35, 30), (75, 70), (255, 255, 255), -1)
    result = detector.step(active, learning_rate=0)
    assert len(result.boxes) >= 1


@pytest.fixture
def live_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), LabHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_http_endpoints(live_server):
    with urlopen(live_server + "/api/health", timeout=4) as response:
        assert json.load(response)["status"] == "ok"
    with urlopen(live_server + "/api/curriculum", timeout=4) as response:
        assert len(json.load(response)) == 24
    with urlopen(live_server + "/", timeout=4) as response:
        assert b"VISION" in response.read()
    image = encode_image(make_sample(160, 128))
    request = Request(live_server + "/api/process", json.dumps({"image": image, "algorithm": "otsu"}).encode(),
                      {"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=10) as response:
        assert json.load(response)["metrics"]["algorithm"] == "otsu"
