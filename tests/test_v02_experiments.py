import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from opencv_tutorial.algorithms import make_sample
from opencv_tutorial.experiments import STUDIES, apply_noise, run_study
from opencv_tutorial.server import ablation_payload, encode_image
from test_server_and_motion import live_server  # noqa: F401


def test_noise_reproducibility_and_safe_input():
    img = make_sample(160, 128)
    first = apply_noise(img, "gaussian", 15, 7)
    second = apply_noise(img, "gaussian", 15, 7)
    np.testing.assert_array_equal(first.observed, second.observed)
    np.testing.assert_array_equal(first.clean, img)
    assert not np.array_equal(first.observed, img)
    pepper = apply_noise(img, "salt_pepper", 25, 55)
    assert np.count_nonzero(pepper.observed != img) > 0
    with pytest.raises(ValueError):
        apply_noise(img, "bad")
    with pytest.raises(ValueError):
        apply_noise(img, "gaussian", seed=-1)


def test_sweep_fixes_input_reference_and_parameters():
    img = make_sample(160, 128)
    study = run_study(img, "gaussian", "sigma", noise="gaussian", strength=20, seed=19)
    assert len(study["candidates"]) == len(STUDIES["gaussian"]["sigma"])
    assert study["study_resolution"] == [160, 128]
    assert all(row["quality"]["quality_available"] for row in study["candidates"])
    assert all(row["quality"]["before"] == study["candidates"][0]["quality"]["before"]
               for row in study["candidates"])
    assert len({row["parameters"]["sigma"] for row in study["candidates"]}) == 4
    with pytest.raises(ValueError):
        run_study(img, "otsu", "sigma")


def test_no_ground_truth_does_not_fabricate_scores():
    study = run_study(make_sample(160, 128), "canny", "threshold", noise="none")
    assert all(row["quality"]["quality_available"] is False for row in study["candidates"])
    clean = run_study(make_sample(160, 128), "median", "kernel", noise="none")
    assert all(row["quality"]["quality_available"] is False for row in clean["candidates"])


def test_large_study_downsamples_once():
    study = run_study(make_sample(768, 480), "median", "kernel", noise="gaussian")
    w, h = study["study_resolution"]
    assert w * h <= 110_000


def test_ablation_http_contract(live_server):
    encoded = encode_image(make_sample(160, 128))
    body = {"image": encoded, "algorithm": "gaussian", "study": "sigma", "seed": 7}
    request = Request(live_server + "/api/ablate", json.dumps(body).encode(),
                      {"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=20) as res:
        output = json.load(res)
    assert len(output["candidates"]) == 4
    assert output["candidates"][0]["image"].startswith("data:image/png;base64,")
    assert output["candidates"][0]["quality"]["quality_available"]
    with urlopen(live_server + "/api/studies") as res:
        assert "gaussian" in json.load(res)
    body["study"] = "unsafe"
    bad = Request(live_server + "/api/ablate", json.dumps(body).encode(),
                  {"Content-Type": "application/json"}, method="POST")
    with pytest.raises(HTTPError) as err:
        urlopen(bad, timeout=20)
    assert err.value.code == 400
