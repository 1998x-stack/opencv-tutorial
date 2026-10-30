"""End-to-end CLI report and contact-sheet validation, isolated in tmp_path."""
import json
import subprocess
import sys

import cv2

from opencv_tutorial.algorithms import make_sample


def test_ablate_cli_report(tmp_path):
    inp = tmp_path / "input.png"
    out = tmp_path / "report.json"
    assert cv2.imwrite(str(inp), make_sample(160, 128))
    cmd = [sys.executable, "-m", "opencv_tutorial", "ablate",
           "--input", str(inp), "--algorithm", "gaussian", "--study", "sigma",
           "--seed", "19", "--output", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True)
    assert "contact_sheet" in result.stdout
    report = json.loads(out.read_text())
    assert report["seed"] == 19
    assert len(report["candidates"]) == 4
    assert report["candidates"][0]["quality"]["quality_available"]
    sheet = cv2.imread(str(out.with_suffix(".png")))
    assert sheet is not None and sheet.shape[0] == 128
    assert sheet.shape[1] == 160 * 6
