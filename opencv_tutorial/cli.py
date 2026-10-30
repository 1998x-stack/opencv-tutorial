"""CLI entry point for repeatable image and video experiments."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .algorithms import make_sample
from .curriculum import get_curriculum
from .experiments import STUDIES, run_study
from .geometry import detect_document, warp_document
from .intermediate import benchmark_synthetic
from .motion import MotionDetector
from .optical_flow import sparse_lk
from .panorama import stitch_pair
from .stereo import stereo_depth, stereo_fixture_report
from .pose3d import pnp_fixture_report, vo_fixture_report
from .onnx_lab import make_test_onnx, run_onnx, benchmark_onnx
from .pipeline import ALGORITHMS, run_pipeline
from .server import serve
from .tracking import CentroidTracker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opencv-lab", description="OpenCV from scratch laboratory")
    sub = parser.add_subparsers(dest="command", required=True)
    p_server = sub.add_parser("serve", help="Start interactive learning portal")
    p_server.add_argument("--host", default="127.0.0.1")
    p_server.add_argument("--port", type=int, default=8765)
    p_sample = sub.add_parser("sample", help="Generate synthetic image")
    p_sample.add_argument("--output", type=Path, default=Path("sample.png"))
    p_run = sub.add_parser("process", help="Apply algorithm to image")
    p_run.add_argument("--input", type=Path, required=True)
    p_run.add_argument("--output", type=Path, required=True)
    p_run.add_argument("--algorithm", choices=ALGORITHMS, default="gaussian")
    p_run.add_argument("--kernel", type=int, default=5)
    p_run.add_argument("--sigma", type=float, default=1.2)
    p_run.add_argument("--threshold", type=int, default=120)
    p_bench = sub.add_parser("benchmark", help="Time image algorithms for repeatable inputs")
    p_bench.add_argument("--rounds", type=int, default=30)
    p_bench.add_argument("--output", type=Path)
    p_doc = sub.add_parser("scan", help="Locate and rectify a document")
    p_doc.add_argument("--input", type=Path, required=True)
    p_doc.add_argument("--output", type=Path, required=True)
    p_motion = sub.add_parser("motion", help="Analyze a locally stored video")
    p_motion.add_argument("--input", type=Path, required=True)
    p_motion.add_argument("--output", type=Path, required=True)
    p_study = sub.add_parser("ablate", help="Run a fixed-input reproducible parameter study")
    p_study.add_argument("--input", type=Path, required=True)
    p_study.add_argument("--algorithm", choices=sorted(STUDIES), default="gaussian")
    p_study.add_argument("--study", choices=["kernel", "sigma", "threshold"], default="sigma")
    p_study.add_argument("--noise", choices=["none", "gaussian", "salt_pepper"], default="gaussian")
    p_study.add_argument("--strength", type=float, default=18)
    p_study.add_argument("--seed", type=int, default=7)
    p_study.add_argument("--output", type=Path, required=True)
    p_pan = sub.add_parser("panorama", help="ORB/RANSAC panorama from two overlapping photos")
    p_pan.add_argument("--first", type=Path, required=True)
    p_pan.add_argument("--second", type=Path, required=True)
    p_pan.add_argument("--output", type=Path, required=True)
    p_flow = sub.add_parser("flow", help="Sparse pyramidal LK on two consecutive frames")
    p_flow.add_argument("--first", type=Path, required=True)
    p_flow.add_argument("--second", type=Path, required=True)
    p_flow.add_argument("--output", type=Path, required=True)
    p_track = sub.add_parser("track", help="MOG2 + centroid association over local video")
    p_track.add_argument("--input", type=Path, required=True)
    p_track.add_argument("--output", type=Path, required=True)
    p_eval = sub.add_parser("evaluate-intermediate", help="Synthetic geometric sanity checks")
    p_eval.add_argument("--seed", type=int, default=17)
    p_eval.add_argument("--output", type=Path)
    p_stereo = sub.add_parser("stereo", help="Calibrated rectified SGBM depth estimation")
    p_stereo.add_argument("--left", type=Path, required=True)
    p_stereo.add_argument("--right", type=Path, required=True)
    p_stereo.add_argument("--fx", type=float, required=True, help="Focal length in pixels")
    p_stereo.add_argument("--baseline", type=float, required=True, help="Camera baseline in meters")
    p_stereo.add_argument("--num-disparities", type=int, default=64)
    p_stereo.add_argument("--output", type=Path, required=True)
    p_stereo.add_argument("--report", type=Path)
    p_advanced = sub.add_parser("evaluate-advanced", help="Stereo/PnP/VO numeric fixtures")
    p_advanced.add_argument("--seed", type=int, default=17)
    p_advanced.add_argument("--output", type=Path)
    p_toy = sub.add_parser("onnx-fixture", help="Generate an offline tiny ONNX sanity model")
    p_toy.add_argument("--output", type=Path, required=True)
    p_infer = sub.add_parser("onnx-infer", help="Run local ONNX model via OpenCV DNN")
    p_infer.add_argument("--model", type=Path, required=True)
    p_infer.add_argument("--input", type=Path, required=True)
    p_infer.add_argument("--width", type=int, default=16)
    p_infer.add_argument("--height", type=int, default=16)
    p_infer.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    p_infer.add_argument("--sha256", default=None)
    p_bdn = sub.add_parser("bench-dnn", help="Warmup/repeat local OpenCV DNN forward")
    p_bdn.add_argument("--model", type=Path, required=True)
    p_bdn.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    p_bdn.add_argument("--rounds", type=int, default=15)
    p_bdn.add_argument("--warmups", type=int, default=3)
    p_bdn.add_argument("--image-size", type=int, default=16)
    p_bdn.add_argument("--sha256", default=None)
    p_bdn.add_argument("--output", type=Path)
    sub.add_parser("course", help="Print the 24-week curriculum as JSON")
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(args.host, args.port)
        return 0
    if args.command == "course":
        print(json.dumps(get_curriculum(), indent=2))
        return 0
    if args.command == "sample":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), make_sample()):
            raise RuntimeError("Failed to save sample")
        return 0
    if args.command == "benchmark":
        if args.rounds < 1 or args.rounds > 1000:
            parser.error("rounds must be in [1,1000]")
        image = make_sample(512, 384)
        records = []
        for name in ALGORITHMS:
            run_pipeline(image, name)  # warmup
            latencies = []
            for _ in range(args.rounds):
                start = perf_counter()
                run_pipeline(image, name)
                latencies.append((perf_counter() - start) * 1000)
            records.append({"algorithm": name, "runs": args.rounds,
                            "median_ms": round(float(np.median(latencies)), 3),
                            "p95_ms": round(float(np.percentile(latencies, 95)), 3)})
        report = {"opencv": cv2.__version__, "size": [512, 384], "results": records}
        content = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(content, encoding="utf-8")
        print(content)
        return 0
    if args.command == "evaluate-advanced":
        output = {"stereo": stereo_fixture_report(args.seed),
                  "pnp": pnp_fixture_report(args.seed), "vo": vo_fixture_report(args.seed),
                  "assumptions": ["Synthetic known calibration/geometry",
                                  "Monocular VO translation has unknown scale"]}
        data = json.dumps(output, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(data, encoding="utf-8")
        print(data)
        return 0
    if args.command == "stereo":
        left, right = cv2.imread(str(args.left)), cv2.imread(str(args.right))
        if left is None or right is None:
            raise FileNotFoundError("Cannot read stereo pair")
        result = stereo_depth(left, right, fx_px=args.fx,
                              baseline_m=args.baseline, num_disparities=args.num_disparities)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), result.visualization):
            raise RuntimeError("Cannot write stereo output")
        report = json.dumps(result.stats, indent=2, allow_nan=False)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report, encoding="utf-8")
        print(report)
        return 0
    if args.command == "onnx-fixture":
        digest = make_test_onnx(args.output)
        print(json.dumps({"model": str(args.output), "sha256": digest,
                          "purpose": "I/O fixture only; no semantics or trained classes"}))
        return 0
    if args.command == "onnx-infer":
        image = cv2.imread(str(args.input))
        if image is None:
            raise FileNotFoundError("Cannot decode image")
        report = run_onnx(args.model, image, input_width=args.width,
                          input_height=args.height, backend=args.backend,
                          expected_sha256=args.sha256)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    if args.command == "bench-dnn":
        report = benchmark_onnx(args.model, backend=args.backend,
                                rounds=args.rounds, warmups=args.warmups,
                                image_size=args.image_size, expected_sha256=args.sha256)
        data = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(data, encoding="utf-8")
        print(data)
        return 0
    if args.command == "evaluate-intermediate":
        result = benchmark_synthetic(seed=args.seed)
        report = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
        print(report)
        return 0
    if args.command in ("panorama", "flow"):
        first, second = cv2.imread(str(args.first)), cv2.imread(str(args.second))
        if first is None or second is None:
            raise FileNotFoundError("Cannot decode first or second image")
        if args.command == "panorama":
            result = stitch_pair(first, second)
            output, report = result.image, result.stats
        else:
            result = sparse_lk(first, second)
            output, report = result.annotated, result.stats
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), output):
            raise RuntimeError("Could not write output")
        print(json.dumps(report, allow_nan=False))
        return 0
    if args.command == "track":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(args.input))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video {args.input}")
        detector, tracker = MotionDetector(), CentroidTracker()
        writer = None
        count, max_tracks = 0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                motion = detector.step(frame)
                tracks = tracker.update(motion.boxes)
                annotated = tracker.annotate(frame)
                if writer is None:
                    h, w = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    writer = cv2.VideoWriter(str(args.output),
                        cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 25., (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot initialize video output")
                writer.write(annotated)
                count += 1
                max_tracks = max(max_tracks, len(tracks))
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        print(json.dumps({"frames": count, "max_active_tracks": max_tracks,
                          "output": str(args.output)}))
        return 0
    if args.command == "ablate":
        image = cv2.imread(str(args.input))
        if image is None:
            raise FileNotFoundError(f"Cannot decode {args.input}")
        study = run_study(image, args.algorithm, args.study, noise=args.noise,
                          strength=args.strength, seed=args.seed)
        # Never serialize full pixels to JSON. Save an inspectable contact sheet.
        panels = [study["clean"], study["observed"]]
        for row in study["candidates"]:
            output = row["image"]
            if output.ndim == 2:
                output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
            panels.append(output)
        tiled = np.hstack(panels)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        contact_path = args.output.with_suffix(".png")
        if not cv2.imwrite(str(contact_path), tiled):
            raise RuntimeError("Cannot write ablation contact sheet")
        report = {k: v for k, v in study.items()
                  if k not in ("clean", "observed", "candidates")}
        report["contact_sheet"] = str(contact_path)
        report["panels"] = ["clean", "noisy"] + [str(r["value"]) for r in study["candidates"]]
        report["candidates"] = [{k: v for k, v in row.items() if k != "image"}
                                for row in study["candidates"]]
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"report": str(args.output), "contact_sheet": str(contact_path)}))
        return 0
    if args.command in ("process", "scan", "motion"):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.command == "motion":
            cap = cv2.VideoCapture(str(args.input))
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video {args.input}")
            writer = None
            detector = MotionDetector()
            count = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    result = detector.step(frame)
                    if writer is None:
                        h, w = frame.shape[:2]
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps if fps > 0 else 25.0, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError("Could not open output video writer")
                    writer.write(result.annotated)
                    count += 1
            finally:
                cap.release()
                if writer is not None:
                    writer.release()
            print(json.dumps({"processed_frames": count, "output": str(args.output)}))
            return 0
        image = cv2.imread(str(args.input))
        if image is None:
            raise FileNotFoundError(f"Cannot decode {args.input}")
        if args.command == "scan":
            corners = detect_document(image)
            if corners is None:
                raise RuntimeError("No quadrilateral document found")
            output = warp_document(image, corners)
        else:
            result = run_pipeline(image, args.algorithm, args.kernel, args.sigma, args.threshold)
            output = result.image
            print(json.dumps(result.metrics, indent=2))
        if not cv2.imwrite(str(args.output), output):
            raise RuntimeError("Failed to write output image")
        return 0
    return 1
