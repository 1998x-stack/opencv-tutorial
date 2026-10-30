"""Single source of truth for the full 24-week teaching syllabus."""

TRACKS = [
    ("Beginner", [
        ("Image representations", "Pixels, dtype, color spaces, NumPy views, cv::Mat", "Build a pixel inspector and channel explorer"),
        ("Geometric transforms", "Resize, interpolation, affine transforms, warp mapping", "Compare interpolation artifacts"),
        ("Convolution and denoising", "Correlation, Gaussian, median, bilateral filtering", "Reproduce Gaussian filtering from first principles"),
        ("Gradients and edges", "Sobel, Scharr, Laplacian, Canny", "Visualize horizontal and vertical gradients"),
        ("Thresholding", "Histograms, Otsu, adaptive thresholding", "Compare segmentation under uneven lighting"),
        ("Morphological processing", "Erosion, dilation, opening, closing, components", "Remove noise and count connected regions"),
        ("Geometry and shapes", "Contours, moments, Hough transform, bounding boxes", "Build shape and area measurements"),
        ("Project: document scanner", "Perspective warp and end-to-end image pipeline", "Detect corners and rectify a paper document"),
    ]),
    ("Intermediate", [
        ("Keypoints", "Harris, Shi-Tomasi, FAST, ORB, SIFT", "Visualize stable keypoints across transformations"),
        ("Descriptors and matching", "Hamming, BFMatcher, FLANN, ratio test", "Measure match quality under image rotation"),
        ("Robust geometry", "Homography, RANSAC, inliers and outliers", "Recover a transform from noisy correspondences"),
        ("Project: panorama", "Warp, compositing, image mosaics", "Implement pairwise panorama construction"),
        ("Motion segmentation", "Frame differencing, MOG2, KNN", "Foreground segmentation with failure-case analysis"),
        ("Optical flow", "Lucas-Kanade, Farneback, pyramids", "Visualize sparse and dense motion fields"),
        ("Temporal tracking", "Kalman filtering, association, tracking identity", "Plot persistent motion trajectories"),
        ("Project: video analytics", "Video I/O, motion, events, profiling", "Implement an offline motion video report"),
    ]),
    ("Advanced", [
        ("Multiview geometry", "Fundamental and essential matrices, PnP", "Estimate relative camera pose and reprojection error"),
        ("Stereo and depth", "Rectification, disparity, stereo matching", "Evaluate disparity against ground truth"),
        ("Visual odometry", "Feature tracking, camera motion, scale ambiguity", "Plot a simplified relative-motion trajectory"),
        ("DNN inference", "ONNX, normalization, NMS, OpenCV DNN", "Benchmark preprocessing + inference + postprocessing"),
        ("C++ internals", "Memory layout, ROI, copy semantics, SIMD", "Compare NumPy/OpenCV/C++ code paths"),
        ("Performance lab", "Threading, frame latency, memory, GPU backends", "Measure P50/P95 latency with fixed workloads"),
        ("Reliability & evaluation", "Golden datasets, ablations, fault injection", "Build reproducible quality and performance gates"),
        ("Final integrated system", "Design, implementation, testing, deployment", "Ship a complete offline CV application"),
    ]),
]


def get_curriculum() -> list[dict]:
    return [
        {"week": n + 1, "id": f"W{n + 1:02d}", "level": level,
         "title": title, "topics": topics, "deliverable": deliverable,
         "status": "implemented" if n + 1 in (3, 4, 5, 6, 7, 8, 9, 10, 11, 13) else "planned"}
        for n, (level, title, topics, deliverable) in enumerate(
            (level, *item) for level, items in TRACKS for item in items
        )
    ]
