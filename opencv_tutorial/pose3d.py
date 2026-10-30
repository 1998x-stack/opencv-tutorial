"""Calibrated perspective-n-point, relative pose and scale-ambiguous visual odometry."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PoseEstimate:
    rotation: np.ndarray
    translation: np.ndarray
    inliers: np.ndarray
    stats: dict


def _camera(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if (K.shape != (3, 3) or not np.all(np.isfinite(K))
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-7)):
        raise ValueError("Camera matrix must be a finite 3x3 pinhole calibration")
    return K


def _points(a: np.ndarray, dimensions: int, minimum: int) -> np.ndarray:
    p = np.asarray(a, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != dimensions or len(p) < minimum
            or len(p) > 50000 or not np.all(np.isfinite(p))):
        raise ValueError(f"Expected at least {minimum} finite {dimensions}D points")
    return np.ascontiguousarray(p)


def solve_pnp_ransac(object_xyz: np.ndarray, image_uv: np.ndarray, K: np.ndarray,
                     *, reprojection_px: float = 3.0, iterations: int = 150) -> PoseEstimate:
    """Solve metric camera-from-world pose when 3D object coordinates have known units."""
    xyz, uv, K = _points(object_xyz, 3, 6), _points(image_uv, 2, 6), _camera(K)
    if len(xyz) != len(uv):
        raise ValueError("2D/3D correspondence lengths must match")
    if not 0 < reprojection_px <= 20 or not 10 <= iterations <= 10000:
        raise ValueError("Invalid RANSAC thresholds")
    ok, rvec, tvec, idx = cv2.solvePnPRansac(
        xyz, uv, K, None, flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=reprojection_px, iterationsCount=iterations, confidence=0.999)
    if not ok or idx is None or len(idx) < 6:
        raise ValueError("PnP failed: insufficient geometrically consistent correspondences")
    inliers = idx[:, 0].astype(np.int32)
    # Refine using consensus points only; never optimize against rejected outliers.
    rvec, tvec = cv2.solvePnPRefineLM(xyz[inliers], uv[inliers], K, None, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(xyz[inliers], rvec, tvec, K, None)
    residual = np.linalg.norm(projected.reshape(-1, 2) - uv[inliers], axis=1)
    positive = (xyz[inliers] @ R[2, :] + tvec[2, 0]) > 0
    if np.mean(positive) < 0.9:
        raise ValueError("PnP failed cheirality check")
    return PoseEstimate(R, tvec.reshape(3), inliers,
                        {"method": "PnP-RANSAC + LM", "points": len(xyz),
                         "inliers": len(inliers), "inlier_ratio": len(inliers) / len(xyz),
                         "reprojection_rmse_px": float(np.sqrt(np.mean(residual**2))),
                         "translation_units": "input world coordinate units"})


def relative_pose(uv_a: np.ndarray, uv_b: np.ndarray, K: np.ndarray,
                  *, threshold_px: float = 1.5) -> PoseEstimate:
    """Estimate 2-view essential geometry; translation is a direction, NOT meters."""
    a, b, K = _points(uv_a, 2, 8), _points(uv_b, 2, 8), _camera(K)
    if len(a) != len(b) or not 0 < threshold_px <= 10:
        raise ValueError("Invalid 2-view correspondence configuration")
    E, initial = cv2.findEssentialMat(a, b, K, method=cv2.RANSAC,
                                      prob=0.999, threshold=threshold_px)
    if E is None or initial is None:
        raise ValueError("Essential matrix estimation failed")
    if E.shape != (3, 3):
        # OpenCV may produce stacked 3x3 essential candidates; take best via recovery.
        candidates = np.vsplit(E, E.shape[0] // 3)
    else:
        candidates = [E]
    solution = None
    for candidate in candidates:
        count, R, t, mask = cv2.recoverPose(candidate, a, b, K, mask=initial.copy())
        if solution is None or count > solution[0]:
            solution = (count, R, t, mask)
    assert solution is not None
    count, R, t, mask = solution
    indices = np.flatnonzero(mask.ravel() > 0)
    if count < 8 or len(indices) < 8:
        raise ValueError("Relative pose degenerate: too few cheirality-consistent inliers")
    return PoseEstimate(R, t.reshape(3), indices.astype(np.int32),
                        {"method": "Essential RANSAC + recoverPose", "matches": len(a),
                         "inliers": len(indices), "inlier_ratio": len(indices) / len(a),
                         "translation_scale": "unknown (unit direction only)"})


def integrate_unit_poses(relatives: list[PoseEstimate]) -> list[list[float]]:
    """Camera centers in arbitrary units, one unit of translation per 2-view estimate."""
    R_wc = np.eye(3, dtype=np.float64)  # world -> camera orientation
    t_wc = np.zeros(3)
    centers = [[0.0, 0.0, 0.0]]
    for pose in relatives:
        R_wc = pose.rotation @ R_wc
        t_wc = pose.rotation @ t_wc + pose.translation
        center = -R_wc.T @ t_wc
        centers.append([float(v) for v in center])
    return centers


def synthetic_correspondences(seed: int = 17, *, motion: float = .18,
                              count: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ground-truth 3D scene/camera, with a translation and small rotation."""
    rng = np.random.default_rng(seed)
    xyz = np.column_stack((rng.uniform(-2, 2, count), rng.uniform(-1.5, 1.5, count),
                           rng.uniform(4, 10, count))).astype(np.float64)
    K = np.array([[530., 0, 320.], [0, 530., 240.], [0, 0, 1.]])
    rvec = np.array([.015, -.025, .01], dtype=np.float64)
    t = np.array([motion, -.035, .025], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    uv_a, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, None)
    uv_b, _ = cv2.projectPoints(xyz, rvec, t, K, None)
    return xyz, uv_a.reshape(-1, 2), uv_b.reshape(-1, 2), K, R, t


def pnp_fixture_report(seed: int = 17) -> dict:
    xyz, _, uv_b, K, R_true, t_true = synthetic_correspondences(seed)
    rng = np.random.default_rng(seed + 1)
    noisy = uv_b + rng.normal(0, .4, size=uv_b.shape)
    bad = rng.choice(len(xyz), 15, replace=False)
    noisy[bad] = rng.uniform([0, 0], [640, 480], size=(len(bad), 2))
    estimate = solve_pnp_ransac(xyz, noisy, K)
    R_delta = estimate.rotation @ R_true.T
    angle = float(np.rad2deg(np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1, 1))))
    return {"seed": seed, "stats": estimate.stats,
            "rotation_error_deg": angle,
            "translation_l2_error_world_units": float(np.linalg.norm(estimate.translation - t_true))}


def vo_fixture_report(seed: int = 17) -> dict:
    xyz, a, b, K, R_true, t_true = synthetic_correspondences(seed)
    _ = xyz
    pose = relative_pose(a, b, K)
    R_delta = pose.rotation @ R_true.T
    angle = float(np.rad2deg(np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1, 1))))
    cosine = float(np.dot(pose.translation, t_true) / np.linalg.norm(t_true))
    return {"seed": seed, "rotation_error_deg": angle,
            "translation_direction_cosine": cosine, "stats": pose.stats,
            "trajectory_arbitrary_units": integrate_unit_poses([pose])}


def monocular_motion(first: np.ndarray, second: np.ndarray, K: np.ndarray) -> tuple[PoseEstimate, np.ndarray]:
    """Experimental 2-view monocular VO: optical flow -> essential -> unit-direction pose.

    Assumes calibrated pinhole camera, rigid scene, useful parallax and nonzero motion.
    No automatic metric scale, loop closure, IMU fusion or SLAM optimization.
    """
    from .optical_flow import sparse_lk

    flow = sparse_lk(first, second, max_corners=850)
    if len(flow.points_first) < 20:
        raise ValueError("Insufficient tracked features for monocular relative pose")
    pose = relative_pose(flow.points_first, flow.points_second, K)
    return pose, flow.annotated


def pose_fixture_image(seed: int = 17) -> np.ndarray:
    xyz, _, uv_b, K, _, _ = synthetic_correspondences(seed)
    estimate = solve_pnp_ransac(xyz, uv_b, K)
    frame = np.full((480, 640, 3), (23, 28, 37), dtype=np.uint8)
    for p in uv_b:
        cv2.circle(frame, tuple(np.rint(p).astype(int)), 2, (51, 185, 238), -1)
    rvec, _ = cv2.Rodrigues(estimate.rotation)
    cv2.drawFrameAxes(frame, K, None, rvec,
                      estimate.rotation @ np.array([0., 0., 6.]) + estimate.translation,
                      0.8, 2)
    cv2.putText(frame, "PnP: XYZ points and calibrated axis", (15, 29),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (233, 233, 233), 1, cv2.LINE_AA)
    return frame
