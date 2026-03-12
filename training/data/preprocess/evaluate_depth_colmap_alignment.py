#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os.path as osp
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


def quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def parse_colmap_cameras(cameras_txt: str) -> Dict[int, Dict[str, float]]:
    camera_map: Dict[int, Dict[str, float]] = {}
    with open(cameras_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
                fx = fy = params[0]
                cx = params[1]
                cy = params[2]
            elif model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"]:
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
            else:
                raise ValueError(f"Unsupported COLMAP camera model: {model}")

            camera_map[camera_id] = {
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }
    return camera_map


def parse_colmap_images(images_txt: str, frame_regex: re.Pattern) -> List[Dict]:
    entries: List[Dict] = []
    with open(images_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue

            # COLMAP images.txt stores two lines per image:
            # (1) pose/intrinsics reference line
            # (2) optional POINTS2D list line.
            # We only want line (1); line (2) can also have >=10 tokens.
            try:
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
            except ValueError:
                continue

            image_name = parts[9]
            if "." not in osp.basename(image_name):
                continue

            match = frame_regex.search(osp.basename(image_name))
            if match is None:
                continue
            frame_idx = int(match.group(1))

            entries.append(
                {
                    "image_id": image_id,
                    "camera_id": camera_id,
                    "image_name": image_name,
                    "frame_idx": frame_idx,
                    "R": quat_to_rotmat(qw, qx, qy, qz),
                    "t": np.array([tx, ty, tz], dtype=np.float64),
                }
            )

    entries.sort(key=lambda x: x["image_id"])
    return entries


def parse_colmap_points(points3d_txt: str) -> np.ndarray:
    points = []
    with open(points3d_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, z = map(float, parts[1:4])
            points.append([x, y, z])
    if len(points) == 0:
        raise RuntimeError(f"No points parsed from {points3d_txt}")
    return np.asarray(points, dtype=np.float64)


def load_depth(
    depth_path: str,
    depth_format: str,
    depth_key: str,
    maps_metadata_path: Optional[str],
    depth_dtype: str,
) -> np.ndarray:
    if depth_format == "auto":
        if depth_path.endswith(".npy"):
            depth_format = "npy"
        elif depth_path.endswith(".npz"):
            depth_format = "npz"
        elif depth_path.endswith(".bin"):
            depth_format = "bin"
        else:
            raise ValueError(f"Cannot infer depth format from: {depth_path}")

    if depth_format == "npy":
        arr = np.load(depth_path)
    elif depth_format == "npz":
        with np.load(depth_path) as data:
            if depth_key not in data:
                raise KeyError(
                    f"Depth key '{depth_key}' not in {depth_path}. Available: {list(data.files)}"
                )
            arr = data[depth_key]
    elif depth_format == "bin":
        if maps_metadata_path is None:
            raise ValueError("maps_metadata_path is required for bin depth format")
        with open(maps_metadata_path, "r") as f:
            metadata = json.load(f)
        n_frames = int(metadata["n_frames"])
        height = int(metadata["height"])
        width = int(metadata["width"])
        arr = np.fromfile(depth_path, dtype=np.dtype(depth_dtype)).reshape(n_frames, height, width)
    else:
        raise ValueError(f"Unsupported depth format: {depth_format}")

    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f"Depth array must be [N,H,W], got shape {arr.shape}")
    return arr.astype(np.float32)


def load_score_map(score_map_path: Optional[str], depth_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    if score_map_path is None:
        return None
    n_frames, height, width = depth_shape
    score = np.fromfile(score_map_path, dtype=np.float16).reshape(n_frames, height, width).astype(np.float32)
    return score


def robust_linear_fit_huber(x: np.ndarray, y: np.ndarray, iters: int = 8) -> Tuple[float, float]:
    # Fit y ~= s*x + b via IRLS with Huber weights.
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]

    for _ in range(iters):
        residual = y - (A @ coef)
        median = np.median(residual)
        mad = np.median(np.abs(residual - median)) + 1e-9
        scale = 1.4826 * mad
        c = 1.345 * scale + 1e-9
        w = np.ones_like(residual)
        outlier_mask = np.abs(residual) > c
        w[outlier_mask] = c / np.abs(residual[outlier_mask])
        Aw = A * w[:, None]
        yw = y * w
        coef = np.linalg.lstsq(Aw, yw, rcond=None)[0]

    s, b = float(coef[0]), float(coef[1])
    return s, b


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    eps = 1e-6
    abs_err = np.abs(pred - gt)
    sq_err = (pred - gt) ** 2
    abs_rel = abs_err / np.maximum(np.abs(gt), eps)
    rmse = np.sqrt(np.mean(sq_err))
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(rmse),
        "median_abs_err": float(np.median(abs_err)),
        "p90_abs_err": float(np.quantile(abs_err, 0.90)),
        "abs_rel": float(np.mean(abs_rel)),
    }


def build_pairs(
    points_world: np.ndarray,
    image_entries: List[Dict],
    camera_map: Dict[int, Dict[str, float]],
    depth: np.ndarray,
    score_map: Optional[np.ndarray],
    score_thres: float,
    min_depth: float,
    max_depth: float,
    use_z_buffer: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    n_frames, dh, dw = depth.shape

    all_pred = []
    all_gt = []
    per_frame_ratio: Dict[int, np.ndarray] = {}

    for entry in image_entries:
        frame_idx = entry["frame_idx"]
        if frame_idx < 0 or frame_idx >= n_frames:
            continue

        camera_id = entry["camera_id"]
        if camera_id not in camera_map:
            continue

        cam = camera_map[camera_id]
        sx = dw / cam["width"]
        sy = dh / cam["height"]

        R = entry["R"]
        t = entry["t"]

        points_cam = (R @ points_world.T).T + t[None]
        z = points_cam[:, 2]
        valid = z > 1e-9
        if not np.any(valid):
            continue

        points_cam = points_cam[valid]
        z = z[valid]

        u = cam["fx"] * (points_cam[:, 0] / z) + cam["cx"]
        v = cam["fy"] * (points_cam[:, 1] / z) + cam["cy"]

        ud = np.rint(u * sx).astype(np.int32)
        vd = np.rint(v * sy).astype(np.int32)
        inside = (ud >= 0) & (ud < dw) & (vd >= 0) & (vd < dh)
        if not np.any(inside):
            continue

        ud = ud[inside]
        vd = vd[inside]
        z = z[inside]

        if use_z_buffer:
            zbuf = np.full((dh, dw), np.inf, dtype=np.float64)
            for uu, vv, zz in zip(ud, vd, z):
                if zz < zbuf[vv, uu]:
                    zbuf[vv, uu] = zz
            valid_px = np.isfinite(zbuf)
            if not np.any(valid_px):
                continue
            gt = zbuf[valid_px].astype(np.float32)
            pred = depth[frame_idx][valid_px]
            if score_map is not None and score_thres >= 0:
                score_vals = score_map[frame_idx][valid_px]
                score_valid = score_vals >= score_thres
                pred = pred[score_valid]
                gt = gt[score_valid]
        else:
            pred = depth[frame_idx, vd, ud]
            gt = z.astype(np.float32)
            if score_map is not None and score_thres >= 0:
                score_vals = score_map[frame_idx, vd, ud]
                score_valid = score_vals >= score_thres
                pred = pred[score_valid]
                gt = gt[score_valid]

        valid_depth = np.isfinite(pred) & np.isfinite(gt)
        valid_depth &= pred > min_depth
        if max_depth > 0:
            valid_depth &= pred < max_depth
        valid_depth &= gt > 1e-6

        pred = pred[valid_depth]
        gt = gt[valid_depth]

        if pred.size < 8:
            continue

        all_pred.append(pred)
        all_gt.append(gt)
        per_frame_ratio[frame_idx] = gt / np.maximum(pred, 1e-6)

    if len(all_pred) == 0:
        return np.array([]), np.array([]), {}

    return np.concatenate(all_pred), np.concatenate(all_gt), per_frame_ratio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Numerically evaluate depth-vs-COLMAP geometry consistency."
    )
    parser.add_argument("--vipe-results-dir", type=str, default=None)
    parser.add_argument("--colmap-dir", type=str, default=None)
    parser.add_argument("--depth-path", type=str, default=None)
    parser.add_argument("--maps-metadata-path", type=str, default=None)
    parser.add_argument("--score-map-path", type=str, default=None)
    parser.add_argument("--depth-format", type=str, default="auto", choices=["auto", "npy", "npz", "bin"])
    parser.add_argument("--depth-key", type=str, default="depths")
    parser.add_argument("--depth-dtype", type=str, default="float16")
    parser.add_argument("--frame-pattern", type=str, default=r"frame_(\d+)")
    parser.add_argument("--score-thres", type=float, default=-1.0, help="Use >=0 to filter by score map.")
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=-1.0)
    parser.add_argument("--use-z-buffer", action="store_true")
    parser.add_argument("--out-json", type=str, default=None)
    return parser.parse_args()


def resolve_paths(args):
    if args.vipe_results_dir is not None:
        vipe_dir = args.vipe_results_dir
        colmap_dir = args.colmap_dir or osp.join(vipe_dir, "colmap")
        depth_path = args.depth_path or osp.join(vipe_dir, "depth.bin")
        maps_metadata_path = args.maps_metadata_path or osp.join(vipe_dir, "maps_metadata.json")
        score_map_path = args.score_map_path or osp.join(vipe_dir, "score_maps.bin")
    else:
        if args.colmap_dir is None or args.depth_path is None:
            raise ValueError("Provide either --vipe-results-dir OR both --colmap-dir and --depth-path")
        colmap_dir = args.colmap_dir
        depth_path = args.depth_path
        maps_metadata_path = args.maps_metadata_path
        score_map_path = args.score_map_path

    cameras_txt = osp.join(colmap_dir, "cameras.txt")
    images_txt = osp.join(colmap_dir, "images.txt")
    points3d_txt = osp.join(colmap_dir, "points3D.txt")

    for path in [cameras_txt, images_txt, points3d_txt, depth_path]:
        if not osp.isfile(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    if args.depth_format in ["bin", "auto"] and depth_path.endswith(".bin"):
        if maps_metadata_path is None or not osp.isfile(maps_metadata_path):
            raise FileNotFoundError("maps_metadata.json is required for .bin depth")

    if score_map_path is not None and not osp.isfile(score_map_path):
        score_map_path = None

    return cameras_txt, images_txt, points3d_txt, depth_path, maps_metadata_path, score_map_path


def main():
    args = parse_args()
    frame_regex = re.compile(args.frame_pattern)

    cameras_txt, images_txt, points3d_txt, depth_path, maps_metadata_path, score_map_path = resolve_paths(args)

    camera_map = parse_colmap_cameras(cameras_txt)
    image_entries = parse_colmap_images(images_txt, frame_regex=frame_regex)
    points_world = parse_colmap_points(points3d_txt)

    depth = load_depth(
        depth_path=depth_path,
        depth_format=args.depth_format,
        depth_key=args.depth_key,
        maps_metadata_path=maps_metadata_path,
        depth_dtype=args.depth_dtype,
    )
    score_map = load_score_map(score_map_path, depth_shape=depth.shape)

    pred, gt, per_frame_ratio = build_pairs(
        points_world=points_world,
        image_entries=image_entries,
        camera_map=camera_map,
        depth=depth,
        score_map=score_map,
        score_thres=args.score_thres,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        use_z_buffer=args.use_z_buffer,
    )

    if pred.size == 0:
        raise RuntimeError("No valid depth/geometry pairs found.")

    raw_metrics = compute_metrics(pred, gt)
    s, b = robust_linear_fit_huber(pred, gt)
    aligned_pred = s * pred + b
    aligned_metrics = compute_metrics(aligned_pred, gt)

    ratio_values = []
    for _, ratio in per_frame_ratio.items():
        ratio_values.append(float(np.median(ratio)))
    ratio_values = np.asarray(ratio_values, dtype=np.float64)
    ratio_log_std = float(np.std(np.log(np.maximum(ratio_values, 1e-9)))) if ratio_values.size > 0 else float("nan")
    ratio_cv = float(np.std(ratio_values) / (np.mean(ratio_values) + 1e-9)) if ratio_values.size > 0 else float("nan")

    pearson = float(np.corrcoef(pred, gt)[0, 1]) if pred.size > 1 else float("nan")

    result = {
        "files": {
            "cameras_txt": cameras_txt,
            "images_txt": images_txt,
            "points3D_txt": points3d_txt,
            "depth_path": depth_path,
            "maps_metadata_path": maps_metadata_path,
            "score_map_path": score_map_path,
        },
        "counts": {
            "num_colmap_images": len(image_entries),
            "num_colmap_points3d": int(points_world.shape[0]),
            "num_pairs_used": int(pred.size),
            "num_frames_with_pairs": int(len(per_frame_ratio)),
        },
        "depth_shape": list(depth.shape),
        "fit": {
            "linear_s": float(s),
            "linear_b": float(b),
        },
        "metrics_raw": raw_metrics,
        "metrics_aligned_linear": aligned_metrics,
        "correlation": {
            "pearson_pred_vs_colmap_z": pearson,
        },
        "scale_consistency": {
            "per_frame_ratio_median": float(np.median(ratio_values)) if ratio_values.size > 0 else float("nan"),
            "per_frame_ratio_cv": ratio_cv,
            "per_frame_ratio_log_std": ratio_log_std,
        },
    }

    print(json.dumps(result, indent=2))
    if args.out_json is not None:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved report: {args.out_json}")


if __name__ == "__main__":
    main()
