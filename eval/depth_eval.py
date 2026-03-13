#!/usr/bin/env python3
"""
Depth evaluation for VGGT vs ViPE pseudo-GT metric depth.

Since VGGT predicts normalized/affine-invariant depth while ViPE provides metric depth,
two alignment strategies are used:

  1. Sim(3)-scaled  — Apply the Sim(3) scale from camera-center alignment to convert
                      VGGT world-space depths to metric, then compute metrics directly.

  2. Affine-invariant — Fit per-frame least-squares (scale, shift) so that
                        s * pred + t ≈ gt, removing any residual scale/shift.

Source of predicted depth
  Default: demo_colmap.py saves depths.npy (shape [N, 518, 518]) — per-frame dense
  depth predictions from VGGT.  This is the primary evaluation path.

  Fallback (--use-ply): project sparse PLY points into each predicted camera and
  compare against GT depth at corresponding pixels.  Used when depths.npy is missing
  or for comparison with the old evaluation method.

Usage
  python eval/depth_eval.py \\
    --gt-root   /path/to/dataset \\
    --vanilla-root   /path/to/vanilla_eval_exps/forced_mask_conf \\
    --finetuned-root /path/to/finetuned_eval_exps/forced_mask_conf \\
    --split-file /path/to/dataset/splits/train.txt \\
    --output-json eval_outputs/depth_eval_train.json

Outputs (per sequence, per model):
  abs_rel_sim3         — AbsRel after Sim(3)-scale alignment
  rmse_sim3            — RMSE  after Sim(3)-scale alignment
  delta1_sim3          — % pixels with max(pred/gt, gt/pred) < 1.25  (Sim(3)-scaled)
  abs_rel_affine       — AbsRel after per-frame affine (scale+shift) alignment
  rmse_affine          — RMSE  after per-frame affine alignment
  delta1_affine        — delta<1.25 after affine alignment
  depth_temporal_std   — mean pixel-wise temporal std of predicted depth (npy path only)
  num_valid_pairs      — total pixel pairs used across all frames in the sequence
  num_frames_ok        — frames with enough valid pairs (>= min_pairs_per_frame)
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _cv2 = None
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_PAIRS_PER_FRAME = 50    # frames with fewer valid pairs are skipped
MIN_FRAMES_PER_SEQ = 5     # sequences with fewer ok frames are skipped
EPS = 1e-8
FRAME_RE = re.compile(r"frame_(\d+)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Depth evaluation: VGGT depths.npy vs GT metric depth.")
    p.add_argument("--gt-root",        required=True,  help="Root containing GT sequences.")
    p.add_argument("--vanilla-root",   required=True,
                   help="Root containing vanilla model outputs.")
    p.add_argument("--finetuned-root", required=True,
                   help="Root containing finetuned model outputs.")
    p.add_argument("--split-file",     default=None,
                   help="Sequence list file (one name per line).")
    p.add_argument("--gt-depth-dir",   default="depth",
                   help="Sub-folder under each GT sequence for depth maps (default: 'depth').")
    p.add_argument("--gt-subdir",      default="colmap",
                   help="Sub-folder under each GT sequence for COLMAP cameras (default: 'colmap').")
    p.add_argument("--pred-subdir",    default="sparse",
                   help="Sub-folder under each predicted sequence folder (default: 'sparse').")
    p.add_argument("--pred-depth-file", default="depths.npy",
                   help="Filename for predicted depth array under each pred sequence folder (default: 'depths.npy').")
    p.add_argument("--output-json",    default=None,   help="Path to save JSON report.")
    p.add_argument("--use-ply",        action="store_true",
                   help="Fall back to sparse PLY projection for depth (old method).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# COLMAP binary readers  (no pycolmap dependency)
# ---------------------------------------------------------------------------

@dataclass
class ColmapCamera:
    camera_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ColmapImage:
    image_id: int
    R: np.ndarray        # (3,3) rotation  cam-from-world
    t: np.ndarray        # (3,)  translation cam-from-world
    center: np.ndarray   # (3,)  camera center in world
    camera_id: int
    name: str            # image filename (basename only)


def _read_cameras_bin(path: Path) -> Dict[int, ColmapCamera]:
    cameras: Dict[int, ColmapCamera] = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            # model_id 0 = SIMPLE_PINHOLE (f, cx, cy)
            # model_id 1 = PINHOLE        (fx, fy, cx, cy)
            if model_id == 0:
                f_val, cx, cy = struct.unpack("<3d", f.read(24))
                fx = fy = f_val
            elif model_id == 1:
                fx, fy, cx, cy = struct.unpack("<4d", f.read(32))
            else:
                # Unsupported model — skip by reading remaining params
                # (We cannot know param count without a full model table here;
                #  add more cases if needed.)
                raise ValueError(f"Unsupported COLMAP camera model_id={model_id}")
            cameras[cam_id] = ColmapCamera(cam_id, int(width), int(height), fx, fy, cx, cy)
    return cameras


def _quat_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < EPS:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)


def _read_images_bin(path: Path) -> Dict[str, ColmapImage]:
    images: Dict[str, ColmapImage] = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = Path(name_bytes.decode()).name    # basename only
            num_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts2d * 24)                   # skip points2D
            R = _quat_to_rotmat(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=np.float64)
            center = -R.T @ t
            images[name] = ColmapImage(image_id, R, t, center, camera_id, name)
    return images


def _read_images_txt(path: Path) -> Dict[str, ColmapImage]:
    """Fallback: read images.txt instead of images.bin."""
    images: Dict[str, ColmapImage] = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    # Each image uses 2 lines; odd lines are the data lines
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = Path(parts[9]).name
        R = _quat_to_rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        center = -R.T @ t
        images[name] = ColmapImage(image_id, R, t, center, camera_id, name)
    return images


def load_colmap_model(model_dir: Path) -> Tuple[Dict[int, ColmapCamera], Dict[str, ColmapImage]]:
    cameras_bin = model_dir / "cameras.bin"
    images_bin = model_dir / "images.bin"
    cameras_txt = model_dir / "cameras.txt"
    images_txt = model_dir / "images.txt"

    if cameras_bin.is_file():
        cameras = _read_cameras_bin(cameras_bin)
    elif cameras_txt.is_file():
        cameras = _parse_cameras_txt(cameras_txt)
    else:
        raise FileNotFoundError(f"No cameras.bin or cameras.txt in {model_dir}")

    if images_bin.is_file():
        images = _read_images_bin(images_bin)
    elif images_txt.is_file():
        images = _read_images_txt(images_txt)
    else:
        raise FileNotFoundError(f"No images.bin or images.txt in {model_dir}")

    return cameras, images


def _parse_cameras_txt(path: Path) -> Dict[int, ColmapCamera]:
    cameras: Dict[int, ColmapCamera] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1].upper()
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = params[0]
                cx = params[1]
                cy = params[2]
            else:  # PINHOLE, OPENCV, etc.
                fx = params[0]
                fy = params[1]
                cx = params[2]
                cy = params[3]
            cameras[cam_id] = ColmapCamera(cam_id, width, height, fx, fy, cx, cy)
    return cameras


# ---------------------------------------------------------------------------
# PLY reader  (binary little-endian, float xyz)
# ---------------------------------------------------------------------------

def read_ply_points(ply_path: Path) -> np.ndarray:
    """Return (N, 3) float64 XYZ array from a binary PLY file."""
    with open(ply_path, "rb") as f:
        # Parse header
        n_vertices = 0
        header_end = 0
        prop_order: List[str] = []
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property float"):
                prop_order.append(line.split()[-1])
            elif line == "end_header":
                header_end = f.tell()
                break

        if n_vertices == 0:
            return np.zeros((0, 3), dtype=np.float64)

        # Each vertex: 3 floats (xyz) + remaining properties (uchar rgb etc.)
        # We only need xyz — read all vertices and take first 3 floats
        n_float_props = sum(1 for p in prop_order if p in ("x", "y", "z"))
        n_uchar_props = len(prop_order) - n_float_props
        bytes_per_vertex = n_float_props * 4 + n_uchar_props * 1

        raw = np.frombuffer(f.read(n_vertices * bytes_per_vertex), dtype=np.uint8)
        raw = raw.reshape(n_vertices, bytes_per_vertex)

        # Extract xyz floats from first 12 bytes
        xyz = np.frombuffer(raw[:, :12].tobytes(), dtype=np.float32).reshape(n_vertices, 3)
    return xyz.astype(np.float64)


# ---------------------------------------------------------------------------
# Sim(3) alignment  (Umeyama, camera centers)
# ---------------------------------------------------------------------------

def umeyama_sim3(
    src: np.ndarray,   # (N, 3) predicted camera centers
    dst: np.ndarray,   # (N, 3) GT camera centers
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (scale, R, t, aligned_src) solving dst ≈ scale * R @ src + t."""
    n = src.shape[0]
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    var_src = (src_c ** 2).sum() / n
    cov = (dst_c.T @ src_c) / n
    U, d, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    scale = float((d * np.diag(S)).sum() / (var_src + EPS))
    t = dst_mean - scale * (R @ src_mean)
    aligned = (scale * (R @ src.T)).T + t
    return scale, R, t, aligned


# ---------------------------------------------------------------------------
# Per-frame depth metrics
# ---------------------------------------------------------------------------

def _affine_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Fit s, b via least squares such that s*pred + b ≈ gt.
    Returns aligned prediction.
    """
    A = np.stack([pred, np.ones_like(pred)], axis=1)   # (N, 2)
    result, _, _, _ = np.linalg.lstsq(A, gt, rcond=None)
    s, b = result
    return s * pred + b


def compute_depth_metrics(
    pred_metric: np.ndarray,   # (N,) predicted depth, already Sim(3)-scaled to metric
    gt_depth:    np.ndarray,   # (N,) GT metric depth
) -> Dict[str, float]:
    """
    Compute depth metrics for one set of valid pixel pairs.
    Returns both Sim(3)-scaled and affine-invariant metrics.
    """
    mask = (gt_depth > EPS) & (pred_metric > EPS)
    if mask.sum() < MIN_PAIRS_PER_FRAME:
        return {}

    p = pred_metric[mask]
    g = gt_depth[mask]

    # --- Sim(3)-scaled metrics ---
    rel = np.abs(p - g) / (g + EPS)
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / (g + EPS), g / (p + EPS))
    d1 = float((ratio < 1.25).mean())

    # --- Affine-invariant metrics ---
    p_aff = _affine_align(p, g)
    mask2 = p_aff > EPS
    if mask2.sum() >= MIN_PAIRS_PER_FRAME:
        rel_aff = np.abs(p_aff[mask2] - g[mask2]) / (g[mask2] + EPS)
        rmse_aff = float(np.sqrt(np.mean((p_aff[mask2] - g[mask2]) ** 2)))
        ratio_aff = np.maximum(p_aff[mask2] / (g[mask2]+EPS), g[mask2] / (p_aff[mask2]+EPS))
        d1_aff = float((ratio_aff < 1.25).mean())
        abs_rel_aff = float(rel_aff.mean())
    else:
        rmse_aff = abs_rel_aff = d1_aff = float("nan")

    return {
        "abs_rel_sim3":   float(rel.mean()),
        "rmse_sim3":      rmse,
        "delta1_sim3":    d1,
        "abs_rel_affine": abs_rel_aff,
        "rmse_affine":    rmse_aff,
        "delta1_affine":  d1_aff,
        "n_pairs":        int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

def _resize_depth(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a 2-D depth map to (h, w) using bilinear interpolation."""
    if arr.shape == (h, w):
        return arr
    if _HAS_CV2:
        return _cv2.resize(arr.astype(np.float32), (w, h), interpolation=_cv2.INTER_LINEAR).astype(np.float64)
    # Fallback: nearest-neighbour via index arithmetic (no extra deps)
    src_h, src_w = arr.shape
    row_idx = np.clip(np.round(np.linspace(0, src_h - 1, h)).astype(int), 0, src_h - 1)
    col_idx = np.clip(np.round(np.linspace(0, src_w - 1, w)).astype(int), 0, src_w - 1)
    return arr[np.ix_(row_idx, col_idx)].astype(np.float64)


def evaluate_sequence_from_npy(
    pred_model_dir:  Path,
    pred_depth_file: str,
    gt_depth_dir:    Path,
    gt_cameras:      Dict[int, ColmapCamera],
    gt_images:       Dict[str, ColmapImage],
) -> Optional[Dict]:
    """
    Evaluate depth using depths.npy saved by demo_colmap.py (shape [N, 518, 518]).
    Returns aggregated metrics dict or an error dict.
    """
    pred_depth_npy = pred_model_dir / pred_depth_file
    if not pred_depth_npy.is_file():
        return {"error": f"missing pred depth: {pred_depth_npy}"}

    gt_depth_npy = gt_depth_dir / "depths.npy"
    if not gt_depth_npy.is_file():
        return {"error": f"missing GT depth: {gt_depth_npy}"}

    # Load predicted COLMAP model (for Sim(3) scale)
    try:
        pred_cameras, pred_images = load_colmap_model(pred_model_dir)
    except Exception as e:
        return {"error": f"load_pred_model: {e}"}

    # Common frames (order by gt image id for temporal consistency)
    common_names = sorted(
        [n for n in gt_images if n in pred_images],
        key=lambda n: gt_images[n].image_id,
    )
    if len(common_names) < MIN_FRAMES_PER_SEQ:
        return {"error": f"too few common frames: {len(common_names)}"}

    # Sim(3) scale: predicted camera centers → GT camera centers
    gt_centers  = np.stack([gt_images[n].center  for n in common_names])
    pred_centers = np.stack([pred_images[n].center for n in common_names])
    try:
        sim3_scale, _, _, _ = umeyama_sim3(pred_centers, gt_centers)
    except Exception as e:
        return {"error": f"sim3: {e}"}

    if sim3_scale <= 0 or not np.isfinite(sim3_scale):
        return {"error": f"invalid sim3_scale={sim3_scale}"}

    # Load depth arrays
    try:
        pred_depth_arr = np.load(pred_depth_npy, mmap_mode="r")   # (N, 518, 518)
        gt_depth_arr   = np.load(gt_depth_npy,   mmap_mode="r")   # (N_gt, H_gt, W_gt)
    except Exception as e:
        return {"error": f"load_depth_arrays: {e}"}

    n_pred_frames = pred_depth_arr.shape[0]
    n_gt_frames, gt_H, gt_W = gt_depth_arr.shape

    all_metrics: List[Dict] = []
    pred_frames_resized: List[np.ndarray] = []  # for depth_temporal_std

    for name in common_names:
        m = FRAME_RE.search(name)
        if m is None:
            continue
        frame_idx = int(m.group(1))
        if frame_idx < 0 or frame_idx >= n_pred_frames or frame_idx >= n_gt_frames:
            continue

        # Predicted depth: scale to metric, resize to GT resolution
        pred_slice = np.array(pred_depth_arr[frame_idx], dtype=np.float64)
        pred_metric_full = sim3_scale * pred_slice                   # (518, 518)
        pred_metric_resized = _resize_depth(pred_metric_full, gt_H, gt_W)  # (gt_H, gt_W)

        # GT depth
        gt_frame = np.array(gt_depth_arr[frame_idx], dtype=np.float64)    # (gt_H, gt_W)

        # Mask: only where GT > 0
        mask = gt_frame > EPS
        if mask.sum() < MIN_PAIRS_PER_FRAME:
            continue

        frame_metrics = compute_depth_metrics(
            pred_metric_resized.ravel(), gt_frame.ravel()
        )
        if frame_metrics:
            all_metrics.append(frame_metrics)
            # Store masked-predicted depth for temporal std (use GT mask)
            pred_masked = np.where(mask, pred_metric_resized, 0.0)
            pred_frames_resized.append(pred_masked)

    if len(all_metrics) < MIN_FRAMES_PER_SEQ:
        return {"error": f"too few valid frames: {len(all_metrics)}"}

    def _wavg(key: str) -> float:
        vals    = [m[key]       for m in all_metrics if key in m and np.isfinite(m[key])]
        weights = [m["n_pairs"] for m in all_metrics if key in m and np.isfinite(m[key])]
        if not vals:
            return float("nan")
        return float(np.average(vals, weights=weights))

    # depth_temporal_std: mean pixel-wise std across frames (GT-masked pixels only)
    depth_temporal_std = float("nan")
    if len(pred_frames_resized) >= 2:
        depth_stack = np.stack(pred_frames_resized, axis=0)   # (T, gt_H, gt_W)
        # Compute GT mask union across frames to only average where depth is valid
        gt_mask_union = np.zeros((gt_H, gt_W), dtype=bool)
        for name in common_names:
            m_re = FRAME_RE.search(name)
            if m_re is None:
                continue
            fi = int(m_re.group(1))
            if fi < 0 or fi >= n_gt_frames:
                continue
            gt_mask_union |= (np.array(gt_depth_arr[fi], dtype=np.float64) > EPS)
        pixel_stds = np.std(depth_stack, axis=0)              # (gt_H, gt_W)
        if gt_mask_union.sum() > 0:
            depth_temporal_std = float(np.mean(pixel_stds[gt_mask_union]))

    total_pairs = sum(m["n_pairs"] for m in all_metrics)
    return {
        "abs_rel_sim3":        _wavg("abs_rel_sim3"),
        "rmse_sim3":           _wavg("rmse_sim3"),
        "delta1_sim3":         _wavg("delta1_sim3"),
        "abs_rel_affine":      _wavg("abs_rel_affine"),
        "rmse_affine":         _wavg("rmse_affine"),
        "delta1_affine":       _wavg("delta1_affine"),
        "depth_temporal_std":  depth_temporal_std,
        "num_valid_pairs":     total_pairs,
        "num_frames_ok":       len(all_metrics),
        "sim3_scale":          sim3_scale,
    }


def evaluate_sequence_ply(
    pred_model_dir: Path,
    gt_depth_dir:   Path,
    gt_cameras:     Dict[int, ColmapCamera],
    gt_images:      Dict[str, ColmapImage],
) -> Optional[Dict]:
    """
    Legacy PLY-projection-based depth evaluation (sparse proxy).
    Used when --use-ply is set or when depths.npy is unavailable.
    """
    ply_path = pred_model_dir / "points.ply"
    if not ply_path.is_file():
        return {"error": "missing points.ply"}

    depth_npy = gt_depth_dir / "depths.npy"
    if not depth_npy.is_file():
        return {"error": f"missing GT depth: {depth_npy}"}

    # Load predicted model
    try:
        pred_cameras, pred_images = load_colmap_model(pred_model_dir)
    except Exception as e:
        return {"error": f"load_pred_model: {e}"}

    # Load PLY
    try:
        pts_world = read_ply_points(ply_path)   # (N, 3)
    except Exception as e:
        return {"error": f"read_ply: {e}"}

    if len(pts_world) == 0:
        return {"error": "empty PLY"}

    # Common frames
    common_names = [n for n in gt_images if n in pred_images]
    if len(common_names) < MIN_FRAMES_PER_SEQ:
        return {"error": f"too few common frames: {len(common_names)}"}

    # Sim(3) from predicted camera centers to GT camera centers
    gt_centers = np.stack([gt_images[n].center for n in common_names])
    pred_centers = np.stack([pred_images[n].center for n in common_names])
    try:
        sim3_scale, _, _, _ = umeyama_sim3(pred_centers, gt_centers)
    except Exception as e:
        return {"error": f"sim3: {e}"}

    if sim3_scale <= 0 or not np.isfinite(sim3_scale):
        return {"error": f"invalid sim3_scale={sim3_scale}"}

    # Load GT depth array (memory-mapped to avoid loading all frames at once)
    try:
        gt_depth_arr = np.load(depth_npy, mmap_mode="r")   # (N_frames, H, W)
    except Exception as e:
        return {"error": f"load_gt_depth: {e}"}

    n_frames_arr, gt_H, gt_W = gt_depth_arr.shape

    # --- Per-frame projection and metric accumulation ---
    all_metrics: List[Dict] = []

    for name in common_names:
        pred_img = pred_images[name]
        pred_cam = pred_cameras.get(pred_img.camera_id)
        if pred_cam is None:
            continue

        # Get GT depth frame index from filename  (frame_XXXXXX.jpg → index XXXXXX)
        m = FRAME_RE.search(name)
        if m is None:
            continue
        depth_idx = int(m.group(1))
        if depth_idx < 0 or depth_idx >= n_frames_arr:
            continue

        gt_frame = gt_depth_arr[depth_idx].astype(np.float32)   # (H, W)

        # Project PLY points into predicted camera
        R_p = pred_img.R
        t_p = pred_img.t
        pts_cam = (R_p @ pts_world.T).T + t_p

        valid = pts_cam[:, 2] > EPS
        pts_cam = pts_cam[valid]
        if len(pts_cam) == 0:
            continue

        z = pts_cam[:, 2]
        u = pred_cam.fx * pts_cam[:, 0] / z + pred_cam.cx
        v = pred_cam.fy * pts_cam[:, 1] / z + pred_cam.cy

        W_p = pred_cam.width
        H_p = pred_cam.height
        finite = np.isfinite(u) & np.isfinite(v)
        u = u[finite]; v = v[finite]; z = z[finite]
        if len(z) == 0:
            continue

        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        in_bounds = (ui >= 0) & (ui < W_p) & (vi >= 0) & (vi < H_p)
        ui = ui[in_bounds]; vi = vi[in_bounds]; z = z[in_bounds]
        if len(z) == 0:
            continue

        pred_metric = sim3_scale * z

        if W_p == gt_W and H_p == gt_H:
            ui_gt = ui; vi_gt = vi
        else:
            ui_gt = np.clip(np.round(ui * gt_W / W_p).astype(np.int32), 0, gt_W - 1)
            vi_gt = np.clip(np.round(vi * gt_H / H_p).astype(np.int32), 0, gt_H - 1)

        gt_sampled = gt_frame[vi_gt, ui_gt].astype(np.float64)
        frame_metrics = compute_depth_metrics(pred_metric, gt_sampled)
        if frame_metrics:
            all_metrics.append(frame_metrics)

    if len(all_metrics) < MIN_FRAMES_PER_SEQ:
        return {"error": f"too few valid frames: {len(all_metrics)}"}

    def _wavg(key: str) -> float:
        vals    = [m[key]       for m in all_metrics if key in m and np.isfinite(m[key])]
        weights = [m["n_pairs"] for m in all_metrics if key in m and np.isfinite(m[key])]
        if not vals:
            return float("nan")
        return float(np.average(vals, weights=weights))

    total_pairs = sum(m["n_pairs"] for m in all_metrics)
    return {
        "abs_rel_sim3":   _wavg("abs_rel_sim3"),
        "rmse_sim3":      _wavg("rmse_sim3"),
        "delta1_sim3":    _wavg("delta1_sim3"),
        "abs_rel_affine": _wavg("abs_rel_affine"),
        "rmse_affine":    _wavg("rmse_affine"),
        "delta1_affine":  _wavg("delta1_affine"),
        "num_valid_pairs": total_pairs,
        "num_frames_ok":  len(all_metrics),
        "sim3_scale":     sim3_scale,
    }


def evaluate_sequence(
    pred_model_dir:  Path,
    pred_depth_file: str,
    gt_depth_dir:    Path,
    gt_cameras:      Dict[int, ColmapCamera],
    gt_images:       Dict[str, ColmapImage],
    use_ply:         bool = False,
) -> Optional[Dict]:
    """Route to npy-based (default) or PLY-based (fallback) evaluation."""
    pred_depth_npy = pred_model_dir / pred_depth_file
    if use_ply or not pred_depth_npy.is_file():
        return evaluate_sequence_ply(pred_model_dir, gt_depth_dir, gt_cameras, gt_images)
    return evaluate_sequence_from_npy(
        pred_model_dir, pred_depth_file, gt_depth_dir, gt_cameras, gt_images
    )


# ---------------------------------------------------------------------------
# Sequence list helpers
# ---------------------------------------------------------------------------

def load_sequences(split_file: Optional[Path], gt_root: Path) -> List[str]:
    if split_file is not None and split_file.is_file():
        seqs = []
        for line in split_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                seqs.append(line)
        return seqs
    # Discover automatically
    return sorted(p.name for p in gt_root.iterdir() if p.is_dir())


def resolve_pred_dir(pred_root: Path, seq: str, pred_subdir: str) -> Optional[Path]:
    d = pred_root / seq / pred_subdir
    if d.is_dir():
        return d
    d2 = pred_root / seq
    if d2.is_dir():
        return d2
    return None


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(rows: List[Dict]) -> None:
    headers = [
        ("Sequence",        34),
        ("V_AbsRel_S3",     12),
        ("V_d1_S3",         9),
        ("V_AbsRel_Af",     12),
        ("V_d1_Af",         9),
        ("V_TmpStd",        9),
        ("F_AbsRel_S3",     12),
        ("F_d1_S3",         9),
        ("F_AbsRel_Af",     12),
        ("F_d1_Af",         9),
        ("F_TmpStd",        9),
    ]
    hdr = " ".join(f"{h:>{w}}" for h, w in headers)
    print(hdr)
    print("-" * len(hdr))

    def fv(d: Optional[Dict], key: str) -> str:
        if d is None or "error" in d:
            return "err"
        v = d.get(key)
        if v is None or not np.isfinite(v):
            return "nan"
        return f"{v:.4f}"

    v_rows, f_rows = [], []
    for row in rows:
        v = row.get("vanilla")
        ft = row.get("finetuned")
        parts = [
            f"{row['sequence']:<34}",
            f"{fv(v,  'abs_rel_sim3'):>12}",
            f"{fv(v,  'delta1_sim3'):>9}",
            f"{fv(v,  'abs_rel_affine'):>12}",
            f"{fv(v,  'delta1_affine'):>9}",
            f"{fv(v,  'depth_temporal_std'):>9}",
            f"{fv(ft, 'abs_rel_sim3'):>12}",
            f"{fv(ft, 'delta1_sim3'):>9}",
            f"{fv(ft, 'abs_rel_affine'):>12}",
            f"{fv(ft, 'delta1_affine'):>9}",
            f"{fv(ft, 'depth_temporal_std'):>9}",
        ]
        print(" ".join(parts))
        if v and "error" not in v:
            v_rows.append(v)
        if ft and "error" not in ft:
            f_rows.append(ft)

    print("-" * len(hdr))

    def mean_metric(rows_: List[Dict], key: str) -> str:
        vals = [r[key] for r in rows_ if np.isfinite(r.get(key, float("nan")))]
        return f"{np.mean(vals):.4f}" if vals else "nan"

    keys = ["abs_rel_sim3", "delta1_sim3", "abs_rel_affine", "delta1_affine", "depth_temporal_std"]
    print(f"\n{'Means':}")
    print(f"  vanilla  : " + "  ".join(f"{k}={mean_metric(v_rows, k)}" for k in keys))
    print(f"  finetuned: " + "  ".join(f"{k}={mean_metric(f_rows, k)}" for k in keys))

    # Win rates
    lower_better = ["abs_rel_sim3", "rmse_sim3", "abs_rel_affine", "rmse_affine", "depth_temporal_std"]
    higher_better = ["delta1_sim3", "delta1_affine"]
    print("\nWin-rate (finetuned better):")
    for key in ["abs_rel_sim3", "delta1_sim3", "abs_rel_affine", "delta1_affine", "depth_temporal_std"]:
        wins = ties = total = 0
        for row in rows:
            v = row.get("vanilla")
            ft = row.get("finetuned")
            if not v or not ft or "error" in v or "error" in ft:
                continue
            vv = v.get(key)
            fv_ = ft.get(key)
            if vv is None or fv_ is None or not np.isfinite(vv) or not np.isfinite(fv_):
                continue
            total += 1
            if abs(fv_ - vv) < 1e-8:
                ties += 1
            elif key in lower_better:
                wins += (1 if fv_ < vv else 0)
            else:
                wins += (1 if fv_ > vv else 0)
        rate = f"{wins}/{total} ({wins/total:.2f})" if total else "n/a"
        print(f"  {key}: {rate}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    gt_root = Path(args.gt_root)
    vanilla_root = Path(args.vanilla_root)
    finetuned_root = Path(args.finetuned_root)
    split_file = Path(args.split_file) if args.split_file else None

    sequences = load_sequences(split_file, gt_root)
    rows: List[Dict] = []

    for seq in sequences:
        row: Dict = {"sequence": seq, "vanilla": None, "finetuned": None}

        # Load GT cameras (for Sim(3) computation)
        gt_colmap_dir = gt_root / seq / args.gt_subdir
        if not gt_colmap_dir.is_dir():
            gt_colmap_dir = gt_root / seq
        gt_depth_dir = gt_root / seq / args.gt_depth_dir

        try:
            gt_cameras, gt_images = load_colmap_model(gt_colmap_dir)
        except Exception as e:
            row["error"] = f"gt_load: {e}"
            rows.append(row)
            print(f"[SKIP] {seq}: gt_load error: {e}")
            continue

        for tag, pred_root in (("vanilla", vanilla_root), ("finetuned", finetuned_root)):
            pred_dir = resolve_pred_dir(pred_root, seq, args.pred_subdir)
            if pred_dir is None:
                row[tag] = {"error": "missing pred dir"}
                continue
            result = evaluate_sequence(
                pred_dir,
                args.pred_depth_file,
                gt_depth_dir,
                gt_cameras,
                gt_images,
                use_ply=args.use_ply,
            )
            row[tag] = result

        rows.append(row)

        # Quick progress line
        v = row.get("vanilla") or {}
        ft = row.get("finetuned") or {}
        v_str = f"AbsRel_S3={v.get('abs_rel_sim3', 'n/a'):.4f}" if v and "error" not in v else v.get(
            "error", "err")
        ft_str = f"AbsRel_S3={ft.get('abs_rel_sim3', 'n/a'):.4f}" if ft and "error" not in ft else ft.get(
            "error", "err")
        print(f"[{seq}]  vanilla={v_str}  finetuned={ft_str}")

    print("\n" + "=" * 80)
    print_summary(rows)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "config": {
                "gt_root":        str(gt_root),
                "vanilla_root":   str(vanilla_root),
                "finetuned_root": str(finetuned_root),
                "split_file":     str(split_file) if split_file else None,
                "gt_depth_dir":   args.gt_depth_dir,
                "pred_subdir":    args.pred_subdir,
            },
            "results": rows,
        }
        with out_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved JSON: {out_path}")


if __name__ == "__main__":
    main()
