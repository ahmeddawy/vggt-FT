# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import re
import json
import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap
from PIL import Image


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (defaults to <scene_dir>/sparse)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/bucket/dawy/vggt_finetune/checkpoint_5.pt",
        help="Path to model checkpoint (.pt).",
    )
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=8, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=4096, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default=None,
        help="Optional mask directory. If not set, tries <scene_dir>/masks.",
    )
    parser.add_argument(
        "--ignore_masks",
        action="store_true",
        default=False,
        help="Disable mask-based confidence suppression even if masks exist.",
    )
    parser.add_argument(
        "--mask_background_value",
        type=int,
        default=0,
        help="Mask pixel value treated as background/valid region (kept).",
    )
    parser.add_argument(
        "--mask_conf_suppressed_value",
        type=float,
        default=1.0,
        help="Confidence value assigned to masked-out pixels (for expp1 confidence, 1.0 is minimum).",
    )
    parser.add_argument(
        "--disable_mask_conf_suppression",
        action="store_true",
        default=False,
        help="If set, masks are used only for diagnostics; confidence is not externally suppressed.",
    )
    parser.add_argument(
        "--save_conf_mask_report",
        action="store_true",
        default=False,
        help="Save conf-mask diagnostics JSON to <output_dir>/conf_mask_report.json.",
    )
    parser.add_argument(
        "--save_depth",
        action="store_true",
        default=True,
        help="Save per-frame depth maps as <output_dir>/depths.npy (indexed by frame number, shape [max_frame+1, H, W]).",
    )
    parser.add_argument(
        "--save_depth_conf",
        action="store_true",
        default=False,
        help="Save per-frame depth confidence maps as <output_dir>/depth_confs.npy (shape [N, H, W]).",
    )
    return parser.parse_args()


def _extract_frame_index(image_name: str):
    stem = Path(image_name).stem
    # Matches names like frame_000123 or 000123
    m = re.search(r"(\d+)$", stem)
    if m is None:
        return None
    return int(m.group(1))


def _resolve_mask_path(image_name: str, masks_dir: Path, fallback_idx: int):
    stem = Path(image_name).stem
    frame_idx = _extract_frame_index(image_name)

    candidates = [masks_dir / f"{stem}.png"]
    if frame_idx is not None:
        candidates.append(masks_dir / f"{frame_idx:05d}.png")
        candidates.append(masks_dir / f"{frame_idx:06d}.png")
    candidates.append(masks_dir / f"{fallback_idx:05d}.png")
    candidates.append(masks_dir / f"{fallback_idx:06d}.png")

    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_keep_mask_square(mask_path: Path, target_size: int, mask_background_value: int):
    """Load semantic/binary mask and convert to square target_size keep-mask.

    Returns:
        np.ndarray[bool]: shape (target_size, target_size), True for kept pixels.
    """
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    keep = (mask == mask_background_value).astype(np.uint8)

    h, w = keep.shape
    max_dim = max(h, w)
    top = (max_dim - h) // 2
    left = (max_dim - w) // 2
    square_keep = np.zeros((max_dim, max_dim), dtype=np.uint8)  # padded area is suppressed
    square_keep[top : top + h, left : left + w] = keep

    keep_img = Image.fromarray(square_keep * 255, mode="L")
    keep_img = keep_img.resize((target_size, target_size), resample=Image.NEAREST)
    keep_square = np.array(keep_img) > 127
    return keep_square


def build_keep_masks_for_scene(
    image_paths,
    masks_dir: Path,
    target_size: int,
    mask_background_value: int,
):
    keep_masks = []
    found = 0
    missing = 0

    for i, image_path in enumerate(image_paths):
        mask_path = _resolve_mask_path(image_path, masks_dir, fallback_idx=i)
        if mask_path is None:
            keep_masks.append(np.ones((target_size, target_size), dtype=bool))
            missing += 1
            continue
        keep_masks.append(
            _load_keep_mask_square(mask_path, target_size=target_size, mask_background_value=mask_background_value)
        )
        found += 1

    keep_masks = np.stack(keep_masks, axis=0)  # [S, H, W]
    return keep_masks, found, missing


def suppress_conf_with_keep_masks(conf_map, keep_masks, suppressed_value):
    conf_map = np.array(conf_map, copy=True)
    if conf_map.shape != keep_masks.shape:
        raise ValueError(f"Conf/mask shape mismatch: {conf_map.shape} vs {keep_masks.shape}")
    conf_map[~keep_masks] = suppressed_value
    return conf_map


def compute_conf_mask_stats(conf_map, keep_masks, conf_threshold):
    if conf_map.shape != keep_masks.shape:
        raise ValueError(f"Conf/mask shape mismatch: {conf_map.shape} vs {keep_masks.shape}")

    valid_mask = keep_masks.astype(bool)
    masked_mask = ~valid_mask

    valid_vals = conf_map[valid_mask]
    masked_vals = conf_map[masked_mask]

    def _mean_or_nan(arr):
        return float(arr.mean()) if arr.size > 0 else float("nan")

    def _rate_or_nan(arr, thres):
        return float((arr >= thres).mean()) if arr.size > 0 else float("nan")

    return {
        "conf_threshold": float(conf_threshold),
        "valid_count": int(valid_vals.size),
        "masked_count": int(masked_vals.size),
        "valid_mean": _mean_or_nan(valid_vals),
        "masked_mean": _mean_or_nan(masked_vals),
        "valid_p50": float(np.percentile(valid_vals, 50)) if valid_vals.size > 0 else float("nan"),
        "masked_p50": float(np.percentile(masked_vals, 50)) if masked_vals.size > 0 else float("nan"),
        "valid_keep_rate": _rate_or_nan(valid_vals, conf_threshold),
        "masked_keep_rate": _rate_or_nan(masked_vals, conf_threshold),
    }


def print_conf_mask_stats(tag, stats):
    print(
        f"[{tag}] conf-thr={stats['conf_threshold']:.3f} | "
        f"valid(mean={stats['valid_mean']:.4f}, p50={stats['valid_p50']:.4f}, keep={stats['valid_keep_rate']:.4f}, n={stats['valid_count']}) | "
        f"masked(mean={stats['masked_mean']:.4f}, p50={stats['masked_p50']:.4f}, keep={stats['masked_keep_rate']:.4f}, n={stats['masked_count']})"
    )


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))
    sparse_reconstruction_dir = args.output_dir if args.output_dir is not None else os.path.join(args.scene_dir, "sparse")

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    model = model.to(device)
    print(f"Model loaded from {ckpt_path}")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # Optional scene masks (semantic/binary) to suppress confidence on masked regions.
    keep_masks = None
    if not args.ignore_masks:
        masks_dir = Path(args.masks_dir) if args.masks_dir else Path(args.scene_dir) / "masks"
        if masks_dir.is_dir():
            keep_masks, found_mask_count, missing_mask_count = build_keep_masks_for_scene(
                image_path_list,
                masks_dir,
                target_size=vggt_fixed_resolution,
                mask_background_value=args.mask_background_value,
            )
            print(
                f"Using masks from {masks_dir} | found={found_mask_count}, missing={missing_mask_count}, "
                f"target_size={vggt_fixed_resolution}"
            )
        else:
            print(f"No masks directory found at {masks_dir}; continuing without masks.")

    # Run VGGT to estimate camera and depth (with 518x518 inference resolution).
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)

    conf_mask_report = None
    if keep_masks is not None:
        raw_stats = compute_conf_mask_stats(depth_conf, keep_masks, args.conf_thres_value)
        print_conf_mask_stats("raw_conf", raw_stats)

        conf_mask_report = {
            "scene_dir": str(args.scene_dir),
            "checkpoint": str(args.checkpoint),
            "conf_threshold": float(args.conf_thres_value),
            "suppression_applied": not args.disable_mask_conf_suppression,
            "mask_conf_suppressed_value": float(args.mask_conf_suppressed_value),
            "raw": raw_stats,
            "post": None,
        }

        if args.disable_mask_conf_suppression:
            print("Mask confidence suppression is disabled; using raw model confidence.")
        else:
            depth_conf = suppress_conf_with_keep_masks(
                depth_conf,
                keep_masks,
                suppressed_value=args.mask_conf_suppressed_value,
            )
            post_stats = compute_conf_mask_stats(depth_conf, keep_masks, args.conf_thres_value)
            print_conf_mask_stats("post_masked_conf", post_stats)
            conf_mask_report["post"] = post_stats
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    print(f"Saving reconstruction to {sparse_reconstruction_dir}")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    if args.save_conf_mask_report and conf_mask_report is not None:
        report_path = os.path.join(sparse_reconstruction_dir, "conf_mask_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(conf_mask_report, f, indent=2)
        print(f"Saved confidence mask report to {report_path}")

    if args.save_depth:
        # Build frame-number-indexed array: depths[frame_idx] = depth for frame_XXXXXX.jpg
        # This matches depth_eval.py which indexes by the number extracted from the filename.
        frame_nums = []
        for p in image_path_list:
            m = re.search(r"(\d+)$", Path(p).stem)
            frame_nums.append(int(m.group(1)) if m else None)
        valid_nums = [n for n in frame_nums if n is not None]
        max_frame = max(valid_nums) if valid_nums else len(frame_nums) - 1
        # Squeeze trailing channel dim if present: (N, H, W, 1) -> (N, H, W)
        dm = depth_map.squeeze(-1) if depth_map.ndim == 4 else depth_map
        H_d, W_d = dm.shape[1], dm.shape[2]
        depth_indexed = np.zeros((max_frame + 1, H_d, W_d), dtype=dm.dtype)
        for i, fn in enumerate(frame_nums):
            if fn is not None:
                depth_indexed[fn] = dm[i]
        depth_save_path = os.path.join(sparse_reconstruction_dir, "depths.npy")
        np.save(depth_save_path, depth_indexed)
        print(f"Saved depth maps to {depth_save_path} (shape={depth_indexed.shape})")

    if args.save_depth_conf:
        conf_save_path = os.path.join(sparse_reconstruction_dir, "depth_confs.npy")
        np.save(conf_save_path, depth_conf)
        print(f"Saved depth confidence maps to {conf_save_path} (shape={depth_conf.shape})")

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(sparse_reconstruction_dir, "points.ply"))

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
