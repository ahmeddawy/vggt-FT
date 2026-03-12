#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize RGB, original depth, masked depth, and masked-out pixels as a 2x2 video."
        )
    )
    parser.add_argument("--seq-dir", type=str, required=True, help="Sequence directory.")
    parser.add_argument("--images-dir", type=str, default=None, help="Optional custom images directory.")
    parser.add_argument("--depth-path", type=str, default=None, help="Optional custom original depth .npy/.npz path.")
    parser.add_argument(
        "--masked-depth-path",
        type=str,
        default=None,
        help="Optional custom masked depth .npy/.npz path.",
    )
    parser.add_argument("--depth-key", type=str, default="depths", help="Key for .npz depth files.")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output mp4 path (default: <seq-dir>/masked_depth_preview.mp4).",
    )
    parser.add_argument("--fps", type=float, default=12.0, help="Output video FPS.")
    parser.add_argument("--max-frames", type=int, default=-1, help="Max frames to render; -1 means all.")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index.")
    parser.add_argument(
        "--colormap",
        type=str,
        default="turbo",
        choices=["turbo", "inferno", "magma", "jet"],
        help="Depth colormap.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    seq_dir = Path(args.seq_dir)
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    images_dir = Path(args.images_dir) if args.images_dir else seq_dir / "images"
    depth_path = Path(args.depth_path) if args.depth_path else seq_dir / "depth" / "depths.npy"
    masked_depth_path = (
        Path(args.masked_depth_path) if args.masked_depth_path else seq_dir / "masked_depth" / "depths.npy"
    )
    output_path = (
        Path(args.output_path) if args.output_path else seq_dir / "masked_depth_preview.mp4"
    )

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not depth_path.is_file():
        raise FileNotFoundError(f"Original depth not found: {depth_path}")
    if not masked_depth_path.is_file():
        raise FileNotFoundError(f"Masked depth not found: {masked_depth_path}")

    return seq_dir, images_dir, depth_path, masked_depth_path, output_path


def list_images(images_dir: Path) -> List[Path]:
    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    images.sort()
    if len(images) == 0:
        raise RuntimeError(f"No images found in {images_dir}")
    return images


def load_depth(path: Path, depth_key: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, mmap_mode="r")
    elif path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if depth_key not in data:
                raise KeyError(f"Depth key '{depth_key}' not found in {path}. Available: {list(data.files)}")
            arr = data[depth_key]
    else:
        raise ValueError(f"Unsupported depth format: {path}")

    if arr.ndim != 3:
        raise ValueError(f"Depth array must be [N,H,W], got shape {arr.shape} ({path})")
    return arr


def get_cv_colormap(name: str) -> int:
    if name == "turbo":
        return cv2.COLORMAP_TURBO
    if name == "inferno":
        return cv2.COLORMAP_INFERNO
    if name == "magma":
        return cv2.COLORMAP_MAGMA
    if name == "jet":
        return cv2.COLORMAP_JET
    raise ValueError(f"Unknown colormap: {name}")


def depth_to_color(depth: np.ndarray, cmap: int) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() < 16:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    vals = depth[valid]
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if hi <= lo:
        hi = lo + 1e-6

    depth_norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cmap)
    depth_color[~valid] = 0
    return depth_color


def put_title(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    _, images_dir, depth_path, masked_depth_path, output_path = resolve_paths(args)

    image_paths = list_images(images_dir)
    depth = load_depth(depth_path, args.depth_key)
    masked_depth = load_depth(masked_depth_path, args.depth_key)

    if depth.shape[1:] != masked_depth.shape[1:]:
        raise ValueError(f"Depth shape mismatch: {depth.shape} vs {masked_depth.shape}")

    total = min(len(image_paths), depth.shape[0], masked_depth.shape[0])
    start = max(0, int(args.start_frame))
    end = total if args.max_frames < 0 else min(total, start + int(args.max_frames))
    if start >= end:
        raise ValueError(f"Empty frame range: start={start}, end={end}, total={total}")

    h, w = depth.shape[1], depth.shape[2]
    video_size = (w * 2, h * 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(args.fps), video_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    cmap = get_cv_colormap(args.colormap)

    for i in range(start, end):
        rgb = cv2.imread(str(image_paths[i]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read image: {image_paths[i]}")
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

        d = np.asarray(depth[i], dtype=np.float32)
        d_masked = np.asarray(masked_depth[i], dtype=np.float32)

        d_color = depth_to_color(d, cmap)
        d_masked_color = depth_to_color(d_masked, cmap)

        removed = ((d > 0) & (d_masked <= 0)).astype(np.uint8)
        removed_vis = np.zeros((h, w, 3), dtype=np.uint8)
        removed_vis[..., 2] = removed * 255

        rgb_panel = put_title(rgb, "RGB")
        d_panel = put_title(d_color, "Original Depth")
        d_masked_panel = put_title(d_masked_color, "Masked Depth")
        removed_panel = put_title(removed_vis, "Masked-Out Pixels")

        top = np.concatenate([rgb_panel, d_panel], axis=1)
        bot = np.concatenate([d_masked_panel, removed_panel], axis=1)
        frame = np.concatenate([top, bot], axis=0)

        cv2.putText(
            frame,
            f"frame {i}",
            (video_size[0] - 220, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    print(f"Saved video: {output_path}")
    print(f"Rendered frames: {start}..{end - 1} (count={end - start})")


if __name__ == "__main__":
    main()

