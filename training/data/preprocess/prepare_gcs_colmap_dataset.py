#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import random
from typing import List, Optional

import numpy as np


def count_colmap_images(images_txt_path: str) -> int:
    count = 0
    with open(images_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10:
                count += 1
    return count


def resolve_depth_path(seq_dir: str) -> Optional[str]:
    depth_dir = osp.join(seq_dir, "depth")
    if not osp.isdir(depth_dir):
        return None

    for fname in ["depths.npy", "depths.npz"]:
        p = osp.join(depth_dir, fname)
        if osp.isfile(p):
            return p

    for fname in sorted(os.listdir(depth_dir)):
        if fname.endswith("_depths.npy") or fname.endswith("_depths.npz"):
            return osp.join(depth_dir, fname)
    return None


def maybe_convert_npz_to_npy(depth_path: str, depth_key: str = "depths") -> str:
    if not depth_path.endswith(".npz"):
        return depth_path
    npy_path = osp.splitext(depth_path)[0] + ".npy"
    if osp.isfile(npy_path):
        return npy_path

    with np.load(depth_path) as data:
        if depth_key not in data:
            raise KeyError(
                f"Depth key '{depth_key}' not found in {depth_path}. "
                f"Available keys: {list(data.files)}"
            )
        arr = data[depth_key]
    np.save(npy_path, arr.astype(np.float32))
    return npy_path


def main():
    parser = argparse.ArgumentParser(description="Prepare GCS COLMAP dataset split files.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to dataset root.")
    parser.add_argument("--out-dir", type=str, required=True, help="Where train.txt/test.txt will be saved.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio in [0, 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--min-frames", type=int, default=24, help="Minimum COLMAP frames per sequence.")
    parser.add_argument(
        "--convert-npz-to-npy",
        action="store_true",
        help="If set, convert depth npz files to npy for faster training IO.",
    )
    args = parser.parse_args()

    if not osp.isdir(args.dataset_dir):
        raise ValueError(f"Dataset dir does not exist: {args.dataset_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    valid_sequences: List[str] = []
    skipped = []

    for seq_name in sorted(os.listdir(args.dataset_dir)):
        seq_dir = osp.join(args.dataset_dir, seq_name)
        if not osp.isdir(seq_dir):
            continue

        cameras_txt = osp.join(seq_dir, "colmap", "cameras.txt")
        images_txt = osp.join(seq_dir, "colmap", "images.txt")
        images_dir = osp.join(seq_dir, "images")
        depth_path = resolve_depth_path(seq_dir)

        if not (osp.isfile(cameras_txt) and osp.isfile(images_txt) and osp.isdir(images_dir) and depth_path):
            skipped.append((seq_name, "missing colmap/images/depth"))
            continue

        n_frames = count_colmap_images(images_txt)
        if n_frames < args.min_frames:
            skipped.append((seq_name, f"only {n_frames} frames (< {args.min_frames})"))
            continue

        if args.convert_npz_to_npy:
            depth_path = maybe_convert_npz_to_npy(depth_path)

        valid_sequences.append(seq_name)

    if len(valid_sequences) == 0:
        raise RuntimeError("No valid sequences found.")

    rng = random.Random(args.seed)
    rng.shuffle(valid_sequences)

    n_val = int(round(len(valid_sequences) * args.val_ratio))
    n_val = max(1, n_val) if len(valid_sequences) > 1 else 0
    n_val = min(n_val, len(valid_sequences) - 1) if len(valid_sequences) > 1 else 0

    val_sequences = sorted(valid_sequences[:n_val])
    train_sequences = sorted(valid_sequences[n_val:])

    train_txt = osp.join(args.out_dir, "train.txt")
    test_txt = osp.join(args.out_dir, "test.txt")
    with open(train_txt, "w") as f:
        f.write("\n".join(train_sequences) + ("\n" if len(train_sequences) > 0 else ""))
    with open(test_txt, "w") as f:
        f.write("\n".join(val_sequences) + ("\n" if len(val_sequences) > 0 else ""))

    print(f"Valid sequences: {len(valid_sequences)}")
    print(f"Train sequences: {len(train_sequences)} -> {train_txt}")
    print(f"Test sequences : {len(val_sequences)} -> {test_txt}")
    print(f"Skipped        : {len(skipped)}")
    if len(skipped) > 0:
        print("First skipped examples:")
        for seq_name, reason in skipped[:10]:
            print(f"  - {seq_name}: {reason}")


if __name__ == "__main__":
    main()
