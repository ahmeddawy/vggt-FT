#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import re
import tempfile
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import Imath
import numpy as np
import OpenEXR


FRAME_RE = re.compile(r"frame_(\d+)")


@dataclass
class SequenceResult:
    seq_name: str
    ok: bool
    reason: str


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


def ensure_colmap_layout(seq_dir: str, use_symlink: bool) -> Tuple[bool, str]:
    colmap_dir = osp.join(seq_dir, "colmap")
    os.makedirs(colmap_dir, exist_ok=True)

    for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
        dst = osp.join(colmap_dir, fname)
        if osp.isfile(dst):
            continue

        src = osp.join(seq_dir, fname)
        if not osp.isfile(src):
            # points3D.txt may be missing in a few sequences; cameras/images are mandatory.
            if fname in ["cameras.txt", "images.txt"]:
                return False, f"missing source file: {src}"
            continue

        if use_symlink:
            src_abs = osp.abspath(src)
            if osp.lexists(dst):
                os.remove(dst)
            os.symlink(src_abs, dst)
        else:
            with open(src, "rb") as fin, open(dst, "wb") as fout:
                fout.write(fin.read())

    return True, "ok"


def resolve_depth_zip(seq_name: str, seq_dir: str) -> Optional[str]:
    depth_dir = osp.join(seq_dir, "depth")
    if not osp.isdir(depth_dir):
        return None

    preferred = [
        osp.join(depth_dir, f"{seq_name}.zip"),
        osp.join(seq_dir, f"{seq_name}.zip"),
    ]
    for p in preferred:
        if osp.isfile(p):
            return p

    zip_files = sorted(
        osp.join(depth_dir, f) for f in os.listdir(depth_dir) if f.lower().endswith(".zip")
    )
    if len(zip_files) > 0:
        return zip_files[0]
    return None


def read_exr_depth_from_zip(zf: zipfile.ZipFile, member_name: str) -> np.ndarray:
    # Some OpenEXR builds cannot consume zip stream objects directly.
    # Read bytes from zip and decode via a temporary EXR file path.
    with zf.open(member_name) as f:
        exr_bytes = f.read()

    with tempfile.NamedTemporaryFile(suffix=".exr") as tmp:
        tmp.write(exr_bytes)
        tmp.flush()

        exr = OpenEXR.InputFile(tmp.name)
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        z_raw = exr.channel("Z", Imath.PixelType(Imath.PixelType.HALF))
        exr.close()

    depth = np.frombuffer(z_raw, dtype=np.float16).reshape((height, width))
    return depth.astype(np.float32, copy=False)


def convert_depth_zip_to_npy(
    zip_path: str,
    out_npy_path: str,
    out_dtype: str,
) -> Tuple[bool, str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".exr")]
        if len(members) == 0:
            return False, f"no .exr files found in {zip_path}"

        parsed: List[Tuple[int, str]] = []
        for m in members:
            stem = osp.splitext(osp.basename(m))[0]
            try:
                frame_idx = int(stem)
            except ValueError:
                continue
            parsed.append((frame_idx, m))
        if len(parsed) == 0:
            return False, f"no numeric frame names found in {zip_path}"

        parsed.sort(key=lambda x: x[0])
        max_idx = parsed[-1][0]

        # Probe shape from first readable frame.
        first_depth = None
        for _, member_name in parsed:
            try:
                first_depth = read_exr_depth_from_zip(zf, member_name)
                break
            except Exception:
                continue
        if first_depth is None:
            return False, f"failed to decode any EXR in {zip_path}"

        h, w = first_depth.shape
        dtype = np.float16 if out_dtype == "float16" else np.float32

        depth_mm = np.lib.format.open_memmap(
            out_npy_path,
            mode="w+",
            dtype=dtype,
            shape=(max_idx + 1, h, w),
        )
        depth_mm[:] = np.nan

        for frame_idx, member_name in parsed:
            try:
                d = read_exr_depth_from_zip(zf, member_name)
            except Exception:
                continue
            if d.shape != (h, w):
                continue
            depth_mm[frame_idx] = d.astype(dtype, copy=False)

        del depth_mm
    return True, "ok"


def infer_num_image_frames(seq_dir: str) -> Optional[int]:
    images_dir = osp.join(seq_dir, "images")
    if not osp.isdir(images_dir):
        return None
    max_idx = -1
    for fname in os.listdir(images_dir):
        match = FRAME_RE.search(fname)
        if match is None:
            continue
        idx = int(match.group(1))
        max_idx = max(max_idx, idx)
    if max_idx < 0:
        return None
    return max_idx + 1


def process_sequence(seq_dir: str, out_dtype: str, overwrite_depth: bool, use_symlink: bool) -> SequenceResult:
    seq_name = osp.basename(seq_dir.rstrip("/"))

    images_dir = osp.join(seq_dir, "images")
    depth_dir = osp.join(seq_dir, "depth")
    if not osp.isdir(images_dir):
        return SequenceResult(seq_name, False, "missing images/")
    if not osp.isdir(depth_dir):
        return SequenceResult(seq_name, False, "missing depth/")

    ok, reason = ensure_colmap_layout(seq_dir, use_symlink=use_symlink)
    if not ok:
        return SequenceResult(seq_name, False, reason)

    out_npy = osp.join(depth_dir, "depths.npy")
    if osp.isfile(out_npy) and not overwrite_depth:
        return SequenceResult(seq_name, True, "already prepared")

    depth_zip = resolve_depth_zip(seq_name, seq_dir)
    if depth_zip is None:
        return SequenceResult(seq_name, False, "no depth zip found")

    ok, reason = convert_depth_zip_to_npy(depth_zip, out_npy, out_dtype=out_dtype)
    if not ok:
        return SequenceResult(seq_name, False, reason)

    # Soft validation.
    colmap_images_txt = osp.join(seq_dir, "colmap", "images.txt")
    if osp.isfile(colmap_images_txt):
        colmap_count = count_colmap_images(colmap_images_txt)
        depth_shape = np.load(out_npy, mmap_mode="r").shape
        if colmap_count > depth_shape[0]:
            return SequenceResult(
                seq_name,
                False,
                f"depth len {depth_shape[0]} < COLMAP images {colmap_count}",
            )

    img_count = infer_num_image_frames(seq_dir)
    if img_count is not None:
        depth_shape = np.load(out_npy, mmap_mode="r").shape
        if img_count > depth_shape[0]:
            return SequenceResult(
                seq_name,
                False,
                f"depth len {depth_shape[0]} < image frames {img_count}",
            )

    return SequenceResult(seq_name, True, "prepared")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare sequence folders for VGGT GCSColmapDataset: ensure colmap/ layout and depths.npy."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Dataset root with per-sequence folders.",
    )
    parser.add_argument(
        "--depth-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="dtype for saved depth/depths.npy.",
    )
    parser.add_argument(
        "--overwrite-depth",
        action="store_true",
        help="Recreate depth/depths.npy even if it already exists.",
    )
    parser.add_argument(
        "--copy-colmap-files",
        action="store_true",
        help="Copy root cameras/images/points files instead of symlinking into colmap/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    if not osp.isdir(dataset_dir):
        raise ValueError(f"dataset dir does not exist: {dataset_dir}")

    # Accept either:
    # 1) dataset root containing many sequence folders, or
    # 2) a single sequence folder.
    if osp.isdir(osp.join(dataset_dir, "images")) and osp.isdir(osp.join(dataset_dir, "depth")):
        seq_dirs = [dataset_dir]
    else:
        seq_dirs = [
            osp.join(dataset_dir, d)
            for d in sorted(os.listdir(dataset_dir))
            if osp.isdir(osp.join(dataset_dir, d))
        ]
    if len(seq_dirs) == 0:
        raise RuntimeError(f"no sequence folders found under {dataset_dir}")

    results: List[SequenceResult] = []
    for seq_dir in seq_dirs:
        res = process_sequence(
            seq_dir=seq_dir,
            out_dtype=args.depth_dtype,
            overwrite_depth=args.overwrite_depth,
            use_symlink=not args.copy_colmap_files,
        )
        results.append(res)
        status = "OK" if res.ok else "FAIL"
        print(f"[{status}] {res.seq_name}: {res.reason}")

    ok_count = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count
    print(f"\nDone. total={len(results)} ok={ok_count} fail={fail_count}")

    if fail_count > 0:
        print("\nFailed sequences:")
        for r in results:
            if not r.ok:
                print(f"  - {r.seq_name}: {r.reason}")


if __name__ == "__main__":
    main()
