#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import json
import os
import os.path as osp
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image


SEQ_SKIP_NAMES = {"reports", "splits"}
MASK_NAME_PATTERN = re.compile(r"^(\d+)\.png$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create masked depth maps for each sequence by applying per-frame segmentation masks. "
            "Masked pixels are set to 0.0 depth (ignored by training)."
        )
    )
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root containing sequence dirs.")
    parser.add_argument("--mask-root", type=str, required=True, help="Path to mask root containing <seq>.zip and <seq>.txt.")
    parser.add_argument(
        "--keep-classes",
        type=str,
        nargs="+",
        default=["background"],
        help="Class names to keep as valid depth. Everything else will be masked to 0.",
    )
    parser.add_argument(
        "--depth-filename",
        type=str,
        default="depths.npy",
        help="Depth filename under each sequence depth/ directory.",
    )
    parser.add_argument(
        "--output-dir-name",
        type=str,
        default="masked_depth",
        help="Output folder name under each sequence where masked depth will be saved.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="depths.npy",
        help="Output file name inside <seq>/<output-dir-name>/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional path to save per-sequence processing report as JSON.",
    )
    return parser.parse_args()


def read_class_map(path: Path) -> Dict[int, str]:
    id_to_name: Dict[int, str] = {}
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\s*:\s*(.+)$", line)
        if match is None:
            continue
        cls_id = int(match.group(1))
        cls_name = match.group(2).strip()
        id_to_name[cls_id] = cls_name
    return id_to_name


def read_mask_index(zip_path: Path) -> Dict[int, str]:
    frame_to_member: Dict[int, str] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            base = osp.basename(member)
            match = MASK_NAME_PATTERN.match(base)
            if match is None:
                continue
            frame_idx = int(match.group(1))
            frame_to_member[frame_idx] = member
    return frame_to_member


def resolve_sequences(dataset_root: Path) -> List[Path]:
    seq_dirs = []
    for p in sorted(dataset_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name in SEQ_SKIP_NAMES:
            continue
        seq_dirs.append(p)
    return seq_dirs


def get_keep_ids(id_to_name: Dict[int, str], keep_classes: Set[str]) -> Set[int]:
    keep_ids = set()
    for cls_id, cls_name in id_to_name.items():
        if cls_name.lower() in keep_classes:
            keep_ids.add(cls_id)
    return keep_ids


def process_sequence(
    seq_dir: Path,
    mask_root: Path,
    keep_classes: Set[str],
    depth_filename: str,
    output_dir_name: str,
    output_filename: str,
    overwrite: bool,
) -> Tuple[bool, str, Dict]:
    seq_name = seq_dir.name
    depth_path = seq_dir / "depth" / depth_filename
    mask_zip = mask_root / f"{seq_name}.zip"
    mask_txt = mask_root / f"{seq_name}.txt"
    out_dir = seq_dir / output_dir_name
    out_path = out_dir / output_filename

    meta = {
        "seq_name": seq_name,
        "depth_path": str(depth_path),
        "mask_zip": str(mask_zip),
        "mask_txt": str(mask_txt),
        "output_path": str(out_path),
    }

    if not depth_path.exists():
        return False, "missing depth file", meta
    if not mask_zip.exists():
        return False, "missing mask zip", meta
    if not mask_txt.exists():
        return False, "missing mask txt", meta
    if out_path.exists() and not overwrite:
        meta["status"] = "skipped_exists"
        return True, "exists", meta

    depth_arr = np.load(depth_path, mmap_mode="r")
    if depth_arr.ndim != 3:
        return False, f"depth must be [N,H,W], got {depth_arr.shape}", meta

    num_frames, depth_h, depth_w = depth_arr.shape
    id_to_name = read_class_map(mask_txt)
    keep_ids = get_keep_ids(id_to_name, keep_classes)

    meta["depth_shape"] = [int(num_frames), int(depth_h), int(depth_w)]
    meta["num_class_ids"] = int(len(id_to_name))
    meta["keep_ids"] = sorted(int(x) for x in keep_ids)

    if len(keep_ids) == 0:
        return False, "no keep class IDs found in mask txt", meta

    frame_to_member = read_mask_index(mask_zip)
    if len(frame_to_member) != num_frames:
        return False, f"frame count mismatch depth={num_frames} mask={len(frame_to_member)}", meta

    missing_frame_ids = [i for i in range(num_frames) if i not in frame_to_member]
    if missing_frame_ids:
        return False, f"mask zip missing frame ids, first={missing_frame_ids[:5]}", meta

    out_dir.mkdir(parents=True, exist_ok=True)
    out_arr = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=depth_arr.dtype,
        shape=depth_arr.shape,
    )

    total_masked_pixels = 0
    total_pixels = int(num_frames * depth_h * depth_w)
    problematic_shapes = 0

    with zipfile.ZipFile(mask_zip, "r") as zf:
        for frame_idx in range(num_frames):
            member = frame_to_member[frame_idx]
            mask_img = np.array(Image.open(io.BytesIO(zf.read(member))))
            if mask_img.ndim != 2:
                return False, f"mask is not single-channel at frame {frame_idx}", meta
            if mask_img.shape != (depth_h, depth_w):
                problematic_shapes += 1
                return False, (
                    f"mask/depth shape mismatch at frame {frame_idx}: "
                    f"mask={mask_img.shape} depth={(depth_h, depth_w)}"
                ), meta

            keep_mask = np.isin(mask_img, list(keep_ids))
            frame_depth = np.array(depth_arr[frame_idx], copy=True)
            frame_depth[~keep_mask] = 0.0
            out_arr[frame_idx] = frame_depth
            total_masked_pixels += int((~keep_mask).sum())

    del out_arr

    meta["total_pixels"] = total_pixels
    meta["total_masked_pixels"] = total_masked_pixels
    meta["masked_ratio"] = float(total_masked_pixels / max(total_pixels, 1))
    meta["problematic_shapes"] = problematic_shapes
    meta["status"] = "ok"
    return True, "ok", meta


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    mask_root = Path(args.mask_root)
    keep_classes = {x.strip().lower() for x in args.keep_classes if x.strip()}

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset-root does not exist: {dataset_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"mask-root does not exist: {mask_root}")
    if not keep_classes:
        raise ValueError("keep-classes is empty.")

    seq_dirs = resolve_sequences(dataset_root)

    report = {
        "dataset_root": str(dataset_root),
        "mask_root": str(mask_root),
        "keep_classes": sorted(keep_classes),
        "output_dir_name": args.output_dir_name,
        "output_filename": args.output_filename,
        "total_sequences_seen": len(seq_dirs),
        "ok": 0,
        "failed": 0,
        "skipped_exists": 0,
        "rows": [],
    }

    for seq_dir in seq_dirs:
        ok, reason, meta = process_sequence(
            seq_dir=seq_dir,
            mask_root=mask_root,
            keep_classes=keep_classes,
            depth_filename=args.depth_filename,
            output_dir_name=args.output_dir_name,
            output_filename=args.output_filename,
            overwrite=args.overwrite,
        )
        meta["reason"] = reason
        report["rows"].append(meta)
        if ok:
            if reason == "exists":
                report["skipped_exists"] += 1
            else:
                report["ok"] += 1
        else:
            report["failed"] += 1

    print(
        json.dumps(
            {
                "total_sequences_seen": report["total_sequences_seen"],
                "ok": report["ok"],
                "failed": report["failed"],
                "skipped_exists": report["skipped_exists"],
                "keep_classes": report["keep_classes"],
            },
            indent=2,
        )
    )

    if args.report_json is not None:
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()

