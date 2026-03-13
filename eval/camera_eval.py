#!/usr/bin/env python3
"""
Camera evaluation suite for VGGT vs ViPE pseudo-GT.

Metrics per sequence (for vanilla and finetuned models):
  - ATE RMSE (camera-center, after Sim(3) alignment)
  - RPE rotation (degrees)
  - RPE translation direction (degrees)
  - RPE translation vector error (meters, after Sim(3))
  - Relative focal error (mean of fx/fy relative errors)

Usage example:
  python eval/camera_eval.py \
    --gt-root /mnt/bucket/dawy/vggt_finetune/dataset \
    --vanilla-root /mnt/bucket/dawy/vggt_finetune/eval_outputs/50_epoch/forced_mask_conf \
    --finetuned-root /mnt/bucket/dawy/vggt_finetune/eval_outputs/10_epoch_learned_masking/forced_mask_conf \
    --split-file /mnt/bucket/dawy/vggt_finetune/dataset/splits_gcs/test.txt \
    --output-json /mnt/bucket/dawy/vggt_finetune/eval_outputs/camera_eval_test.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pycolmap


EPS = 1e-12


@dataclass
class CameraFrame:
    image_id: int
    Rcw: np.ndarray  # (3, 3), camera-from-world rotation
    tcw: np.ndarray  # (3,), camera-from-world translation
    center: np.ndarray  # (3,), camera center in world
    fx: float
    fy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate camera metrics against ViPE pseudo-GT.")
    parser.add_argument("--gt-root", type=str, required=True, help="Root containing GT sequences.")
    parser.add_argument("--vanilla-root", type=str, required=True, help="Root containing vanilla outputs.")
    parser.add_argument("--finetuned-root", type=str, required=True, help="Root containing finetuned outputs.")
    parser.add_argument("--split-file", type=str, default=None, help="Optional sequence list file.")
    parser.add_argument(
        "--gt-subdir",
        type=str,
        default="colmap",
        help="GT model subdir under sequence folder. Fallback to sequence root if missing.",
    )
    parser.add_argument(
        "--pred-subdir",
        type=str,
        default="sparse",
        help="Predicted model subdir under sequence folder. Fallback to sequence root if missing.",
    )
    parser.add_argument(
        "--rpe-delta",
        type=int,
        default=1,
        help="Frame gap for RPE (e.g., 1 compares consecutive frames).",
    )
    parser.add_argument(
        "--rpe-min-gt-motion",
        type=float,
        default=0.0,
        help="Ignore RPE translation-direction pairs with GT motion below this threshold (meters).",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save detailed JSON report.")
    return parser.parse_args()


def has_colmap_model_dir(model_dir: Path) -> bool:
    return (model_dir / "cameras.bin").is_file() or (model_dir / "cameras.txt").is_file()


def resolve_model_dir(root: Path, seq: str, preferred_subdir: str) -> Optional[Path]:
    seq_dir = root / seq
    preferred = seq_dir / preferred_subdir
    if has_colmap_model_dir(preferred):
        return preferred
    if has_colmap_model_dir(seq_dir):
        return seq_dir
    return None


def load_reconstruction(model_dir: Path) -> pycolmap.Reconstruction:
    recon = pycolmap.Reconstruction()
    if (model_dir / "cameras.bin").is_file():
        recon.read_binary(str(model_dir))
        return recon
    if (model_dir / "cameras.txt").is_file():
        recon.read_text(str(model_dir))
        return recon
    raise FileNotFoundError(f"No COLMAP model found in {model_dir}")


def camera_fx_fy(camera) -> Tuple[float, float]:
    if hasattr(camera, "calibration_matrix"):
        k = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        if k.shape == (3, 3):
            return float(k[0, 0]), float(k[1, 1])

    params = np.asarray(camera.params, dtype=np.float64)
    model_str = str(getattr(camera, "model_name", getattr(camera, "model", ""))).upper()
    single_focal_models = (
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL",
        "RADIAL_FISHEYE",
    )
    if any(m in model_str for m in single_focal_models):
        f = float(params[0]) if params.size > 0 else float("nan")
        return f, f
    if params.size >= 2:
        return float(params[0]), float(params[1])
    if params.size == 1:
        f = float(params[0])
        return f, f
    return float("nan"), float("nan")


def frames_and_order_from_reconstruction(recon: pycolmap.Reconstruction) -> Tuple[Dict[str, CameraFrame], List[str]]:
    frames: Dict[str, CameraFrame] = {}
    order: List[str] = []
    for image_id in sorted(recon.images.keys()):
        image = recon.images[image_id]
        name = Path(image.name).name
        Rcw = np.asarray(image.cam_from_world.rotation.matrix(), dtype=np.float64)
        tcw = np.asarray(image.cam_from_world.translation, dtype=np.float64)
        center = -Rcw.T @ tcw
        camera = recon.cameras[image.camera_id]
        fx, fy = camera_fx_fy(camera)
        frames[name] = CameraFrame(image_id=image_id, Rcw=Rcw, tcw=tcw, center=center, fx=fx, fy=fy)
        order.append(name)
    return frames, order


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    cos_theta = np.clip((np.trace(R1 @ R2.T) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < EPS or n2 < EPS:
        return None
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns Sim(3) parameters solving dst ~= s * R * src + t
    and aligned source points.
    """
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    var_src = (src_c**2).sum() / n
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


def evaluate_model(
    gt_frames: Dict[str, CameraFrame],
    gt_order: List[str],
    pred_frames: Dict[str, CameraFrame],
    rpe_delta: int,
    rpe_min_gt_motion: float,
) -> Optional[Dict[str, float]]:
    common = [name for name in gt_order if name in pred_frames]
    if len(common) < max(3, rpe_delta + 2):
        return None

    gt_centers = np.stack([gt_frames[name].center for name in common], axis=0)
    pred_centers = np.stack([pred_frames[name].center for name in common], axis=0)
    sim3_scale, sim3_R, sim3_t, pred_centers_aligned = umeyama_sim3(pred_centers, gt_centers)

    ate_rmse = float(np.sqrt(np.mean(np.sum((pred_centers_aligned - gt_centers) ** 2, axis=1))))

    pred_rot_aligned = {
        name: pred_frames[name].Rcw @ sim3_R.T for name in common
    }

    rot_abs_errors = [
        rotation_error_deg(gt_frames[name].Rcw, pred_rot_aligned[name])
        for name in common
    ]

    rpe_rot_errors: List[float] = []
    rpe_tdir_errors: List[float] = []
    rpe_t_l2_errors: List[float] = []
    rpe_pairs_used = 0
    for i in range(len(common) - rpe_delta):
        name_i = common[i]
        name_j = common[i + rpe_delta]

        R_rel_gt = gt_frames[name_j].Rcw @ gt_frames[name_i].Rcw.T
        R_rel_pr = pred_rot_aligned[name_j] @ pred_rot_aligned[name_i].T
        rpe_rot_errors.append(rotation_error_deg(R_rel_gt, R_rel_pr))

        t_gt = gt_frames[name_j].center - gt_frames[name_i].center
        t_pr = pred_centers_aligned[i + rpe_delta] - pred_centers_aligned[i]
        if np.linalg.norm(t_gt) < rpe_min_gt_motion:
            continue
        tdir_err = angle_between_deg(t_gt, t_pr)
        if tdir_err is not None:
            rpe_tdir_errors.append(tdir_err)
            rpe_t_l2_errors.append(float(np.linalg.norm(t_pr - t_gt)))
            rpe_pairs_used += 1

    fx_rel_errors: List[float] = []
    fy_rel_errors: List[float] = []
    for name in common:
        gt_fx = gt_frames[name].fx
        gt_fy = gt_frames[name].fy
        pr_fx = pred_frames[name].fx
        pr_fy = pred_frames[name].fy
        if np.isfinite(gt_fx) and np.isfinite(pr_fx) and abs(gt_fx) > EPS:
            fx_rel_errors.append(abs(pr_fx - gt_fx) / abs(gt_fx))
        if np.isfinite(gt_fy) and np.isfinite(pr_fy) and abs(gt_fy) > EPS:
            fy_rel_errors.append(abs(pr_fy - gt_fy) / abs(gt_fy))

    focal_rel = np.nan
    focal_vals = fx_rel_errors + fy_rel_errors
    if focal_vals:
        focal_rel = float(np.mean(focal_vals))

    # --- Temporal consistency metrics ---
    # pose_accel_mean: mean 3D acceleration of camera centers (jitter measure)
    # pose_vel_std: std of frame-to-frame speed (consistency measure)
    pose_accel_mean = np.nan
    pose_vel_std = np.nan
    if len(pred_centers_aligned) >= 3:
        vels = np.linalg.norm(np.diff(pred_centers_aligned, axis=0), axis=1)  # (N-1,)
        pose_vel_std = float(np.std(vels))
        accels = pred_centers_aligned[2:] - 2 * pred_centers_aligned[1:-1] + pred_centers_aligned[:-2]
        pose_accel_mean = float(np.mean(np.linalg.norm(accels, axis=1)))

    out = {
        "num_common_frames": int(len(common)),
        "ate_rmse": ate_rmse,
        "abs_rot_deg_mean": float(np.mean(rot_abs_errors)) if rot_abs_errors else np.nan,
        "rpe_rot_deg_mean": float(np.mean(rpe_rot_errors)) if rpe_rot_errors else np.nan,
        "rpe_tdir_deg_mean": float(np.mean(rpe_tdir_errors)) if rpe_tdir_errors else np.nan,
        "rpe_t_l2_m_mean": float(np.mean(rpe_t_l2_errors)) if rpe_t_l2_errors else np.nan,
        "rpe_pairs_used": int(rpe_pairs_used),
        "focal_rel_mean": focal_rel,
        "pose_accel_mean": pose_accel_mean,
        "pose_vel_std": pose_vel_std,
        "sim3_scale": float(sim3_scale),
        "sim3_rotation_det": float(np.linalg.det(sim3_R)),
        "sim3_translation_norm": float(np.linalg.norm(sim3_t)),
    }
    return out


def read_split_sequences(split_file: Path) -> List[str]:
    seqs: List[str] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        seqs.append(line)
    return seqs


def discover_sequences(gt_root: Path, vanilla_root: Path, finetuned_root: Path) -> List[str]:
    gt = {p.name for p in gt_root.iterdir() if p.is_dir()}
    va = {p.name for p in vanilla_root.iterdir() if p.is_dir()}
    ft = {p.name for p in finetuned_root.iterdir() if p.is_dir()}
    return sorted(gt & va & ft)


def metric_mean(rows: Iterable[Dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def print_summary(per_seq_rows: List[Dict]) -> None:
    headers = [
        ("Sequence", 34),
        ("N", 4),
        ("V_ATE", 9),
        ("V_RPEr", 9),
        ("V_RPEt°", 9),
        ("V_RPEtm", 9),
        ("V_frel", 9),
        ("V_Accel", 9),
        ("V_VelStd", 9),
        ("F_ATE", 9),
        ("F_RPEr", 9),
        ("F_RPEt°", 9),
        ("F_RPEtm", 9),
        ("F_frel", 9),
        ("F_Accel", 9),
        ("F_VelStd", 9),
    ]
    line = " ".join([f"{h:>{w}}" for h, w in headers])
    print(line)
    print("-" * len(line))

    for row in per_seq_rows:
        v = row.get("vanilla")
        f = row.get("finetuned")
        n = 0
        if v is not None and "num_common_frames" in v:
            n = v["num_common_frames"]
        elif f is not None and "num_common_frames" in f:
            n = f["num_common_frames"]

        def fv(d: Optional[Dict], key: str) -> str:
            if d is None or key not in d or not np.isfinite(d[key]):
                return "nan"
            return f"{d[key]:.4f}"

        parts = [
            f"{row['sequence']:<34}",
            f"{n:>4}",
            f"{fv(v, 'ate_rmse'):>9}",
            f"{fv(v, 'rpe_rot_deg_mean'):>9}",
            f"{fv(v, 'rpe_tdir_deg_mean'):>9}",
            f"{fv(v, 'rpe_t_l2_m_mean'):>9}",
            f"{fv(v, 'focal_rel_mean'):>9}",
            f"{fv(v, 'pose_accel_mean'):>9}",
            f"{fv(v, 'pose_vel_std'):>9}",
            f"{fv(f, 'ate_rmse'):>9}",
            f"{fv(f, 'rpe_rot_deg_mean'):>9}",
            f"{fv(f, 'rpe_tdir_deg_mean'):>9}",
            f"{fv(f, 'rpe_t_l2_m_mean'):>9}",
            f"{fv(f, 'focal_rel_mean'):>9}",
            f"{fv(f, 'pose_accel_mean'):>9}",
            f"{fv(f, 'pose_vel_std'):>9}",
        ]
        print(" ".join(parts))

    vanilla_rows = [r["vanilla"] for r in per_seq_rows if r.get("vanilla") is not None]
    ft_rows = [r["finetuned"] for r in per_seq_rows if r.get("finetuned") is not None]

    print("-" * len(line))
    print("Means:")
    print(
        "  vanilla  "
        f"ATE={metric_mean(vanilla_rows, 'ate_rmse'):.4f}, "
        f"RPEr={metric_mean(vanilla_rows, 'rpe_rot_deg_mean'):.4f}, "
        f"RPEt°={metric_mean(vanilla_rows, 'rpe_tdir_deg_mean'):.4f}, "
        f"RPEtm={metric_mean(vanilla_rows, 'rpe_t_l2_m_mean'):.4f}, "
        f"frel={metric_mean(vanilla_rows, 'focal_rel_mean'):.4f}, "
        f"accel={metric_mean(vanilla_rows, 'pose_accel_mean'):.4f}, "
        f"vel_std={metric_mean(vanilla_rows, 'pose_vel_std'):.4f}"
    )
    print(
        "  finetuned"
        f" ATE={metric_mean(ft_rows, 'ate_rmse'):.4f}, "
        f"RPEr={metric_mean(ft_rows, 'rpe_rot_deg_mean'):.4f}, "
        f"RPEt°={metric_mean(ft_rows, 'rpe_tdir_deg_mean'):.4f}, "
        f"RPEtm={metric_mean(ft_rows, 'rpe_t_l2_m_mean'):.4f}, "
        f"frel={metric_mean(ft_rows, 'focal_rel_mean'):.4f}, "
        f"accel={metric_mean(ft_rows, 'pose_accel_mean'):.4f}, "
        f"vel_std={metric_mean(ft_rows, 'pose_vel_std'):.4f}"
    )

    win_metrics = [
        "ate_rmse",
        "rpe_rot_deg_mean",
        "rpe_tdir_deg_mean",
        "rpe_t_l2_m_mean",
        "focal_rel_mean",
        "pose_accel_mean",
        "pose_vel_std",
    ]
    wins = {m: 0 for m in win_metrics}
    ties = {m: 0 for m in win_metrics}
    total = {m: 0 for m in win_metrics}
    for row in per_seq_rows:
        v = row.get("vanilla")
        f = row.get("finetuned")
        if v is None or f is None:
            continue
        for m in win_metrics:
            if m not in v or m not in f:
                continue
            if not np.isfinite(v[m]) or not np.isfinite(f[m]):
                continue
            total[m] += 1
            if abs(f[m] - v[m]) < 1e-12:
                ties[m] += 1
            elif f[m] < v[m]:
                wins[m] += 1

    print("Win-rate (finetuned better, lower is better):")
    for m in win_metrics:
        if total[m] == 0:
            print(f"  {m}: n=0")
            continue
        win_rate = wins[m] / total[m]
        print(f"  {m}: {wins[m]}/{total[m]} (ties={ties[m]}) => {win_rate:.3f}")


def main() -> None:
    args = parse_args()
    gt_root = Path(args.gt_root)
    vanilla_root = Path(args.vanilla_root)
    finetuned_root = Path(args.finetuned_root)

    if args.split_file:
        sequences = read_split_sequences(Path(args.split_file))
    else:
        sequences = discover_sequences(gt_root, vanilla_root, finetuned_root)

    per_seq_rows: List[Dict] = []

    for seq in sequences:
        row: Dict = {"sequence": seq, "vanilla": None, "finetuned": None}

        gt_model_dir = resolve_model_dir(gt_root, seq, args.gt_subdir)
        if gt_model_dir is None:
            row["error"] = "missing_gt_model"
            per_seq_rows.append(row)
            continue

        try:
            gt_recon = load_reconstruction(gt_model_dir)
            gt_frames, gt_order = frames_and_order_from_reconstruction(gt_recon)
        except Exception as exc:
            row["error"] = f"gt_load_error: {exc}"
            per_seq_rows.append(row)
            continue

        for tag, pred_root in (("vanilla", vanilla_root), ("finetuned", finetuned_root)):
            pred_model_dir = resolve_model_dir(pred_root, seq, args.pred_subdir)
            if pred_model_dir is None:
                row[tag] = None
                continue
            try:
                pred_recon = load_reconstruction(pred_model_dir)
                pred_frames, _ = frames_and_order_from_reconstruction(pred_recon)
                row[tag] = evaluate_model(
                    gt_frames=gt_frames,
                    gt_order=gt_order,
                    pred_frames=pred_frames,
                    rpe_delta=args.rpe_delta,
                    rpe_min_gt_motion=args.rpe_min_gt_motion,
                )
            except Exception as exc:
                row[tag] = {"error": f"{exc}"}
        per_seq_rows.append(row)

    print_summary(per_seq_rows)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "config": {
                "gt_root": str(gt_root),
                "vanilla_root": str(vanilla_root),
                "finetuned_root": str(finetuned_root),
                "split_file": args.split_file,
                "gt_subdir": args.gt_subdir,
                "pred_subdir": args.pred_subdir,
                "rpe_delta": args.rpe_delta,
                "rpe_min_gt_motion": args.rpe_min_gt_motion,
            },
            "results": per_seq_rows,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Saved JSON report to: {out_path}")


if __name__ == "__main__":
    main()
