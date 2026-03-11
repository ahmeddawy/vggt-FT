#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an aggregator-only checkpoint for head-randomization experiments."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to source checkpoint (full VGGT checkpoint).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write aggregator-only checkpoint.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="aggregator.",
        help="State-dict key prefix to keep (default: aggregator.).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(str(input_path), map_location="cpu")
    model_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    if not isinstance(model_state, dict):
        raise ValueError(f"Unexpected checkpoint format in {input_path}.")

    kept_state = {k: v for k, v in model_state.items() if k.startswith(args.prefix)}
    if not kept_state:
        raise ValueError(
            f"No parameters matched prefix '{args.prefix}' in checkpoint: {input_path}"
        )

    output_checkpoint = {
        "model": kept_state,
        "meta": {
            "source_checkpoint": str(input_path),
            "kept_prefix": args.prefix,
            "num_total_params": len(model_state),
            "num_kept_params": len(kept_state),
        },
    }
    torch.save(output_checkpoint, str(output_path))

    print(f"Source checkpoint: {input_path}")
    print(f"Output checkpoint: {output_path}")
    print(f"Kept tensors: {len(kept_state)} / {len(model_state)}")


if __name__ == "__main__":
    main()
