#!/usr/bin/env python3
"""Entry point for training mmFormer on MyoPS-380 (Protocol: myops380_train304_test76_last_v1)."""
import argparse
import os
import sys

import torch
from networks.mmformer2d import mmFormer2D
from trainer import trainer_myops


def parse_args():
    parser = argparse.ArgumentParser(description="Train mmFormer on MyoPS-380 benchmark.")
    parser.add_argument("--dataset", type=str, default="MyoPS380", help="Dataset name")
    parser.add_argument("--root_path", type=str, required=True, help="Path to Processed_data")
    parser.add_argument("--list_dir", type=str, default="lists/lists_MyoPS380", help="Path to splits list directory")
    parser.add_argument("--output", type=str, default="outputs/runs", help="Output directory")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--max_epochs", type=int, default=300, help="Total training epochs budget")
    parser.add_argument("--batch_size", type=int, default=8, help="Microbatch size")
    parser.add_argument("--effective_batch_size", type=int, default=16, help="Target effective batch size")
    parser.add_argument("--base_lr", type=float, default=0.001, help="Base learning rate (MedIA paper: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--img_size", type=int, default=128, help="Image resolution")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of canonical classes")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--cpu_threads", type=int, default=2, help="PyTorch CPU threads")
    return parser.parse_args()


def main():
    args = parse_args()

    run_name = f"mmFormer_{args.dataset}_Seed_{args.seed}"
    snapshot_path = os.path.join(args.output, run_name)
    os.makedirs(snapshot_path, exist_ok=True)

    model = mmFormer2D(num_cls=args.num_classes, num_modals=3).cuda()

    trainer_myops(args, model, snapshot_path)


if __name__ == "__main__":
    main()
