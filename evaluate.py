#!/usr/bin/env python3
"""Shared Independent Evaluator for mmFormer on MyoPS-380 Test Set.
Protocol: myops380_voxel_v3 (76 test volumes from test_vol.txt)
Empty-Mask Policy:
- Both empty: excluded from patient average
- One empty: Dice=0, IoU=0, HD95/ASD=undefined (excluded from distance average)
Output:
- test_predictions/<CaseID>_pred.npz (uint8 3D arrays)
- test_results/metrics.json
- test_results/per_case.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_myops import MyopsDataset
from metrics.surface_distance import protocol_for_dataset, dataset_rows, summarize_dataset_rows
from networks.mmformer2d import mmFormer2D


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate mmFormer on MyoPS-380 test set.")
    parser.add_argument("--root_path", type=str, required=True, help="Path to Processed_data")
    parser.add_argument("--list_dir", type=str, default="lists/lists_MyoPS380", help="Directory containing test_vol.txt")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing trained run / checkpoints")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint last.pth (defaults to output_dir/checkpoints/last.pth)")
    parser.add_argument("--img_size", type=int, default=128, help="Image resolution")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of canonical classes")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size of slices for volume inference")
    parser.add_argument("--cpu_threads", type=int, default=2, help="CPU threads")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    return parser.parse_args()


@torch.no_grad()
def predict_test_volume(
    model: torch.nn.Module,
    vol_image: torch.Tensor,
    img_size: int = 128,
    batch_size: int = 8,
    device: str = "cuda"
) -> np.ndarray:
    """Predict a 3D volume (3, H, W, D) slice-by-slice along depth D.
    Returns: numpy uint8 array of shape (H, W, D) with canonical labels.
    """
    _, height, width, depth = vol_image.shape
    prediction_3d = np.zeros((height, width, depth), dtype=np.uint8)

    model.eval()
    model.is_training = False

    for start_idx in range(0, depth, batch_size):
        end_idx = min(start_idx + batch_size, depth)
        curr_len = end_idx - start_idx

        # Extract slices: (3, H, W, B) -> (B, 3, H, W)
        slices = vol_image[:, :, :, start_idx:end_idx].permute(3, 0, 1, 2).to(device)

        if slices.shape[-2:] != (img_size, img_size):
            inputs = F.interpolate(slices, size=(img_size, img_size), mode="bilinear", align_corners=False)
        else:
            inputs = slices

        # Forward through mmFormer2D (single-pass, full modalities mask)
        fuse_pred = model(inputs, mask=None)  # (B, num_classes, H, W)

        if fuse_pred.shape[-2:] != (height, width):
            fuse_pred = F.interpolate(fuse_pred.float(), size=(height, width), mode="bilinear", align_corners=False)

        preds = torch.argmax(fuse_pred, dim=1).cpu().numpy().astype(np.uint8)

        for b in range(curr_len):
            prediction_3d[:, :, start_idx + b] = preds[b]

    return prediction_3d


def main():
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)

    results_dir = os.path.join(args.output_dir, "test_results")
    predictions_dir = os.path.join(args.output_dir, "test_predictions")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    log_file = os.path.join(results_dir, "eval.log")
    logging.basicConfig(
        filename=log_file, level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info("==============================================================================")
    logging.info("🔬 EVALUATING MMFORMER ON MYOPS-380 TEST SET (76 VOLUMES)")
    logging.info("Protocol: myops380_voxel_v3 (Unit voxel grid, Empty-Mask Policy)")
    logging.info("==============================================================================")

    # 1. Checkpoint resolution
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = os.path.join(args.output_dir, "checkpoints", "last.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logging.info(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    model = mmFormer2D(num_cls=args.num_classes, num_modals=3)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    # Strip module. prefix if saved from DataParallel
    cleaned_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state)
    model.to(args.device)
    model.eval()
    model.is_training = False

    # 2. Dataset loader for test volumes
    test_dataset = MyopsDataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="test_vol",
        transform=None,
        label_order="legacy"
    )
    logging.info(f"Loaded {len(test_dataset)} test volumes from {args.list_dir}/test_vol.txt")

    all_rows = []
    case_predictions = {}

    start_eval_time = time.time()

    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        case_name = sample["case_name"]
        vol_image = sample["image"]  # (3, H, W, D)
        vol_label = sample["label"].numpy()  # (H, W, D)

        pred_3d = predict_test_volume(
            model=model,
            vol_image=vol_image,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device
        )

        case_predictions[case_name] = pred_3d

        # Save individual prediction npz
        npz_file = os.path.join(predictions_dir, f"{case_name}_pred.npz")
        np.savez_compressed(npz_file, prediction=pred_3d)

        # Compute metric rows for this case
        rows = dataset_rows(pred_3d, vol_label, case_name, dataset_id="myops380")
        all_rows.extend(rows)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(test_dataset):
            logging.info(f"Processed [{idx+1:02d}/{len(test_dataset):02d}] {case_name}")

    eval_duration = time.time() - start_eval_time
    logging.info(f"Inference complete in {eval_duration:.1f}s ({eval_duration/len(test_dataset):.2f}s per volume)")

    # 3. Summarize metrics
    summary = summarize_dataset_rows(all_rows, dataset_id="myops380")

    # Save per_case.csv
    csv_file = os.path.join(results_dir, "per_case.csv")
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_file, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    # Save metrics.json
    metrics_json_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_json_file, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    # 4. Print clean summary table
    print("\n" + "=" * 80)
    print("📊 BENCHMARK METRIC SUMMARY: mmFormer (myops380_train304_test76_last_v1)")
    print("=" * 80)
    print(f"{'Region / Metric':<35} | {'Dice (%)':<15} | {'HD95 (voxels)':<15}")
    print("-" * 80)

    regions_to_print = [
        ("scar", "Scar"),
        ("edema_inclusive", "Edema (Inclusive / AAR)"),
        ("edema", "Edema (Exclusive)"),
        ("myocardial_ring", "Myocardium Ring"),
        ("normal_myocardium", "Normal Myocardium"),
    ]

    for reg_key, display_name in regions_to_print:
        reg_dict = summary.get(reg_key, {})
        dice_val = reg_dict.get("mean_dice", None)
        hd_val = reg_dict.get("mean_hd95_voxel", None)

        dice_str = f"{dice_val * 100:.2f}%" if dice_val is not None else "N/A"
        hd_str = f"{hd_val:.2f}" if hd_val is not None else "N/A"
        print(f"{display_name:<35} | {dice_str:<15} | {hd_str:<15}")

    avg_path = summary.get("avg_pathology_inclusive_dice", None)
    avg_path_str = f"{avg_path * 100:.2f}%" if avg_path is not None else "N/A"
    avg_hd = summary.get("avg_pathology_inclusive_hd95_voxel", None)
    avg_hd_str = f"{avg_hd:.2f}" if avg_hd is not None else "N/A"
    print("-" * 80)
    print(f"{'Avg. Pathology (Inclusive)':<35} | {avg_path_str:<15} | {avg_hd_str:<15}")
    print("=" * 80)
    logging.info(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
