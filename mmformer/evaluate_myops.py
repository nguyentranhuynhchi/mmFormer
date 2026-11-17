"""Evaluation script for mmFormer on MyoPS380 and MyoPS++ test sets.

Protocols:
  - MyoPS380: myops380_voxel_v3 (Shared Evaluator, 76 Cases, Voxel units)
  - MyoPS++:  myopspp_unet2d_roi128_1mm_predicted_roi_mm_v1 (16 Cases, Physical mm units)
Outputs:
  1) <run_dir>/test_predictions/<CaseID>_pred.npz (Canonical label format, label 3 is Scar)
  2) <output_dir>/per_case.csv (Full per-case, per-region breakdown)
  3) <output_dir>/metrics.json (Official benchmark summary matching SCAR schema)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

# Ensure repo root is at the head of sys.path and remove pkg_dir to prevent module shadowing
repo_root = Path(__file__).resolve().parent.parent
pkg_dir = str(Path(__file__).resolve().parent)
while pkg_dir in sys.path:
    sys.path.remove(pkg_dir)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
from scipy.ndimage import zoom
import torch

from mmformer.mmformer import Model
from mmformer.data.dataset_myops import (
    MyopsVolumeDataset,
    canonicalize_label,
    normalize_dataset_name,
    resolve_default_split_dir,
)
from mmformer.utils.metrics import (
    BENCHMARK_PROTOCOLS,
    BENCHMARK_PROTOCOL,
    dataset_rows,
    summarize_dataset_rows,
)

logger = logging.getLogger("mmFormer_Eval")


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def json_safe(val: Any) -> Any:
    if isinstance(val, dict):
        return {str(k): json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [json_safe(v) for v in val]
    if isinstance(val, (float, np.floating)):
        return float(val) if math.isfinite(val) else None
    if isinstance(val, np.generic):
        return val.item()
    return val


def format_master_clinical_table(summary: dict[str, Any], dataset: str = "myops380") -> str:
    """Format Master Benchmark clinical table matching specification."""
    norm_ds = normalize_dataset_name(dataset)
    proto = BENCHMARK_PROTOCOLS.get(norm_ds, BENCHMARK_PROTOCOLS["myops380"])
    dist_unit = proto["distance_unit"]

    regions = [
        "normal_myocardium",
        "edema",
        "scar",
        "edema_inclusive",
        "myocardial_ring",
    ]
    hd_label = f"HD95 ({dist_unit})" if dist_unit == "mm" else "HD95 (vox)"
    asd_label = f"ASD ({dist_unit})" if dist_unit == "mm" else "ASD (vox)"

    lines = [
        "================================================================================",
        f"MASTER BENCHMARK CLINICAL EVALUATION RESULTS (Protocol: {proto['id']})",
        "--------------------------------------------------------------------------------",
        f"{'Region':<22} | {'Dice Mean':<10} | {'IoU Mean':<10} | {hd_label:<12} | {asd_label:<10}",
        "--------------------------------------------------------------------------------",
    ]
    for r_name in regions:
        reg = summary.get(r_name, {})
        d = reg.get("mean_dice")
        i = reg.get("mean_iou")
        h = reg.get(f"mean_hd95_{dist_unit}") if reg.get(f"mean_hd95_{dist_unit}") is not None else reg.get("mean_hd95_voxel")
        a = reg.get(f"mean_asd_{dist_unit}") if reg.get(f"mean_asd_{dist_unit}") is not None else reg.get("mean_asd_voxel")
        d_str = f"{d:.4f}" if d is not None else "N/A"
        i_str = f"{i:.4f}" if i is not None else "N/A"
        h_str = f"{h:.2f}" if h is not None else "N/A"
        a_str = f"{a:.2f}" if a is not None else "N/A"
        lines.append(f"{r_name:<22} | {d_str:<10} | {i_str:<10} | {h_str:<12} | {a_str:<10}")
    lines.append("--------------------------------------------------------------------------------")
    apd = summary.get("avg_pathology_dice")
    apid = summary.get("avg_pathology_inclusive_dice")
    apd_str = f"{apd:.4f}" if apd is not None else "N/A"
    apid_str = f"{apid:.4f}" if apid is not None else "N/A"
    lines.append(f"{'avg_pathology_dice':<22} | {apd_str:<10} | {'':<10} | {'':<12} | {'':<10}")
    lines.append(f"{'avg_pathology_inclusive_dice':<22} | {apid_str:<10} | {'':<10} | {'':<12} | {'':<10}")
    lines.append("================================================================================")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate mmFormer on MyoPS380 and MyoPS++ test cases")
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        dest="checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g. outputs/runs/mmformer_seed1234/checkpoints/mmformer_seed1234.pt)",
    )
    parser.add_argument(
        "--dataset",
        "--dataset_id",
        "--dataset-id",
        dest="dataset",
        type=str,
        default="myops380",
        choices=["myops380", "myopspp", "myopspp_unet2d_roi128_1mm", "myops_380"],
        help="Target dataset benchmark ('myops380' or 'myopspp')",
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        type=str,
        default=os.environ.get("LOCAL_DATASET", "E:/STUDY/DATASET/MyoPS380/Processed_data"),
        help="Path to preprocessed dataset root containing bSSFP, LGE, T2w",
    )
    parser.add_argument(
        "--list-dir",
        "--list_dir",
        "--split-dir",
        "--split_dir",
        dest="list_dir",
        type=str,
        default=None,
        help="Path to split manifests folder containing test_vol.txt (defaults dynamically by dataset)",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports and prediction files",
    )
    parser.add_argument(
        "--save-predictions",
        "--save_predictions",
        dest="save_predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save test_predictions/<CaseID>_pred.npz files (default: True)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cuda', 'cpu')")
    parser.add_argument(
        "--measure-efficiency",
        "--measure_efficiency",
        dest="measure_efficiency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure Complexity and Inference efficiency metrics (default: True)",
    )
    parser.add_argument(
        "--no_measure_efficiency",
        dest="measure_efficiency",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--efficiency-output",
        "--efficiency_output",
        dest="efficiency_output",
        type=str,
        default=None,
        help="Optional destination path for efficiency_metrics.json",
    )
    return parser


def parse_args(args: Any = None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def predict_case_volume(
    model: torch.nn.Module,
    img_bssfp: np.ndarray,
    img_lge: np.ndarray,
    img_t2w: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predict a 3D volume by running 2D slice inference and stacking along depth (H, W, D)."""
    h, w, d = img_bssfp.shape
    prediction_3d = np.zeros((h, w, d), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for slice_idx in range(d):
            s0 = img_bssfp[:, :, slice_idx]
            s1 = img_lge[:, :, slice_idx]
            s2 = img_t2w[:, :, slice_idx]

            # Zoom to 128x128 if needed
            if h != 128 or w != 128:
                s0 = zoom(s0, (128 / h, 128 / w), order=3)
                s1 = zoom(s1, (128 / h, 128 / w), order=3)
                s2 = zoom(s2, (128 / h, 128 / w), order=3)

            stacked = np.stack([s0, s1, s2], axis=0)  # (3, 128, 128)
            tensor = torch.from_numpy(stacked).unsqueeze(0).float().to(device)  # (1, 3, 128, 128)

            fuse_pred = model(tensor)  # (1, 4, 128, 128) probability map
            pred_slice = torch.argmax(fuse_pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            if h != 128 or w != 128:
                pred_slice = zoom(pred_slice, (h / 128, w / 128), order=0)

            prediction_3d[:, :, slice_idx] = pred_slice

    return prediction_3d


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    norm_ds = normalize_dataset_name(args.dataset)
    split_dir = Path(args.list_dir) if args.list_dir else resolve_default_split_dir(norm_ds)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    # Default output directory: <run_dir>/evaluation
    run_dir = ckpt_path.parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "evaluation")
    pred_dir = run_dir / "test_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "evaluate.log", encoding="utf-8"),
        ],
    )

    proto = BENCHMARK_PROTOCOLS.get(norm_ds, BENCHMARK_PROTOCOLS["myops380"])
    logger.info("================================================================================")
    logger.info("mmFormer Evaluation - Dataset: %s | Protocol: %s", norm_ds, proto["id"])
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Splits directory: %s", split_dir)
    logger.info("Prediction directory: %s", pred_dir)
    logger.info("================================================================================")

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    # Initialize model
    model = Model(num_cls=4, num_modals=3)

    # Load weights
    logger.info("Loading checkpoint weights...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully. Trainable params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    if getattr(args, "measure_efficiency", True):
        from mmformer.utils.efficiency import (
            build_efficiency_payload,
            check_rule_compliance,
            count_parameters,
            measure_flops,
            measure_inference_efficiency,
            save_efficiency_report,
        )
        logger.info("--- Efficiency Benchmark Stage 1: FLOPs & Complexity ---")
        params_info = count_parameters(model)
        flops_info = measure_flops(model, input_shape=(1, 3, 128, 128), device=device)
        logger.info(
            "Trainable Params: %.2fM | Total Params: %.2fM | FLOPs: %.2f G",
            params_info["trainable_params_m"],
            params_info["total_params_m"],
            flops_info["flops_g"],
        )

        logger.info("--- Efficiency Benchmark Stage 2: Inference Latency & Peak VRAM ---")
        warmup = getattr(args, "efficiency_warmup", 30 if device.type == "cuda" else 2)
        repeats = getattr(args, "efficiency_repeats", 100 if device.type == "cuda" else 5)
        inference_info = measure_inference_efficiency(
            model,
            input_shape=(1, 3, 128, 128),
            device=device,
            warmup_iters=warmup,
            benchmark_iters=repeats,
        )
        logger.info(
            "Inference Latency: %.2f ms/slice (FPS: %.2f) | Peak VRAM: %.2f MB",
            inference_info["latency_ms_per_slice"],
            inference_info["fps"],
            inference_info["peak_vram_mb"],
        )

        rule_compliant, compliance_warnings, hardware_info = check_rule_compliance(
            device=device,
            batch_size=getattr(args, "batch_size", 16),
            max_epochs=getattr(args, "max_epochs", 300),
            precision="fp32",
            num_workers=0,
            pin_memory=False,
            log=logger,
        )

        eff_payload = build_efficiency_payload(
            model_name="mmFormer",
            model_id="mmformer",
            seed=getattr(args, "seed", 1234),
            params_info=params_info,
            flops_info=flops_info,
            inference_info=inference_info,
            training_info=None,
            rule_compliant=rule_compliant,
            compliance_warnings=compliance_warnings,
            hardware_info=hardware_info,
        )
        eff_out = (
            Path(args.efficiency_output)
            if getattr(args, "efficiency_output", None)
            else (output_dir / "efficiency_metrics.json")
        )
        eff_summary_out = eff_out.parent / "efficiency_summary.txt"
        save_efficiency_report(
            eff_payload,
            [eff_out],
            [eff_summary_out],
            log=logger,
        )

    # Load volume dataset
    dataset = MyopsVolumeDataset(
        data_root=args.data_root,
        list_dir=split_dir,
        split="test_vol",
        dataset=norm_ds,
    )
    logger.info("Total test volume cases: %d", len(dataset))

    all_metric_rows: list[dict[str, Any]] = []
    has_labels = False

    eval_start_time = time.time()

    for idx in range(len(dataset)):
        case_sample = dataset[idx]
        case_name = case_sample["case_name"]
        img_bssfp = case_sample["image"]
        img_lge = case_sample["image1"]
        img_t2w = case_sample["image2"]
        target = case_sample["label"]
        spacing = case_sample.get("spacing", None)

        # Run 3D volume inference
        pred_3d = predict_case_volume(model, img_bssfp, img_lge, img_t2w, device)

        # Save test prediction in canonical format
        if args.save_predictions:
            pred_file = pred_dir / f"{case_name}_pred.npz"
            np.savez_compressed(pred_file, prediction=pred_3d.astype(np.uint8))

        # Compute metrics if ground truth is available
        if target is not None:
            has_labels = True
            rows = dataset_rows(
                prediction=pred_3d,
                target=target,
                case=case_name,
                compute_distance=True,
                dataset=norm_ds,
                spacing=spacing,
            )
            all_metric_rows.extend(rows)

            scar_row = next((r for r in rows if r["region"] == "scar"), None)
            edema_inc_row = next((r for r in rows if r["region"] == "edema_inclusive"), None)
            s_dice = f"{scar_row['dice']:.4f}" if (scar_row and scar_row["dice"] is not None) else "None"
            e_dice = f"{edema_inc_row['dice']:.4f}" if (edema_inc_row and edema_inc_row["dice"] is not None) else "None"
            logger.info("Case [%2d/%2d] %s | Scar Dice: %s | Edema-Inc Dice: %s", idx + 1, len(dataset), case_name, s_dice, e_dice)
        else:
            logger.info("Case [%2d/%2d] %s | Prediction saved (No GT label available)", idx + 1, len(dataset), case_name)

    total_time = time.time() - eval_start_time
    logger.info("Inference complete in %.1f seconds.", total_time)

    if has_labels and all_metric_rows:
        dist_unit = proto["distance_unit"]
        # Write per_case.csv
        csv_file = output_dir / "per_case.csv"
        fieldnames = [
            "case",
            "region",
            "dice",
            "iou",
            "precision",
            "recall",
            f"hd95_{dist_unit}",
            f"asd_{dist_unit}",
            "hd95_voxel",
            "asd_voxel",
            "status",
            "prediction_voxels",
            "target_voxels",
            "hd95_unit",
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in all_metric_rows:
                writer.writerow(row)
        logger.info("Per-case metrics saved to: %s", csv_file)

        # Summarize rows and write metrics.json
        summary = summarize_dataset_rows(all_metric_rows, dataset=norm_ds)
        metrics_payload = {
            "protocol": proto,
            "dataset": norm_ds,
            "distance_unit": dist_unit,
            "summary": summary,
            # Mirror top-level keys for backwards-compatibility with summary tools
            "scar": summary.get("scar", {}),
            "edema": summary.get("edema", {}),
            "edema_inclusive": summary.get("edema_inclusive", {}),
            "normal_myocardium": summary.get("normal_myocardium", {}),
            "myocardial_ring": summary.get("myocardial_ring", {}),
            "provenance": {
                "checkpoint": str(ckpt_path),
                "model_id": "mmformer",
                "dataset": norm_ds,
                "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_cases": len(dataset),
            },
        }
        json_file = output_dir / "metrics.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_safe(metrics_payload), f, indent=2)
        logger.info("Summary metrics saved to: %s", json_file)

        # Print Master Benchmark clinical table to terminal and logger
        clinical_table_str = format_master_clinical_table(summary, dataset=norm_ds)
        for line in clinical_table_str.splitlines():
            logger.info("%s", line)


if __name__ == "__main__":
    main()
