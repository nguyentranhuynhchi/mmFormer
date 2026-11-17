"""Standalone Benchmark Prediction Evaluation script for mmFormer.

Protocols:
  - myops380: myops380_voxel_v3 (Exact SCAR Parity, 76 Cases, Voxel Distance)
  - myopspp:  myopspp_unet2d_roi128_1mm_predicted_roi_mm_v1 (16 Cases, Physical mm Distance)

Evaluates patient volumes across 5 target regions:
  1. Myocardium (MYO / myocardial_ring): IDs [1, 2, 3]
  2. Edema (EDEMA / edema): ID [2] (exclusive)
  3. Scar (SCAR / scar): ID [3]
  4. LGE-abnormal (LGE-abn / edema_inclusive): IDs [2, 3]
  5. Normal Myocardium (NORM / normal_myocardium): ID [1]

Outputs:
  - <output_dir>/per_case.csv
  - <output_dir>/metrics.json
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
from typing import Any, Optional

import numpy as np

# Ensure repo root is at head of sys.path and remove pkg_dir to prevent module shadowing
repo_root = Path(__file__).resolve().parent
pkg_dir = str(repo_root / "mmformer")
while pkg_dir in sys.path:
    sys.path.remove(pkg_dir)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from mmformer.data.dataset_myops import (
    CLASS_NAMES,
    MyopsVolumeDataset,
    normalize_dataset_name,
    resolve_default_split_dir,
)
from mmformer.evaluate_myops import format_master_clinical_table
from mmformer.utils.metrics import (
    BENCHMARK_PROTOCOLS,
    dataset_rows,
    summarize_dataset_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--pred_dir",
        "--pred-dir",
        dest="pred_dir",
        default=None,
        help="Directory containing <CaseID>_pred.npz files",
    )
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        dest="checkpoint",
        default=None,
        help="Optional checkpoint path; if provided and pred_dir is not set, runs evaluate_myops",
    )
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        default=os.environ.get("LOCAL_DATASET", "E:/STUDY/DATASET/MyoPS380/Processed_data"),
        help="Root folder containing bSSFP, LGE, T2w subdirectories",
    )
    parser.add_argument(
        "--dataset",
        "--dataset_id",
        "--dataset-id",
        dest="dataset",
        default="myops380",
        choices=["myops380", "myopspp", "myopspp_unet2d_roi128_1mm", "myops_380"],
        help="Benchmark dataset to evaluate",
    )
    parser.add_argument(
        "--list-dir",
        "--list_dir",
        "--split-dir",
        "--split_dir",
        dest="list_dir",
        default=None,
        help="Directory containing test_vol.txt (defaults dynamically by dataset)",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=None,
        help="Evaluation output directory; defaults to pred_dir parent",
    )
    parser.add_argument("--split", default="test_vol", help="Split name to evaluate (default: test_vol)")
    parser.add_argument(
        "--compute-distance",
        "--compute_distance",
        dest="compute_distance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute Hausdorff and surface distance metrics",
    )
    parser.add_argument(
        "--no_compute_distance",
        dest="compute_distance",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--measure-efficiency",
        "--measure_efficiency",
        dest="measure_efficiency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure Complexity and Inference efficiency metrics (requires --checkpoint)",
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
        default=None,
        help="Optional JSON file to save measured efficiency metrics",
    )
    return parser


def parse_args(args=None):
    return build_parser().parse_args(args)


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


def load_prediction_file(pred_dir: Path, case_name: str) -> np.ndarray:
    candidates = [
        pred_dir / f"{case_name}_pred.npz",
        pred_dir / f"{case_name}.npz",
        pred_dir / "test_predictions" / f"{case_name}_pred.npz",
        pred_dir / "test_predictions" / f"{case_name}.npz",
    ]
    for c in candidates:
        if c.is_file():
            with np.load(c) as d:
                for k in ("prediction", "pred", "arr_0"):
                    if k in d:
                        return np.asarray(d[k], dtype=np.uint8)
    raise FileNotFoundError(f"Prediction file not found for case {case_name} in {pred_dir}")


def main(argv=None):
    args = build_parser().parse_args(argv)

    # If checkpoint is provided and no precomputed pred_dir, delegate to evaluate_myops
    if args.checkpoint and (not args.pred_dir or not Path(args.pred_dir).is_dir()):
        import mmformer.evaluate_myops as eval_module
        return eval_module.main(argv)

    norm_ds = normalize_dataset_name(args.dataset)
    split_dir = Path(args.list_dir) if args.list_dir else resolve_default_split_dir(norm_ds)

    if not args.pred_dir:
        raise ValueError("Must provide either --pred_dir with existing predictions or --checkpoint.")

    pred_dir = Path(args.pred_dir).resolve()
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"pred_dir does not exist: {pred_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (pred_dir.parent if pred_dir.name == "test_predictions" else pred_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "evaluation.log"
    file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
        handlers=[
            file_handler,
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("mmFormer_EvalPreds")

    try:
        logger.info("=== Standalone Evaluation on %s (%s) ===", norm_ds, args.split)
        logger.info("Predictions: %s", pred_dir)
        logger.info("Output: %s", output_dir)

        dataset = MyopsVolumeDataset(
            data_root=args.data_root,
            list_dir=split_dir,
            split=args.split,
            dataset=norm_ds,
        )
        logger.info("Loaded ground truth for %d cases from %s", len(dataset), split_dir)

        benchmark_proto = BENCHMARK_PROTOCOLS.get(norm_ds, BENCHMARK_PROTOCOLS["myops380"])
        dist_unit = benchmark_proto["distance_unit"]

        all_metric_rows: list[dict[str, Any]] = []

        for idx in range(len(dataset)):
            case_sample = dataset[idx]
            case_name = case_sample["case_name"]
            target = case_sample["label"]
            spacing = case_sample.get("spacing", None)

            if target is None:
                continue

            prediction = load_prediction_file(pred_dir, case_name)

            case_rows = dataset_rows(
                prediction=prediction,
                target=target,
                case=case_name,
                compute_distance=args.compute_distance,
                dataset=norm_ds,
                spacing=spacing,
            )
            all_metric_rows.extend(case_rows)

        if not all_metric_rows:
            raise ValueError("No evaluation results were generated.")

        csv_path = output_dir / "per_case.csv"
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
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in all_metric_rows:
                writer.writerow(row)
        logger.info("Saved per-case metrics to: %s", csv_path)

        summary = summarize_dataset_rows(all_metric_rows, dataset=norm_ds)
        metrics_payload = {
            "protocol": benchmark_proto,
            "dataset": norm_ds,
            "distance_unit": dist_unit,
            "summary": summary,
            "scar": summary.get("scar", {}),
            "edema": summary.get("edema", {}),
            "edema_inclusive": summary.get("edema_inclusive", {}),
            "normal_myocardium": summary.get("normal_myocardium", {}),
            "myocardial_ring": summary.get("myocardial_ring", {}),
            "provenance": {
                "pred_dir": str(pred_dir),
                "model_id": "mmformer",
                "dataset": norm_ds,
                "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_cases": len(dataset),
            },
        }
        json_path = output_dir / "metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(metrics_payload), f, indent=2)
        logger.info("Saved summary metrics to: %s", json_path)

        clinical_table_str = format_master_clinical_table(summary, dataset=norm_ds)
        for line in clinical_table_str.splitlines():
            logger.info("%s", line)
    finally:
        file_handler.close()
        logging.getLogger().removeHandler(file_handler)


if __name__ == "__main__":
    main()
