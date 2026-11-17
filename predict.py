"""Single-pass deterministic volume inference script for mmFormer.

Supports:
  - MyoPS380 (76 Held-Out Test Volumes)
  - MyoPS++ (16 Held-Out Test Volumes)

Key Requirements:
  - Checkpoint loading: <run_dir>/checkpoints/mmformer_seed<seed>.pt
  - Single-pass model.eval(), torch.inference_mode()
  - Process all cases from test_vol.txt (76 cases for myops380, 16 cases for myopspp)
  - Dynamic 2D slice inference and zoom if needed
  - Strict NO test-time augmentation (TTA)
  - Strict NO post-processing / largest connected component (LCC)
  - Export: <run_dir>/test_predictions/<CaseID>_pred.npz with key 'prediction' (uint8, [0..3] canonical)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import zoom

# Ensure repo root is at the head of sys.path and remove pkg_dir to prevent module shadowing
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
from mmformer.mmformer import Model
from mmformer.utils.metrics import BENCHMARK_PROTOCOLS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        dest="checkpoint",
        type=str,
        default=None,
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
        help="Inference output directory; defaults to checkpoint run directory",
    )
    parser.add_argument("--split", default="test_vol", help="Split name to predict (default: test_vol)")
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


def parse_args(args=None):
    return build_parser().parse_args(args)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


@torch.inference_mode()
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
    for slice_idx in range(d):
        s0 = img_bssfp[:, :, slice_idx]
        s1 = img_lge[:, :, slice_idx]
        s2 = img_t2w[:, :, slice_idx]

        if h != 128 or w != 128:
            zh = 128.0 / h
            zw = 128.0 / w
            s0 = zoom(s0, (zh, zw), order=1)
            s1 = zoom(s1, (zh, zw), order=1)
            s2 = zoom(s2, (zh, zw), order=1)

        t0 = torch.from_numpy(s0).float().unsqueeze(0).unsqueeze(0)
        t1 = torch.from_numpy(s1).float().unsqueeze(0).unsqueeze(0)
        t2 = torch.from_numpy(s2).float().unsqueeze(0).unsqueeze(0)

        # Shape (1, 3, 128, 128)
        x = torch.cat([t0, t1, t2], dim=1).to(device)

        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        pred_slice = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        if h != 128 or w != 128:
            zh_back = h / 128.0
            zw_back = w / 128.0
            pred_slice = zoom(pred_slice, (zh_back, zw_back), order=0)

        prediction_3d[:, :, slice_idx] = pred_slice

    return prediction_3d


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.checkpoint:
        raise ValueError("Must provide --checkpoint for volume prediction.")

    norm_ds = normalize_dataset_name(args.dataset)
    split_dir = Path(args.list_dir) if args.list_dir else resolve_default_split_dir(norm_ds)

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")

    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    pred_dir = output_dir / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "predict.log"
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
    logger = logging.getLogger("mmFormer_Predict")

    try:
        device = resolve_device(args.device)
        logger.info("=== mmFormer Volume Inference on %s (%s) ===", norm_ds, args.split)
        logger.info("Checkpoint: %s", ckpt_path)
        logger.info("Output: %s", pred_dir)
        logger.info("Device: %s", device)

        # Initialize model
        model = Model(num_cls=4, num_modals=3)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info("Model loaded. Trainable params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

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
            inference_info = measure_inference_efficiency(
                model,
                input_shape=(1, 3, 128, 128),
                device=device,
                warmup_iters=30 if device.type == "cuda" else 2,
                benchmark_iters=100 if device.type == "cuda" else 5,
            )
            logger.info(
                "Inference Latency: %.2f ms/slice (FPS: %.2f) | Peak VRAM: %.2f MB",
                inference_info["latency_ms_per_slice"],
                inference_info["fps"],
                inference_info["peak_vram_mb"],
            )

            rule_compliant, compliance_warnings, hardware_info = check_rule_compliance(
                device=device,
                batch_size=16,
                max_epochs=300,
                precision="fp32",
                num_workers=0,
                pin_memory=False,
                log=logger,
            )

            eff_payload = build_efficiency_payload(
                model_name="mmFormer",
                model_id="mmformer",
                seed=1234,
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
            split=args.split,
            dataset=norm_ds,
        )
        exp_cases = 16 if norm_ds == "myopspp" else 76
        logger.info("Inference dataset contains %d cases (expected: %d)", len(dataset), exp_cases)

        benchmark_proto = BENCHMARK_PROTOCOLS.get(norm_ds, BENCHMARK_PROTOCOLS["myops380"])
        dist_unit = benchmark_proto["distance_unit"]

        for idx in range(len(dataset)):
            case_sample = dataset[idx]
            case_name = case_sample["case_name"]
            img_bssfp = case_sample["image"]
            img_lge = case_sample["image1"]
            img_t2w = case_sample["image2"]
            spacing = case_sample.get("spacing", None)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()

            pred_3d = predict_case_volume(model, img_bssfp, img_lge, img_t2w, device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start_time

            pred_file = pred_dir / f"{case_name}_pred.npz"
            np.savez_compressed(
                pred_file,
                prediction=pred_3d.astype(np.uint8),
                class_names=np.asarray(CLASS_NAMES),
                spacing=np.asarray(spacing if spacing is not None else [np.nan, np.nan, np.nan]),
                affine=np.asarray(case_sample.get("affine", np.full((4, 4), np.nan))),
                spacing_unit=case_sample.get("spacing_unit", dist_unit),
                metric_distance_unit=dist_unit,
                exported_label_order="canonical",
                adapter_version="canonical_v1",
                model_id="mmformer",
                dataset=norm_ds,
            )
            if case_sample.get("has_affine", False):
                try:
                    import nibabel as nib
                    nifti = nib.Nifti1Image(pred_3d.astype(np.uint8), np.asarray(case_sample["affine"]))
                    if case_sample.get("spacing_unit") == "mm":
                        nifti.header.set_xyzt_units("mm")
                    nib.save(nifti, pred_dir / f"{case_name}_pred.nii.gz")
                except Exception:
                    pass

            logger.info("Saved: %s_pred.npz (%d slices, %.3fs)", case_name, pred_3d.shape[2], elapsed)

        logger.info("=== Completed volume inference for %d cases -> %s ===", len(dataset), pred_dir)
    finally:
        file_handler.close()
        logging.getLogger().removeHandler(file_handler)


if __name__ == "__main__":
    main()
