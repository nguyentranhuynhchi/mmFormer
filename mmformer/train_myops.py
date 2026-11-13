"""Training script for mmFormer on MyoPS380 (no-val protocol).

Protocol: myops380_train304_test76_last_v1
Budget: 300 epochs, physical batch 16, 37,200 total updates.
Objective: Loss_total = Loss_fuse + Loss_sep + Loss_prm (CE + SoftDice on 4 canonical classes)
Optimizer: AdamW (lr=0.0003333, wd=0.0001, betas=(0.9, 0.999))
Scheduler: Polynomial decay (power=0.9, zero-based offset)
Checkpoint: <run_dir>/checkpoints/mmformer_seed<seed>.pt (atomic save per epoch)
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
from pathlib import Path
import random
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmformer.mmformer import Model
from mmformer.data.dataset_myops import MyopsDataset, RandomGenerator
from mmformer.utils.criterions import compute_mmformer_loss
from mmformer.utils.efficiency import (
    build_efficiency_payload,
    check_rule_compliance,
    count_parameters,
    measure_flops,
    measure_inference_efficiency,
    save_efficiency_report,
    summarize_training_efficiency,
)
from mmformer.utils.optimization import (
    OPTIMIZATION_PROTOCOL,
    build_optimizer,
    build_scheduler,
    validate_optimization_checkpoint,
)

logger = logging.getLogger("mmFormer_Train")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def resolve_amp(mode: str, device: torch.device) -> torch.dtype | None:
    if mode == "none" or (mode == "auto" and device.type == "cpu"):
        return None
    if device.type != "cuda":
        raise ValueError("Mixed precision is only supported on CUDA.")
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bfloat16 is not supported on this CUDA device.")
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    raise ValueError(f"Unknown AMP mode: {mode}")


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mmFormer on MyoPS380 without validation")
    parser.add_argument(
        "--data-root",
        "--data_root",
        dest="data_root",
        type=str,
        default=os.environ.get("LOCAL_DATASET", "E:/STUDY/DATASET/MyoPS380/Processed_data"),
        help="Path to preprocessed MyoPS380 dataset root containing bSSFP, LGE, T2w",
    )
    parser.add_argument(
        "--list-dir",
        "--list_dir",
        dest="list_dir",
        type=str,
        default=str(repo_root / "splits" / "myops380_no_val"),
        help="Path to split manifests folder containing train.txt and test_vol.txt",
    )
    parser.add_argument(
        "--run-root",
        "--run_root",
        dest="run_root",
        type=str,
        default="outputs/runs",
        help="Root directory for saving checkpoints and training runs",
    )
    parser.add_argument(
        "--run-name",
        "--run_name",
        dest="run_name",
        type=str,
        default=None,
        help="Custom run directory name (default: mmformer_seed<seed>)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed (default: 1234)")
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=16,
        help="Physical batch size (default: 16)",
    )
    parser.add_argument("--lr", type=float, default=0.0003333, help="Base learning rate (default: 0.0003333)")
    parser.add_argument(
        "--weight-decay",
        "--weight_decay",
        dest="weight_decay",
        type=float,
        default=0.0001,
        help="AdamW weight decay (default: 0.0001)",
    )
    parser.add_argument("--epochs", type=int, default=300, help="Total training epochs (default: 300)")
    parser.add_argument(
        "--amp",
        type=str,
        default="auto",
        choices=["auto", "none", "fp16", "bf16"],
        help="Mixed precision mode (use 'none' for FP32 efficiency benchmark)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cuda', 'cpu')")
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=4,
        help="Dataloader worker threads (default: 4; use 0 for efficiency benchmark)",
    )
    parser.add_argument(
        "--pin-memory",
        "--pin_memory",
        dest="pin_memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pin host memory for CUDA transfers (default: True on CUDA)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument(
        "--measure-efficiency",
        "--measure_efficiency",
        dest="measure_efficiency",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure Complexity, Inference, and Training efficiency metrics and export JSON (default: True)",
    )
    parser.add_argument(
        "--efficiency-output",
        "--efficiency_output",
        dest="efficiency_output",
        type=str,
        default="outputs/runs/mmformer/efficiency_metrics.json",
        help="Primary JSON output path for efficiency metrics",
    )
    parser.add_argument(
        "--efficiency-warmup",
        "--efficiency_warmup",
        dest="efficiency_warmup",
        type=int,
        default=30,
        help="Inference warmup iterations (default: 30)",
    )
    parser.add_argument(
        "--efficiency-repeats",
        "--efficiency_repeats",
        dest="efficiency_repeats",
        type=int,
        default=100,
        help="Inference benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--efficiency-only",
        "--efficiency_only",
        dest="efficiency_only",
        action="store_true",
        help="Measure Complexity + Inference efficiency only (skip training loop)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine run directory and checkpoint path
    run_name = args.run_name or f"mmformer_seed{args.seed}"
    run_root_path = Path(args.run_root)
    if run_root_path.name != "runs":
        run_dir = run_root_path / "runs" / run_name
    else:
        run_dir = run_root_path / run_name
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint_dir / f"mmformer_seed{args.seed}.pt"

    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "train.log", encoding="utf-8"),
        ],
    )

    logger.info("================================================================================")
    logger.info("mmFormer MyoPS380 Training - Protocol: myops380_train304_test76_last_v1")
    logger.info("Run directory: %s", run_dir)
    logger.info("================================================================================")

    # Seed
    seed_everything(args.seed, deterministic=True)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp(args.amp, device)
    pin_memory = (device.type == "cuda") if args.pin_memory is None else bool(args.pin_memory)

    # Validate benchmark rule compliance
    rule_compliant, compliance_warnings, hardware_info = check_rule_compliance(
        device=device,
        amp_mode=args.amp,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        batch_size=args.batch_size,
        planned_epochs=args.epochs,
    )
    if args.measure_efficiency or args.efficiency_only:
        logger.info(
            "Efficiency Config | GPU: %s (count=%d) | Precision: %s | num_workers: %d | pin_memory: %s | batch: %d | epochs: %d",
            hardware_info["gpu_name"],
            hardware_info["visible_gpu_count"],
            hardware_info["precision"],
            args.num_workers,
            pin_memory,
            args.batch_size,
            args.epochs,
        )
        if not rule_compliant:
            for warn_msg in compliance_warnings:
                logger.warning("[EFFICIENCY RULE MISMATCH] %s", warn_msg)

    # Initialize 2D mmFormer model (3 modalities, 4 classes)
    model = Model(num_cls=4, num_modals=3)

    # Count parameters
    params_info = count_parameters(model)
    total_params = params_info["total_params"]
    trainable_params = params_info["trainable_params"]

    logger.info(
        "INFO | Device %s | AMP %s | parameters %s (trainable: %s) | train/val slices 1972/0 | effective batch <= %d",
        device.type,
        str(amp_dtype) if amp_dtype else "fp32",
        f"{total_params:,}",
        f"{trainable_params:,}",
        args.batch_size,
    )

    # Save run configuration
    config_dict = {
        "model_id": "mmformer",
        "architecture": "mmFormer-2D",
        "parameters_total": total_params,
        "parameters_trainable": trainable_params,
        "n_classes": 4,
        "num_modals": 3,
        "modalities": ["bSSFP", "LGE", "T2w"],
        "img_size": 128,
        "patch_size": 8,
        "training_protocol": {
            "id": "myops380_train304_test76_last_v1",
            "epochs": args.epochs,
            "physical_batch": args.batch_size,
            "accum_steps": 1,
            "no_val": True,
        },
        "optimization_protocol": OPTIMIZATION_PROTOCOL,
        "base_lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": device.type,
        "amp": str(amp_dtype) if amp_dtype else "none",
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "measure_efficiency": bool(args.measure_efficiency),
    }
    with open(log_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    # Check checkpoint on CPU first so model-only efficiency benchmarks are uncontaminated by optimizer VRAM
    resume_file = args.resume or (ckpt_path if ckpt_path.is_file() else None)
    resume_ckpt: dict[str, Any] | None = None
    if resume_file and Path(resume_file).is_file():
        logger.info("Loading checkpoint metadata from: %s", resume_file)
        resume_ckpt = torch.load(resume_file, map_location="cpu", weights_only=False)
        if resume_ckpt.get("training_complete", False) and not args.efficiency_only:
            logger.info("Training was already marked complete at epoch %d. Nothing to run.", resume_ckpt["epoch"])
            return
        model.load_state_dict(resume_ckpt["model_state_dict"])

    # Ensure FP32 base weights and move model to device
    model.float().to(device)

    efficiency_json_targets = _dedupe_paths(
        [Path(args.efficiency_output), run_dir / "efficiency_metrics.json"]
    )
    efficiency_summary_targets = _dedupe_paths([log_dir / "efficiency_summary.txt"])

    flops_info: dict[str, Any] | None = None
    inference_info: dict[str, Any] | None = None
    if args.measure_efficiency or args.efficiency_only:
        logger.info(
            "Measuring Complexity (FP32 FLOPs on 1x3x128x128) & Inference (Batch=1, %d warmup / %d runs)...",
            args.efficiency_warmup,
            args.efficiency_repeats,
        )
        flops_info = measure_flops(model, input_shape=(1, 3, 128, 128), device=device)
        inference_info = measure_inference_efficiency(
            model,
            input_shape=(1, 3, 128, 128),
            device=device,
            warmup=args.efficiency_warmup,
            repeats=args.efficiency_repeats,
        )
        logger.info(
            "Complexity | Trainable Params: %.2fM (Total: %.2fM) | FLOPs: %.2f G",
            params_info["trainable_params_m"],
            params_info["total_params_m"],
            flops_info["flops_g"],
        )
        logger.info(
            "Inference (B=1, FP32) | Latency: %.2f ms/slice | Peak VRAM: %.2f MB (%.4f GB)",
            inference_info["latency_ms_per_slice"],
            inference_info["peak_vram_mb"],
            inference_info["peak_vram_gb"],
        )

        if args.efficiency_only:
            payload = build_efficiency_payload(
                model_name="mmFormer",
                model_id="mmformer",
                seed=args.seed,
                params_info=params_info,
                flops_info=flops_info,
                inference_info=inference_info,
                training_info=None,
                rule_compliant=rule_compliant,
                compliance_warnings=compliance_warnings,
                hardware_info=hardware_info,
            )
            save_efficiency_report(payload, efficiency_json_targets, efficiency_summary_targets, log=logger)
            return

    # Dataset & Dataloader
    transform = RandomGenerator(output_size=(128, 128))
    train_dataset = MyopsDataset(
        data_root=args.data_root,
        list_dir=args.list_dir,
        split="train",
        transform=transform,
        label_order="legacy",
    )
    logger.info("Loaded train dataset: %d slices", len(train_dataset))

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )

    batches_per_epoch = len(train_loader)
    total_planned_updates = args.epochs * batches_per_epoch  # 300 * 124 = 37,200
    logger.info("Batches per epoch: %d | Total planned updates: %d", batches_per_epoch, total_planned_updates)

    # Clear CUDA cache & reset peak memory stats right before allocating training state
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Optimizer, Scheduler, Scaler
    optimizer = build_optimizer(model, base_lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, total_updates=total_planned_updates)

    use_scaler = (device.type == "cuda" and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_epoch = 1
    global_step = 0
    total_updates = 0
    epoch_times: list[float] = []
    prev_wall_clock_s = 0.0
    max_train_peak_vram_bytes = 0
    max_train_reserved_bytes = 0

    # Resume handling
    if resume_ckpt is not None:
        logger.info("Resuming optimizer, scheduler, and RNG states from checkpoint...")
        validate_optimization_checkpoint(resume_ckpt)
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if use_scaler and "scaler_state_dict" in resume_ckpt and resume_ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

        start_epoch = int(resume_ckpt["epoch"]) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        total_updates = int(resume_ckpt.get("total_updates", 0))
        epoch_times = [float(t) for t in resume_ckpt.get("epoch_times", [])][: max(0, start_epoch - 1)]
        prev_wall_clock_s = float(resume_ckpt.get("total_wall_clock_seconds", 0.0))
        max_train_peak_vram_bytes = int(resume_ckpt.get("train_peak_vram_bytes", 0))
        max_train_reserved_bytes = int(resume_ckpt.get("train_peak_reserved_bytes", 0))

        # Restore RNG states
        if "rng_state" in resume_ckpt:
            rng = resume_ckpt["rng_state"]
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and "cuda" in rng and rng["cuda"] is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["random"])

        logger.info("Resumed from epoch %d (step %d, updates %d)", resume_ckpt["epoch"], global_step, total_updates)
        del resume_ckpt

    # Training loop
    logger.info("Starting training loop from epoch %d to %d...", start_epoch, args.epochs)
    session_start_time = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        _sync_cuda(device)
        epoch_start_time = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_fuse = 0.0
        running_sep = 0.0
        running_prm = 0.0

        for batch_idx, sampled_batch in enumerate(train_loader):
            images = sampled_batch["image"].to(device, non_blocking=pin_memory)
            labels = sampled_batch["label"].to(device, non_blocking=pin_memory)

            optimizer.zero_grad()

            if device.type == "cuda" and amp_dtype is not None:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    fuse_pred, sep_preds, prm_preds = model(images)
                    loss, loss_metrics = compute_mmformer_loss(fuse_pred, sep_preds, prm_preds, labels, num_cls=4)

                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                fuse_pred, sep_preds, prm_preds = model(images)
                loss, loss_metrics = compute_mmformer_loss(fuse_pred, sep_preds, prm_preds, labels, num_cls=4)
                loss.backward()
                optimizer.step()

            # Polynomial LR step after every successful optimizer update
            scheduler.step()
            total_updates += 1
            global_step += 1

            running_loss += loss_metrics["loss_total"]
            running_fuse += loss_metrics["loss_fuse"]
            running_sep += loss_metrics["loss_sep"]
            running_prm += loss_metrics["loss_prm"]

        _sync_cuda(device)
        epoch_duration = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_duration)

        if device.type == "cuda" and torch.cuda.is_available():
            max_train_peak_vram_bytes = max(max_train_peak_vram_bytes, int(torch.cuda.max_memory_allocated(device)))
            max_train_reserved_bytes = max(max_train_reserved_bytes, int(torch.cuda.max_memory_reserved(device)))

        avg_loss = running_loss / batches_per_epoch
        avg_fuse = running_fuse / batches_per_epoch
        avg_sep = running_sep / batches_per_epoch
        avg_prm = running_prm / batches_per_epoch
        current_lr = scheduler.get_last_lr()[0]
        total_wall_clock_s = prev_wall_clock_s + (time.perf_counter() - session_start_time)

        logger.info(
            "Epoch %3d/%3d | Loss: %.4f (Fuse: %.4f, Sep: %.4f, PRM: %.4f) | LR: %.7f | Time: %.2fs | Updates: %d/%d",
            epoch,
            args.epochs,
            avg_loss,
            avg_fuse,
            avg_sep,
            avg_prm,
            current_lr,
            epoch_duration,
            total_updates,
            total_planned_updates,
        )

        # Atomic checkpoint save
        is_final_epoch = (epoch == args.epochs)
        rng_state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()

        checkpoint_data = {
            "epoch": epoch,
            "planned_epochs": args.epochs,
            "completed_epochs": epoch,
            "training_complete": is_final_epoch,
            "global_step": global_step,
            "total_updates": total_updates,
            "total_planned_updates": total_planned_updates,
            "model_id": "mmformer",
            "parameters_count": total_params,
            "epoch_times": epoch_times,
            "total_wall_clock_seconds": total_wall_clock_s,
            "train_peak_vram_bytes": max_train_peak_vram_bytes,
            "train_peak_reserved_bytes": max_train_reserved_bytes,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_scaler else None,
            "rng_state": rng_state,
            "training_protocol": {
                "id": "myops380_train304_test76_last_v1",
                "cohort": "myops380",
                "train_patients": 304,
                "train_slices": 1972,
                "test_patients": 76,
                "epochs": args.epochs,
            },
            "optimization_protocol": OPTIMIZATION_PROTOCOL,
            "loss_metric": {
                "loss_total": avg_loss,
                "loss_fuse": avg_fuse,
                "loss_sep": avg_sep,
                "loss_prm": avg_prm,
            },
        }

        # Safe atomic save: write to tmp file first then rename
        tmp_ckpt = ckpt_path.with_suffix(".tmp")
        torch.save(checkpoint_data, tmp_ckpt)
        tmp_ckpt.replace(ckpt_path)

    logger.info("================================================================================")
    logger.info("Training complete! %d epochs finished. Final checkpoint saved at: %s", args.epochs, ckpt_path)
    logger.info("================================================================================")

    if args.measure_efficiency and flops_info is not None and inference_info is not None:
        final_wall_clock_s = prev_wall_clock_s + (time.perf_counter() - session_start_time)
        training_info = summarize_training_efficiency(
            epoch_times_s=epoch_times,
            total_wall_clock_s=final_wall_clock_s,
            peak_vram_bytes=max_train_peak_vram_bytes,
            planned_epochs=args.epochs,
            warmup_epochs=5,
            peak_reserved_bytes=max_train_reserved_bytes,
        )
        payload = build_efficiency_payload(
            model_name="mmFormer",
            model_id="mmformer",
            seed=args.seed,
            params_info=params_info,
            flops_info=flops_info,
            inference_info=inference_info,
            training_info=training_info,
            rule_compliant=rule_compliant,
            compliance_warnings=compliance_warnings,
            hardware_info=hardware_info,
        )
        save_efficiency_report(payload, efficiency_json_targets, efficiency_summary_targets, log=logger)


if __name__ == "__main__":
    main()
