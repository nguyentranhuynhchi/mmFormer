"""Benchmark Trainer for mmFormer on MyoPS-380.
Complies strictly with QUY_TAC_BENCHMARK_NO_VAL.md:
- Protocol: myops380_train304_test76_last_v1
- 304 train patients (1,972 2D slices), NO validation
- Fixed 300 epochs budget
- Effective batch size = 16 (sample-weighted gradient accumulation)
- AdamW optimizer (weight_decay=1e-4) + Step-level Polynomial LR Decay
- Checkpoint: ONLY last.pth at epoch 300 (no best checkpoint)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset_myops import MyopsDataset, RandomGenerator
from utils.criterions2d import compute_mmformer_loss


def trainer_myops(args, model: nn.Module, snapshot_path: str):
    os.makedirs(snapshot_path, exist_ok=True)
    checkpoints_dir = os.path.join(snapshot_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    log_file = os.path.join(snapshot_path, "train.log")
    logging.basicConfig(
        filename=log_file, level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info("==============================================================================")
    logging.info("🚀 STARTING MMFORMER BENCHMARK TRAINING (NO-VAL PROTOCOL)")
    logging.info("Protocol: myops380_train304_test76_last_v1 (304 train slices=1972, No-Val)")
    logging.info(f"Arguments: {vars(args)}")
    logging.info("==============================================================================")

    # 1. Verify train split SHA-256 integrity
    train_split_path = os.path.join(args.list_dir, "train.txt")
    if os.path.isfile(train_split_path):
        with open(train_split_path, "rb") as fp:
            train_hash = hashlib.sha256(fp.read()).hexdigest()
        logging.info(f"train.txt SHA-256: {train_hash} (Expected: 16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919)")
        if train_hash != "16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919":
            raise RuntimeError(f"train.txt checksum mismatch: {train_hash}. Aborting to protect benchmark integrity.")
    else:
        raise FileNotFoundError(f"train.txt split file not found at: {train_split_path}")

    # Set internal CPU thread limits to prevent CPU thrashing
    cpu_threads = getattr(args, "cpu_threads", 2)
    torch.set_num_threads(cpu_threads)
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    effective_batch_size = getattr(args, "effective_batch_size", 16) or 16
    accum_steps = max(1, effective_batch_size // batch_size)

    logging.info(f"Batch configuration: batch_size={batch_size}, accum_steps={accum_steps}, effective_batch_size={batch_size * accum_steps}")
    logging.info(f"Optimization: max_epochs={args.max_epochs}, base_lr={base_lr}, weight_decay={args.weight_decay}")

    # 2. Dataset & DataLoader (Standard SCAR RandomGenerator)
    db_train = MyopsDataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=RandomGenerator(output_size=(args.img_size, args.img_size)),
        label_order="legacy"
    )
    logging.info(f"Loaded MyoPS-380 train dataset: {len(db_train)} slices")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    num_workers = getattr(args, "num_workers", 2)
    trainloader = DataLoader(
        db_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn, drop_last=False
    )

    model.train()
    model.is_training = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: total={total_params:,} | trainable={trainable_params:,}")

    # 3. Optimizer & LR Scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    writer = SummaryWriter(os.path.join(snapshot_path, "tensorboard"))
    max_epochs = args.max_epochs
    total_updates = max_epochs * math.ceil(len(trainloader) / accum_steps)
    logging.info(f"Total optimizer updates: {total_updates} over {max_epochs} epochs ({len(trainloader)} batches/epoch)")

    update_step = 0
    group_samples = 0
    start_time_all = time.time()

    # 4. Training Loop (300 Epochs, NO validation)
    for epoch_num in range(max_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_fuse_loss = 0.0
        epoch_sep_loss = 0.0
        epoch_prm_loss = 0.0
        batch_count_total = 0

        optimizer.zero_grad(set_to_none=True)
        group_samples = 0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch = sampled_batch["image"].cuda(non_blocking=True)
            one_hot_label_batch = sampled_batch["one_hot_label"].cuda(non_blocking=True)
            mask_batch = sampled_batch["mask"].cuda(non_blocking=True)

            model.is_training = True
            fuse_pred, sep_preds, prm_preds = model(image_batch, mask_batch)

            loss, fuse_loss, sep_loss, prm_loss, fuse_cross = compute_mmformer_loss(
                fuse_pred, sep_preds, prm_preds, one_hot_label_batch, num_cls=num_classes
            )

            curr_bs = image_batch.shape[0]
            # Sample-weighted backpropagation
            (loss * curr_bs).backward()
            group_samples += curr_bs

            epoch_loss += loss.item() * curr_bs
            epoch_fuse_loss += fuse_loss.item() * curr_bs
            epoch_sep_loss += sep_loss.item() * curr_bs
            epoch_prm_loss += prm_loss.item() * curr_bs
            batch_count_total += curr_bs

            # Gradient accumulation step boundary check
            is_boundary = ((i_batch + 1) % accum_steps == 0) or ((i_batch + 1) == len(trainloader))
            if is_boundary:
                # Divide gradients by exact number of samples in the accumulation window
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(group_samples)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                group_samples = 0
                update_step += 1

                # Polynomial LR decay: step-level
                curr_lr = base_lr * (max(0.0, 1.0 - update_step / total_updates)) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr

        epoch_duration = time.time() - epoch_start_time
        avg_loss = epoch_loss / batch_count_total
        avg_fuse = epoch_fuse_loss / batch_count_total
        avg_sep = epoch_sep_loss / batch_count_total
        avg_prm = epoch_prm_loss / batch_count_total
        curr_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("train/total_loss", avg_loss, epoch_num + 1)
        writer.add_scalar("train/fuse_loss", avg_fuse, epoch_num + 1)
        writer.add_scalar("train/sep_loss", avg_sep, epoch_num + 1)
        writer.add_scalar("train/prm_loss", avg_prm, epoch_num + 1)
        writer.add_scalar("train/lr", curr_lr, epoch_num + 1)

        logging.info(
            f"Epoch {epoch_num+1:03d}/{max_epochs} | Loss: {avg_loss:.4f} (Fuse: {avg_fuse:.4f}, Sep: {avg_sep:.4f}, PRM: {avg_prm:.4f}) | LR: {curr_lr:.6f} | {epoch_duration:.1f}s"
        )

        # 5. Save last checkpoint atomically every epoch (NO best.pth)
        full_ckpt = {
            "epoch": epoch_num + 1,
            "planned_epochs": max_epochs,
            "completed_epochs": epoch_num + 1,
            "training_complete": (epoch_num + 1 == max_epochs),
            "checkpoint_selection": "last_completed_epoch",
            "checkpoint_selection_metric": None,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args)
        }
        last_pth = os.path.join(checkpoints_dir, "last.pth")
        torch.save(full_ckpt, last_pth)
        torch.save(full_ckpt, os.path.join(checkpoints_dir, "last.pt"))

    writer.close()
    total_time_hours = (time.time() - start_time_all) / 3600.0
    logging.info(f"🎉 Training finished successfully! Total time: {total_time_hours:.2f} hours. Last checkpoint: {last_pth}")
    return "Training Finished!"
