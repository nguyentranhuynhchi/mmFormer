import os
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from mmformer.mmformer import Model
from mmformer.layers import general_conv2d_prenorm, fusion_prenorm
from mmformer.data.dataset_myops import MyopsDataset, MyopsVolumeDataset, canonicalize_label, RandomGenerator
from mmformer.utils.criterions import compute_mmformer_loss, to_one_hot, cross_entropy_loss, dice_loss
from mmformer.utils.optimization import (
    build_optimizer,
    build_scheduler,
    OPTIMIZATION_PROTOCOL,
    validate_optimization_checkpoint,
)
from mmformer.utils.metrics import (
    BENCHMARK_PROTOCOL,
    binary_metrics,
    dataset_rows,
    summarize_dataset_rows,
)
from mmformer.evaluate_myops import predict_case_volume


class MMFormerBenchmarkTests(unittest.TestCase):
    def test_model_architecture_and_params(self):
        """Verify 2D mmFormer architecture, active parameters, and output shapes."""
        model = Model(num_cls=4, num_modals=3)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertEqual(total_params, trainable_params)
        self.assertEqual(total_params, 24050336)

        # Eval mode
        model.eval()
        dummy_tensor = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out_eval = model(dummy_tensor)
        self.assertEqual(out_eval.shape, (2, 4, 128, 128))
        sums = out_eval.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

        # Train mode
        model.train()
        out_train = model(dummy_tensor)
        self.assertIsInstance(out_train, tuple)
        self.assertEqual(len(out_train), 3)
        fuse_pred, sep_preds, prm_preds = out_train
        self.assertEqual(fuse_pred.shape, (2, 4, 128, 128))
        self.assertEqual(len(sep_preds), 3)
        self.assertEqual(len(prm_preds), 4)

    def test_loss_and_gradients(self):
        """Verify 3-branch loss and non-zero gradient flow across 100% of trainable weights."""
        model = Model(num_cls=4, num_modals=3)
        model.train()
        dummy_input = torch.randn(2, 3, 128, 128)
        dummy_target = torch.randint(0, 4, (2, 128, 128), dtype=torch.long)

        fuse_pred, sep_preds, prm_preds = model(dummy_input)
        loss, loss_metrics = compute_mmformer_loss(fuse_pred, sep_preds, prm_preds, dummy_target, num_cls=4)

        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreater(loss.item(), 0.0)

        loss.backward()

        # All 235 parameter tensors must have non-zero gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} did not receive gradients")
                self.assertFalse(torch.all(param.grad == 0), f"Parameter {name} has all-zero gradients")

    def test_loss_edge_cases_and_shapes(self):
        """Verify loss behavior on edge cases and shape variations: (B, H, W) and (B, 1, H, W)."""
        pred = torch.softmax(torch.randn(2, 4, 128, 128), dim=1)

        # 3D target (B, H, W)
        target_3d = torch.zeros((2, 128, 128), dtype=torch.long)
        oh_3d = to_one_hot(target_3d, num_cls=4)
        self.assertEqual(oh_3d.shape, (2, 4, 128, 128))

        # 4D target (B, 1, H, W)
        target_4d = torch.zeros((2, 1, 128, 128), dtype=torch.long)
        oh_4d = to_one_hot(target_4d, num_cls=4)
        self.assertEqual(oh_4d.shape, (2, 4, 128, 128))

        # Check numerical stability
        ce = cross_entropy_loss(pred, oh_3d)
        dice = dice_loss(pred, oh_3d, num_cls=4)
        self.assertFalse(torch.isnan(ce))
        self.assertFalse(torch.isnan(dice))

    def test_optimization_dynamics(self):
        """Verify AdamW polynomial decay scheduler offset across 37,200 updates."""
        model = torch.nn.Linear(10, 2)
        base_lr = 0.0003333
        optimizer = build_optimizer(model, base_lr=base_lr, weight_decay=0.0001)
        total_updates = 37200
        scheduler = build_scheduler(optimizer, total_updates=total_updates)

        # Step 0
        self.assertAlmostEqual(scheduler.get_last_lr()[0], base_lr, places=7)

        # Update 1 (first step maintains base_lr)
        optimizer.step()
        scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], base_lr, places=7)

        # Update 2 (decay starts)
        optimizer.step()
        scheduler.step()
        expected_lr_2 = base_lr * ((1.0 - 1.0 / total_updates) ** 0.9)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], expected_lr_2, places=7)

        # Halfway step
        for _ in range(3, 18601):
            optimizer.step()
            scheduler.step()
        expected_half = base_lr * (max(0.0, 1.0 - (18600 - 1) / total_updates) ** 0.9)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], expected_half, places=7)

        # Final step 37200
        for _ in range(18601, 37201):
            optimizer.step()
            scheduler.step()
        expected_final = base_lr * (max(0.0, 1.0 - (37200 - 1) / total_updates) ** 0.9)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], expected_final, places=9)
        self.assertGreater(scheduler.get_last_lr()[0], 0.0)

    def test_canonicalize_label(self):
        """Verify target adapter LUT [0, 1, 3, 2]."""
        legacy = np.array([0, 1, 2, 3], dtype=np.uint8)
        canonical = canonicalize_label(legacy, order="legacy")
        np.testing.assert_array_equal(canonical, [0, 1, 3, 2])

        with self.assertRaises(ValueError):
            canonicalize_label(np.array([0, 1, 4]))

    def test_random_generator(self):
        """Verify 100% I-MMSeg augmentation consistency."""
        rg = RandomGenerator(output_size=(128, 128))
        for trial in range(15):
            s = {
                "image": np.random.rand(128, 128).astype(np.float32),
                "image1": np.random.rand(128, 128).astype(np.float32),
                "image2": np.random.rand(128, 128).astype(np.float32),
                "label": np.random.randint(0, 4, (128, 128), dtype=np.uint8),
                "case_name": f"trial_{trial}",
            }
            res = rg(s)
            self.assertEqual(res["image"].shape, (3, 128, 128))
            self.assertEqual(res["label"].shape, (128, 128))
            self.assertTrue(torch.all(res["label"] >= 0) and torch.all(res["label"] <= 3))

    def test_metrics_evaluator(self):
        """Verify shared evaluator metrics under protocol myops380_voxel_v3."""
        pred_vol = np.zeros((128, 128, 10), dtype=np.uint8)
        target_vol = np.zeros((128, 128, 10), dtype=np.uint8)
        pred_vol[40:60, 40:60, 2:8] = 1
        pred_vol[50:55, 50:55, 3:6] = 2
        pred_vol[55:58, 55:58, 4:5] = 3
        target_vol[40:60, 40:60, 2:8] = 1
        target_vol[50:55, 50:55, 3:6] = 2
        target_vol[55:58, 55:58, 4:5] = 3

        rows = dataset_rows(pred_vol, target_vol, case="case_synth")
        self.assertEqual(len(rows), 5)
        summary = summarize_dataset_rows(rows)
        self.assertAlmostEqual(summary["scar"]["mean_dice"], 1.0)
        self.assertAlmostEqual(summary["avg_pathology_inclusive_dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
