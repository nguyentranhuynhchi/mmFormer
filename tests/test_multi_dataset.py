"""tests/test_multi_dataset.py
===========================
Comprehensive Unit Test Suite for mmFormer Multi-Dataset Support (MyoPS380 & MyoPS++),
Double-Swap Prevention, Dynamic Physical mm Spacing, Step Budget Calculations,
CLI Parity, and Model Architecture & Loss Function Preservation.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmformer.data.dataset_myops import (
    CLASS_NAMES,
    MYOPS380_EXPECTED_TEST_VOLUMES,
    MYOPS380_EXPECTED_TRAIN_PATIENTS,
    MYOPS380_EXPECTED_TRAIN_SLICES,
    MYOPSPP_EXPECTED_TEST_VOLUMES,
    MYOPSPP_EXPECTED_TRAIN_PATIENTS,
    MYOPSPP_EXPECTED_TRAIN_SLICES,
    MyopsDataset,
    MyopsVolumeDataset,
    canonicalize_label,
    normalize_dataset_name,
    patient_id,
    resolve_default_split_dir,
)
from mmformer.utils.metrics import (
    BENCHMARK_PROTOCOLS,
    BENCHMARK_PROTOCOL,
    binary_metrics,
    dataset_rows,
    summarize_dataset_rows,
    metric_hd95_slice_robust,
    compute_patient_surface_distances,
)
from mmformer.mmformer import Model
from mmformer.utils.criterions import compute_mmformer_loss, cross_entropy_loss, dice_loss
import mmformer.train_myops as train_pkg
import mmformer.evaluate_myops as eval_pkg
import train_myops as train_root
import evaluate_myops as eval_root
import predict as predict_root
import evaluate_predictions as eval_preds_root


class TestMultiDatasetSplits(unittest.TestCase):
    """Test data split manifests and line counts for both MyoPS380 and MyoPS++."""

    def test_normalize_dataset_name(self):
        self.assertEqual(normalize_dataset_name("myops380"), "myops380")
        self.assertEqual(normalize_dataset_name("myopspp"), "myopspp")
        self.assertEqual(normalize_dataset_name("myopspp_unet2d_roi128_1mm"), "myopspp")
        self.assertEqual(normalize_dataset_name("MYOPSPP"), "myopspp")
        self.assertEqual(normalize_dataset_name("MYOPS380"), "myops380")
        self.assertEqual(normalize_dataset_name(None), "myops380")
        with self.assertRaises(ValueError):
            normalize_dataset_name("invalid_cohort_name")

    def test_myops380_split_manifests(self):
        split_dir = REPO_ROOT / "splits" / "myops380_no_val"
        train_file = split_dir / "train.txt"
        test_file = split_dir / "test_vol.txt"

        self.assertTrue(train_file.is_file(), f"Missing {train_file}")
        self.assertTrue(test_file.is_file(), f"Missing {test_file}")

        train_lines = [l.strip() for l in train_file.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
        test_lines = [l.strip() for l in test_file.read_text(encoding="utf-8-sig").splitlines() if l.strip()]

        self.assertEqual(len(train_lines), MYOPS380_EXPECTED_TRAIN_SLICES, f"Expected {MYOPS380_EXPECTED_TRAIN_SLICES} slices")
        train_patients = {patient_id(l) for l in train_lines}
        self.assertEqual(len(train_patients), MYOPS380_EXPECTED_TRAIN_PATIENTS, f"Expected {MYOPS380_EXPECTED_TRAIN_PATIENTS} patients")

        self.assertEqual(len(test_lines), MYOPS380_EXPECTED_TEST_VOLUMES, f"Expected {MYOPS380_EXPECTED_TEST_VOLUMES} test volumes")
        test_patients = {patient_id(l) for l in test_lines}
        self.assertEqual(len(test_patients), MYOPS380_EXPECTED_TEST_VOLUMES)

        # Zero patient leakage
        self.assertEqual(len(train_patients & test_patients), 0, "Data leakage between train and test in MyoPS380!")

    def test_myopspp_split_manifests(self):
        split_dir = REPO_ROOT / "splits" / "myopspp_no_val"
        train_file = split_dir / "train.txt"
        test_file = split_dir / "test_vol.txt"

        self.assertTrue(train_file.is_file(), f"Missing {train_file}")
        self.assertTrue(test_file.is_file(), f"Missing {test_file}")

        train_lines = [l.strip() for l in train_file.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
        test_lines = [l.strip() for l in test_file.read_text(encoding="utf-8-sig").splitlines() if l.strip()]

        self.assertEqual(len(train_lines), MYOPSPP_EXPECTED_TRAIN_SLICES, f"Expected {MYOPSPP_EXPECTED_TRAIN_SLICES} slices")
        train_patients = {patient_id(l) for l in train_lines}
        self.assertEqual(len(train_patients), MYOPSPP_EXPECTED_TRAIN_PATIENTS, f"Expected {MYOPSPP_EXPECTED_TRAIN_PATIENTS} patients")

        self.assertEqual(len(test_lines), MYOPSPP_EXPECTED_TEST_VOLUMES, f"Expected {MYOPSPP_EXPECTED_TEST_VOLUMES} test volumes")
        test_patients = {patient_id(l) for l in test_lines}
        self.assertEqual(len(test_patients), MYOPSPP_EXPECTED_TEST_VOLUMES)

        # Zero patient leakage
        self.assertEqual(len(train_patients & test_patients), 0, "Data leakage between train and test in MyoPS++!")


class TestLabelTargetAdapter(unittest.TestCase):
    """Test Case 1 & 2: Double-swap prevention and LUT swap vs identity preservation."""

    def test_myops380_label_lut_swap(self):
        """1. test_myops380_label_lut_swap: MyoPS380 applies LUT [0, 1, 3, 2] swapping 2 and 3."""
        # Input has 0 (bg), 1 (myo), 2 (legacy scar), 3 (legacy edema)
        raw_labels = np.array([0, 1, 2, 3], dtype=np.uint8)
        canonical = canonicalize_label(raw_labels, dataset="myops380")

        # In canonical: 0 is bg, 1 is myo, 2 is edema, 3 is scar
        self.assertEqual(canonical[0], 0, "Background should remain 0")
        self.assertEqual(canonical[1], 1, "Normal myocardium should remain 1")
        self.assertEqual(canonical[2], 3, "Legacy Scar (2) must be mapped to Canonical Scar (3)")
        self.assertEqual(canonical[3], 2, "Legacy Edema (3) must be mapped to Canonical Edema (2)")

    def test_myopspp_label_identity_preservation(self):
        """2. test_myopspp_label_identity_preservation: MyoPS++ preserves labels 2 and 3 (NO swap)."""
        # Input has canonical 0 (bg), 1 (myo), 2 (canonical edema), 3 (canonical scar)
        raw_labels = np.array([0, 1, 2, 3], dtype=np.uint8)
        canonical = canonicalize_label(raw_labels, dataset="myopspp")

        # Must be identical mapping (Identity)
        np.testing.assert_array_equal(canonical, raw_labels, "MyoPS++ labels must NOT be swapped!")
        self.assertEqual(canonical[2], 2, "MyoPS++ Edema (2) must remain 2 (NO SWAP)")
        self.assertEqual(canonical[3], 3, "MyoPS++ Scar (3) must remain 3 (NO SWAP)")


class TestSpacingAndMetrics(unittest.TestCase):
    """Test Case 3: Spacing mm extraction and distance calculation."""

    def test_spacing_mm_extraction(self):
        """3. test_spacing_mm_extraction: Dynamic mm spacing [1.0, 1.0, sz] extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create synthetic h5 volume with spacing [1.2, 1.2, 8.0]
            bssfp_dir = tmp_path / "bSSFP" / "test_vol_h5"
            lge_dir = tmp_path / "LGE" / "test_vol_h5"
            t2w_dir = tmp_path / "T2w" / "test_vol_h5"
            for d in (bssfp_dir, lge_dir, t2w_dir):
                d.mkdir(parents=True, exist_ok=True)

            case_name = "test_case01"
            for d in (bssfp_dir, lge_dir, t2w_dir):
                h5_file = d / f"{case_name}.npy.h5"
                with h5py.File(h5_file, "w") as f:
                    f.create_dataset("image", data=np.zeros((128, 128, 10), dtype=np.float32))
                    f.create_dataset("label", data=np.zeros((128, 128, 10), dtype=np.uint8))
                    f.attrs["spacing"] = np.array([1.2, 1.2, 8.0], dtype=np.float64)
                    f.attrs["slice_thickness"] = 8.0

            split_dir = tmp_path / "splits"
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "test_vol.txt").write_text(f"{case_name}\n", encoding="utf-8")

            # Check MyoPS++ volume dataset spacing
            ds_pp = MyopsVolumeDataset(data_root=tmp_path, list_dir=split_dir, dataset="myopspp")
            sample_pp = ds_pp[0]
            self.assertEqual(sample_pp["spacing_unit"], "mm")
            expected_spacing_pp = np.array([1.0, 1.0, 8.0], dtype=np.float64)
            np.testing.assert_allclose(sample_pp["spacing"], expected_spacing_pp)

            # Check MyoPS380 volume dataset spacing
            ds_380 = MyopsVolumeDataset(data_root=tmp_path, list_dir=split_dir, dataset="myops380")
            sample_380 = ds_380[0]
            self.assertEqual(sample_380["spacing_unit"], "voxel")

    def test_metrics_physical_mm_vs_voxel(self):
        """Verify distance metrics apply physical spacing for myopspp and voxel for myops380."""
        # Create synthetic binary shapes separated along z axis
        pred = np.zeros((32, 32, 10), dtype=np.uint8)
        pred[10:20, 10:20, 2] = 3  # Scar at slice 2

        gt = np.zeros((32, 32, 10), dtype=np.uint8)
        gt[10:20, 10:20, 4] = 3  # Scar at slice 4 (2 slices apart)

        # In voxel units: slice difference = 2 voxels
        rows_vox = dataset_rows(pred, gt, case="case01", compute_distance=True, dataset="myops380", spacing=None)
        scar_vox = next(r for r in rows_vox if r["region"] == "scar")
        self.assertEqual(scar_vox["hd95_unit"], "voxel")
        self.assertAlmostEqual(scar_vox["hd95_voxel"], 2.0, delta=0.1)

        # In physical mm with spacing [1.0, 1.0, 5.0]: distance = 2 * 5.0 = 10.0 mm
        spacing_mm = [1.0, 1.0, 5.0]
        rows_mm = dataset_rows(pred, gt, case="case01", compute_distance=True, dataset="myopspp", spacing=spacing_mm)
        scar_mm = next(r for r in rows_mm if r["region"] == "scar")
        self.assertEqual(scar_mm["hd95_unit"], "mm")
        self.assertAlmostEqual(scar_mm["hd95_mm"], 10.0, delta=0.5)

    def test_edema_inclusive_extracted(self):
        """Verify edema_inclusive (labels 2 + 3) is properly computed."""
        pred = np.zeros((32, 32, 5), dtype=np.uint8)
        pred[10:20, 10:20, :] = 2  # Edema
        pred[15:25, 15:25, :] = 3  # Scar

        gt = np.zeros((32, 32, 5), dtype=np.uint8)
        gt[10:20, 10:20, :] = 2
        gt[15:25, 15:25, :] = 3

        rows = dataset_rows(pred, gt, case="case01", compute_distance=True, dataset="myopspp")
        edema_inc = next((r for r in rows if r["region"] == "edema_inclusive"), None)
        self.assertIsNotNone(edema_inc)
        self.assertAlmostEqual(edema_inc["dice"], 1.0)


class TestBudgetCalculation(unittest.TestCase):
    """Test Case 4: Budget steps calculation (37,200 for MyoPS380, 5,700 for MyoPS++)."""

    def test_budget_steps_calculation(self):
        batch_size = 16
        epochs = 300

        # MyoPS380: 1,972 slices
        slices_380 = 1972
        batches_380 = math.ceil(slices_380 / batch_size)  # 124
        self.assertEqual(batches_380, 124, "MyoPS380 batches per epoch must be 124")
        total_steps_380 = epochs * batches_380
        self.assertEqual(total_steps_380, 37200, "MyoPS380 total steps must be exactly 37,200")

        # MyoPS++: 296 slices
        slices_pp = 296
        batches_pp = math.ceil(slices_pp / batch_size)  # 19
        self.assertEqual(batches_pp, 19, "MyoPS++ batches per epoch must be 19")
        total_steps_pp = epochs * batches_pp
        self.assertEqual(total_steps_pp, 5700, "MyoPS++ total steps must be exactly 5,700")


class TestCLIParity(unittest.TestCase):
    """Test Case 5: CLI argument parity between root scripts and mmformer package scripts."""

    def test_cli_argument_parity(self):
        parsers = [
            ("train_pkg", train_pkg.build_parser()),
            ("train_root", train_root.build_parser()),
            ("eval_pkg", eval_pkg.build_parser()),
            ("eval_root", eval_root.build_parser()),
            ("predict_root", predict_root.build_parser()),
            ("eval_preds_root", eval_preds_root.build_parser()),
        ]

        for name, parser in parsers:
            # Check --dataset flag
            actions = {a.dest: a for a in parser._actions}
            self.assertIn("dataset", actions, f"{name} missing --dataset argument")
            dataset_action = actions["dataset"]
            self.assertIn("myops380", dataset_action.choices)
            self.assertIn("myopspp", dataset_action.choices)

            # Check list-dir / split-dir alias
            self.assertIn("list_dir", actions, f"{name} missing list_dir dest")

            # Check measure_efficiency flag
            self.assertIn("measure_efficiency", actions, f"{name} missing measure_efficiency")

            extra = ["--checkpoint", "dummy.pt"] if name in ("eval_pkg", "eval_root", "predict_root") else []
            parsed_pp = parser.parse_args(["--dataset", "myopspp", "--split-dir", "custom/splits", "--no-measure-efficiency"] + extra)
            self.assertEqual(parsed_pp.dataset, "myopspp")
            self.assertEqual(parsed_pp.list_dir, "custom/splits")
            self.assertFalse(parsed_pp.measure_efficiency)

            parsed_380 = parser.parse_args(["--dataset", "myops380", "--list_dir", "other/splits", "--measure_efficiency"] + extra)
            self.assertEqual(parsed_380.dataset, "myops380")
            self.assertEqual(parsed_380.list_dir, "other/splits")
            self.assertTrue(parsed_380.measure_efficiency)


class TestLossAndArchitecturePreservation(unittest.TestCase):
    """Test Case 6: Preserving 3-branch multimodal bridge and native loss."""

    def test_loss_and_architecture_preservation(self):
        # 1. Check model architecture
        model = Model(num_cls=4, num_modals=3)
        self.assertEqual(model.num_modals, 3, "Model must have 3 input modalities")
        self.assertEqual(model.num_cls, 4, "Model must have 4 segmentation classes")

        # Forward pass in train mode (returns 3 outputs: fuse_pred, sep_preds, prm_preds)
        x = torch.randn(2, 3, 128, 128)
        model.train()
        fuse_pred, sep_preds, prm_preds = model(x)

        self.assertEqual(fuse_pred.shape, (2, 4, 128, 128), "Fused output shape must be (B, 4, 128, 128)")
        self.assertEqual(len(sep_preds), 3, "Must have 3 modality-specific predictions")
        self.assertEqual(len(prm_preds), 4, "Must have 4 progressive refinement predictions")

        # Eval mode returns single fused prediction
        model.eval()
        with torch.no_grad():
            eval_pred = model(x)
        self.assertEqual(eval_pred.shape, (2, 4, 128, 128))

        # 2. Check loss function
        labels = torch.randint(0, 4, (2, 128, 128), dtype=torch.long)
        total_loss, loss_metrics = compute_mmformer_loss(
            fuse_pred, sep_preds, prm_preds, labels
        )
        self.assertTrue(torch.isfinite(total_loss), "Loss must be finite")
        self.assertGreater(total_loss.item(), 0.0)

    def test_flexible_path_resolution_roi_and_cache(self):
        """Test flexible path resolution for roi_128_1mm and cache directory layouts in mmFormer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_root = tmp_path / "roi_128_1mm"
            cache_dir = data_root / "cache"
            for mod in ("bSSFP", "LGE", "T2w"):
                (cache_dir / mod / "train_npz").mkdir(parents=True)
            list_dir = tmp_path / "splits"
            list_dir.mkdir(parents=True)
            (list_dir / "train.txt").write_text("Case3002_slice000\n", encoding="utf-8")

            lbl = np.array([[0, 1], [2, 3]], dtype=np.uint8)
            for mod in ("bSSFP", "LGE", "T2w"):
                np.savez(cache_dir / mod / "train_npz" / "Case3002_slice000.npz",
                         image=np.ones((2, 2), dtype=np.float32) * 4.0, label=lbl)

            ds = MyopsDataset(data_root=data_root, list_dir=list_dir, split="train", dataset="myopspp")
            sample = ds[0]
            self.assertEqual(sample["image"].shape, (3, 2, 2))
            np.testing.assert_array_equal(sample["label"].numpy(), lbl)

    def test_flexible_path_resolution_flat_single_root(self):
        """Test flat single-root layout with 3-channel npz files in mmFormer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_root = tmp_path / "roi_128_1mm"
            (data_root / "train_npz").mkdir(parents=True)
            list_dir = tmp_path / "splits"
            list_dir.mkdir(parents=True)
            (list_dir / "train.txt").write_text("Case3002_slice000\n", encoding="utf-8")

            lbl = np.array([[0, 1], [2, 3]], dtype=np.uint8)
            np.savez(data_root / "train_npz" / "Case3002_slice000.npz",
                     image=np.ones((3, 2, 2), dtype=np.float32) * 6.0, label=lbl)

            ds = MyopsDataset(data_root=data_root, list_dir=list_dir, split="train", dataset="myopspp")
            sample = ds[0]
            self.assertEqual(sample["image"].shape, (3, 2, 2))
            np.testing.assert_array_equal(sample["label"].numpy(), lbl)


if __name__ == "__main__":
    unittest.main()
