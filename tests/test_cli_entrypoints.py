"""Unit and integration tests for mmFormer standalone CLI entrypoints: predict.py and evaluate_predictions.py."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluate_predictions as eval_preds
from mmformer.mmformer import Model
import predict


class TestMMFormerCLIEntrypoints(unittest.TestCase):
    def test_predict_parser_defaults_and_aliases(self) -> None:
        parser = predict.build_parser()
        args = parser.parse_args([
            "--ckpt", "checkpoints/model.pt",
            "--data-root", "data/Processed_data",
            "--split-dir", "splits/myopspp_no_val",
            "--output-dir", "runs/out",
            "--dataset", "myopspp",
            "--no-measure-efficiency",
        ])
        self.assertEqual(args.checkpoint, "checkpoints/model.pt")
        self.assertEqual(args.data_root, "data/Processed_data")
        self.assertEqual(args.list_dir, "splits/myopspp_no_val")
        self.assertEqual(args.output_dir, "runs/out")
        self.assertEqual(args.dataset, "myopspp")
        self.assertFalse(args.measure_efficiency)

    def test_evaluate_predictions_parser_defaults_and_aliases(self) -> None:
        parser = eval_preds.build_parser()
        args = parser.parse_args([
            "--pred-dir", "runs/out/test_predictions",
            "--data-root", "data/Processed_data",
            "--list-dir", "splits/myops380_no_val",
            "--output-dir", "runs/eval",
            "--dataset", "myops380",
            "--measure-efficiency",
        ])
        self.assertEqual(args.pred_dir, "runs/out/test_predictions")
        self.assertEqual(args.data_root, "data/Processed_data")
        self.assertEqual(args.list_dir, "splits/myops380_no_val")
        self.assertEqual(args.output_dir, "runs/eval")
        self.assertEqual(args.dataset, "myops380")
        self.assertTrue(args.measure_efficiency)

    def test_predict_and_eval_predictions_pipeline(self) -> None:
        """End-to-end synthetic volume prediction and evaluation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Create a minimal dummy checkpoint
            model = Model(num_cls=4, num_modals=3)
            ckpt_path = tmp_path / "model.pt"
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

            # 2. Create mock dataset with 1 case
            data_root = tmp_path / "Processed_data"
            for mod in ("bSSFP", "LGE", "T2w"):
                (data_root / mod / "test_vol_h5").mkdir(parents=True, exist_ok=True)

            case_name = "case_syn01"
            h, w, d = 16, 16, 2
            dummy_bssfp = np.random.rand(h, w, d).astype(np.float32)
            dummy_lge = np.random.rand(h, w, d).astype(np.float32)
            dummy_t2w = np.random.rand(h, w, d).astype(np.float32)
            # Label with bg=0, normal=1, edema=2, scar=3
            dummy_label = np.zeros((h, w, d), dtype=np.uint8)
            dummy_label[4:12, 4:12, :] = 1
            dummy_label[6:8, 6:8, :] = 2
            dummy_label[8:10, 8:10, :] = 3

            for mod_name, vol, lbl in [
                ("bSSFP", dummy_bssfp, dummy_label),
                ("LGE", dummy_lge, None),
                ("T2w", dummy_t2w, None),
            ]:
                fpath = data_root / mod_name / "test_vol_h5" / f"{case_name}.npy.h5"
                with h5py.File(fpath, "w") as hf:
                    hf.create_dataset("image", data=vol)
                    if lbl is not None:
                        hf.create_dataset("label", data=lbl)

            # 3. Create split dir with test_vol.txt
            split_dir = tmp_path / "splits"
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "test_vol.txt").write_text(f"{case_name}\n", encoding="utf-8")

            # 4. Run predict.py
            pred_out_dir = tmp_path / "predictions_run"
            predict.main([
                "--checkpoint", str(ckpt_path),
                "--data-root", str(data_root),
                "--list-dir", str(split_dir),
                "--output-dir", str(pred_out_dir),
                "--dataset", "myops380",
                "--device", "cpu",
                "--no-measure-efficiency",
            ])

            pred_file = pred_out_dir / "test_predictions" / f"{case_name}_pred.npz"
            self.assertTrue(pred_file.is_file(), f"Expected prediction file {pred_file}")
            with np.load(pred_file) as npz:
                self.assertIn("prediction", npz)
                pred_vol = npz["prediction"]
                self.assertEqual(pred_vol.shape, (h, w, d))
                self.assertEqual(pred_vol.dtype, np.uint8)

            # 5. Run evaluate_predictions.py
            eval_out_dir = tmp_path / "evaluation_run"
            eval_preds.main([
                "--pred-dir", str(pred_out_dir / "test_predictions"),
                "--data-root", str(data_root),
                "--list-dir", str(split_dir),
                "--output-dir", str(eval_out_dir),
                "--dataset", "myops380",
                "--no-compute-distance",  # Fast execution for synthetic small volume
            ])

            per_case_csv = eval_out_dir / "per_case.csv"
            metrics_json = eval_out_dir / "metrics.json"
            self.assertTrue(per_case_csv.is_file(), f"Missing {per_case_csv}")
            self.assertTrue(metrics_json.is_file(), f"Missing {metrics_json}")

            with open(metrics_json, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            self.assertIn("summary", metrics_data)
            self.assertIn("scar", metrics_data["summary"])
            self.assertIn("edema_inclusive", metrics_data["summary"])


if __name__ == "__main__":
    unittest.main()
