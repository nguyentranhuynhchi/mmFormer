"""Comprehensive unit and CLI integration tests for mmFormer computational efficiency benchmarking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmformer.mmformer import Model
from mmformer.evaluate_myops import format_master_clinical_table
from mmformer.train_myops import parse_args
from mmformer.utils.efficiency import (
    BYTES_PER_GB,
    BYTES_PER_MB,
    _bytes_to_mb_gb,
    build_efficiency_payload,
    check_rule_compliance,
    count_parameters,
    format_efficiency_report,
    measure_flops,
    measure_inference_efficiency,
    save_efficiency_report,
    summarize_training_efficiency,
)


class TestEfficiencySuite(unittest.TestCase):
    def test_binary_memory_units(self) -> None:
        """Verify binary 1024-based conversions match PyTorch and NVIDIA standards."""
        self.assertEqual(BYTES_PER_MB, 1024.0**2)
        self.assertEqual(BYTES_PER_GB, 1024.0**3)

        mb, gb = _bytes_to_mb_gb(int(BYTES_PER_GB))
        self.assertEqual(mb, 1024.0)
        self.assertEqual(gb, 1.0)

        mb_half, gb_half = _bytes_to_mb_gb(int(512 * BYTES_PER_MB))
        self.assertEqual(mb_half, 512.0)
        self.assertEqual(gb_half, 0.5)

    def test_mmformer_parameter_counts(self) -> None:
        """Verify mmFormer 2D model param counts match exact 24,050,336 reference."""
        model = Model(num_cls=4, num_modals=3)
        params_info = count_parameters(model)

        self.assertEqual(params_info["total_params"], 24_050_336)
        self.assertEqual(params_info["trainable_params"], 24_050_336)
        self.assertEqual(params_info["trainable_params_excel"], "24.05M")
        self.assertEqual(params_info["total_params_excel"], "24.05M")
        self.assertAlmostEqual(params_info["trainable_params_m"], 24.0503, places=3)

    def test_mmformer_flops_and_inference_cpu(self) -> None:
        """Verify FlopCounterMode counts theoretical FLOPs on (1, 3, 128, 128) FP32 and inference preserves train state."""
        model = Model(num_cls=4, num_modals=3)
        model.train()

        flops_info = measure_flops(model, input_shape=(1, 3, 128, 128), device=torch.device("cpu"))
        self.assertEqual(flops_info["counter"], "torch.utils.flop_counter.FlopCounterMode")
        self.assertEqual(flops_info["flops_total"], 5_363_138_560)
        self.assertAlmostEqual(flops_info["flops_g"], 5.3631, places=3)
        self.assertEqual(flops_info["flops_g_excel"], 5.36)

        # Ensure model training state was restored after measuring FLOPs
        self.assertTrue(model.training)

        inf_info = measure_inference_efficiency(
            model, input_shape=(1, 3, 128, 128), device=torch.device("cpu"), warmup=2, repeats=3
        )
        self.assertTrue(model.training)
        self.assertEqual(inf_info["batch_size"], 1)
        self.assertGreater(inf_info["latency_ms_per_slice"], 0.0)
        self.assertEqual(inf_info["peak_vram_mb"], 0.0)
        self.assertEqual(inf_info["peak_vram_gb"], 0.0)

    def test_warmup_epoch_skipping(self) -> None:
        """Verify training timing skips first 5 warmup epochs and averages epochs 6..300."""
        epoch_times = [100.0] * 5 + [10.0] * 295
        peak_bytes = int(2500 * BYTES_PER_MB)
        summary = summarize_training_efficiency(
            epoch_times_s=epoch_times,
            total_wall_clock_s=sum(epoch_times) + 120.0,
            peak_vram_bytes=peak_bytes,
            planned_epochs=300,
            warmup_epochs=5,
            peak_reserved_bytes=int(3.0 * BYTES_PER_GB),
        )
        self.assertEqual(summary["warmup_epochs_skipped"], 5)
        self.assertEqual(summary["measured_epoch_range"], "6..300")
        self.assertEqual(summary["time_per_epoch_s"], 10.0)
        self.assertEqual(summary["epoch_times_1_to_5_s"], [100.0] * 5)
        self.assertAlmostEqual(summary["total_time_hours"], sum(epoch_times) / 3600.0, places=3)
        self.assertEqual(summary["peak_vram_mb"], 2500.0)
        self.assertAlmostEqual(summary["peak_vram_gb"], 2500.0 / 1024.0, places=4)
        self.assertEqual(summary["peak_vram_reserved_mb"], 3072.0)
        self.assertEqual(summary["peak_vram_reserved_gb"], 3.0)

        # Edge case: training with <= 5 epochs handles gracefully
        short_times = [15.0, 14.0]
        summary_short = summarize_training_efficiency(
            epoch_times_s=short_times,
            total_wall_clock_s=29.5,
            peak_vram_bytes=int(1024 * BYTES_PER_MB),
            planned_epochs=2,
            warmup_epochs=5,
        )
        self.assertEqual(summary_short["warmup_epochs_skipped"], 0)
        self.assertEqual(summary_short["measured_epoch_range"], "1..2")
        self.assertEqual(summary_short["time_per_epoch_s"], 14.5)

    def test_rule_compliance_validation(self) -> None:
        """Verify strict rule compliance checks against NVIDIA A100, FP32, workers=0, pin_memory=True."""
        # Non-compliant CPU run
        ok_cpu, warnings_cpu, hw_cpu = check_rule_compliance(
            device=torch.device("cpu"),
            amp_mode="none",
            num_workers=0,
            pin_memory=True,
            batch_size=16,
            planned_epochs=300,
        )
        self.assertFalse(ok_cpu)
        self.assertTrue(any("CPU" in w for w in warnings_cpu))
        self.assertEqual(hw_cpu["precision"], "fp32")

        # Non-compliant flags
        ok_bad, warnings_bad, _ = check_rule_compliance(
            device=torch.device("cpu"),
            amp_mode="auto",
            num_workers=4,
            pin_memory=False,
            batch_size=8,
            planned_epochs=100,
        )
        self.assertFalse(ok_bad)
        self.assertGreaterEqual(len(warnings_bad), 5)

    def test_build_payload_and_format_report(self) -> None:
        """Verify JSON payload assembly and terminal report output with dual MB / GB units."""
        payload = build_efficiency_payload(
            model_name="mmFormer",
            model_id="mmformer",
            seed=1234,
            params_info={
                "total_params": 24050336,
                "trainable_params": 24050336,
                "total_params_m": 24.0503,
                "trainable_params_m": 24.0503,
                "trainable_params_excel": "24.05M",
                "total_params_excel": "24.05M",
            },
            flops_info={"flops_total": 5363138560, "flops_g": 5.3631},
            inference_info={
                "latency_ms_per_slice": 4.12,
                "warmup_runs": 30,
                "measure_runs": 100,
                "peak_vram_mb": 512.0,
                "peak_vram_gb": 0.5,
                "peak_vram_reserved_mb": 600.0,
            },
            training_info={
                "planned_epochs": 300,
                "completed_epochs": 300,
                "warmup_epochs_skipped": 5,
                "measured_epoch_range": "6..300",
                "time_per_epoch_s": 8.50,
                "total_time_seconds": 2550.0,
                "total_time_hours": 0.7083,
                "total_wall_clock_hours": 0.72,
                "peak_vram_mb": 2400.0,
                "peak_vram_gb": 2.3438,
                "peak_vram_reserved_mb": 2800.0,
            },
            rule_compliant=True,
            compliance_warnings=[],
            hardware_info={
                "gpu_name": "NVIDIA A100-SXM4-40GB",
                "visible_gpu_count": 1,
                "pytorch_version": "2.4.0",
                "cuda_version": "12.1",
                "precision": "fp32",
                "amp_mode": "none",
                "num_workers": 0,
                "pin_memory": True,
                "batch_size": 16,
            },
        )
        self.assertEqual(payload["schema_version"], "efficiency_metrics_v1")
        self.assertEqual(payload["excel_summary"]["Trainable Params (M)"], 24.05)
        self.assertEqual(payload["excel_summary"]["FLOPs (G)"], 5.36)

        report_txt = format_efficiency_report(payload)
        self.assertIn("24.05M", report_txt)
        self.assertIn("5.36 G", report_txt)
        self.assertIn("512.00 MB (0.5000 GB)", report_txt)
        self.assertIn("2400.00 MB (2.3438 GB)", report_txt)
        self.assertIn("EXCEL PASTE ROW", report_txt)

    def test_save_efficiency_report_atomic(self) -> None:
        """Verify atomic writing to target JSON and summary text files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            json_target1 = tmp_path / "sub1" / "metrics.json"
            json_target2 = tmp_path / "sub2" / "metrics.json"
            txt_target = tmp_path / "logs" / "summary.txt"

            payload = build_efficiency_payload(
                model_name="mmFormer",
                model_id="mmformer",
                seed=1234,
                params_info={"trainable_params_m": 24.05, "total_params_m": 24.05, "trainable_params": 24050336, "total_params": 24050336},
                flops_info={"flops_g": 5.36, "flops_total": 5363138560},
                inference_info={"latency_ms_per_slice": 4.0, "peak_vram_mb": 100.0, "peak_vram_gb": 0.1, "warmup_runs": 30, "measure_runs": 100, "peak_vram_reserved_mb": 120.0},
                training_info=None,
                rule_compliant=False,
                compliance_warnings=["CPU test"],
                hardware_info={"gpu_name": "cpu", "visible_gpu_count": 0, "pytorch_version": "2.4", "cuda_version": None, "precision": "fp32", "amp_mode": "none", "num_workers": 0, "pin_memory": False, "batch_size": 16},
            )

            save_efficiency_report(payload, [json_target1, json_target2], [txt_target])

            self.assertTrue(json_target1.is_file())
            self.assertTrue(json_target2.is_file())
            self.assertTrue(txt_target.is_file())

            loaded = json.loads(json_target1.read_text(encoding="utf-8"))
            self.assertEqual(loaded["model_id"], "mmformer")
            self.assertEqual(loaded["excel_summary"]["Trainable Params (M)"], 24.05)

    def test_cli_argument_aliases(self) -> None:
        """Verify hyphen and underscore CLI argument aliases parse identically."""
        sys_argv_hyphen = [
            "train_myops.py",
            "--data-root", "E:/data",
            "--list-dir", "E:/splits",
            "--run-root", "E:/runs",
            "--run-name", "test_run",
            "--batch-size", "16",
            "--weight-decay", "0.0005",
            "--num-workers", "0",
            "--pin-memory",
            "--measure-efficiency",
            "--efficiency-output", "E:/out.json",
            "--efficiency-warmup", "10",
            "--efficiency-repeats", "20",
        ]
        sys_argv_underscore = [
            "train_myops.py",
            "--data_root", "E:/data",
            "--list_dir", "E:/splits",
            "--run_root", "E:/runs",
            "--run_name", "test_run",
            "--batch_size", "16",
            "--weight_decay", "0.0005",
            "--num_workers", "0",
            "--pin_memory",
            "--measure_efficiency",
            "--efficiency_output", "E:/out.json",
            "--efficiency_warmup", "10",
            "--efficiency_repeats", "20",
        ]

        old_argv = sys.argv
        try:
            sys.argv = sys_argv_hyphen
            args_hyphen = parse_args()
            sys.argv = sys_argv_underscore
            args_underscore = parse_args()

            self.assertEqual(args_hyphen.data_root, args_underscore.data_root)
            self.assertEqual(args_hyphen.list_dir, args_underscore.list_dir)
            self.assertEqual(args_hyphen.run_root, args_underscore.run_root)
            self.assertEqual(args_hyphen.run_name, args_underscore.run_name)
            self.assertEqual(args_hyphen.batch_size, args_underscore.batch_size)
            self.assertEqual(args_hyphen.weight_decay, args_underscore.weight_decay)
            self.assertEqual(args_hyphen.num_workers, args_underscore.num_workers)
            self.assertEqual(args_hyphen.pin_memory, args_underscore.pin_memory)
            self.assertEqual(args_hyphen.measure_efficiency, args_underscore.measure_efficiency)
            self.assertEqual(args_hyphen.efficiency_output, args_underscore.efficiency_output)
        finally:
            sys.argv = old_argv

    def test_evaluate_master_benchmark_clinical_table(self) -> None:
        """Verify format_master_clinical_table outputs proper headers, regions, and summary dice."""
        mock_summary = {
            "normal_myocardium": {"mean_dice": 0.8523, "mean_iou": 0.7431, "mean_hd95_voxel": 1.25, "mean_asd_voxel": 0.45},
            "edema": {"mean_dice": 0.6812, "mean_iou": 0.5210, "mean_hd95_voxel": 5.10, "mean_asd_voxel": 1.80},
            "scar": {"mean_dice": 0.7345, "mean_iou": 0.5820, "mean_hd95_voxel": 4.20, "mean_asd_voxel": 1.50},
            "edema_inclusive": {"mean_dice": 0.7410, "mean_iou": 0.5910, "mean_hd95_voxel": 4.60, "mean_asd_voxel": 1.65},
            "myocardial_ring": {"mean_dice": 0.8650, "mean_iou": 0.7620, "mean_hd95_voxel": 1.10, "mean_asd_voxel": 0.40},
            "avg_pathology_dice": 0.7078,
            "avg_pathology_inclusive_dice": 0.7378,
        }
        table_txt = format_master_clinical_table(mock_summary)
        self.assertIn("Region                 | Dice Mean  | IoU Mean   | HD95 (vox)   | ASD (vox)", table_txt)
        self.assertIn("normal_myocardium", table_txt)
        self.assertIn("edema", table_txt)
        self.assertIn("scar", table_txt)
        self.assertIn("edema_inclusive", table_txt)
        self.assertIn("myocardial_ring", table_txt)
        self.assertIn("avg_pathology_dice", table_txt)
        self.assertIn("0.7078", table_txt)
        self.assertIn("avg_pathology_inclusive_dice", table_txt)
        self.assertIn("0.7378", table_txt)

    def test_cli_train_myops_end_to_end_and_resume(self) -> None:
        """Integration test: --efficiency-only mode and smoke training with resume on CPU."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_root = tmp_path / "Processed_data"
            list_dir = tmp_path / "splits"
            run_root = tmp_path / "runs"
            eff_out = tmp_path / "eff" / "efficiency_metrics.json"
            list_dir.mkdir(parents=True)

            for mod in ("bSSFP", "LGE", "T2w"):
                (data_root / mod / "train_npz").mkdir(parents=True)

            train_lines: list[str] = []
            for idx in range(2):
                name = f"case9999_slice{idx:03d}"
                train_lines.append(name)
                label = np.zeros((128, 128), dtype=np.uint8)
                label[40:80, 40:80] = 1
                label[50:65, 50:65] = 2
                label[65:75, 50:65] = 3
                for mod in ("bSSFP", "LGE", "T2w"):
                    img = np.random.rand(128, 128).astype(np.float32)
                    np.savez(data_root / mod / "train_npz" / f"{name}.npz", image=img, label=label)

            (list_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

            # 1) Test --efficiency-only
            cmd_only = [
                sys.executable,
                str(REPO_ROOT / "train_myops.py"),
                "--data-root", str(data_root),
                "--list-dir", str(list_dir),
                "--run-root", str(run_root),
                "--run-name", "smoke_eff_only",
                "--epochs", "1",
                "--batch-size", "2",
                "--amp", "none",
                "--num-workers", "0",
                "--device", "cpu",
                "--efficiency-output", str(eff_out),
                "--efficiency-warmup", "1",
                "--efficiency-repeats", "2",
                "--efficiency-only",
            ]
            res = subprocess.run(cmd_only, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
            self.assertTrue(eff_out.is_file())
            payload = json.loads(eff_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["excel_summary"]["Trainable Params (M)"], 24.05)
            self.assertEqual(payload["excel_summary"]["FLOPs (G)"], 5.36)
            self.assertIsNone(payload["training"])

            # 2) Test 2-epoch training with --amp none --num-workers 0
            eff_out_train = tmp_path / "eff_train" / "efficiency_metrics.json"
            base_cmd = [
                sys.executable,
                str(REPO_ROOT / "train_myops.py"),
                "--data-root", str(data_root),
                "--list-dir", str(list_dir),
                "--run-root", str(run_root),
                "--run-name", "smoke_train",
                "--batch-size", "2",
                "--epochs", "2",
                "--amp", "none",
                "--num-workers", "0",
                "--device", "cpu",
                "--efficiency-output", str(eff_out_train),
                "--efficiency-warmup", "1",
                "--efficiency-repeats", "2",
            ]
            r1 = subprocess.run(base_cmd, capture_output=True, text=True)
            self.assertEqual(r1.returncode, 0, msg=f"STDOUT:\n{r1.stdout}\nSTDERR:\n{r1.stderr}")
            ckpt_file = run_root / "smoke_train" / "checkpoints" / "mmformer_seed1234.pt"
            self.assertTrue(ckpt_file.is_file())
            self.assertTrue(eff_out_train.is_file())
            train_payload = json.loads(eff_out_train.read_text(encoding="utf-8"))
            self.assertEqual(train_payload["training"]["completed_epochs"], 2)
            self.assertEqual(len(train_payload["training"]["all_epoch_times_s"]), 2)
            self.assertTrue((run_root / "smoke_train" / "efficiency_metrics.json").is_file())
            self.assertTrue((run_root / "smoke_train" / "logs" / "efficiency_summary.txt").is_file())

            # 3) Simulate resume from epoch 1
            ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            first_epoch_time = ckpt["epoch_times"][0]
            ckpt["epoch"] = 1
            ckpt["completed_epochs"] = 1
            ckpt["training_complete"] = False
            ckpt["epoch_times"] = [first_epoch_time]
            torch.save(ckpt, ckpt_file)
            eff_out_train.unlink()

            r2 = subprocess.run(base_cmd + ["--resume", str(ckpt_file)], capture_output=True, text=True)
            self.assertEqual(r2.returncode, 0, msg=f"STDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}")
            self.assertTrue(eff_out_train.is_file())
            resumed_payload = json.loads(eff_out_train.read_text(encoding="utf-8"))
            self.assertEqual(resumed_payload["training"]["completed_epochs"], 2)
            self.assertEqual(len(resumed_payload["training"]["all_epoch_times_s"]), 2)
            self.assertAlmostEqual(
                resumed_payload["training"]["all_epoch_times_s"][0],
                round(first_epoch_time, 4),
                places=4,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
