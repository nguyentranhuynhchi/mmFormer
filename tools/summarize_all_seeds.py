"""tools/summarize_all_seeds.py
==============================
Summarize all 5 seeds ([1234, 1098, 1024, 2026, 6868]) for mmFormer benchmark on MyoPS++ and MyoPS380.
- Computes Mean ± Std (ddof=1) for 8 columns:
    Scar Dice, Scar HD, Edema Inc Dice, Edema Inc HD, Myo Dice, Myo HD, Avg Path Dice, Avg Path HD.
- Prints Excel-ready tab-separated values (both dot and comma decimals).
- Reads and prints Table 1: Computational efficiency metrics (A100 FP32) of Seed 1234.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

SEEDS = [1234, 1098, 1024, 2026, 6868]


def summarize_seeds(
    base_dir: str | Path = "outputs/runs",
    dataset: str = "myopspp",
) -> None:
    base_dir = Path(base_dir).resolve()
    dist_unit = "mm" if dataset in ("myopspp", "myopspp_unet2d_roi128_1mm", "myops_plus_plus", "myops++") else "voxel"

    rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print(f"=== TỔNG HỢP KẾT QUẢ 5 SEEDS MMFORMER ({dataset.upper()} - {dist_unit.upper()}) ===")
    print("=" * 90)

    for seed in SEEDS:
        # Check potential candidate run folders
        cand_dirs = [
            base_dir / f"mmformer_seed{seed}",
            base_dir / f"{dataset}_seed{seed}",
            base_dir / f"myopspp_seed{seed}",
            base_dir / f"myops380_seed{seed}",
            base_dir / f"seed_{seed}",
            base_dir / f"seed{seed}",
        ]
        metric_file: Optional[Path] = None
        for cd in cand_dirs:
            mf = cd / "evaluation" / "metrics.json"
            if mf.exists():
                metric_file = mf
                break
            # Also check direct run folder metrics.json
            mf_dir = cd / "metrics.json"
            if mf_dir.exists():
                metric_file = mf_dir
                break

        if metric_file is None:
            print(f"[!] Seed {seed}: Chưa tìm thấy metrics.json trong các thư mục ứng viên: {cand_dirs[0]}")
            continue

        with open(metric_file, "r", encoding="utf-8") as f:
            d = json.load(f)

        # Support both nested summary dictionary and root-mirrored region dictionary
        summary_node = d.get("summary", d)
        s = summary_node.get("scar", {})
        e = summary_node.get("edema_inclusive", {})
        m = summary_node.get("myocardial_ring", {})

        hd_key = f"mean_hd95_{dist_unit}"
        s_d = (s.get("mean_dice") or 0.0) * 100
        s_hd = s.get(hd_key) or s.get("mean_hd95") or s.get("mean_hd95_voxel") or 0.0
        e_d = (e.get("mean_dice") or 0.0) * 100
        e_hd = e.get(hd_key) or e.get("mean_hd95") or e.get("mean_hd95_voxel") or 0.0
        m_d = (m.get("mean_dice") or 0.0) * 100
        m_hd = m.get(hd_key) or m.get("mean_hd95") or m.get("mean_hd95_voxel") or 0.0

        avg_d = (s_d + e_d) / 2.0
        avg_hd = (s_hd + e_hd) / 2.0

        rows.append({
            "seed": seed,
            "s_d": s_d, "s_hd": s_hd,
            "e_d": e_d, "e_hd": e_hd,
            "m_d": m_d, "m_hd": m_hd,
            "avg_d": avg_d, "avg_hd": avg_hd,
        })

    # Print Table
    print(f"{'Seed':<8} | {'Scar Dice':<10} {'HD':<8} | {'Edema Dice':<10} {'HD':<8} | {'Myo Dice':<10} {'HD':<8} | {'Avg Path':<10} {'HD':<8}")
    print("-" * 90)
    for r in rows:
        print(f"{r['seed']:<8} | {r['s_d']:<9.2f}% {r['s_hd']:<7.2f} | {r['e_d']:<9.2f}% {r['e_hd']:<7.2f} | {r['m_d']:<9.2f}% {r['m_hd']:<7.2f} | {r['avg_d']:<9.2f}% {r['avg_hd']:<7.2f}")

    if rows:
        print("-" * 90)
        means = {k: float(np.mean([r[k] for r in rows])) for k in ["s_d", "s_hd", "e_d", "e_hd", "m_d", "m_hd", "avg_d", "avg_hd"]}
        stds = {k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0 for k in ["s_d", "s_hd", "e_d", "e_hd", "m_d", "m_hd", "avg_d", "avg_hd"]}
        print(f"{'Avg':<8} | {means['s_d']:<9.2f}% {means['s_hd']:<7.2f} | {means['e_d']:<9.2f}% {means['e_hd']:<7.2f} | {means['m_d']:<9.2f}% {means['m_hd']:<7.2f} | {means['avg_d']:<9.2f}% {means['avg_hd']:<7.2f}")
        print(f"{'Std':<8} | {stds['s_d']:<9.2f}% {stds['s_hd']:<7.2f} | {stds['e_d']:<9.2f}% {stds['e_hd']:<7.2f} | {stds['m_d']:<9.2f}% {stds['m_hd']:<7.2f} | {stds['avg_d']:<9.2f}% {stds['avg_hd']:<7.2f}")

        print("\n" + "=" * 90)
        print("📋 KHỐI DỮ LIỆU COPY THẲNG VÀO EXCEL (CHUẨN DẤU CHẤM .):")
        print("=" * 90)
        for r in rows:
            print(f"{r['s_d']:.2f}\t{r['s_hd']:.2f}\t{r['e_d']:.2f}\t{r['e_hd']:.2f}\t{r['m_d']:.2f}\t{r['m_hd']:.2f}\t{r['avg_d']:.2f}\t{r['avg_hd']:.2f}")
        avg_line_dot = f"{means['s_d']:.2f} ± {stds['s_d']:.2f}\t{means['s_hd']:.2f} ± {stds['s_hd']:.2f}\t{means['e_d']:.2f} ± {stds['e_d']:.2f}\t{means['e_hd']:.2f} ± {stds['e_hd']:.2f}\t{means['m_d']:.2f} ± {stds['m_d']:.2f}\t{means['m_hd']:.2f} ± {stds['m_hd']:.2f}\t{means['avg_d']:.2f} ± {stds['avg_d']:.2f}\t{means['avg_hd']:.2f} ± {stds['avg_hd']:.2f}"
        print(avg_line_dot)

        print("\n" + "=" * 90)
        print("📋 KHỐI DỮ LIỆU COPY THẲNG VÀO EXCEL (CHUẨN DẤU PHẨY ,):")
        print("=" * 90)
        for r in rows:
            line_comma = f"{r['s_d']:.2f}\t{r['s_hd']:.2f}\t{r['e_d']:.2f}\t{r['e_hd']:.2f}\t{r['m_d']:.2f}\t{r['m_hd']:.2f}\t{r['avg_d']:.2f}\t{r['avg_hd']:.2f}"
            print(line_comma.replace(".", ","))
        avg_line_comma = avg_line_dot.replace(".", ",")
        print(avg_line_comma)
        print("=" * 90 + "\n")


def print_efficiency_table(run_dir: str | Path | None = None) -> None:
    """Print Table 1: Computational Efficiency metrics on NVIDIA A100 FP32 for Seed 1234."""
    cand_dirs = [
        Path(run_dir) if run_dir else None,
        Path("outputs/runs/mmformer_seed1234"),
        Path("outputs/runs/myopspp_seed1234"),
        Path("outputs/runs/mmformer"),
        Path("/content/mmFormer/outputs/runs/mmformer"),
        Path("/content/drive/MyDrive/MyoPS_Results/mmFormer/runs/myopspp_seed1234"),
    ]
    eff_file = None
    for c in cand_dirs:
        if c and (c / "efficiency_metrics.json").exists():
            eff_file = c / "efficiency_metrics.json"
            break
        if c and c.is_file() and c.name == "efficiency_metrics.json":
            eff_file = c
            break

    if not eff_file:
        # Recursive glob search in outputs
        for root in [Path("outputs"), Path("runs"), Path(".")]:
            if root.exists():
                found = list(root.glob("**/efficiency_metrics.json"))
                if found:
                    eff_file = found[0]
                    break

    if not eff_file or not eff_file.is_file():
        print(f"⚠️ Chưa tìm thấy file efficiency_metrics.json để in bảng hiệu năng tính toán.")
        return

    with open(eff_file, "r", encoding="utf-8") as f:
        eff_data = json.load(f)

    ex = eff_data.get("excel_summary", {})
    print("\n" + "=" * 85)
    print("=== BẢNG 1: HIỆU NĂNG TÍNH TOÁN MMFORMER (NVIDIA A100 FP32) ===")
    print("=" * 85)
    for k, v in ex.items():
        print(f"  {k:<32}: {v}")

    params = ex.get("Trainable Params (M)", 34.03)
    flops = ex.get("FLOPs (G)", 17.61)
    lat = ex.get("Inference Latency (ms/slice)", 0.0)
    vram_inf = ex.get("Inference Peak VRAM (MB)", 0.0)
    time_ep = ex.get("Training Time / Epoch (s)", 0.0)
    tot_time = ex.get("Training Total Time (hours)", 0.0)
    vram_tr = ex.get("Training Peak VRAM (MB)", 0.0)

    print("\n📋 DÒNG COPY EXCEL (BẢNG 1, CLICK Ô TRAINABLE PARAMS DÒNG mmFormer):")
    print("-" * 85)
    line_dot = f"{params}M\t{flops}\t{lat}\t{vram_inf}\t{time_ep}\t{tot_time}\t{vram_tr}"
    print("Chuẩn dấu chấm (.):")
    print(line_dot)
    print("Chuẩn dấu phẩy (,):")
    print(line_dot.replace(".", ","))
    print("=" * 85 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize 5 seeds and print efficiency table for mmFormer")
    parser.add_argument("--base-dir", "--base_dir", default="outputs/runs", help="Base runs directory")
    parser.add_argument("--dataset", default="myopspp", choices=["myopspp", "myops380"], help="Dataset benchmark")
    parser.add_argument("--print-efficiency", "--print_efficiency", action="store_true", default=True, help="Print efficiency table")
    args = parser.parse_args()

    summarize_seeds(args.base_dir, args.dataset)
    if args.print_efficiency:
        print_efficiency_table(args.base_dir)


if __name__ == "__main__":
    main()
