"""End-to-End Test Suite for mmFormer Benchmark Pipeline."""
import hashlib
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.dataset_myops import RandomGenerator, canonicalize_label
from networks.mmformer2d import mmFormer2D
from utils.criterions2d import compute_mmformer_loss, dice_loss_2d, softmax_weighted_loss_2d
from metrics.surface_distance import dataset_rows, summarize_dataset_rows


def test_split_hashes():
    print("👉 [1/5] Testing Split Hashes SHA-256...")
    expected = {
        "train.txt": "16248ec3630bb0493fb421aab5ffb4dc9b1d0fd49c8614d69d3098a92305f919",
        "test_vol.txt": "3237b31411833cedcb3ecd86d36edae356bd3f880d3fcfd915066f1c850d9cbc"
    }
    for fname, exp_hash in expected.items():
        fpath = REPO_ROOT / "lists" / "lists_MyoPS380" / fname
        assert fpath.is_file(), f"File {fpath} does not exist!"
        with open(fpath, "rb") as fp:
            actual = hashlib.sha256(fp.read()).hexdigest()
        assert actual == exp_hash, f"Hash mismatch for {fname}: got {actual}"
    print("   ✅ SHA-256 check passed perfectly!")


def test_random_generator():
    print("👉 [2/5] Testing Data Augmentation (RandomGenerator)...")
    augmenter = RandomGenerator(output_size=(128, 128))
    dummy_sample = {
        "image": np.random.rand(128, 128).astype(np.float32),
        "image1": np.random.rand(128, 128).astype(np.float32),
        "image2": np.random.rand(128, 128).astype(np.float32),
        "label": np.random.randint(0, 4, (128, 128), dtype=np.uint8),
        "case_name": "test_case_001"
    }
    out = augmenter(dummy_sample)
    assert out["image"].shape == (3, 128, 128), f"Wrong image shape: {out['image'].shape}"
    assert out["label"].shape == (128, 128), f"Wrong label shape: {out['label'].shape}"
    assert out["one_hot_label"].shape == (4, 128, 128), f"Wrong one-hot shape: {out['one_hot_label'].shape}"
    assert out["mask"].shape == (3,), f"Wrong mask shape: {out['mask'].shape}"
    print("   ✅ Augmentation & tensor shapes verified!")


def test_model_forward():
    print("👉 [3/5] Testing mmFormer2D Architecture & Forward Pass...")
    model = mmFormer2D(num_cls=4, num_modals=3)
    
    # 1. Training mode
    model.train()
    model.is_training = True
    x = torch.randn(2, 3, 128, 128)
    mask = torch.tensor([[True, True, True], [True, False, True]])
    fuse_pred, sep_preds, prm_preds = model(x, mask)

    assert fuse_pred.shape == (2, 4, 128, 128), f"Wrong fuse_pred shape: {fuse_pred.shape}"
    assert len(sep_preds) == 3, f"Expected 3 separate preds, got {len(sep_preds)}"
    assert len(prm_preds) == 4, f"Expected 4 deep supervision preds, got {len(prm_preds)}"

    # 2. Inference mode
    model.eval()
    model.is_training = False
    with torch.no_grad():
        inf_pred = model(x)
    assert inf_pred.shape == (2, 4, 128, 128), f"Wrong inference shape: {inf_pred.shape}"
    print(f"   ✅ Model forward passed! Parameters: {sum(p.numel() for p in model.parameters()):,}")


def test_loss_and_backward():
    print("👉 [4/5] Testing Native mmFormer Objective & Backward Pass...")
    model = mmFormer2D(num_cls=4, num_modals=3)
    model.train()
    model.is_training = True

    x = torch.randn(2, 3, 128, 128)
    mask = torch.tensor([[True, True, True], [True, True, True]])
    target_labels = torch.randint(0, 4, (2, 128, 128))
    target_one_hot = F.one_hot(target_labels, num_classes=4).permute(0, 3, 1, 2).float()

    fuse_pred, sep_preds, prm_preds = model(x, mask)
    loss, fuse_loss, sep_loss, prm_loss, fuse_cross = compute_mmformer_loss(
        fuse_pred, sep_preds, prm_preds, target_one_hot, num_cls=4
    )

    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss is not positive!"

    loss.backward()

    # Check gradients
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients computed!"
    print(f"   ✅ Loss backward passed! Total loss: {loss.item():.4f} (Fuse: {fuse_loss.item():.4f}, Sep: {sep_loss.item():.4f}, PRM: {prm_loss.item():.4f})")


def test_evaluator_metrics():
    print("👉 [5/5] Testing Surface Distance Evaluator (myops380_voxel_v3)...")
    pred = np.zeros((128, 128, 6), dtype=np.uint8)
    target = np.zeros((128, 128, 6), dtype=np.uint8)

    # Put sample lesions
    target[40:60, 40:60, 2:4] = 1  # Myo
    target[45:50, 45:50, 2:4] = 2  # Edema
    target[50:55, 50:55, 2:4] = 3  # Scar

    pred[40:60, 40:60, 2:4] = 1
    pred[45:50, 45:50, 2:4] = 2
    pred[50:55, 50:55, 2:4] = 3

    rows = dataset_rows(pred, target, "case_test_01", dataset_id="myops380")
    summary = summarize_dataset_rows(rows, dataset_id="myops380")

    assert summary["scar"]["mean_dice"] == 1.0, f"Expected 1.0, got {summary['scar']['mean_dice']}"
    assert summary["edema_inclusive"]["mean_dice"] == 1.0
    assert summary["avg_pathology_inclusive_dice"] == 1.0
    print("   ✅ Evaluator passed with 100% precision on perfect prediction!")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧪 RUNNING COMPREHENSIVE MMFORMER PIPELINE AUDIT")
    print("=" * 80)
    test_split_hashes()
    test_random_generator()
    test_model_forward()
    test_loss_and_backward()
    test_evaluator_metrics()
    print("\n" + "=" * 80)
    print("🎉 ALL 5/5 UNIT & INTEGRATION TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 80 + "\n")
