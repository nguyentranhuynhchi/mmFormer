"""MyoPS380 and MyoPS++ dataset and I-MMSeg augmentation for mmFormer.

Protocols:
  - MyoPS380: myops380_train304_test76_last_v1 (304 train patients / 1,972 slices, 76 test volumes)
  - MyoPS++:  myopspp_bc80_train64_test16_last_v1 (64 train patients / 296 slices, 16 test volumes)

Target Adapter & Double-Swap Prevention:
  - MyoPS380: applies LUT [0, 1, 3, 2] converting legacy (0:bg, 1:normal, 2:scar, 3:edema)
              to canonical (0:bg, 1:normal, 2:edema, 3:scar).
  - MyoPS++: disk cache is ALREADY canonical -> Identity mapping (NO LUT swap).

Distance Metrics:
  - MyoPS380: Voxel distance (spacing=None, metric_distance_unit='voxel')
  - MyoPS++: Physical mm with dynamic spacing [1.0, 1.0, sz] (metric_distance_unit='mm')

Stacks the 3 modalities [bSSFP, LGE, T2w] into a (3, H, W) tensor for the multi-branch mmFormer.
"""
from __future__ import annotations

import os
from pathlib import Path
import random
import re
from typing import Any, Sequence

import h5py
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_NAMES = ("background", "normal_myocardium", "edema", "scar")
CANONICAL_LABEL_ORDER = "background_normal_edema_scar"
LEGACY_LABEL_ORDER = "background_normal_scar_edema"

LABEL_ORDERS = {
    "canonical": CANONICAL_LABEL_ORDER,
    "legacy": LEGACY_LABEL_ORDER,
    "identity": CANONICAL_LABEL_ORDER,
    "identity_mapping": CANONICAL_LABEL_ORDER,
    CANONICAL_LABEL_ORDER: CANONICAL_LABEL_ORDER,
    LEGACY_LABEL_ORDER: LEGACY_LABEL_ORDER,
}

MYOPS380_EXPECTED_TRAIN_SLICES = 1972
MYOPS380_EXPECTED_TRAIN_PATIENTS = 304
MYOPS380_EXPECTED_TEST_VOLUMES = 76

MYOPSPP_EXPECTED_TRAIN_SLICES = 296
MYOPSPP_EXPECTED_TRAIN_PATIENTS = 64
MYOPSPP_EXPECTED_TEST_VOLUMES = 16


def normalize_dataset_name(dataset: str | None) -> str:
    """Normalize dataset identifier string to 'myops380' or 'myopspp'."""
    if dataset is None:
        return "myops380"
    ds = str(dataset).strip().lower()
    if not ds:
        return "myops380"
    if ds in ("myopspp", "myopspp_unet2d_roi128_1mm", "myops_plus_plus", "myops++"):
        return "myopspp"
    if ds in ("myops380", "myops_380", "default"):
        return "myops380"
    raise ValueError(f"Unknown dataset identifier: {dataset!r}; expected 'myops380' or 'myopspp'")


def resolve_default_split_dir(dataset: str | None) -> Path:
    """Resolves default split list directory based on dataset name."""
    norm_ds = normalize_dataset_name(dataset)
    if norm_ds == "myopspp":
        cand = PROJECT_ROOT / "splits" / "myopspp_no_val"
    else:
        cand = PROJECT_ROOT / "splits" / "myops380_no_val"
    return cand


def resolve_label_order(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return LABEL_ORDERS[str(value).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown label order {value!r}; use 'canonical', 'legacy', or 'identity'.") from exc


def canonicalize_label(
    label: np.ndarray,
    order: str = "legacy",
    dataset: str = "myops380",
) -> np.ndarray:
    """Convert label to canonical format [0:bg, 1:normal, 2:edema, 3:scar].

    - If dataset == 'myops380' (or order == 'legacy'):
        Applies LUT [0, 1, 3, 2] to convert legacy -> canonical.
    - If dataset == 'myopspp' (or order in ('canonical', 'identity')):
        Identity mapping (keeps array unchanged, NEVER swaps).
    """
    label = np.asarray(label)
    if not np.isfinite(label).all() or not np.equal(label, np.round(label)).all():
        raise ValueError("Segmentation masks must contain finite integer class IDs.")
    if label.size == 0 or label.min() < 0 or label.max() > 3:
        raise ValueError(f"Expected class IDs 0..3; found {np.unique(label).tolist()}.")
    result = label.astype(np.uint8, copy=True)

    norm_ds = normalize_dataset_name(dataset)
    if norm_ds == "myopspp":
        # MyoPS++ disk cache is ALREADY canonical; strictly preserve identity mapping!
        return result

    # For MyoPS380, check order
    resolved_order = resolve_label_order(order)
    if resolved_order == LEGACY_LABEL_ORDER:
        lut = np.asarray([0, 1, 3, 2], dtype=np.uint8)
        result = lut[result]
    return result


def patient_id(sample_name: str) -> str:
    """Extract patient ID from caseXXXX_sliceNNN or caseXXXX / CaseXXXX."""
    name = str(sample_name).strip()
    for extension in (".npy.h5", ".nii.gz", ".npz", ".h5", ".nii"):
        if name.endswith(extension):
            name = name[:-len(extension)]
            break
    if not name or name in {".", ".."} or any(c in name for c in ("/", "\\", ":")):
        raise ValueError(f"Invalid sample name {sample_name!r}; manifests must contain basenames.")
    return re.sub(r"_slice\d+$", "", name)


def random_rot_flip(image, image1, image2, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    image1 = np.rot90(image1, k)
    image2 = np.rot90(image2, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    image1 = np.flip(image1, axis=axis).copy()
    image2 = np.flip(image2, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, image1, image2, label


def random_rotate(image, image1, image2, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    image1 = ndimage.rotate(image1, angle, order=0, reshape=False)
    image2 = ndimage.rotate(image2, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, image1, image2, label


def add_gaussian_noise(image, image1, image2, label, mean=0, std=0.2):
    noise = np.random.normal(mean, std, image.shape)
    image = image + noise
    image1 = image1 + noise
    image2 = image2 + noise
    image = np.clip(image, 0, 1)
    image1 = np.clip(image1, 0, 1)
    image2 = np.clip(image2, 0, 1)
    return image, image1, image2, label


class RandomGenerator:
    """Reproduces 100% I-MMSeg augmentation (6 steps, synced RNG across modalities)."""

    def __init__(self, output_size: Sequence[int] = (128, 128)):
        self.output_size = tuple(output_size)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        image, image1, image2, label = sample["image"], sample["image1"], sample["image2"], sample["label"]

        # Step 1: Rotation / Flip / Noise (if / elif / elif chain)
        if random.random() > 0.5:
            image, image1, image2, label = random_rot_flip(image, image1, image2, label)
        elif random.random() > 0.5:
            image, image1, image2, label = random_rotate(image, image1, image2, label)
        elif random.random() > 0.5:
            image, image1, image2, label = add_gaussian_noise(image, image1, image2, label)

        # Step 2: Translation shift
        if random.random() > 0.5:
            shift_x = random.randint(-5, 5)
            shift_y = random.randint(-5, 5)
            image = np.roll(image, (shift_x, shift_y), axis=(0, 1))
            image1 = np.roll(image1, (shift_x, shift_y), axis=(0, 1))
            image2 = np.roll(image2, (shift_x, shift_y), axis=(0, 1))
            label = np.roll(label, (shift_x, shift_y), axis=(0, 1))

        # Step 3: Brightness
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255)
            image1 = np.clip(image1 * factor, 0, 255)
            image2 = np.clip(image2 * factor, 0, 255)

        # Step 4: Gamma
        if random.random() > 0.5:
            gamma = random.uniform(0.8, 1.2)
            image = np.power(image / 255.0, gamma) * 255.0
            image1 = np.power(image1 / 255.0, gamma) * 255.0
            image2 = np.power(image2 / 255.0, gamma) * 255.0

        # Step 5: Random crop
        if random.random() > 0.5:
            top = random.randint(0, image.shape[0] // 5)
            left = random.randint(0, image.shape[1] // 5)
            image = image[top:top + self.output_size[0], left:left + self.output_size[1]]
            image1 = image1[top:top + self.output_size[0], left:left + self.output_size[1]]
            image2 = image2[top:top + self.output_size[0], left:left + self.output_size[1]]
            label = label[top:top + self.output_size[0], left:left + self.output_size[1]]

        # Step 6: Zoom / resize if necessary
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image1 = zoom(image1, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image2 = zoom(image2, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # Stack into (3, H, W) tensor
        stacked = np.stack([image.astype(np.float32), image1.astype(np.float32), image2.astype(np.float32)], axis=0)
        image_tensor = torch.from_numpy(stacked).float()
        label_tensor = torch.from_numpy(label.astype(np.int64)).long()

        return {
            "image": image_tensor,
            "label": label_tensor,
            "case_name": sample.get("case_name", ""),
            "dataset": sample.get("dataset", "myops380"),
        }


def read_split_lines(list_dir: str | Path, split: str) -> list[str]:
    list_path = Path(list_dir) / f"{split}.txt"
    if not list_path.is_file():
        # Fallback candidates
        candidates = [
            Path(list_dir) / f"{split}.txt",
            Path(list_dir) / f"{split}",
        ]
        if split in ("test", "test_vol", "volume"):
            candidates.extend([Path(list_dir) / "test_vol.txt", Path(list_dir) / "test.txt"])
        elif split in ("train", "train_slices"):
            candidates.extend([Path(list_dir) / "train.txt", Path(list_dir) / "train_slices.txt"])
        actual_path = None
        for c in candidates:
            if c.is_file():
                actual_path = c
                break
        if actual_path is None:
            raise FileNotFoundError(f"Missing split manifest: {list_path}")
        list_path = actual_path

    lines = [line.strip() for line in list_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return lines


class MyopsDataset(Dataset):
    """2D Slice Dataset for MyoPS380 and MyoPS++ training.

    Reads 3 aligned modalities (bSSFP, LGE, T2w) from train_npz/ folders,
    converts legacy target labels to canonical labels via canonicalize_label(),
    and outputs 3-channel (3, 128, 128) tensors.
    """

    def __init__(
        self,
        data_root: str | Path,
        list_dir: str | Path | None = None,
        split: str = "train",
        transform: Any = None,
        label_order: str | None = None,
        dataset: str = "myops380",
    ):
        self.dataset = normalize_dataset_name(dataset)
        self.data_root = Path(data_root)
        if list_dir is None:
            self.list_dir = resolve_default_split_dir(self.dataset)
        else:
            self.list_dir = Path(list_dir)

        self.split = split
        self.transform = transform

        if label_order is None:
            self.label_order = "canonical" if self.dataset == "myopspp" else "legacy"
        else:
            self.label_order = label_order

        self.sample_list = read_split_lines(self.list_dir, split)

        # Support nested Processed_data root if needed
        root = self.data_root
        if not (root / "bSSFP").exists() and (root / "Processed_data" / "bSSFP").exists():
            root = root / "Processed_data"

        self.modality_roots = [
            root / "bSSFP",
            root / "LGE",
            root / "T2w",
        ]

    def __len__(self) -> int:
        return len(self.sample_list)

    def _read_slice(self, root: Path, sample_name: str) -> tuple[np.ndarray, np.ndarray | None]:
        file_path = root / "train_npz" / f"{sample_name}.npz"
        if not file_path.is_file():
            alt = root / "test_npz" / f"{sample_name}.npz"
            file_path = alt if alt.is_file() else file_path
        if not file_path.is_file():
            raise FileNotFoundError(f"Could not find slice file: {file_path}")

        with np.load(file_path, allow_pickle=False) as data:
            image = data["image"].astype(np.float32)
            label = data["label"] if "label" in data else None
        return image, label

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_name = self.sample_list[idx]

        # Read bSSFP, LGE, T2w
        img_bssfp, label = self._read_slice(self.modality_roots[0], sample_name)
        img_lge, _ = self._read_slice(self.modality_roots[1], sample_name)
        img_t2w, _ = self._read_slice(self.modality_roots[2], sample_name)

        if label is None:
            raise ValueError(f"Target label missing in {sample_name}")

        # Canonicalize label from legacy to canonical
        label = canonicalize_label(label, order=self.label_order, dataset=self.dataset)

        sample = {
            "image": img_bssfp,
            "image1": img_lge,
            "image2": img_t2w,
            "label": label,
            "case_name": sample_name,
            "dataset": self.dataset,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        else:
            stacked = np.stack([img_bssfp, img_lge, img_t2w], axis=0)
            sample = {
                "image": torch.from_numpy(stacked).float(),
                "label": torch.from_numpy(label.astype(np.int64)).long(),
                "case_name": sample_name,
                "dataset": self.dataset,
            }

        return sample


class MyopsVolumeDataset(Dataset):
    """3D Volume Dataset for MyoPS380 and MyoPS++ testing / evaluation.

    Reads 3D volumes (H, W, D) for test cases from test_vol_h5/ folders.
    Extracts spacing metadata dynamically:
      - If dataset == 'myopspp': extracts slice_thickness or spacing sz and sets spacing=[1.0, 1.0, float(sz)].
      - If dataset == 'myops380': retains original spacing (or None).
    Canonicalizes the ground truth masks if present.
    """

    def __init__(
        self,
        data_root: str | Path,
        list_dir: str | Path | None = None,
        split: str = "test_vol",
        label_order: str | None = None,
        dataset: str = "myops380",
    ):
        self.dataset = normalize_dataset_name(dataset)
        self.data_root = Path(data_root)
        if list_dir is None:
            self.list_dir = resolve_default_split_dir(self.dataset)
        else:
            self.list_dir = Path(list_dir)

        self.split = split

        if label_order is None:
            self.label_order = "canonical" if self.dataset == "myopspp" else "legacy"
        else:
            self.label_order = label_order

        self.sample_list = read_split_lines(self.list_dir, split)

        # Support nested Processed_data root if needed
        root = self.data_root
        if not (root / "bSSFP").exists() and (root / "Processed_data" / "bSSFP").exists():
            root = root / "Processed_data"

        self.modality_roots = [
            root / "bSSFP",
            root / "LGE",
            root / "T2w",
        ]

    def __len__(self) -> int:
        return len(self.sample_list)

    def _read_volume(self, root: Path, case_name: str) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        h5_path = root / "test_vol_h5" / f"{case_name}.npy.h5"
        if not h5_path.is_file():
            # Check alternative extensions
            for ext in (".h5", ".npy.h5"):
                cand = root / "test_vol_h5" / f"{case_name}{ext}"
                if cand.is_file():
                    h5_path = cand
                    break

        if h5_path.is_file():
            with h5py.File(h5_path, "r") as f:
                image = f["image"][:]
                label = None
                for k in ("label", "gt", "segmentation"):
                    if k in f:
                        label = f[k][:]
                        break
                metadata = dict(f.attrs)
                for key in ("spacing", "affine", "slice_thickness"):
                    if key in f:
                        metadata[key] = f[key][:]
            return np.asarray(image, dtype=np.float32), label, metadata

        # Fallback: Reassemble slices if .npy.h5 is not present
        slice_dir = root / "train_npz"
        if not slice_dir.exists():
            slice_dir = root / "test_npz"
        paths = list(slice_dir.glob(f"{case_name}_slice*.npz"))
        indexed = []
        for path in paths:
            match = re.fullmatch(re.escape(case_name) + r"_slice(\d+)", path.stem)
            if match:
                indexed.append((int(match[1]), path))
        indexed.sort()
        if not indexed:
            raise FileNotFoundError(f"Neither test_vol_h5 nor slice files found for {case_name} in {root}")

        slices = []
        labels = []
        metadata = {}
        for _, p in indexed:
            with np.load(p, allow_pickle=False) as d:
                slices.append(d["image"])
                if "label" in d:
                    labels.append(d["label"])
                for k in ("spacing", "affine", "slice_thickness"):
                    if k in d and k not in metadata:
                        metadata[k] = d[k]
        image = np.stack(slices, axis=-1).astype(np.float32)
        label = np.stack(labels, axis=-1) if labels else None
        return image, label, metadata

    def __getitem__(self, idx: int) -> dict[str, Any]:
        case_name = self.sample_list[idx]

        img_bssfp, label, meta = self._read_volume(self.modality_roots[0], case_name)
        img_lge, _, _ = self._read_volume(self.modality_roots[1], case_name)
        img_t2w, _, _ = self._read_volume(self.modality_roots[2], case_name)

        if label is not None:
            label = canonicalize_label(label, order=self.label_order, dataset=self.dataset)

        # Spacing extraction logic:
        raw_spacing = meta.get("spacing", None)
        raw_thickness = meta.get("slice_thickness", None)

        if self.dataset == "myopspp":
            # For MyoPS++: dynamic physical spacing [1.0, 1.0, sz]
            sz = 1.0
            if raw_thickness is not None:
                th = np.asarray(raw_thickness, dtype=np.float64)
                sz = float(th.item() if th.size == 1 else th[0])
            elif raw_spacing is not None:
                sp = np.asarray(raw_spacing, dtype=np.float64)
                sz = float(sp[2]) if sp.size >= 3 else float(sp.item() if sp.size == 1 else 1.0)
            spacing = np.array([1.0, 1.0, float(sz)], dtype=np.float64)
            spacing_unit = "mm"
        else:
            # For MyoPS380: default spacing is None (voxel units)
            spacing = raw_spacing
            spacing_unit = "voxel"

        return {
            "image": img_bssfp,
            "image1": img_lge,
            "image2": img_t2w,
            "label": label,
            "case_name": case_name,
            "patient_id": patient_id(case_name),
            "spacing": spacing,
            "affine": meta.get("affine", None),
            "spacing_unit": spacing_unit,
            "dataset": self.dataset,
        }
