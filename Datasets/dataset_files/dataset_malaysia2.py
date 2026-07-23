from __future__ import annotations

import csv
import shutil
import yaml
import numpy as np
from pathlib import Path
from typing import Final

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
import cv2
import os

MAX_NICKNAME_LEN: Final = 15

class MALAYSIA2_dataset(DatasetVSLAMLab):
    """MALAYSIA2 dataset helper for VSLAM-LAB benchmark with local images."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "malaysia2") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load YAML settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Folder containing sequence subfolders
        self.image_folder: Path = Path(cfg["image_folder"])

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names

        # Default FPS if not specified in YAML
        self.rgb_hz: float = cfg.get("rgb_hz", 30.0)


    def download_sequence_data(self, sequence_name: str) -> None:
        """Copy local images into VSLAM-LAB structure, keeping original filenames."""
        sequence_path = self.dataset_path / sequence_name
        rgb_folder = sequence_path / "rgb_0"
        source_folder = self.image_folder / sequence_name

        if not source_folder.exists():
            print(f"Error: Source folder {source_folder} does not exist")
            return

        images = sorted(
            list(source_folder.glob("*.png")) +
            list(source_folder.glob("*.jpg")) +
            list(source_folder.glob("*.jpeg"))
        )

        if not images:
            print(f"No images found in {source_folder}")
            return

        rgb_folder.mkdir(parents=True, exist_ok=True)

        print(f"Copying {len(images)} images (original filenames preserved)...")

        for img_path in images:
            out_path = rgb_folder / img_path.name
            if out_path.exists():
                continue
            shutil.copy2(img_path, out_path)

        self.create_rgb_csv(sequence_name)
        self.create_groundtruth_csv(sequence_name)
        self.create_calibration_yaml(sequence_name)

    def create_rgb_folder(self, sequence_name: str) -> None:
        pass

    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_folder = sequence_path / "rgb_0"
        rgb_csv = sequence_path / "rgb.csv"
        tmp_csv = rgb_csv.with_suffix(".csv.tmp")

        images = sorted(
            list(rgb_folder.glob("*.png")) +
            list(rgb_folder.glob("*.jpg")) +
            list(rgb_folder.glob("*.jpeg"))
        )
        if not images:
            print(f"No images found in {rgb_folder}")
            return

        with open(tmp_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"])
            for i, img_path in enumerate(images):
                timestamp_ns = int(i * 1e9 / self.rgb_hz)
                writer.writerow([timestamp_ns, str(img_path.relative_to(sequence_path)), "", ""])

        tmp_csv.replace(rgb_csv)


    def create_calibration_yaml(self, sequence_name: str) -> None:
        """Create calibration.yaml for the sequence."""
        sequence_path = self.dataset_path / sequence_name
        rgb_folder = sequence_path / "rgb_0"
        images = sorted(
            list(rgb_folder.glob("*.png")) +
            list(rgb_folder.glob("*.jpg")) +
            list(rgb_folder.glob("*.jpeg"))
        )

        if not images:
            print(f"No images found in {rgb_folder}")
            return

        # Read first image for resolution
        img = cv2.imread(str(images[0]))
        height, width = img.shape[:2]
        fps = self.rgb_hz

        # Approximate intrinsics
        fx = fy = 0
        cx = width / 2
        cy = height / 2

        rgb0 = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "unknown",
            "focal_length": [float(fx), float(fy)],
            "principal_point": [float(cx), float(cy)],
            "fps": float(fps),
            "T_BS": np.eye(4),
        }

        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])


    def create_groundtruth_csv(self, sequence_name: str) -> None:
        """Generate dummy groundtruth.csv with zero poses."""
        sequence_path = self.dataset_path / sequence_name
        gt_csv = sequence_path / "groundtruth.csv"
        tmp_csv = gt_csv.with_suffix(".csv.tmp")

        rgb_folder = sequence_path / "rgb_0"
        images = sorted(rgb_folder.glob("*.png"))
        if not images:
            print(f"No images found in {rgb_folder}")
            return

        with open(tmp_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
            for i, _ in enumerate(images):
                timestamp_ns = int(i * 1e9 / self.rgb_hz)
                writer.writerow([timestamp_ns, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        tmp_csv.replace(gt_csv)
        print(f"Groundtruth CSV created for {sequence_name} with {len(images)} frames")


    def remove_unused_files(self, sequence_name: str) -> None:
        """Optional: remove temporary files (not needed here)."""
        pass