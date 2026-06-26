from __future__ import annotations

import csv
import yaml
import numpy as np
from pathlib import Path
from typing import Final

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
import cv2
import os

MAX_NICKNAME_LEN: Final = 15

class TANK_dataset(DatasetVSLAMLab):
    """TANK dataset helper for VSLAM-LAB benchmark with local images."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "tank") -> None:
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
        """Extract frames from MP4(s), resize to 1920x1080, and format into VSLAM-LAB."""
        sequence_path = self.dataset_path / sequence_name
        rgb_folder    = sequence_path / "rgb_0"

        rgb_folder.mkdir(parents=True, exist_ok=True)

        # ── 영상 파일 찾기 ─────────────────────────────────
        base_name   = sequence_name  # e.g. "r01"
        video_paths = sorted(self.image_folder.glob(f"{base_name}-*.MP4"))

        # 단일 파일도 지원 (r01.MP4)
        single_path = self.image_folder / f"{base_name}.MP4"
        if not video_paths and single_path.exists():
            video_paths = [single_path]

        if not video_paths:
            print(f"Error: No video files found for {sequence_name}")
            return

        print(f"Found {len(video_paths)} video(s): {[p.name for p in video_paths]}")

        # ── 프레임 추출 ────────────────────────────────────
        target_size = (1920, 1080)
        i = 0

        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1:
                self.rgb_hz = fps

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Extracting {total_frames} frames from {video_path.name}...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                resized  = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                out_path = rgb_folder / f"img_{i:04d}.png"
                cv2.imwrite(str(out_path), resized)
                i += 1

            cap.release()

        print(f"Extracted {i} frames total (resized to 1920x1080)")

        self.create_rgb_csv(sequence_name)
        self.create_groundtruth_csv(sequence_name)
        self.create_calibration_yaml(sequence_name)

    def create_rgb_folder(self, sequence_name: str) -> None:
        pass

    def create_rgb_csv(self, sequence_name: str) -> None:
        """Generate rgb.csv for the sequence."""
        sequence_path = self.dataset_path / sequence_name
        rgb_folder = sequence_path / "rgb_0"
        rgb_csv = sequence_path / "rgb.csv"
        tmp_csv = rgb_csv.with_suffix(".csv.tmp")

        images = sorted(rgb_folder.glob("*.png"))
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
        images = sorted(rgb_folder.glob("*.png"))

        if not images:
            print(f"No images found in {rgb_folder}")
            return

        img = cv2.imread(str(images[0]))
        if img is None:
            print(f"Failed to read image: {images[0]}")
            return

        height, width = img.shape[:2]
        fps = self.rgb_hz

        fx = fy = 0  # approximate
        cx = width / 2
        cy = height / 2

        rgb0 = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "unknown",
            "focal_length": [float(fx), float(fy)],
            "principal_point": [float(cx), float(cy)],
            "fps": float(fps),
            "T_BS": np.eye(4)
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