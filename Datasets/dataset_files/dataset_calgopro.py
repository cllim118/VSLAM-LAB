from __future__ import annotations

import csv
import yaml
import numpy as np
from pathlib import Path
from urllib.parse import urljoin
from typing import Final, Any
from collections.abc import Iterable

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from path_constants import Retention, BENCHMARK_RETENTION

import os
import cv2

MAX_NICKNAME_LEN: Final = 15


class CALGOPRO_dataset(DatasetVSLAMLab):
    """CALGOPRO dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "calgopro") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        # Load settings
        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Get download url
        self.video_folder: str = cfg["video_folder"]

        # Sequence nicknames
        self.sequence_nicknames = self.sequence_names


    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_folder = self.dataset_path / sequence_name / "rgb_0"
        if not sequence_folder.exists():
            os.makedirs(sequence_folder)
        
        video_path = Path(self.video_folder) / f"{sequence_name}.MP4"  

        # Check if the video exists
        if not video_path.exists():
            print(f"Error: Video file {video_path} not found in {self.video_folder}.")
            return

        # Convert the video into images
        self.video_to_images(video_path, sequence_folder, sequence_name)

        # Generate CSV files
        self.create_rgb_csv(sequence_name)
        self.create_groundtruth_csv(sequence_name)
        self.create_calibration_yaml(sequence_name)


    def video_to_images(self, video_path: Path, output_folder: Path, sequence_name: str) -> None:
        cap = cv2.VideoCapture(str(video_path)) 

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}.")
            return

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read() 

            if ret:
                frame_filename = output_folder / f"img_{frame_count:04d}.png"
                cv2.imwrite(str(frame_filename), frame)
    
                frame_count += 1
            else:
                break

        cap.release()  


    def create_rgb_folder(self, sequence_name: str) -> None:
        pass
            
    
    def create_rgb_csv(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        rgb_folder = sequence_path / "rgb_0"
        rgb_csv = sequence_path / "rgb.csv"
        tmp_rgb_csv = rgb_csv.with_suffix(".csv.tmp")

        # Open video file
        video_path_mp4 = Path(self.video_folder) / f"{sequence_name}.MP4"
        cap = cv2.VideoCapture(str(video_path_mp4))

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path_mp4}.")
            return

        # Open the CSV writer
        with open(tmp_rgb_csv, "w", newline="", encoding="utf-8") as fout:
            w = csv.writer(fout)
            w.writerow(["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"])

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()

                if ret:
                    # Extract timestamp in milliseconds
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    # Convert milliseconds to nanoseconds
                    timestamp_ns = int(timestamp_ms * 1e6)

                    # Save the RGB frame
                    csv_filename = f"rgb_0/img_{frame_count:04d}.png"
                    cv2.imwrite(str(csv_filename), frame)

                    # Write to CSV
                    w.writerow([timestamp_ns, str(csv_filename), "", ""])

                    frame_count += 1
                else:
                    break

        cap.release()
        tmp_rgb_csv.replace(rgb_csv)

        
    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path = self.dataset_path / sequence_name
        video_path = Path(self.video_folder) / f"{sequence_name}.MP4"
        calibration_yaml = sequence_path / "calibration.yaml"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Could not open video {video_path}")
            return

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Approximate intrinsics
        import math

        fx = 0
        fy = fx 
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
        sequence_path = self.dataset_path / sequence_name
        gt_csv = sequence_path / "groundtruth.csv"
        tmp_gt_csv = gt_csv.with_suffix(".csv.tmp")

        video_path_mp4 = Path(self.video_folder) / f"{sequence_name}.MP4"
        cap = cv2.VideoCapture(str(video_path_mp4))

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path_mp4}.")
            return

        with open(tmp_gt_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])

            while cap.isOpened():
                ret, _ = cap.read()
                if not ret:
                    break

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_ns = int(timestamp_ms * 1e6)

                # All zeros translation
                writer.writerow([timestamp_ns, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        cap.release()
        tmp_gt_csv.replace(gt_csv)


    def remove_unused_files(self, sequence_name: str) -> None:
        pass


