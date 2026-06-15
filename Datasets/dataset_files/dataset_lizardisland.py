from __future__ import annotations

import os
import csv
import yaml
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Final, Any
from pyproj import Transformer
import math

from utilities import SCRIPT_LABEL, print_msg
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab

IMAGE_CROP: Final = {"feb": [0,0], "mar": [0,0], "sep": [0,0]}


class LIZARDISLAND_dataset(DatasetVSLAMLab):
    """LIZARDISLAND dataset helper for VSLAM-LAB benchmark."""

    def __init__(self, benchmark_path: str | Path, dataset_name: str = "lizardisland") -> None:
        super().__init__(dataset_name, Path(benchmark_path))

        with open(self.yaml_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        self.local_raw_data_path: str = cfg["local_raw_data_path"]
        self.rgb_hz: float = cfg.get("rgb_hz", 30)
        self.image_resolution = cfg.get("target_resolution", [640, 480])
        self.subsets = cfg.get("subsets", {})
        self.combined = cfg.get("combined", {})

    def download_sequence_data(self, sequence_name: str) -> None:
        sequence_path: Path = self.dataset_path / sequence_name
        rgb_path: Path = sequence_path / "rgb_0"
        if rgb_path.exists():
            return

        if sequence_name in self.subsets.keys():
            self.download_sequence_data(self.subsets.get(sequence_name)[0])
            self.download_subsequence(sequence_name)
            return

        if sequence_name in self.combined.keys():
            for subset in self.combined.get(sequence_name):
                self.download_sequence_data(subset)
            self.download_combined_subsequence(sequence_name)
            return

        # ── paths ─────────────────────────────────────────
        raw_path: Path = sequence_path / "raw"
        rgb_csv: Path  = sequence_path / "rgb.csv"
        gt_csv: Path   = sequence_path / "groundtruth.csv"
        tmp_rgb: Path  = rgb_csv.with_suffix(".csv.tmp")
        tmp_gt: Path   = gt_csv.with_suffix(".csv.tmp")

        rgb_path.mkdir(parents=True, exist_ok=True)
        #raw_path.mkdir(parents=True, exist_ok=True)

        year = '25' if 'sep' in sequence_name else '24'
        local_sequence_path = Path(self.local_raw_data_path) / f"LIRS_{sequence_name.capitalize()}_{year}"

        print(f"local_sequence_path: {local_sequence_path}")
        if not local_sequence_path.exists():
            print(f"Error: {local_sequence_path} does not exist.")
            return

        # ── collect images ────────────────────────────────
        image_files = sorted([
            Path(root) / file
            for root, _, files in os.walk(local_sequence_path)
            for file in files
            if file.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        print_msg(SCRIPT_LABEL, f"Found {len(image_files)} images.")

        # ── GPS ───────────────────────────────────────────
        gps_lookup = self._load_gps_lookup(sequence_name)
        origin_latlon = next(
            (gps_lookup[f.name.upper()] for f in image_files if f.name.upper() in gps_lookup),
            None
        )
        if origin_latlon is None:
            print_msg(SCRIPT_LABEL, "WARNING: No GPS fixes found — groundtruth will be all zeros.")

        # ── compute output resolution (aspect ratio 유지) ─
        new_W, new_H = self._compute_resolution(image_files[0], sequence_name)
        print_msg(SCRIPT_LABEL, f"Output resolution: {new_W}x{new_H}")

        # ── write csv + images ────────────────────────────
        timestamp_increment = int(1_000_000_000 / self.rgb_hz)

        with open(tmp_rgb, "w", newline="") as f_rgb, \
             open(tmp_gt,  "w", newline="") as f_gt:

            writer_rgb = csv.writer(f_rgb)
            writer_gt  = csv.writer(f_gt)
            writer_rgb.writerow(["ts_rgb_0 (ns)", "path_rgb_0", "sequence_name"])
            writer_gt.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])

            for i, src_file in enumerate(tqdm(image_files, desc="Processing images")):
                ts_ns = i * timestamp_increment
                raw_filename = raw_path / src_file.name
                rgb_filename = rgb_path / src_file.name

                #shutil.copy(src_file, raw_filename)

                try:
                    with Image.open(src_file) as img:
                        w, h = img.size
                        crop = IMAGE_CROP.get(sequence_name, [0, 0])
                        img_cropped = img.crop((0, 0, w - crop[0], h - crop[1]))
                        img_resized = img_cropped.resize((new_W, new_H), Image.Resampling.LANCZOS)
                        img_resized.save(rgb_filename)
                except Exception as e:
                    print_msg(SCRIPT_LABEL, f"Error processing {src_file}: {e}")
                    continue

                writer_rgb.writerow([ts_ns, f"rgb_0/{src_file.name}", sequence_name])

                key = src_file.name.upper()
                if key in gps_lookup and origin_latlon is not None:
                    lat, lon, alt = gps_lookup[key]
                    tx, ty, tz = self._latlon_to_enu(lat, lon, alt, *origin_latlon)
                else:
                    tx, ty, tz = 0.0, 0.0, 0.0
                writer_gt.writerow([ts_ns, tx, ty, tz, 0, 0, 0, 1])

        tmp_rgb.replace(rgb_csv)
        tmp_gt.replace(gt_csv)

        self.create_calibration_yaml(sequence_name)


    def _compute_resolution(self, sample_image: Path, sequence_name: str) -> tuple[int, int]:
        """target_resolution 유지하면서 aspect ratio 보존"""
        with Image.open(sample_image) as img:
            w, h = img.size
            crop = IMAGE_CROP.get(sequence_name, [0, 0])
            w -= crop[0]
            h -= crop[1]

        target_pixels = self.image_resolution[0] * self.image_resolution[1]
        new_H = int(np.sqrt(target_pixels * h / w))
        new_W = int(target_pixels / new_H)
        return new_W, new_H


    def download_subsequence(self, sequence_name: str) -> None:
        import pandas as pd

        sequence_path: Path = self.dataset_path / sequence_name
        rgb_path: Path = sequence_path / "rgb_0"
        rgb_csv: Path  = sequence_path / "rgb.csv"
        gt_csv: Path   = sequence_path / "groundtruth.csv"
        if rgb_path.exists():
            print("rgb_path exists.")
            return
        rgb_path.mkdir(parents=True, exist_ok=True)

        parent_sequence = self.subsets.get(sequence_name)[0]
        parent_sequence_path: Path = self.dataset_path / parent_sequence

        df_rgb = pd.read_csv(parent_sequence_path / "rgb.csv")
        df_gt  = pd.read_csv(parent_sequence_path / "groundtruth.csv")

        target_image_name = self.subsets.get(sequence_name)[1]
        radius = self.subsets.get(sequence_name)[2]
        ref_idx = df_rgb.index[df_rgb['path_rgb_0'] == target_image_name].tolist()[0]

        ref_x = df_gt.at[ref_idx, 'tx (m)']
        ref_y = df_gt.at[ref_idx, 'ty (m)']
        ref_z = df_gt.at[ref_idx, 'tz (m)']

        distances = np.sqrt(
            (df_gt['tx (m)'] - ref_x)**2 +
            (df_gt['ty (m)'] - ref_y)**2 +
            (df_gt['tz (m)'] - ref_z)**2
        )
        mask = distances <= radius
        df_rgb_sub = df_rgb[mask].copy().reset_index(drop=True)
        df_gt_sub  = df_gt[mask].copy().reset_index(drop=True)

        for _, row in df_rgb_sub.iterrows():
            rel_path = row['path_rgb_0']
            full_src = os.path.abspath(parent_sequence_path / rel_path)
            full_dst = os.path.abspath(sequence_path / rel_path)
            if os.path.exists(full_dst) or os.path.islink(full_dst):
                os.remove(full_dst)
            os.symlink(full_src, full_dst)

        df_rgb_sub.to_csv(rgb_csv, index=False)
        df_gt_sub.to_csv(gt_csv,  index=False)
        self.create_calibration_yaml(sequence_name)


    def download_combined_subsequence(self, sequence_name: str) -> None:
        import pandas as pd

        sequence_path: Path = self.dataset_path / sequence_name
        rgb_path: Path = sequence_path / "rgb_0"
        rgb_csv: Path  = sequence_path / "rgb.csv"
        gt_csv: Path   = sequence_path / "groundtruth.csv"
        if rgb_path.exists():
            return
        rgb_path.mkdir(parents=True, exist_ok=True)

        dfs_rgb, dfs_pose = [], []
        for subset in self.combined.get(sequence_name):
            parent_path: Path = self.dataset_path / subset
            dfs_rgb.append(pd.read_csv(parent_path / "rgb.csv"))
            dfs_pose.append(pd.read_csv(parent_path / "groundtruth.csv"))
            for _, row in dfs_rgb[-1].iterrows():
                rel_path = row['path_rgb_0']
                full_src = os.path.abspath(parent_path / rel_path)
                full_dst = os.path.abspath(sequence_path / rel_path)
                if os.path.exists(full_dst) or os.path.islink(full_dst):
                    os.remove(full_dst)
                os.symlink(full_src, full_dst)

        pd.concat(dfs_rgb,  ignore_index=True).to_csv(rgb_csv, index=False)
        pd.concat(dfs_pose, ignore_index=True).to_csv(gt_csv,  index=False)
        self.create_calibration_yaml(sequence_name)

    def create_rgb_folder(self, sequence_name: str) -> None:
        pass

    def create_rgb_csv(self, sequence_name: str) -> None:
        pass

    def create_calibration_yaml(self, sequence_name: str) -> None:
        sequence_path: Path = self.dataset_path / sequence_name
        rgb_path: Path = sequence_path / "rgb_0"

        images = list(rgb_path.glob("*.jpg")) + \
                list(rgb_path.glob("*.jpeg")) + \
                list(rgb_path.glob("*.png"))

        if not images:
            print_msg(SCRIPT_LABEL, f"No images found in {rgb_path}")
            return

        with Image.open(images[0]) as img:
            new_W, new_H = img.size

        rgb0: dict[str, Any] = {
            "cam_name": "rgb_0",
            "cam_type": "rgb",
            "cam_model": "unknown",
            "focal_length": [0.0, 0.0],
            "principal_point": [new_W / 2.0, new_H / 2.0],
            "fps": float(self.rgb_hz),
            "T_BS": np.eye(4).tolist(),
        }
        self.write_calibration_yaml(sequence_name=sequence_name, rgb=[rgb0])

    def create_groundtruth_csv(self, sequence_name: str) -> None:
        pass

    def remove_unused_files(self, sequence_name: str) -> None:
        pass

    def _load_gps_lookup(self, sequence_name: str) -> dict[str, tuple[float, float, float]]:
        gps_files = []
        year = '25' if 'sep' in sequence_name else '24'
        local_sequence_path = Path(self.local_raw_data_path) / f"LIRS_{sequence_name.capitalize()}_{year}"
        base = local_sequence_path / "South_Palfrey_1"

        for gopro_dir in sorted(base.iterdir()):
            for f in gopro_dir.glob("*.csv"):
                print(f"Found GPS file: {f}")
                gps_files.append(f)

        lookup: dict[str, tuple[float, float, float]] = {}
        for csv_path in gps_files:
            with open(csv_path, newline='') as f:
                for row in csv.reader(f):
                    if len(row) < 4:
                        continue
                    fname = row[0].strip().upper()
                    try:
                        lat, lon, alt = float(row[1]), float(row[2]), float(row[3])
                        lookup[fname] = (lat, lon, alt)
                    except ValueError:
                        continue
        return lookup

    def _latlon_to_enu(self, lat, lon, alt, lat0, lon0, alt0) -> tuple[float, float, float]:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
        x0, y0, z0 = transformer.transform(lon0, lat0, alt0)
        xp, yp, zp = transformer.transform(lon, lat, alt)

        lat0_r = math.radians(lat0)
        lon0_r = math.radians(lon0)
        dx, dy, dz = xp - x0, yp - y0, zp - z0

        e = -math.sin(lon0_r)*dx + math.cos(lon0_r)*dy
        n = (-math.sin(lat0_r)*math.cos(lon0_r)*dx
             - math.sin(lat0_r)*math.sin(lon0_r)*dy
             + math.cos(lat0_r)*dz)
        u = (math.cos(lat0_r)*math.cos(lon0_r)*dx
             + math.cos(lat0_r)*math.sin(lon0_r)*dy
             + math.sin(lat0_r)*dz)
        return e, n, u