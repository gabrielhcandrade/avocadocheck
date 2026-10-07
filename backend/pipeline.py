#!/usr/bin/env python3
"""
Single script pipeline for estimating the ripening stage and the remaining
green life of Hass avocados from RGB images of the public Hass Avocado Ripening
Photographic Dataset, and for recommending the destination of a lot.

The script follows the architecture of the project proposal and keeps the same
methodological guarantees used in the previous studies of the group:

1. Scan the dataset folder and the annotation spreadsheet. Every file name
   encodes the storage condition, the day of the experiment, the fruit number,
   the side of the fruit and the ripening stage, e.g. T10_d05_012_a_3.jpg is
   fruit 12 stored under condition T10, photographed on day 5, side a, stage 3.
   The spreadsheet is the reference annotation, the file name is only used to
   cross check it.
2. Build the two targets of the multitask problem.
   stage           : ripening index 1 to 5, a classification target.
   days_remaining  : number of days until the fruit first reaches the ready
                     stage (--ready-stage), computed from the daily trajectory
                     of each fruit, a regression target. Fruits that leave the
                     experiment before reaching the ready stage are right
                     censored, they keep their stage label but have no days
                     target.
3. Detect and crop the fruit with YOLO (stage [1] of the proposal). Three
   detectors are available, a YOLO-World open vocabulary model prompted with
   "avocado", a standard COCO YOLO model or any custom YOLO weights trained on
   avocado boxes, and a classical threshold detector that exploits the white
   studio background. When YOLO finds no box the threshold detector is used
   as a fallback, and the method used for every image is logged.
4. Extract a hand crafted colour and texture descriptor from every crop, the
   HSV histogram baseline of the proposal.
5. Run an exploratory data analysis of the stages, of the storage conditions
   and of the remaining days.
6. Build leakage aware outer folds. All the images of the same fruit, both
   sides and every day of its trajectory, always stay in the same fold.
7. Train and evaluate baseline models with nested cross validation, a majority
   class and a mean predictor, a linear model and a Random Forest, over nested
   feature sets.
8. Train the proposed multitask CNN, a pretrained backbone with one head for
   the stage and one head for the remaining days, on exactly the same outer
   folds, together with the classification only CNN that reproduces the
   baseline of the dataset article. Grad-CAM maps are produced for both heads.
9. Report accuracy, balanced accuracy, macro F1 and quadratic weighted kappa
   for the stage, MAE, RMSE, R2 and the share of predictions within one and two
   days for the remaining days, fruit level bootstrap confidence intervals,
   confusion matrices and diagnostic plots.
10. Simulate lots from the out of fold predictions and apply the transparent
   lot decision rule of the proposal, then compare the recommended destination
   with the destination that the true labels would have produced, under an
   asymmetric cost.

Important notes
- The remaining days are only valid for the storage condition in which they
  were observed. The ripening speed depends strongly on the temperature, so the
  condition is always reported, the lot simulation is done per condition and
  the condition can be given to the models as an input with --use-condition.
- The meaning of the stage labels and of the storage codes must be checked in
  the data article. The default ready stage is 4, change it with
  --ready-stage if the article defines the consumption point differently.
- The logistic thresholds of the lot rule are PoC values scaled to the length
  of the experiment, they are NOT real sea freight times. Under cold storage
  the dataset covers about 20 days of green life, so a 25 day voyage cannot be
  simulated directly.
- No result is invented by this script. Every number in the output folder comes
  from the data that is actually read from disk.

Example
    python pipeline.py --data-dir "Hass Avocado Ripening Photographic Dataset"
    python pipeline.py --data-dir "..." --limit-fruits 40 --skip-cnn   (quick test)
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import re
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:  # optional dependency, the CNN part is skipped gracefully without it
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models as tv_models
    from torchvision import transforms as T

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# T10_d05_012_a_3 -> condition T10, day 5, fruit 12, side a, stage 3
FILE_PATTERN = re.compile(
    r"^(?P<condition>T\d+|Tam)_d(?P<day>\d+)_(?P<sample>\d+)_(?P<side>[ab])_(?P<stage>\d)$",
    re.IGNORECASE,
)

STAGES = (1, 2, 3, 4, 5)
DESTINATIONS = ("export_sea", "domestic_distant", "local_market", "processing")
DESTINATION_RANK = {"processing": 0, "local_market": 1, "domestic_distant": 2, "export_sea": 3}

# fruit mask on the white studio background, HSV values in the 0 to 255 range
MASK_MIN_SATURATION = 50
MASK_MAX_VALUE = 120


@dataclass
class Config:
    data_dir: Path
    annotations: Optional[Path]
    out_dir: Path
    conditions: List[str]
    ready_stage: int
    detector: str
    yolo_weights: Optional[str]
    yolo_prompts: List[str]
    yolo_conf: float
    yolo_imgsz: int
    min_box_area: float
    crop_pad: float
    crop_size: int
    no_crop_cache: bool
    no_feature_cache: bool
    n_splits: int
    n_inner_splits: int
    random_state: int
    limit_fruits: Optional[int]
    n_bootstrap: int
    n_permutations: int
    n_jobs: int
    models: List[str]
    skip_cnn: bool
    backbones: List[str]
    cnn_modes: List[str]
    use_condition: bool
    epochs: int
    batch_size: int
    lr: float
    img_size: int
    patience: int
    reg_weight: float
    num_workers: int
    cnn_max_folds: Optional[int]
    fit_final_cnn: bool
    gradcam_per_stage: int
    device: str
    lot_size: int
    lots_per_day: int
    lot_day_spread: int
    lot_quantile: float
    export_days: float
    domestic_days: float
    safety_margin: float
    review_confidence: float
    cost_upgrade: float
    cost_downgrade: float
    cost_review: float
    figure_formats: List[str]


# ---------------------------------------------------------------------------
# 1. Dataset scan
# ---------------------------------------------------------------------------

def parse_file_name(stem: str) -> Optional[Dict]:
    """Decode the annotation that the dataset encodes in every file name."""
    match = FILE_PATTERN.match(stem)
    if match is None:
        return None
    return {
        "condition": match.group("condition"),
        "day": int(match.group("day")),
        "sample": int(match.group("sample")),
        "side": match.group("side").lower(),
        "stage": int(match.group("stage")),
    }


def find_annotation_file(cfg: Config) -> Optional[Path]:
    if cfg.annotations is not None:
        if not cfg.annotations.exists():
            raise FileNotFoundError(f"Annotation file not found: {cfg.annotations}")
        return cfg.annotations
    candidates = sorted(cfg.data_dir.rglob("*.xlsx")) + sorted(cfg.data_dir.rglob("*.csv"))
    candidates = [c for c in candidates if not c.name.startswith("~$")]
    return candidates[0] if candidates else None


def load_annotations(path: Path) -> pd.DataFrame:
    """
    Read the annotation spreadsheet and normalise its column names.

    The columns are matched by keyword so that small changes in the published
    header, e.g. 'Ripening Index Classification', do not break the script.
    """
    raw = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path, sheet_name=0)
    mapping = {}
    for col in raw.columns:
        low = str(col).strip().lower()
        if "file" in low:
            mapping[col] = "file_stem"
        elif "time" in low:
            mapping[col] = "timestamp"
        elif "storage" in low or "group" in low:
            mapping[col] = "condition"
        elif "sample" in low:
            mapping[col] = "sample"
        elif "day" in low:
            mapping[col] = "day"
        elif "ripen" in low or "class" in low or "stage" in low:
            mapping[col] = "stage"
    ann = raw.rename(columns=mapping)
    required = {"file_stem", "condition", "sample", "day", "stage"}
    missing = required - set(ann.columns)
    if missing:
        raise ValueError(f"Columns {sorted(missing)} not found in {path}. Header: {list(raw.columns)}")
    ann["file_stem"] = ann["file_stem"].astype(str).map(lambda s: Path(s.strip()).stem)
    ann["condition"] = ann["condition"].astype(str).str.strip()
    for col in ["sample", "day", "stage"]:
        ann[col] = pd.to_numeric(ann[col], errors="coerce")
    ann = ann.dropna(subset=["sample", "day", "stage"])
    ann[["sample", "day", "stage"]] = ann[["sample", "day", "stage"]].astype(int)
    keep = [c for c in ["file_stem", "timestamp", "condition", "sample", "day", "stage"] if c in ann.columns]
    return ann[keep].drop_duplicates("file_stem").reset_index(drop=True)


def scan_dataset(cfg: Config) -> pd.DataFrame:
    """
    Build the image inventory.

    The spreadsheet is the reference. Images are paired with it by file stem,
    and the annotation decoded from the file name is used as a consistency
    check. Without a spreadsheet the file names alone are used.
    """
    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Folder not found: {cfg.data_dir}")

    images = {
        p.stem: p
        for p in sorted(cfg.data_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not images:
        raise FileNotFoundError(f"No image file found under {cfg.data_dir}")
    print(f"[SCAN] {len(images)} image files found.")

    ann_path = find_annotation_file(cfg)
    if ann_path is not None:
        ann = load_annotations(ann_path)
        print(f"[SCAN] {len(ann)} annotated rows read from {ann_path.name}")
    else:
        print("[WARNING] No annotation spreadsheet found, the file names are used as annotation.")
        rows = []
        for stem in images:
            parsed = parse_file_name(stem)
            if parsed is not None:
                rows.append({"file_stem": stem, **{k: v for k, v in parsed.items() if k != "side"}})
        ann = pd.DataFrame(rows)

    ann["path"] = ann["file_stem"].map(lambda s: str(images[s]) if s in images else None)
    n_missing = int(ann["path"].isna().sum())
    if n_missing:
        print(f"[WARNING] {n_missing} annotated rows have no image file on disk and were dropped.")
        ann = ann.dropna(subset=["path"]).copy()
    n_unannotated = len(set(images) - set(ann["file_stem"]))
    if n_unannotated:
        print(f"[WARNING] {n_unannotated} image files have no annotation row and were ignored.")

    # cross check the spreadsheet against the file name
    parsed = ann["file_stem"].map(parse_file_name)
    ann["side"] = parsed.map(lambda p: p["side"] if p else "a")
    mismatch = 0
    for col in ["condition", "day", "sample", "stage"]:
        from_name = parsed.map(lambda p, c=col: p[c] if p else None)
        bad = from_name.notna() & (from_name.astype(str).str.lower() != ann[col].astype(str).str.lower())
        mismatch += int(bad.sum())
    if mismatch:
        print(f"[WARNING] {mismatch} label values disagree between the spreadsheet and the file names. "
              "The spreadsheet value is kept.")

    ann["fruit_id"] = ann["condition"] + "_" + ann["sample"].map(lambda s: f"{s:03d}")
    ann["fruit_day_id"] = ann["fruit_id"] + "_d" + ann["day"].map(lambda d: f"{d:02d}")
    ann["image_id"] = ann["file_stem"]

    if cfg.conditions:
        ann = ann[ann["condition"].isin(cfg.conditions)].copy()
        print(f"[SCAN] Restricted to conditions {cfg.conditions}, {len(ann)} images left.")

    if cfg.limit_fruits:
        rng = np.random.default_rng(cfg.random_state)
        fruits = ann["fruit_id"].unique()
        keep = rng.choice(fruits, size=min(cfg.limit_fruits, len(fruits)), replace=False)
        ann = ann[ann["fruit_id"].isin(keep)].copy()
        print(f"[SCAN] --limit-fruits: {len(keep)} fruits and {len(ann)} images kept.")

    return ann.sort_values(["condition", "sample", "day", "side"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Targets
# ---------------------------------------------------------------------------

def build_targets(inventory: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Derive the remaining days from the daily trajectory of every fruit.

    ready_day       first day on which the fruit reaches --ready-stage.
    days_remaining  ready_day minus the day of the photo, clipped at zero, so
                    a fruit that is already ready has zero days left.
    censored        the fruit left the experiment before reaching the ready
                    stage. Its true ready day is unknown, only a lower bound is
                    known, so it is excluded from the regression target.
    """
    df = inventory.copy()
    ready = df[df["stage"] >= cfg.ready_stage].groupby("fruit_id")["day"].min()
    last = df.groupby("fruit_id")["day"].max()
    df["ready_day"] = df["fruit_id"].map(ready)
    df["last_observed_day"] = df["fruit_id"].map(last)
    df["censored"] = df["ready_day"].isna()
    df["days_remaining"] = (df["ready_day"] - df["day"]).clip(lower=0)
    df["has_days_target"] = ~df["censored"]

    # the ripening index must never go back in time for the same fruit
    ordered = df.sort_values(["fruit_id", "day"]).groupby("fruit_id")["stage"]
    non_monotonic = int(ordered.apply(lambda s: (s.diff().dropna() < 0).any()).sum())
    if non_monotonic:
        print(f"[WARNING] {non_monotonic} fruits have a stage that decreases over time.")

    n_fruits = df["fruit_id"].nunique()
    n_cens = df.loc[df["censored"], "fruit_id"].nunique()
    print(f"[TARGET] ready stage = {cfg.ready_stage}. {n_fruits} fruits, {n_cens} right censored "
          f"(no days target), {int(df['has_days_target'].sum())} images with a days target.")
    return df


# ---------------------------------------------------------------------------
# 3. Fruit detection and crop (YOLO)
# ---------------------------------------------------------------------------

def fruit_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Boolean fruit mask on the white studio background.

    A pixel belongs to the fruit when it is saturated (green skin) or dark
    (purple to black ripe skin). Only the largest connected component is kept
    and its holes are filled, which removes specular highlights.
    """
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"))
    mask = (hsv[..., 1] >= MASK_MIN_SATURATION) | (hsv[..., 2] <= MASK_MAX_VALUE)
    mask = ndi.binary_opening(mask, iterations=2)
    labels, n_comp = ndi.label(mask)
    if n_comp == 0:
        return mask
    sizes = ndi.sum_labels(mask, labels, index=np.arange(1, n_comp + 1))
    largest = labels == (int(np.argmax(sizes)) + 1)
    return ndi.binary_fill_holes(largest)


def threshold_box(img: Image.Image) -> Optional[Tuple[np.ndarray, float]]:
    """Bounding box of the fruit mask, computed on a downsized copy."""
    w, h = img.size
    factor = min(1.0, 400.0 / max(w, h))
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    mask = fruit_mask(np.asarray(small))
    if mask.sum() < 50:
        return None
    rows, cols = np.nonzero(mask)
    box = np.array([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1], dtype=float) / factor
    return box, float("nan")


def build_yolo(cfg: Config):
    """
    Load the YOLO detector.

    yolo-world : open vocabulary YOLO, the classes are given as text prompts
                 (--yolo-prompts avocado). No training is needed, which suits a
                 PoC, but the box quality must be checked in fig_detections.
    yolo       : standard YOLO. The COCO weights have no avocado class, so the
                 most confident box of any class is used. This mode is meant
                 for custom weights (--yolo-weights best.pt) trained on a few
                 hundred labelled photos of the group's own dataset.
    """
    try:
        if cfg.detector == "yolo-world":
            from ultralytics import YOLOWorld

            model = YOLOWorld(cfg.yolo_weights or "yolov8s-worldv2.pt")
            model.set_classes(list(cfg.yolo_prompts))
        else:
            from ultralytics import YOLO

            model = YOLO(cfg.yolo_weights or "yolov8n.pt")
        print(f"[DETECT] {cfg.detector} loaded ({cfg.yolo_weights or 'default weights'})")
        return model
    except Exception as exc:
        print(f"[WARNING] YOLO could not be loaded ({exc}). The threshold detector is used instead.")
        return None


def yolo_boxes(model, paths: Sequence[str], cfg: Config) -> List[Optional[Tuple[np.ndarray, float]]]:
    """Most confident valid YOLO box of every image, or None."""
    device = None if cfg.device == "auto" else cfg.device
    results = model.predict(source=list(paths), conf=cfg.yolo_conf, imgsz=cfg.yolo_imgsz,
                            verbose=False, device=device)
    out: List[Optional[Tuple[np.ndarray, float]]] = []
    for res in results:
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            out.append(None)
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        h, w = res.orig_shape[:2]
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) / float(w * h)
        valid = (area >= cfg.min_box_area) & (area <= 0.98)
        if not valid.any():
            out.append(None)
            continue
        best = int(np.argmax(np.where(valid, conf, -1.0)))
        out.append((xyxy[best], float(conf[best])))
    return out


def square_crop(img: Image.Image, box: np.ndarray, pad: float, size: int) -> Image.Image:
    """Square crop centred on the box, padded with white outside the image."""
    x1, y1, x2, y2 = [float(v) for v in box]
    side = int(round(max(x2 - x1, y2 - y1) * (1.0 + pad)))
    side = max(side, 8)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left, top = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    w, h = img.size
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    src = img.crop((max(left, 0), max(top, 0), min(left + side, w), min(top + side, h)))
    canvas.paste(src, (max(-left, 0), max(-top, 0)))
    return canvas.resize((size, size), Image.BILINEAR)


def run_detection(inventory: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Detect the fruit in every image and save a square crop to out_dir/crops.

    Crops and the detection log are cached, a second run only processes the
    images that are not in the cache.
    """
    crop_dir = cfg.out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.out_dir / "detections.csv"

    done = pd.DataFrame()
    if log_path.exists() and not cfg.no_crop_cache:
        done = pd.read_csv(log_path)
        done = done[done["crop_path"].map(lambda p: Path(str(p)).exists())]
        done = done[done["image_id"].isin(inventory["image_id"])]
        print(f"[CACHE] Reusing {len(done)} cached crops from {crop_dir}")

    todo = inventory[~inventory["image_id"].isin(done.get("image_id", pd.Series(dtype=str)))]
    if todo.empty:
        return done.reset_index(drop=True)

    model = build_yolo(cfg) if cfg.detector in ("yolo", "yolo-world") else None
    records: List[Dict] = []
    total = len(todo)
    chunk = 32
    t0 = time.time()
    for start in range(0, total, chunk):
        part = todo.iloc[start:start + chunk]
        paths = part["path"].tolist()
        yolo_out: List[Optional[Tuple[np.ndarray, float]]] = [None] * len(paths)
        if model is not None:
            try:
                yolo_out = yolo_boxes(model, paths, cfg)
            except Exception as exc:  # keep the run alive on a single bad batch
                print(f"[WARNING] YOLO failed on a batch ({exc}), threshold fallback used.")

        for (_, row), det in zip(part.iterrows(), yolo_out):
            try:
                with Image.open(row["path"]) as img:
                    img = img.convert("RGB")
                    method = cfg.detector if det is not None else "threshold"
                    if cfg.detector == "none":
                        det, method = (np.array([0, 0, img.size[0], img.size[1]], float), float("nan")), "none"
                    elif det is None:
                        det = threshold_box(img)
                    if det is None:
                        det, method = (np.array([0, 0, img.size[0], img.size[1]], float), float("nan")), "full_image"
                    box, conf = det
                    crop = square_crop(img, box, cfg.crop_pad, cfg.crop_size)
                    crop_path = crop_dir / f"{row['image_id']}.jpg"
                    crop.save(crop_path, quality=95)
                    w, h = img.size
            except Exception as exc:
                print(f"[WARNING] Failed on {row['path']}: {exc}")
                continue
            records.append({
                "image_id": row["image_id"],
                "crop_path": str(crop_path),
                "detect_method": method,
                "detect_conf": conf,
                "box_x1": box[0], "box_y1": box[1], "box_x2": box[2], "box_y2": box[3],
                "box_area_ratio": float((box[2] - box[0]) * (box[3] - box[1]) / (w * h)),
            })

        n = min(start + chunk, total)
        if n % 512 < chunk or n == total:
            print(f"[DETECT] {n}/{total} images processed ({time.time() - t0:.0f} s)", flush=True)
            # checkpoint, an interrupted run restarts from here
            pd.concat([done, pd.DataFrame(records)], ignore_index=True).to_csv(log_path, index=False)
        if n % 2048 < chunk:
            gc.collect()

    log = pd.concat([done, pd.DataFrame(records)], ignore_index=True)
    log.to_csv(log_path, index=False)
    print(f"[DETECT] Methods used: {log['detect_method'].value_counts().to_dict()}")
    return log


def plot_detections(data: pd.DataFrame, cfg: Config, n: int = 12) -> None:
    """Visual quality check of the boxes, one image per stage and condition when possible."""
    if data.empty:
        return
    sample = data.groupby(["condition", "stage"]).head(1)
    sample = sample.sample(n=min(n, len(sample)), random_state=cfg.random_state)
    cols = 4
    rows = int(math.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    colours = {"yolo-world": "#2ca02c", "yolo": "#2ca02c", "threshold": "#ff7f0e"}
    for ax, (_, row) in zip(axes, sample.iterrows()):
        with Image.open(row["path"]) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            draw.rectangle([row["box_x1"], row["box_y1"], row["box_x2"], row["box_y2"]],
                           outline=colours.get(row["detect_method"], "#d62728"), width=6)
            ax.imshow(img)
        ax.set_title(f"{row['image_id']}\n{row['detect_method']}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")
    fig.tight_layout()
    save_figure(fig, cfg.out_dir / "fig_detections", cfg)


# ---------------------------------------------------------------------------
# 4. Hand crafted colour and texture features (baseline)
# ---------------------------------------------------------------------------

H_BINS, S_BINS, V_BINS = 16, 8, 8


def crop_features(crop_path: Path) -> Optional[Dict[str, float]]:
    """
    Colour and texture descriptor of the fruit pixels of one crop.

    Only the pixels inside the fruit mask are used, so the white background
    does not dilute the colour statistics. Hue is in the PIL 0 to 255 range.
    """
    with Image.open(crop_path) as img:
        rgb = np.asarray(img.convert("RGB"))
    mask = fruit_mask(rgb)
    if mask.sum() < 100:
        return None
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV")).astype(float)
    gray = rgb.astype(float).mean(axis=2)

    hh, ss, vv = hsv[..., 0][mask], hsv[..., 1][mask], hsv[..., 2][mask]
    r, g, b = (rgb[..., i].astype(float)[mask] for i in range(3))

    feats: Dict[str, float] = {}
    for name, ch in [("h", hh), ("s", ss), ("v", vv), ("r", r), ("g", g), ("b", b)]:
        feats[f"{name}_mean"] = float(ch.mean())
        feats[f"{name}_std"] = float(ch.std())
    for q in (10, 50, 90):
        feats[f"v_p{q}"] = float(np.percentile(vv, q))
        feats[f"s_p{q}"] = float(np.percentile(ss, q))
    feats["dark_fraction"] = float((vv < 70).mean())
    feats["green_fraction"] = float(((hh >= 30) & (hh <= 80) & (ss >= 60)).mean())
    feats["purple_fraction"] = float(((hh >= 180) | (hh <= 15)).mean())
    feats["g_minus_r"] = float((g - r).mean())
    feats["excess_green"] = float((2 * g - r - b).mean())

    for name, ch, bins in [("h", hh, H_BINS), ("s", ss, S_BINS), ("v", vv, V_BINS)]:
        hist, _ = np.histogram(ch, bins=bins, range=(0, 256))
        hist = hist / max(1, hist.sum())
        for i, value in enumerate(hist):
            feats[f"hist_{name}_{i:02d}"] = float(value)

    # texture: gradient magnitude and local contrast inside the fruit,
    # a proxy for the roughness and the gloss of the skin
    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)
    inner = ndi.binary_erosion(mask, iterations=3)
    if inner.sum() < 50:
        inner = mask
    feats["texture_grad_mean"] = float(grad[inner].mean())
    feats["texture_grad_std"] = float(grad[inner].std())
    small = gray[::2, ::2]
    local_var = ndi.uniform_filter(small ** 2, size=5) - ndi.uniform_filter(small, size=5) ** 2
    feats["texture_local_std"] = float(np.sqrt(np.clip(local_var, 0, None))[inner[::2, ::2]].mean())
    feats["specular_fraction"] = float((vv > 220).mean())
    feats["fruit_pixels"] = float(mask.sum())
    return feats


STAT_COLUMNS = (
    [f"{c}_{s}" for c in "hsvrgb" for s in ("mean", "std")]
    + [f"{c}_p{q}" for c in "vs" for q in (10, 50, 90)]
    + ["dark_fraction", "green_fraction", "purple_fraction", "g_minus_r", "excess_green"]
)
HIST_COLUMNS = (
    [f"hist_h_{i:02d}" for i in range(H_BINS)]
    + [f"hist_s_{i:02d}" for i in range(S_BINS)]
    + [f"hist_v_{i:02d}" for i in range(V_BINS)]
)
TEXTURE_COLUMNS = ["texture_grad_mean", "texture_grad_std", "texture_local_std", "specular_fraction"]


def build_feature_table(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Extract features for every crop, with an on disk cache."""
    cache = cfg.out_dir / "features_cache.csv"
    cached = pd.DataFrame()
    if cache.exists() and not cfg.no_feature_cache:
        cached = pd.read_csv(cache)
        cached = cached[cached["image_id"].isin(data["image_id"])]
        print(f"[CACHE] Reusing {len(cached)} cached feature rows from {cache}")

    todo = data[~data["image_id"].isin(cached.get("image_id", pd.Series(dtype=str)))]
    rows: List[Dict] = []
    total = len(todo)
    for i, (_, row) in enumerate(todo.iterrows(), start=1):
        try:
            feats = crop_features(Path(row["crop_path"]))
        except Exception as exc:  # keep the run alive on a corrupted file
            print(f"[WARNING] Failed on {row['crop_path']}: {exc}")
            feats = None
        if feats is not None:
            rows.append({"image_id": row["image_id"], **feats})
        if i % 1000 == 0 or i == total:
            print(f"[FEATURES] {i}/{total} crops processed", flush=True)
            pd.concat([cached, pd.DataFrame(rows)], ignore_index=True).to_csv(cache, index=False)

    features = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
    if not features.empty:
        features.to_csv(cache, index=False)
        print(f"[FEATURES] {len(features)} rows saved to {cache}")
    return features


def condition_columns(data: pd.DataFrame) -> List[str]:
    return [c for c in data.columns if c.startswith("cond_")]


def feature_sets(data: pd.DataFrame) -> Dict[str, List[str]]:
    available = set(data.columns)
    sets = {
        "F1_color_stats": STAT_COLUMNS,
        "F2_hsv_histogram": HIST_COLUMNS,
        "F3_color_full": STAT_COLUMNS + HIST_COLUMNS,
        "F4_color_texture": STAT_COLUMNS + HIST_COLUMNS + TEXTURE_COLUMNS,
    }
    cond = condition_columns(data)
    if len(cond) > 1:
        sets["F5_color_texture_condition"] = STAT_COLUMNS + HIST_COLUMNS + TEXTURE_COLUMNS + cond
    return {k: [c for c in v if c in available] for k, v in sets.items()}


# ---------------------------------------------------------------------------
# 5. Exploratory data analysis
# ---------------------------------------------------------------------------

def run_eda(data: pd.DataFrame, detections: pd.DataFrame, cfg: Config) -> Dict:
    per_fruit = data.groupby("fruit_id").agg(
        condition=("condition", "first"), n_images=("image_id", "size"),
        n_days=("day", "nunique"), ready_day=("ready_day", "first"),
        censored=("censored", "first"), last_stage=("stage", "max"),
    )
    days = data.loc[data["has_days_target"], ["condition", "days_remaining"]]

    summary = {
        "n_images": int(len(data)),
        "n_fruits": int(per_fruit.shape[0]),
        "n_fruit_days": int(data["fruit_day_id"].nunique()),
        "ready_stage": cfg.ready_stage,
        "images_per_condition": data["condition"].value_counts().to_dict(),
        "fruits_per_condition": per_fruit["condition"].value_counts().to_dict(),
        "images_per_stage": data["stage"].value_counts().sort_index().to_dict(),
        "stage_by_condition": pd.crosstab(data["condition"], data["stage"]).to_dict(orient="index"),
        "censored_fruits_per_condition": per_fruit[per_fruit["censored"]]["condition"].value_counts().to_dict(),
        "experiment_length_days": data.groupby("condition")["day"].max().to_dict(),
        "ready_day_by_condition": {
            cond: {k: float(v) for k, v in grp["ready_day"].describe().items()}
            for cond, grp in per_fruit.dropna(subset=["ready_day"]).groupby("condition")
        },
        "days_remaining_by_condition": {
            cond: {k: float(v) for k, v in grp["days_remaining"].describe().items()}
            for cond, grp in days.groupby("condition")
        },
        "stage_imbalance_ratio": float(
            data["stage"].value_counts().max() / data["stage"].value_counts().min()
        ),
    }
    if not detections.empty:
        summary["detection_methods"] = detections["detect_method"].value_counts().to_dict()
        summary["box_area_ratio"] = {
            k: float(v) for k, v in detections["box_area_ratio"].describe().items()
        }

    path = cfg.out_dir / "eda_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[EDA] Summary written to {path}")

    conds = sorted(data["condition"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ct = pd.crosstab(data["condition"], data["stage"]).reindex(columns=STAGES, fill_value=0)
    ct.plot(kind="bar", stacked=True, ax=axes[0, 0], colormap="RdYlGn_r", edgecolor="white")
    axes[0, 0].set_xlabel("Storage condition")
    axes[0, 0].set_ylabel("Number of images")
    axes[0, 0].set_title("Ripening stage per storage condition")
    axes[0, 0].legend(title="Stage")
    axes[0, 0].tick_params(axis="x", rotation=0)

    for cond in conds:
        traj = data[data["condition"] == cond].groupby("day")["stage"].agg(["mean", "std"])
        axes[0, 1].plot(traj.index, traj["mean"], marker="o", ms=3, label=cond)
        axes[0, 1].fill_between(traj.index, traj["mean"] - traj["std"], traj["mean"] + traj["std"], alpha=0.15)
    axes[0, 1].axhline(cfg.ready_stage, color="k", ls="--", lw=1)
    axes[0, 1].set_xlabel("Day of experiment")
    axes[0, 1].set_ylabel("Ripening stage, mean and sd")
    axes[0, 1].set_title("Ripening trajectories, dashed line = ready stage")
    axes[0, 1].legend(title="Condition")

    for cond in conds:
        vals = days[days["condition"] == cond]["days_remaining"]
        if len(vals):
            axes[1, 0].hist(vals, bins=np.arange(-0.5, vals.max() + 1.5, 1), alpha=0.5, label=cond)
    axes[1, 0].set_xlabel("Days remaining until the ready stage")
    axes[1, 0].set_ylabel("Number of images")
    axes[1, 0].set_title("Regression target distribution")
    axes[1, 0].legend(title="Condition")

    ready = per_fruit.dropna(subset=["ready_day"])
    box_data = [ready[ready["condition"] == c]["ready_day"].to_numpy() for c in conds]
    axes[1, 1].boxplot(box_data)
    axes[1, 1].set_xticks(range(1, len(conds) + 1), conds)
    axes[1, 1].set_xlabel("Storage condition")
    axes[1, 1].set_ylabel("Day on which the ready stage is reached")
    axes[1, 1].set_title("Green life per fruit, the temperature effect")
    fig.tight_layout()
    save_figure(fig, cfg.out_dir / "fig_eda", cfg)
    return summary


def save_figure(fig, stem: Path, cfg: Config) -> None:
    for ext in cfg.figure_formats:
        fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Leakage aware splits
# ---------------------------------------------------------------------------

def make_outer_splits(data: pd.DataFrame, cfg: Config) -> np.ndarray:
    """
    Outer folds grouped by fruit.

    Every fruit is photographed on both sides and on every day, so a random
    image split would put almost identical photos of the same fruit in train
    and test. The fruit is therefore the grouping unit. The folds are
    stratified by storage condition, which is constant within a fruit, so that
    every fold contains every condition in the same proportion.
    """
    groups = data["fruit_id"].to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = min(cfg.n_splits, n_groups)
    if n_splits < 2:
        raise RuntimeError("Not enough fruits to build a cross validation.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    fold = np.full(len(data), -1, dtype=int)
    for k, (_, test_idx) in enumerate(splitter.split(data, data["condition"], groups=groups)):
        fold[test_idx] = k
    out = data[["image_id", "fruit_id", "condition", "day", "stage", "days_remaining"]].copy()
    out["outer_fold"] = fold
    out.to_csv(cfg.out_dir / "splits.csv", index=False)
    print(f"[SPLIT] {n_splits} outer folds over {n_groups} fruits. Fold assignment saved to splits.csv")
    return fold


# ---------------------------------------------------------------------------
# 7. Metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_F1": float(f1_score(y_true, y_pred, average="macro", labels=list(STAGES), zero_division=0)),
        "QWK": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(STAGES))),
        "MAE_stage": float(np.mean(np.abs(y_true - y_pred))),
        "within_1_stage": float(np.mean(np.abs(y_true - y_pred) <= 1)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAPE is not reported because the target is zero for every ready fruit."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "bias": float(err.mean()),
        "within_1_day": float(np.mean(np.abs(err) <= 1.0)),
        "within_2_days": float(np.mean(np.abs(err) <= 2.0)),
        # over estimating the green life is the dangerous error for export
        "share_overestimated_2d": float(np.mean(err > 2.0)),
    }


TASKS = {
    "stage": {"metrics": classification_metrics, "key": "macro_F1", "higher_is_better": True},
    "days": {"metrics": regression_metrics, "key": "MAE", "higher_is_better": False},
}


def bootstrap_ci(task: str, y: np.ndarray, pred: np.ndarray, groups: np.ndarray, cfg: Config) -> Dict[str, List[float]]:
    """Fruit level bootstrap so that dependent images are resampled together."""
    if cfg.n_bootstrap <= 0:
        return {}
    metric_fn = TASKS[task]["metrics"]
    rng = np.random.default_rng(cfg.random_state)
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    collected: Dict[str, List[float]] = {}
    for _ in range(cfg.n_bootstrap):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[g] for g in picked])
        if len(np.unique(y[idx])) < 2:
            continue
        for key, value in metric_fn(y[idx], pred[idx]).items():
            collected.setdefault(key, []).append(value)
    return {
        key: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        for key, v in collected.items() if v
    }


def summarise_predictions(task: str, y: np.ndarray, pred: np.ndarray, groups: np.ndarray,
                          folds: np.ndarray, cfg: Config) -> Dict[str, float]:
    """Pooled metrics, mean and sd over the outer folds and bootstrap CI."""
    ok = np.isfinite(pred)
    metric_fn = TASKS[task]["metrics"]
    pooled = metric_fn(y[ok], pred[ok])
    per_fold = pd.DataFrame([
        metric_fn(y[ok & (folds == k)], pred[ok & (folds == k)])
        for k in sorted(np.unique(folds[ok]))
    ])
    row: Dict[str, float] = {"n_evaluated": int(ok.sum()), "n_folds_evaluated": int(len(per_fold))}
    row.update({f"pooled_{k}": v for k, v in pooled.items()})
    row.update({f"foldmean_{k}": float(per_fold[k].mean()) for k in per_fold.columns})
    row.update({f"foldsd_{k}": float(per_fold[k].std()) for k in per_fold.columns})
    for key, (lo, hi) in bootstrap_ci(task, y[ok], pred[ok], groups[ok], cfg).items():
        row[f"ci95_{key}_low"] = lo
        row[f"ci95_{key}_high"] = hi
    return row


# ---------------------------------------------------------------------------
# 8. Baseline models on hand crafted features, nested cross validation
# ---------------------------------------------------------------------------

def build_models(task: str, cfg: Config) -> Dict[str, Tuple[Pipeline, Dict, str]]:
    rs = cfg.random_state
    if task == "stage":
        catalogue = {
            "baseline_majority": (Pipeline([("model", DummyClassifier(strategy="most_frequent"))]), {}),
            "logistic": (
                Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000))]),
                {"model__C": [0.01, 0.1, 1.0, 10.0]},
            ),
            "random_forest": (
                Pipeline([("model", RandomForestClassifier(random_state=rs, n_jobs=cfg.n_jobs,
                                                           class_weight="balanced"))]),
                {"model__n_estimators": [300], "model__max_depth": [None, 16],
                 "model__min_samples_leaf": [1, 5], "model__max_features": ["sqrt", 0.3]},
            ),
        }
        scoring = "f1_macro"
    else:
        catalogue = {
            "baseline_mean": (Pipeline([("model", DummyRegressor(strategy="mean"))]), {}),
            "ridge": (
                Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
                {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            ),
            "random_forest": (
                Pipeline([("model", RandomForestRegressor(random_state=rs, n_jobs=cfg.n_jobs))]),
                {"model__n_estimators": [300], "model__max_depth": [None, 16],
                 "model__min_samples_leaf": [1, 5], "model__max_features": ["sqrt", 0.3]},
            ),
        }
        scoring = "neg_mean_absolute_error"
    if cfg.models:
        catalogue = {k: v for k, v in catalogue.items() if k in cfg.models or k.startswith("baseline")}
    return {k: (est, grid, scoring) for k, (est, grid) in catalogue.items()}


def predict_proba_aligned(model, X: pd.DataFrame) -> np.ndarray:
    """Class probabilities in the fixed STAGES order, zero for unseen classes."""
    proba = np.zeros((len(X), len(STAGES)))
    raw = model.predict_proba(X)
    for j, cls in enumerate(model.classes_):
        if int(cls) in STAGES:
            proba[:, STAGES.index(int(cls))] = raw[:, j]
    return proba


def nested_cv(task: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, folds: np.ndarray,
              estimator: Pipeline, grid: Dict, scoring: str, cfg: Config
              ) -> Tuple[np.ndarray, Optional[np.ndarray], List[Dict]]:
    """Outer grouped CV with an inner grouped grid search on the training part."""
    oof = np.full(len(y), np.nan, dtype=float)
    proba = np.full((len(y), len(STAGES)), np.nan) if task == "stage" else None
    chosen: List[Dict] = []

    for k in sorted(np.unique(folds)):
        train, test = folds != k, folds == k
        X_tr, y_tr, g_tr = X[train], y[train], groups[train]
        if grid:
            n_inner = min(cfg.n_inner_splits, len(np.unique(g_tr)))
            search = GridSearchCV(clone(estimator), grid, scoring=scoring,
                                  cv=GroupKFold(n_splits=max(2, n_inner)), n_jobs=cfg.n_jobs, refit=True)
            search.fit(X_tr, y_tr, groups=g_tr)
            model = search.best_estimator_
            chosen.append({"outer_fold": int(k), **search.best_params_})
        else:
            model = clone(estimator).fit(X_tr, y_tr)
            chosen.append({"outer_fold": int(k)})

        oof[test] = model.predict(X[test])
        if proba is not None:
            proba[test] = predict_proba_aligned(model, X[test])
    if task == "days":
        oof = np.clip(oof, 0, None)
    return oof, proba, chosen


def prepare_X(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    X = data[columns].replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median(numeric_only=True))


def task_rows(data: pd.DataFrame, task: str) -> np.ndarray:
    return np.ones(len(data), bool) if task == "stage" else data["has_days_target"].to_numpy()


def task_target(data: pd.DataFrame, task: str) -> np.ndarray:
    col = "stage" if task == "stage" else "days_remaining"
    return data[col].to_numpy(dtype=float)


def run_sklearn_experiments(data: pd.DataFrame, cfg: Config, registry: Dict) -> Dict[str, List[Dict]]:
    """Every feature set x every model, for both tasks."""
    results: Dict[str, List[Dict]] = {"stage": [], "days": []}
    sets = feature_sets(data)

    for task in ("stage", "days"):
        rows = task_rows(data, task)
        sub = data[rows].reset_index(drop=True)
        y = task_target(sub, task)
        groups = sub["fruit_id"].to_numpy()
        folds = sub["outer_fold"].to_numpy()
        catalogue = build_models(task, cfg)
        best = {"score": -np.inf}

        for set_name, columns in sets.items():
            if not columns:
                continue
            X = prepare_X(sub, columns)
            for model_name, (estimator, grid, scoring) in catalogue.items():
                print(f"[CV] {task} | {set_name} x {model_name} ({len(columns)} features)")
                oof, proba, chosen = nested_cv(task, X, y, groups, folds, estimator, grid, scoring, cfg)
                row = {"task": task, "feature_set": set_name, "n_features": len(columns), "model": model_name}
                row.update(summarise_predictions(task, y, oof, groups, folds, cfg))
                results[task].append(row)

                key = f"{model_name}|{set_name}"
                full_pred = np.full(len(data), np.nan)
                full_pred[rows] = oof
                entry = {"pred": full_pred, "model": model_name, "feature_set": set_name, "metrics": row}
                if proba is not None:
                    full_proba = np.full((len(data), len(STAGES)), np.nan)
                    full_proba[rows] = proba
                    entry["proba"] = full_proba
                registry[task][key] = entry

                score = row[f"pooled_{TASKS[task]['key']}"] * (1 if TASKS[task]["higher_is_better"] else -1)
                if not model_name.startswith("baseline") and score > best["score"]:
                    best = {"score": score, "model": model_name, "feature_set": set_name,
                            "columns": columns, "estimator": estimator, "grid": grid,
                            "scoring": scoring, "oof": oof}

        if np.isfinite(best["score"]):
            finalize_sklearn_model(task, sub, y, groups, folds, best, cfg)
    return results


def permutation_test(task: str, X: pd.DataFrame, y: np.ndarray, blocks: np.ndarray, folds: np.ndarray,
                     estimator: Pipeline, observed: float, cfg: Config) -> Optional[Dict[str, float]]:
    """
    Permute the target between fruit-day blocks (both sides of a fruit on one
    day keep the same permuted label) and rerun the outer CV without the inner
    search, to check that the observed score is better than chance.
    """
    if cfg.n_permutations <= 0:
        return None
    metric_fn = TASKS[task]["metrics"]
    key = TASKS[task]["key"]
    sign = 1 if TASKS[task]["higher_is_better"] else -1
    rng = np.random.default_rng(cfg.random_state + 1)
    unique = np.unique(blocks)
    block_target = {b: y[blocks == b][0] for b in unique}
    better = 0
    for _ in range(cfg.n_permutations):
        mapping = dict(zip(unique, rng.permutation(list(block_target.values()))))
        y_perm = np.array([mapping[b] for b in blocks], dtype=float)
        oof = np.full(len(y), np.nan)
        for k in sorted(np.unique(folds)):
            train, test = folds != k, folds == k
            oof[test] = clone(estimator).fit(X[train], y_perm[train]).predict(X[test])
        if sign * metric_fn(y_perm, oof)[key] >= sign * observed:
            better += 1
    return {"n_permutations": int(cfg.n_permutations),
            "p_value": float((better + 1) / (cfg.n_permutations + 1)),
            f"observed_{key}": float(observed)}


def finalize_sklearn_model(task: str, sub: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                           folds: np.ndarray, best: Dict, cfg: Config) -> None:
    X = prepare_X(sub, best["columns"])
    key = TASKS[task]["key"]
    observed = TASKS[task]["metrics"](y, best["oof"])[key]

    perm = permutation_test(task, X, y, sub["fruit_day_id"].to_numpy(), folds, best["estimator"], observed, cfg)
    if perm:
        (cfg.out_dir / f"permutation_test_{task}.json").write_text(
            json.dumps({**perm, "model": best["model"], "feature_set": best["feature_set"]}, indent=2),
            encoding="utf-8")
        print(f"[PERMUTATION] {task}: p = {perm['p_value']:.4f}")

    if best["grid"]:
        n_inner = min(cfg.n_inner_splits, len(np.unique(groups)))
        search = GridSearchCV(clone(best["estimator"]), best["grid"], scoring=best["scoring"],
                              cv=GroupKFold(n_splits=max(2, n_inner)), n_jobs=cfg.n_jobs, refit=True)
        search.fit(X, y, groups=groups)
        final_model, final_params = search.best_estimator_, search.best_params_
    else:
        final_model, final_params = clone(best["estimator"]).fit(X, y), {}

    joblib.dump(
        {"model": final_model, "columns": best["columns"], "task": task,
         "target": "ripening_stage" if task == "stage" else "days_remaining_to_ready_stage",
         "ready_stage": cfg.ready_stage, "feature_set": best["feature_set"], "params": final_params,
         "config": {k: str(v) for k, v in asdict(cfg).items()}},
        cfg.out_dir / f"final_model_{task}.joblib",
    )
    print(f"[MODEL] {task}: final {best['model']} / {best['feature_set']} saved, best_params={final_params}")

    try:
        imp = permutation_importance(final_model, X, y, n_repeats=5, random_state=cfg.random_state,
                                     scoring=best["scoring"], n_jobs=cfg.n_jobs)
        imp_df = (pd.DataFrame({"feature": best["columns"], "importance": imp.importances_mean,
                                "std": imp.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
        imp_df.to_csv(cfg.out_dir / f"feature_importance_{task}.csv", index=False)
        plot_importance(imp_df, task, cfg)
    except Exception as exc:
        print(f"[WARNING] Permutation importance failed: {exc}")


# ---------------------------------------------------------------------------
# 9. Multitask CNN (proposed model) and Grad-CAM
# ---------------------------------------------------------------------------

if HAS_TORCH:

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    class CropDataset(Dataset):
        """Crops with stage index, remaining days, regression mask and condition vector."""

        def __init__(self, frame: pd.DataFrame, conditions: List[str], use_condition: bool, transform):
            self.paths = frame["crop_path"].tolist()
            self.stage = (frame["stage"].to_numpy(dtype=int) - 1).tolist()
            self.days = frame["days_remaining"].fillna(0.0).to_numpy(dtype=np.float32).tolist()
            self.mask = frame["has_days_target"].to_numpy(dtype=np.float32).tolist()
            cond_index = {c: i for i, c in enumerate(conditions)}
            self.cond = frame["condition"].map(cond_index).to_numpy(dtype=int).tolist()
            self.n_cond = len(conditions) if use_condition else 0
            self.transform = transform

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, i: int):
            with Image.open(self.paths[i]) as img:
                x = self.transform(img.convert("RGB"))
            cond = torch.zeros(self.n_cond)
            if self.n_cond:
                cond[self.cond[i]] = 1.0
            return x, self.stage[i], torch.tensor(self.days[i]), torch.tensor(self.mask[i]), cond

    def build_backbone(name: str, pretrained: bool = True):
        """Pretrained backbone without its classifier, its feature size and the Grad-CAM layer."""
        if name == "resnet18":
            net = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None)
            dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, dim, net.layer4
        if name == "resnet50":
            net = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT if pretrained else None)
            dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, dim, net.layer4
        if name == "efficientnet_b0":
            net = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            dim = net.classifier[-1].in_features
            net.classifier = nn.Identity()
            return net, dim, net.features[-1]
        if name == "convnext_tiny":
            net = tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            dim = net.classifier[-1].in_features
            net.classifier[-1] = nn.Identity()
            return net, dim, net.features[-1]
        raise ValueError(f"Unknown backbone: {name}")

    def build_backbone_safe(name: str, pretrained: bool):
        """Pretrained weights when they can be downloaded, random weights otherwise (with a warning)."""
        if pretrained:
            try:
                return build_backbone(name, True)
            except Exception as exc:
                print(f"[WARNING] ImageNet weights for {name} could not be loaded ({exc}). "
                      "Random initialisation is used, the results will be much worse.")
        return build_backbone(name, False)

    class MultiTaskNet(nn.Module):
        """
        Shared backbone with two heads.
        Head A: ripening stage, 5 logits, classification.
        Head B: remaining days, 1 output in units of y_scale, regression.
        The optional condition vector is concatenated to the image embedding.
        """

        def __init__(self, backbone: str, n_classes: int, n_cond: int, pretrained: bool = True):
            super().__init__()
            self.backbone, dim, self.cam_layer = build_backbone_safe(backbone, pretrained)
            self.n_cond = n_cond
            self.dropout = nn.Dropout(0.2)
            self.cls_head = nn.Linear(dim + n_cond, n_classes)
            self.reg_head = nn.Sequential(nn.Linear(dim + n_cond, 128), nn.ReLU(), nn.Linear(128, 1))

        def forward(self, x, cond=None):
            z = self.backbone(x)
            if self.n_cond and cond is not None:
                z = torch.cat([z, cond], dim=1)
            z = self.dropout(z)
            return self.cls_head(z), self.reg_head(z).squeeze(1)

    def make_transforms(img_size: int, train: bool):
        """
        Augmentation for robustness to illumination and orientation. The hue
        jitter is kept very small on purpose, because the hue of the skin is
        the signal that the model must learn.
        """
        if train:
            return T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(180, fill=(255, 255, 255)),
                T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def resolve_device(cfg: Config) -> "torch.device":
        if cfg.device != "auto":
            return torch.device(cfg.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def make_loader(frame, conditions, cfg, train: bool):
        ds = CropDataset(frame, conditions, cfg.use_condition, make_transforms(cfg.img_size, train))
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train, num_workers=cfg.num_workers,
                          pin_memory=torch.cuda.is_available(), drop_last=train and len(ds) > cfg.batch_size)

    def masked_reg_loss(pred, target, mask):
        loss = F.smooth_l1_loss(pred, target, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    def train_cnn(train_df: pd.DataFrame, val_df: pd.DataFrame, backbone: str, mode: str,
                  conditions: List[str], cfg: Config, tag: str) -> Tuple["MultiTaskNet", float, pd.DataFrame]:
        """
        Train one network with early stopping on an inner validation set made of
        fruits that belong to the training part of the outer fold only.
        """
        device = resolve_device(cfg)
        torch.manual_seed(cfg.random_state)
        y_scale = float(max(train_df.loc[train_df["has_days_target"], "days_remaining"].max(), 1.0))
        reg_weight = cfg.reg_weight if mode == "multitask" else 0.0

        model = MultiTaskNet(backbone, len(STAGES), len(conditions) if cfg.use_condition else 0).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
        use_amp = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        ce = nn.CrossEntropyLoss(label_smoothing=0.05)

        train_loader = make_loader(train_df, conditions, cfg, train=True)
        val_loader = make_loader(val_df, conditions, cfg, train=False)

        best_loss, best_state, wait, history = np.inf, None, 0, []
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            t0, run_loss, n_seen = time.time(), 0.0, 0
            for x, stage, days, mask, cond in train_loader:
                x, stage = x.to(device), stage.to(device)
                days, mask, cond = days.to(device) / y_scale, mask.to(device), cond.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits, reg = model(x, cond)
                    loss = ce(logits, stage)
                    if reg_weight > 0:
                        loss = loss + reg_weight * masked_reg_loss(reg.float(), days, mask)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run_loss += float(loss.item()) * len(x)
                n_seen += len(x)
            sched.step()

            probs, reg_pred = predict_cnn_loader(model, val_loader, device, y_scale)
            val_ce = float(F.nll_loss(torch.log(torch.tensor(probs).clamp(min=1e-8)),
                                      torch.tensor(val_df["stage"].to_numpy() - 1)).item())
            m = val_df["has_days_target"].to_numpy()
            val_mae = float(np.mean(np.abs(reg_pred[m] - val_df["days_remaining"].to_numpy()[m]))) if m.any() else np.nan
            val_f1 = f1_score(val_df["stage"], probs.argmax(1) + 1, average="macro")
            # same objective as training, the regression term in y_scale units
            val_loss = val_ce + (reg_weight * val_mae / y_scale if reg_weight > 0 and np.isfinite(val_mae) else 0.0)
            history.append({"tag": tag, "epoch": epoch, "train_loss": run_loss / max(1, n_seen),
                            "val_ce": val_ce, "val_macro_F1": val_f1, "val_MAE_days": val_mae,
                            "val_objective": val_loss, "seconds": time.time() - t0})
            print(f"[CNN] {tag} epoch {epoch}/{cfg.epochs} train_loss={run_loss / max(1, n_seen):.4f} "
                  f"val_F1={val_f1:.3f} val_MAE={val_mae:.2f} d ({time.time() - t0:.0f} s)")

            if val_loss < best_loss - 1e-4:
                best_loss, wait = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg.patience:
                    print(f"[CNN] {tag} early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model, y_scale, pd.DataFrame(history)

    @torch.no_grad()
    def predict_cnn_loader(model, loader, device, y_scale: float) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        probs, regs = [], []
        for x, _, _, _, cond in loader:
            logits, reg = model(x.to(device), cond.to(device))
            probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            regs.append(reg.float().cpu().numpy() * y_scale)
        return np.concatenate(probs), np.clip(np.concatenate(regs), 0, None)

    def grad_cam(model, x, cond, head: str, class_idx: int = 0) -> np.ndarray:
        """
        Grad-CAM on the last convolutional block for the stage head (selected
        class logit) or for the days head (regression output).
        """
        store: Dict[str, "torch.Tensor"] = {}

        def fwd_hook(_module, _inp, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        handle = model.cam_layer.register_forward_hook(fwd_hook)
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits, reg = model(x, cond)
            score = logits[0, class_idx] if head == "stage" else reg[0]
            score.backward()
        handle.remove()
        act, grad = store["act"].detach(), store["grad"].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / cam.max().clamp(min=1e-8)).cpu().numpy()

    def plot_gradcam(model, test_df: pd.DataFrame, conditions: List[str], y_scale: float,
                     cfg: Config, stem: str) -> None:
        if cfg.gradcam_per_stage <= 0 or test_df.empty:
            return
        device = next(model.parameters()).device
        picks = pd.concat([
            g.sample(n=min(cfg.gradcam_per_stage, len(g)), random_state=cfg.random_state)
            for _, g in test_df.groupby("stage")
        ]).reset_index(drop=True)
        tf = make_transforms(cfg.img_size, train=False)
        fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
        axes = np.atleast_2d(axes)
        for i, (_, row) in enumerate(picks.iterrows()):
            with Image.open(row["crop_path"]) as img:
                img = img.convert("RGB")
                x = tf(img).unsqueeze(0).to(device)
                shown = np.asarray(img.resize((cfg.img_size, cfg.img_size)))
            cond = torch.zeros(1, len(conditions) if cfg.use_condition else 0, device=device)
            if cfg.use_condition:
                cond[0, conditions.index(row["condition"])] = 1.0
            with torch.no_grad():
                logits, reg = model(x, cond)
            pred_stage = int(logits.argmax(1).item())
            pred_days = float(reg.item() * y_scale)
            cam_stage = grad_cam(model, x, cond, "stage", pred_stage)
            cam_days = grad_cam(model, x, cond, "days")
            axes[i, 0].imshow(shown)
            true_days = "censored" if not row["has_days_target"] else f"{row['days_remaining']:.0f} d"
            axes[i, 0].set_title(f"{row['image_id']}\ntrue stage {row['stage']}, {true_days}", fontsize=8)
            axes[i, 1].imshow(shown)
            axes[i, 1].imshow(cam_stage, cmap="jet", alpha=0.45)
            axes[i, 1].set_title(f"Grad-CAM stage head, pred {pred_stage + 1}", fontsize=8)
            axes[i, 2].imshow(shown)
            axes[i, 2].imshow(cam_days, cmap="jet", alpha=0.45)
            axes[i, 2].set_title(f"Grad-CAM days head, pred {pred_days:.1f} d", fontsize=8)
            for ax in axes[i]:
                ax.axis("off")
        fig.tight_layout()
        save_figure(fig, cfg.out_dir / stem, cfg)

    def inner_split(frame: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=cfg.random_state)
        tr, va = next(gss.split(frame, groups=frame["fruit_id"]))
        return frame.iloc[tr].reset_index(drop=True), frame.iloc[va].reset_index(drop=True)


def run_cnn_experiments(data: pd.DataFrame, cfg: Config, registry: Dict) -> Dict[str, List[Dict]]:
    """Train every backbone x mode on the same outer folds as the baselines."""
    results: Dict[str, List[Dict]] = {"stage": [], "days": []}
    if cfg.skip_cnn:
        return results
    if not HAS_TORCH:
        print("[WARNING] PyTorch/torchvision are not installed, the CNN part is skipped. "
              "Install them with: pip install torch torchvision")
        return results

    device = resolve_device(cfg)
    print(f"[CNN] Device: {device}")
    if device.type == "cpu":
        print("[WARNING] Training on CPU is very slow. Use a GPU (e.g. Google Colab) or "
              "--cnn-max-folds 1 --epochs 3 for a first test.")

    conditions = sorted(data["condition"].unique())
    folds_all = data["outer_fold"].to_numpy()
    fold_ids = sorted(np.unique(folds_all))
    if cfg.cnn_max_folds and cfg.cnn_max_folds < len(fold_ids):
        fold_ids = fold_ids[: cfg.cnn_max_folds]
        print(f"[WARNING] --cnn-max-folds {cfg.cnn_max_folds}: the CNN rows of the metrics are computed on "
              f"{len(fold_ids)} of {len(np.unique(folds_all))} folds only (see n_folds_evaluated), "
              "they are not directly comparable with the baselines. Use all folds for the final report.")
    histories = []

    for backbone in cfg.backbones:
        for mode in cfg.cnn_modes:
            name = f"cnn_{backbone}_{mode}"
            probs = np.full((len(data), len(STAGES)), np.nan)
            days_pred = np.full(len(data), np.nan)
            for k in fold_ids:
                train_df = data[folds_all != k].reset_index(drop=True)
                test_idx = np.flatnonzero(folds_all == k)
                test_df = data.iloc[test_idx].reset_index(drop=True)
                tr, va = inner_split(train_df, cfg)
                model, y_scale, hist = train_cnn(tr, va, backbone, mode, conditions, cfg, f"{name} fold {k}")
                histories.append(hist.assign(model=name, outer_fold=k))
                loader = make_loader(test_df, conditions, cfg, train=False)
                p, d = predict_cnn_loader(model, loader, next(model.parameters()).device, y_scale)
                probs[test_idx], days_pred[test_idx] = p, d
                if k == fold_ids[0]:
                    try:
                        plot_gradcam(model, test_df, conditions, y_scale, cfg, f"fig_gradcam_{backbone}_{mode}")
                    except Exception as exc:
                        print(f"[WARNING] Grad-CAM failed: {exc}")
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            stage_pred = np.where(np.isfinite(probs[:, 0]), probs.argmax(1) + 1, np.nan)
            for task, pred in (("stage", stage_pred), ("days", days_pred)):
                if task == "days" and mode != "multitask":
                    continue
                rows = task_rows(data, task)
                y = task_target(data, task)
                row = {"task": task, "feature_set": "image_crop", "n_features": cfg.img_size, "model": name}
                row.update(summarise_predictions(task, y[rows], pred[rows], data["fruit_id"].to_numpy()[rows],
                                                 folds_all[rows], cfg))
                results[task].append(row)
                entry = {"pred": np.where(rows, pred, np.nan), "model": name, "feature_set": "image_crop",
                         "metrics": row}
                if task == "stage":
                    entry["proba"] = probs
                registry[task][f"{name}|image_crop"] = entry

    if histories:
        pd.concat(histories, ignore_index=True).to_csv(cfg.out_dir / "cnn_training_history.csv", index=False)
    if cfg.fit_final_cnn:
        fit_final_cnn(data, conditions, results, cfg)
    return results


def fit_final_cnn(data: pd.DataFrame, conditions: List[str], results: Dict, cfg: Config) -> None:
    """Refit the best multitask CNN on all the data, for the Streamlit/Gradio interface."""
    days_rows = [r for r in results["days"] if r["model"].startswith("cnn_")]
    if days_rows:
        best = min(days_rows, key=lambda r: r["pooled_MAE"])["model"]
    else:
        best = f"cnn_{cfg.backbones[0]}_{cfg.cnn_modes[0]}"
    backbone = best[len("cnn_"):].rsplit("_", 1)[0]
    mode = best.rsplit("_", 1)[1]
    if mode == "only":  # cnn_<backbone>_cls_only
        backbone, mode = best[len("cnn_"):-len("_cls_only")], "cls_only"
    tr, va = inner_split(data.reset_index(drop=True), cfg)
    model, y_scale, hist = train_cnn(tr, va, backbone, mode, conditions, cfg, f"final {best}")
    torch.save(
        {"state_dict": model.state_dict(), "backbone": backbone, "mode": mode, "y_scale": y_scale,
         "img_size": cfg.img_size, "conditions": conditions, "use_condition": cfg.use_condition,
         "ready_stage": cfg.ready_stage, "stages": list(STAGES), "crop_pad": cfg.crop_pad,
         "crop_size": cfg.crop_size, "detector": cfg.detector},
        cfg.out_dir / "cnn_final.pt",
    )
    hist.to_csv(cfg.out_dir / "cnn_final_history.csv", index=False)
    print(f"[MODEL] Final CNN {best} saved to cnn_final.pt")


# ---------------------------------------------------------------------------
# 10. Lot decision rule and simulation
# ---------------------------------------------------------------------------

def decide_destination(days: np.ndarray, stage: np.ndarray, conf: Optional[np.ndarray], cfg: Config) -> str:
    """
    Transparent lot rule of the proposal.

    A lot is as good as its worst fruits, so the decision uses a low quantile
    of the remaining days, e.g. the 10th percentile when --lot-quantile 0.9,
    meaning that 90 percent of the fruits must have enough green life.
    """
    q = float(np.percentile(days, 100.0 * (1.0 - cfg.lot_quantile)))
    if conf is not None and float(np.mean(conf)) < cfg.review_confidence:
        return "human_review"
    if q >= cfg.export_days + cfg.safety_margin:
        return "export_sea"
    if q >= cfg.domestic_days + cfg.safety_margin:
        return "domestic_distant"
    if float(np.mean(stage >= 5)) >= 0.5:
        return "processing"
    return "local_market"


def aggregate_fruit_day(data: pd.DataFrame, stage_proba: np.ndarray, days_pred: np.ndarray) -> pd.DataFrame:
    """Average the two sides of a fruit photographed on the same day."""
    df = data[["fruit_day_id", "fruit_id", "condition", "day", "stage", "days_remaining",
               "has_days_target", "outer_fold"]].copy()
    for j, s in enumerate(STAGES):
        df[f"p_stage_{s}"] = stage_proba[:, j]
    df["days_pred"] = days_pred
    df = df.dropna(subset=["days_pred", "p_stage_1"])
    agg = df.groupby("fruit_day_id").agg({
        "fruit_id": "first", "condition": "first", "day": "first", "stage": "first",
        "days_remaining": "first", "has_days_target": "first", "outer_fold": "first",
        "days_pred": "mean", **{f"p_stage_{s}": "mean" for s in STAGES},
    }).reset_index()
    p = agg[[f"p_stage_{s}" for s in STAGES]].to_numpy()
    agg["stage_pred"] = p.argmax(1) + 1
    agg["confidence"] = p.max(1)
    return agg


def pick_lot_models(registry: Dict) -> Optional[Tuple[str, str]]:
    """Best multitask CNN if available, otherwise best baseline classifier and regressor."""
    days = {k: v for k, v in registry["days"].items() if not v["model"].startswith("baseline")}
    stage = {k: v for k, v in registry["stage"].items()
             if not v["model"].startswith("baseline") and "proba" in v}
    if not days or not stage:
        return None
    cnn_days = {k: v for k, v in days.items() if k.startswith("cnn_") and k in stage}
    if cnn_days:
        best = min(cnn_days, key=lambda k: cnn_days[k]["metrics"]["pooled_MAE"])
        return best, best
    best_days = min(days, key=lambda k: days[k]["metrics"]["pooled_MAE"])
    best_stage = max(stage, key=lambda k: stage[k]["metrics"]["pooled_macro_F1"])
    return best_stage, best_days


def run_lot_simulation(data: pd.DataFrame, registry: Dict, cfg: Config) -> Optional[Dict]:
    """
    Build synthetic lots from the out of fold predictions and compare the
    recommended destination with the destination given by the true labels.

    A lot is drawn from fruits of one storage condition photographed on days d
    to d + --lot-day-spread, which mimics fruits harvested or cooled at slightly
    different moments. Only fruits with a known ready day are used.
    """
    picked = pick_lot_models(registry)
    if picked is None:
        print("[LOTS] No model with both heads available, lot simulation skipped.")
        return None
    stage_key, days_key = picked
    fd = aggregate_fruit_day(data, registry["stage"][stage_key]["proba"], registry["days"][days_key]["pred"])
    fd = fd[fd["has_days_target"]].reset_index(drop=True)
    fd.to_csv(cfg.out_dir / "predictions_fruit_day.csv", index=False)

    fruit_level = {
        "stage": classification_metrics(fd["stage"], fd["stage_pred"]),
        "days": regression_metrics(fd["days_remaining"], fd["days_pred"]),
    }

    rng = np.random.default_rng(cfg.random_state + 7)
    records = []
    for cond, grp in fd.groupby("condition"):
        for day in sorted(grp["day"].unique()):
            pool = grp[(grp["day"] >= day) & (grp["day"] <= day + cfg.lot_day_spread)]
            pool = pool.drop_duplicates("fruit_id")
            if len(pool) < max(3, cfg.lot_size // 4):
                continue
            size = min(cfg.lot_size, len(pool))
            for lot in range(cfg.lots_per_day):
                lot_df = pool.iloc[rng.choice(len(pool), size=size, replace=False)]
                pred = decide_destination(lot_df["days_pred"].to_numpy(), lot_df["stage_pred"].to_numpy(),
                                          lot_df["confidence"].to_numpy(), cfg)
                true = decide_destination(lot_df["days_remaining"].to_numpy(), lot_df["stage"].to_numpy(),
                                          None, cfg)
                records.append({
                    "condition": cond, "start_day": int(day), "lot": lot, "lot_size": size,
                    "true_destination": true, "predicted_destination": pred,
                    "true_q_days": float(np.percentile(lot_df["days_remaining"], 100 * (1 - cfg.lot_quantile))),
                    "pred_q_days": float(np.percentile(lot_df["days_pred"], 100 * (1 - cfg.lot_quantile))),
                    "mean_confidence": float(lot_df["confidence"].mean()),
                })

    lots = pd.DataFrame(records)
    if lots.empty:
        print("[LOTS] Not enough fruits to build lots.")
        return None

    def lot_cost(row) -> Tuple[str, float]:
        if row["predicted_destination"] == "human_review":
            return "human_review", cfg.cost_review
        diff = DESTINATION_RANK[row["predicted_destination"]] - DESTINATION_RANK[row["true_destination"]]
        if diff > 0:
            return "upgrade_unfit", cfg.cost_upgrade * diff
        if diff < 0:
            return "downgrade", cfg.cost_downgrade * (-diff)
        return "correct", 0.0

    outcome = lots.apply(lot_cost, axis=1, result_type="expand")
    lots["outcome"], lots["cost"] = outcome[0], outcome[1]
    lots.to_csv(cfg.out_dir / "lot_decisions.csv", index=False)

    unfit_export = (lots["predicted_destination"] == "export_sea") & (lots["true_destination"] != "export_sea")
    summary = {
        "stage_model": stage_key,
        "days_model": days_key,
        "fruit_day_level_metrics": fruit_level,
        "rule": {"lot_quantile": cfg.lot_quantile, "export_days": cfg.export_days,
                 "domestic_days": cfg.domestic_days, "safety_margin": cfg.safety_margin,
                 "review_confidence": cfg.review_confidence, "lot_size": cfg.lot_size,
                 "lot_day_spread": cfg.lot_day_spread},
        "n_lots": int(len(lots)),
        "outcome_share": lots["outcome"].value_counts(normalize=True).to_dict(),
        "share_unfit_lots_sent_to_export": float(unfit_export.mean()),
        "mean_cost_per_lot": float(lots["cost"].mean()),
        "true_destination_share": lots["true_destination"].value_counts(normalize=True).to_dict(),
        "by_condition": {
            cond: {"n_lots": int(len(g)), "accuracy": float((g["outcome"] == "correct").mean()),
                   "share_unfit_export": float(((g["predicted_destination"] == "export_sea")
                                                & (g["true_destination"] != "export_sea")).mean()),
                   "mean_cost": float(g["cost"].mean())}
            for cond, g in lots.groupby("condition")
        },
    }
    (cfg.out_dir / "lot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[LOTS] {len(lots)} lots simulated with {stage_key} + {days_key}. "
          f"Correct {summary['outcome_share'].get('correct', 0):.1%}, "
          f"unfit lots sent to export {summary['share_unfit_lots_sent_to_export']:.1%}")
    plot_lot_confusion(lots, cfg)
    return summary


# ---------------------------------------------------------------------------
# 11. Figures
# ---------------------------------------------------------------------------

def plot_importance(imp_df: pd.DataFrame, task: str, cfg: Config) -> None:
    top = imp_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(top))))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#8172b3")
    ax.set_xlabel("Drop in macro F1 when permuted" if task == "stage" else "Increase in MAE when permuted (days)")
    ax.set_title(f"Permutation importance, final {task} model")
    fig.tight_layout()
    save_figure(fig, cfg.out_dir / f"fig_feature_importance_{task}", cfg)


def plot_confusion(y: np.ndarray, pred: np.ndarray, title: str, stem: str, cfg: Config) -> None:
    ok = np.isfinite(pred)
    cm = confusion_matrix(y[ok].astype(int), pred[ok].astype(int), labels=list(STAGES))
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(STAGES)):
        for j in range(len(STAGES)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(STAGES)), STAGES)
    ax.set_yticks(range(len(STAGES)), STAGES)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_figure(fig, cfg.out_dir / stem, cfg)


def plot_days_diagnostics(data: pd.DataFrame, pred: np.ndarray, title: str, cfg: Config) -> None:
    ok = data["has_days_target"].to_numpy() & np.isfinite(pred)
    y = data["days_remaining"].to_numpy()[ok]
    p = pred[ok]
    cond = data["condition"].to_numpy()[ok]
    residual = p - y
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for c in np.unique(cond):
        sel = cond == c
        axes[0].scatter(y[sel] + rng.uniform(-0.25, 0.25, sel.sum()), p[sel], s=5, alpha=0.3, label=c)
    lims = [0, max(y.max(), p.max())]
    axes[0].plot(lims, lims, "k--", lw=1)
    axes[0].set_xlabel("Observed days remaining")
    axes[0].set_ylabel("Out of fold prediction (days)")
    axes[0].set_title(title, fontsize=9)
    axes[0].legend(title="Condition", markerscale=3)

    bins = np.arange(0, y.max() + 1)
    mae_by_day = [np.mean(np.abs(residual[y == d])) if (y == d).any() else np.nan for d in bins]
    bias_by_day = [np.mean(residual[y == d]) if (y == d).any() else np.nan for d in bins]
    axes[1].plot(bins, mae_by_day, marker="o", label="MAE")
    axes[1].plot(bins, bias_by_day, marker="s", label="bias (pred - true)")
    axes[1].axhline(0, color="k", lw=1, ls="--")
    axes[1].set_xlabel("Observed days remaining")
    axes[1].set_ylabel("Days")
    axes[1].set_title("Error versus remaining green life")
    axes[1].legend()

    axes[2].hist(residual, bins=40, color="#55a868", edgecolor="white")
    axes[2].axvline(0, color="k", lw=1, ls="--")
    axes[2].set_xlabel("Residual, predicted minus observed (days)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Positive residual = green life overestimated")
    fig.tight_layout()
    save_figure(fig, cfg.out_dir / "fig_diagnostics_days", cfg)


def plot_model_comparison(results: Dict[str, pd.DataFrame], cfg: Config) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, task, metric, label in [
        (axes[0], "stage", "pooled_macro_F1", "Pooled out of fold macro F1"),
        (axes[1], "days", "pooled_MAE", "Pooled out of fold MAE (days)"),
    ]:
        res = results.get(task)
        if res is None or res.empty:
            ax.axis("off")
            continue
        pivot = res.pivot_table(index="feature_set", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax, edgecolor="white")
        ax.set_ylabel(label)
        ax.set_xlabel("Input")
        ax.set_title(f"Model comparison, {task}")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Model", fontsize=7)
    fig.tight_layout()
    save_figure(fig, cfg.out_dir / "fig_model_comparison", cfg)


def plot_lot_confusion(lots: pd.DataFrame, cfg: Config) -> None:
    order_true = [d for d in DESTINATIONS if d in set(lots["true_destination"])]
    order_pred = [d for d in list(DESTINATIONS) + ["human_review"] if d in set(lots["predicted_destination"])]
    ct = pd.crosstab(lots["true_destination"], lots["predicted_destination"]).reindex(
        index=order_true, columns=order_pred, fill_value=0)
    fig, ax = plt.subplots(figsize=(1.6 * len(order_pred) + 2, 1.0 * len(order_true) + 2))
    im = ax.imshow(ct.to_numpy(), cmap="Oranges")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, str(ct.iat[i, j]), ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(order_pred)), order_pred, rotation=30, ha="right")
    ax.set_yticks(range(len(order_true)), order_true)
    ax.set_xlabel("Recommended destination")
    ax.set_ylabel("Destination from the true labels")
    ax.set_title("Simulated lots, above the diagonal = value lost, below = risk")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_figure(fig, cfg.out_dir / "fig_lot_decisions", cfg)


def save_predictions(data: pd.DataFrame, registry: Dict, cfg: Config) -> None:
    base = data[["image_id", "fruit_id", "condition", "day", "side", "stage", "days_remaining",
                 "has_days_target", "outer_fold"]]
    for task in ("stage", "days"):
        frames = []
        for entry in registry[task].values():
            df = base.copy()
            df["model"], df["feature_set"] = entry["model"], entry["feature_set"]
            df["predicted"] = entry["pred"]
            if "proba" in entry:
                for j, s in enumerate(STAGES):
                    df[f"p_stage_{s}"] = entry["proba"][:, j]
                df["confidence"] = np.nanmax(entry["proba"], axis=1)
            frames.append(df.dropna(subset=["predicted"]))
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(cfg.out_dir / f"predictions_{task}.csv", index=False)


def write_results(results: Dict[str, List[Dict]], registry: Dict, data: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for task in ("stage", "days"):
        if not results[task]:
            continue
        key = f"pooled_{TASKS[task]['key']}"
        df = pd.DataFrame(results[task]).sort_values(key, ascending=not TASKS[task]["higher_is_better"])
        df = df.reset_index(drop=True)
        df.to_csv(cfg.out_dir / f"metrics_{task}.csv", index=False)
        out[task] = df

        best = df[~df["model"].str.startswith("baseline")].iloc[0]
        entry = registry[task][f"{best['model']}|{best['feature_set']}"]
        title = f"{best['model']} / {best['feature_set']}"
        if task == "stage":
            plot_confusion(data["stage"].to_numpy(), entry["pred"], title, "fig_confusion_stage", cfg)
        else:
            plot_days_diagnostics(data, entry["pred"], title, cfg)
    save_predictions(data, registry, cfg)
    print(f"[RESULTS] metrics_stage.csv, metrics_days.csv and predictions written to {cfg.out_dir}")
    return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_argument_group("data")
    g.add_argument("--data-dir", type=Path, required=True, help="Root folder of the Hass avocado dataset")
    g.add_argument("--annotations", type=Path, default=None, help="Spreadsheet, found automatically by default")
    g.add_argument("--out-dir", type=Path, default=Path("outputs_avocado"))
    g.add_argument("--conditions", nargs="*", default=[], help="e.g. T10, empty = all conditions")
    g.add_argument("--ready-stage", type=int, default=4, help="Stage that defines the consumption point")
    g.add_argument("--limit-fruits", type=int, default=None, help="Random subset of fruits, for quick tests")

    g = parser.add_argument_group("detection")
    g.add_argument("--detector", choices=["yolo-world", "yolo", "threshold", "none"], default="yolo-world")
    g.add_argument("--yolo-weights", type=str, default=None, help="Custom weights, e.g. runs/detect/train/weights/best.pt")
    g.add_argument("--yolo-prompts", nargs="*", default=["avocado"])
    g.add_argument("--yolo-conf", type=float, default=0.05)
    g.add_argument("--yolo-imgsz", type=int, default=640)
    g.add_argument("--min-box-area", type=float, default=0.02, help="Minimum box area as a share of the image")
    g.add_argument("--crop-pad", type=float, default=0.08)
    g.add_argument("--crop-size", type=int, default=320)
    g.add_argument("--no-crop-cache", action="store_true")
    g.add_argument("--no-feature-cache", action="store_true")

    g = parser.add_argument_group("validation")
    g.add_argument("--n-splits", type=int, default=5)
    g.add_argument("--n-inner-splits", type=int, default=3)
    g.add_argument("--random-state", type=int, default=42)
    g.add_argument("--n-bootstrap", type=int, default=1000)
    g.add_argument("--n-permutations", type=int, default=0)
    g.add_argument("--n-jobs", type=int, default=-1)
    g.add_argument("--models", nargs="*", default=[], help="Baseline models to run, baselines always run")

    g = parser.add_argument_group("cnn")
    g.add_argument("--skip-cnn", action="store_true")
    g.add_argument("--backbones", nargs="*", default=["resnet18"],
                   choices=["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"])
    g.add_argument("--cnn-modes", nargs="*", default=["multitask", "cls_only"], choices=["multitask", "cls_only"])
    g.add_argument("--use-condition", action="store_true", help="Give the storage condition to the CNN")
    g.add_argument("--epochs", type=int, default=15)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--img-size", type=int, default=224)
    g.add_argument("--patience", type=int, default=4)
    g.add_argument("--reg-weight", type=float, default=1.0, help="Weight of the days loss in the multitask loss")
    g.add_argument("--num-workers", type=int, default=2)
    g.add_argument("--cnn-max-folds", type=int, default=None, help="Only the first k outer folds, to save time")
    g.add_argument("--fit-final-cnn", action="store_true")
    g.add_argument("--gradcam-per-stage", type=int, default=2)
    g.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, mps")

    g = parser.add_argument_group("lot decision rule")
    g.add_argument("--lot-size", type=int, default=20)
    g.add_argument("--lots-per-day", type=int, default=20)
    g.add_argument("--lot-day-spread", type=int, default=2)
    g.add_argument("--lot-quantile", type=float, default=0.9,
                   help="Share of fruits that must have enough days, 0.9 uses the 10th percentile")
    g.add_argument("--export-days", type=float, default=10.0, help="PoC logistic time for export, in days")
    g.add_argument("--domestic-days", type=float, default=4.0)
    g.add_argument("--safety-margin", type=float, default=2.0)
    g.add_argument("--review-confidence", type=float, default=0.6)
    g.add_argument("--cost-upgrade", type=float, default=10.0, help="Cost per level of a lot sent too far")
    g.add_argument("--cost-downgrade", type=float, default=1.0, help="Cost per level of a lot downgraded")
    g.add_argument("--cost-review", type=float, default=0.5)
    g.add_argument("--figure-formats", nargs="*", default=["png", "pdf"])
    args = parser.parse_args(argv)
    return Config(**vars(args))


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CONFIG] {json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2)}")
    (cfg.out_dir / "config.json").write_text(
        json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2), encoding="utf-8")

    inventory = scan_dataset(cfg)
    inventory = build_targets(inventory, cfg)

    detections = run_detection(inventory, cfg)
    data = inventory.merge(detections, on="image_id", how="inner")
    data.to_csv(cfg.out_dir / "file_inventory.csv", index=False)
    plot_detections(data, cfg)

    features = build_feature_table(data, cfg)
    data = data.merge(features, on="image_id", how="inner").reset_index(drop=True)
    for cond in sorted(data["condition"].unique()):
        data[f"cond_{cond}"] = (data["condition"] == cond).astype(float)
    print(f"[DATA] {len(data)} images with crop and features.")

    run_eda(data, detections, cfg)
    data["outer_fold"] = make_outer_splits(data, cfg)

    registry: Dict[str, Dict] = {"stage": {}, "days": {}}
    results = run_sklearn_experiments(data, cfg, registry)
    cnn_results = run_cnn_experiments(data, cfg, registry)
    for task in results:
        results[task].extend(cnn_results[task])

    tables = write_results(results, registry, data, cfg)
    plot_model_comparison(tables, cfg)
    run_lot_simulation(data, registry, cfg)

    print("\n[SUMMARY] Top rows of metrics_stage.csv")
    cols = ["feature_set", "model", "pooled_accuracy", "pooled_macro_F1", "pooled_QWK", "n_folds_evaluated"]
    if "stage" in tables:
        print(tables["stage"][cols].head(10).to_string(index=False))
    print("\n[SUMMARY] Top rows of metrics_days.csv")
    cols = ["feature_set", "model", "pooled_MAE", "pooled_RMSE", "pooled_R2", "pooled_within_2_days"]
    if "days" in tables:
        print(tables["days"][cols].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
