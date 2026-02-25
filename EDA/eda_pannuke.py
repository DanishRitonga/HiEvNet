# %%
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

# %%
script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data" / "PanNuke" / "data"

folds = [1, 2, 3]
dfs = []
for f in folds:
    parquet_file = data_dir / f"fold{f}-00000-of-00001.parquet"
    df_fold = pd.read_parquet(parquet_file)
    print(f"Fold {f} shape: {df_fold.shape}")
    dfs.append(df_fold)

dfs = pd.concat(dfs, ignore_index=True)

# %%

config_path = script_dir / "pannuke_utils" / "config.json"


def _get_config_map(config_path: Path) -> tuple[dict[int, str], dict[int, str]]:
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["category"], config["tissue"]


CATEGORY_MAP, TISSUE_MAP = _get_config_map(config_path)


# %%
def _decode_image_bytes(byte_data) -> np.ndarray:
    image = Image.open(io.BytesIO(byte_data))
    return np.array(image).astype(np.uint8)


def decode_roi_bytes(df: pd.DataFrame, row_index: int) -> np.ndarray:
    row = df.iloc[row_index]
    byte_data = row["image"]["bytes"]

    return _decode_image_bytes(byte_data)


def decode_ins_bytes(
    df: pd.DataFrame, row_index: int, ins_index: int = 0
) -> np.ndarray:
    row = df.iloc[row_index]
    byte_data = row["instances"][ins_index]["bytes"]

    return _decode_image_bytes(byte_data)


# %%
def _get_bbox(
    mask: np.ndarray, format: str = "xyxy"
) -> Optional[tuple[int, int, int, int]]:
    y, x = np.where(mask > 0)

    if x.size and y.size:
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        if format == "xyxy":
            return (x_min, y_min, x_max, y_max)
        elif format == "xywh":
            return (x_min, y_min, x_max - x_min, y_max - y_min)

    return None


def _get_yolo_bbox(mask: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    bbox = _get_bbox(mask, format="xywh")
    if bbox is not None:
        x_min, y_min, w, h = bbox
        h_img, w_img = mask.shape[:2]

        return (
            (x_min + w / 2.0) / w_img,
            (y_min + h / 2.0) / h_img,
            w / w_img,
            h / h_img,
        )

    return None


# %%
def _get_gt_df(df: pd.DataFrame) -> pd.DataFrame:
    gt_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        gts = []
        for ins, cat in zip(row["instances"], row["categories"]):
            mask = _decode_image_bytes(ins["bytes"])
            bbox = _get_yolo_bbox(mask)
            if bbox is not None:
                gt = (cat, *bbox)
                gts.append(gt)
        gt_list.append(gts)

    df["yolo_gt"] = gt_list

    return df


# %%
