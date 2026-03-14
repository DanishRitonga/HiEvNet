from collections.abc import Generator

import cv2
import numpy as np
import polars as pl

from ._base import BaseDataIngestor


class ParquetIngestor(BaseDataIngestor):  # noqa: D101
    def process_item(self, row: dict) -> Generator[tuple[str, np.ndarray, np.ndarray, dict], None, None]:
        """Takes a registry row representing a Parquet file, extracts the ROIs,
        and yields them in the standardized Method 3 format.
        """  # noqa: D205
        parquet_path = row['image_path']
        base_roi_name = row['roi_id']  # e.g., "train-00000"

        # 1. Peek at the Schema (Lazy Evaluation)
        lf = pl.scan_parquet(parquet_path)
        schema = lf.collect_schema()

        rgb_col, mask_col, cat_col = self._identify_columns(schema)

        if not all([rgb_col, mask_col, cat_col]):
            raise ValueError(
                f'Could not map all columns in {parquet_path}. RGB: {rgb_col}, Masks: {mask_col}, Cats: {cat_col}'
            )

        lf = lf.with_row_index('internal_roi_id')

        # Table 1: RGB Images
        df_rgb = lf.select(['internal_roi_id', rgb_col]).collect()

        # Table 2: Masks & Categories (Exploded)
        df_masks = lf.select(['internal_roi_id', mask_col, cat_col]).explode([mask_col, cat_col]).drop_nulls().collect()

        masks_by_roi = {}
        if not df_masks.is_empty():
            for sub_df in df_masks.partition_by('internal_roi_id'):
                roi_key = sub_df['internal_roi_id'][0]
                masks_by_roi[roi_key] = sub_df

        # Now iterate through the RGB images
        for rgb_row in df_rgb.iter_rows(named=True):
            internal_id = rgb_row['internal_roi_id']

            rgb_struct = rgb_row[rgb_col]
            rgb_bytes = rgb_struct['bytes'] if isinstance(rgb_struct, dict) else rgb_struct

            image_array = self._decode_image(rgb_bytes, is_mask=False)
            h, w = image_array.shape[:2]

            instance_matrix = np.zeros((h, w), dtype=np.int32)
            cats = []

            # Fast O(1) dictionary lookup instead of filtering in a loop
            if internal_id in masks_by_roi:
                roi_masks_df = masks_by_roi[internal_id]

                for instance_id, mask_row in enumerate(roi_masks_df.iter_rows(named=True), start=1):
                    mask_struct = mask_row[mask_col]
                    mask_bytes = mask_struct['bytes'] if isinstance(mask_struct, dict) else mask_struct

                    category = mask_row[cat_col]

                    mask_array = self._decode_image(mask_bytes, is_mask=True)

                    if mask_array.ndim > 2:
                        mask_array = mask_array[:, :, 0]

                    instance_matrix[mask_array > 0] = instance_id

                    category = self.standardize_label(category)
                    cats.append(category)

            cats = np.array(cats, dtype=object)

            global_roi_id = f'{base_roi_name}_roi_{internal_id}'
            yield (global_roi_id, image_array, instance_matrix, cats)

    def _identify_columns(self, schema: pl.Schema) -> tuple[str, str, str]:
        """Dynamically identifies columns based on HuggingFace/Parquet Struct schemas."""
        rgb_col, mask_col, cat_col = None, None, None

        for col_name, dtype in schema.items():
            # Match RGB: Struct with 'bytes' (or raw Binary fallback)
            if isinstance(dtype, pl.Struct) or dtype == pl.Binary:
                rgb_col = col_name

            # Match Masks: List of Structs (or List of Binary fallback)
            elif isinstance(dtype, pl.List) and (isinstance(dtype.inner, pl.Struct) or dtype.inner == pl.Binary):
                mask_col = col_name

            # Match Categories: List of Integers
            elif isinstance(dtype, pl.List) and dtype.inner in [pl.Int64, pl.Int32, pl.UInt32, pl.Int8]:
                cat_col = col_name

        return rgb_col, mask_col, cat_col

    def _decode_image(self, byte_string: bytes, is_mask: bool = False) -> np.ndarray:
        """Helper to convert raw bytes back into numpy arrays using OpenCV."""
        # Convert bytes to a 1D uint8 numpy array
        np_arr = np.frombuffer(byte_string, np.uint8)

        # Decode the array. Use IMREAD_UNCHANGED for masks to preserve exact values (e.g., boolean/binary masks),
        # use IMREAD_COLOR for RGB images to ensure 3 channels.
        flags = cv2.IMREAD_UNCHANGED if is_mask else cv2.IMREAD_COLOR
        decoded_img = cv2.imdecode(np_arr, flags)

        if decoded_img is None:
            raise ValueError('OpenCV failed to decode the byte array.')

        # OpenCV loads images in BGR format by default. Convert to RGB.
        if not is_mask and len(decoded_img.shape) == 3:
            decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)

        return decoded_img
