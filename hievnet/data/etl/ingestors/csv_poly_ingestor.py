import cv2
import numpy as np
import polars as pl

from ._base import BaseDataIngestor


class CSVPolygonIngestor(BaseDataIngestor):
    def process_item(self, row: dict) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        image_path = row['image_path']
        mask_path = row['mask_path']
        roi_id = row['roi_id']

        # 1. Fetch column mapping
        col_map = self.config.get('csv_column_map', {})
        col_x = col_map.get('x_coords')
        col_y = col_map.get('y_coords')
        col_cat = col_map.get('category')

        if not all([col_x, col_y, col_cat]):
            raise KeyError('Missing required csv_column_map keys. Need x_coords, y_coords, and category.')

        # 2. Load the RGB Image
        image_array = cv2.imread(image_path)
        if image_array is None:
            raise ValueError(f'Failed to read image at {image_path}')
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        h, w = image_array.shape[:2]

        # 3. Initialize Canvas and Categories
        instance_matrix = np.zeros((h, w), dtype=np.int32)
        cats = [0]

        # 4. Read the CSV using Polars
        try:
            df = pl.read_csv(mask_path)
        except Exception as e:
            raise ValueError(f'Failed to read CSV at {mask_path}: {e}')

        # 5. Row-by-Row Extraction
        for instance_id, cell_row in enumerate(df.iter_rows(named=True), start=1):
            # Grab the comma-separated strings
            x_str = cell_row[col_x]
            y_str = cell_row[col_y]

            # Skip empty or malformed rows
            if not x_str or not y_str:
                continue

            # Split the strings and cast directly to an integer numpy array
            x_arr = np.array(x_str.split(','), dtype=np.int32)
            y_arr = np.array(y_str.split(','), dtype=np.int32)

            # Zip them together into the (N, 2) shape OpenCV expects
            pts = np.column_stack((x_arr, y_arr))

            # Rasterize
            cv2.fillPoly(instance_matrix, [pts], instance_id)

            # Extract and Standardize the Category (using the 'group' column)
            raw_category = cell_row[col_cat]
            standardized_category = self.standardize_label(raw_category)
            cats.append(standardized_category)

        # 6. Lock the categories array
        cats_array = np.array(cats, dtype=np.int16)

        tissue_origin = self.resolve_tissue()

        return (roi_id, image_array, instance_matrix, cats_array, tissue_origin)
