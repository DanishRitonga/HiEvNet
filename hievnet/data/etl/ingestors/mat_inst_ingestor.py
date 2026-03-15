import cv2
import numpy as np
from scipy.io import loadmat

from ._base import BaseDataIngestor


class MatInstanceIngestor(BaseDataIngestor):
    def process_item(self, row: dict) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        image_path = row['image_path']
        mask_path = row['mask_path']
        roi_id = row['roi_id']

        # 1. Load the RGB Image
        image_array = cv2.imread(image_path)
        if image_array is None:
            raise ValueError(f'Failed to read image at {image_path}')
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # 2. Load the .mat dictionary using SciPy
        try:
            mat_data = loadmat(mask_path)
        except Exception as e:
            raise ValueError(f'Failed to read .mat file at {mask_path}: {e}')

        # 3. Extract the Instance Matrix directly
        if 'inst_map' not in mat_data:
            raise KeyError(f"'inst_map' key not found in {mask_path}")

        # The authors already built our target matrix!
        # We just cast it to int32 to match our pipeline standard.
        instance_matrix = mat_data['inst_map'].astype(np.int32)

        # 4. Extract and Standardize Categories
        if 'inst_type' not in mat_data:
            raise KeyError(f"'inst_type' key not found in {mask_path}")

        # MATLAB arrays often load as 2D column vectors (e.g., shape (N, 1)).
        # We flatten it to a standard 1D NumPy array.
        raw_types = mat_data['inst_type'].flatten()

        # Initialize our standard array with the background at index 0
        cats = [0]

        # The README specifies 'inst_type' is in order of the inst_map IDs (1 to N).
        # We iterate through the raw integers, standardize them, and append.
        for raw_cat in raw_types:
            standardized_category = self.standardize_label(raw_cat)
            cats.append(standardized_category)

        # 5. Lock the categories array
        cats_array = np.array(cats, dtype=np.int16)

        # Optional Sanity Check: Ensure our categories array matches the highest ID in the matrix
        max_id = np.max(instance_matrix)
        if max_id >= len(cats_array):
            print(
                f'Warning: ROI {roi_id} has a max instance ID of {max_id}, '
                f'but only provided {len(cats_array) - 1} labels.'
            )

        tissue_origin = self.resolve_tissue()

        return (roi_id, image_array, instance_matrix, cats_array, tissue_origin)
