import cv2
import numpy as np
import orjson

from ._base import BaseDataIngestor


class GeoJSONIngestor(BaseDataIngestor):
    def process_item(self, row: dict) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        image_path = row['image_path']
        mask_path = row['mask_path']
        roi_id = row['roi_id']

        # 1. Load the RGB Image
        image_array = cv2.imread(image_path)
        if image_array is None:
            raise ValueError(f'Failed to read image at {image_path}')
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        h, w = image_array.shape[:2]

        # 2. Initialize the Canvas and Categories Array
        instance_matrix = np.zeros((h, w), dtype=np.int32)

        # Index 0 is permanently reserved for the background class
        cats = [0]

        # 3. Parse the GeoJSON using the fast Rust backend
        with open(mask_path, 'rb') as f:
            geo_data = orjson.loads(f.read())

        # 4. Rasterize Polygons
        features = geo_data.get('features', [])

        for instance_id, feature in enumerate(features, start=1):
            geom_type = feature.get('geometry', {}).get('type')

            if geom_type not in ['Polygon', 'MultiPolygon']:
                continue

            coordinates = feature['geometry']['coordinates']
            properties = feature.get('properties', {})

            # --- THE ONTOLOGY GATEKEEPER ---
            # Extract the RAW string category from the dataset
            raw_category = self._extract_category(properties, default='unlabeled')

            # Instantly standardize it using the Base Class method
            standardized_category = self.standardize_label(raw_category)

            # Rasterize
            if geom_type == 'Polygon':
                exterior_ring = coordinates[0]
                pts = np.array(exterior_ring, dtype=np.int32)
                cv2.fillPoly(instance_matrix, [pts], instance_id)

            elif geom_type == 'MultiPolygon':
                for poly_coords in coordinates:
                    exterior_ring = poly_coords[0]
                    pts = np.array(exterior_ring, dtype=np.int32)
                    cv2.fillPoly(instance_matrix, [pts], instance_id)

            # Append the strictly standardized string
            cats.append(standardized_category)

        # 5. Lock the categories into a NumPy object array for safe string storage
        cats_array = np.array(cats, dtype=np.int16)

        tissue_origin = self.resolve_tissue()

        return (roi_id, image_array, instance_matrix, cats_array, tissue_origin)

    def _extract_category(self, properties: dict, default: str) -> str:
        """Extracts the exact classification name provided by the dataset authors."""
        if 'classification' in properties and 'name' in properties['classification']:
            return str(properties['classification']['name'])

        if 'classId' in properties:
            return str(properties['classId'])

        return default
