# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from hievnet.data.etl import CSVPolygonIngestor, ETLConfig

pl.Config.set_fmt_str_lengths(100)


yaml_path = Path(__file__).parent.joinpath('dataset.yaml')

print('1. Parsing Configuration...')
try:
    config_manager = ETLConfig(yaml_path)

    # Fetch PanopTILs config
    dataset_name = 'PanopTILs'
    panoptils_config = config_manager.get_dataset_config(dataset_name)

    # Inject the namespace map
    panoptils_config['namespace_map'] = config_manager.get_namespace_map(dataset_name)

    print(f'✅ {dataset_name} config loaded.')
    print(f'   Column Map: {panoptils_config.get("csv_column_map")}')
except Exception as e:
    print(f'❌ Config parsing failed: {e}')

print('\n2. Initializing Ingestor & Building Registry...')
try:
    ingestor = CSVPolygonIngestor(config=panoptils_config)
    registry = ingestor.get_registry()

    print('✅ Registry built successfully. Preview:')
    print(registry.head(3))

    if registry.is_empty():
        print('❌ Registry is empty! Check your directory paths and regex/extension rules.')

except Exception as e:
    print(f'❌ Ingestor initialization failed: {e}')

# %%
print('\n3. Testing Pixel Extraction on the First CSV File...')

first_row = registry.row(5, named=True)
print(f'Processing ROI: {first_row["roi_id"]}')

try:
    # Extract the arrays
    roi_id, image_array, instance_matrix, cats_array = ingestor.process_item(first_row)

    print(f'\n✅ Successfully Extracted ROI: {roi_id}')
    print(f'   -> Image Array Shape: {image_array.shape}, dtype: {image_array.dtype}')
    print(f'   -> Instance Matrix Shape: {instance_matrix.shape}, dtype: {instance_matrix.dtype}')

    # Validate instances
    unique_instances = np.unique(instance_matrix)
    num_instances = len(unique_instances) - 1  # Subtract 1 for background (0)
    print(f'   -> Extracted {num_instances} polygons.')
    print(f'   -> First 5 standardized categories: {cats_array[:5]}')

    assert len(cats_array) == num_instances + 1, 'Category array length mismatch!'

except ValueError as e:
    print(f'\n❌ Pipeline stopped by Fail-Loud Gatekeeper: {e}')
except Exception as e:
    print(f'\n❌ Extraction failed: {e}')

print('\n4. Plotting visual sanity check...')
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image_array)
axes[0].set_title(f'RGB Image ({roi_id})')
axes[0].axis('off')

# Mask the background (0) so it doesn't colorize the empty space
masked_instance = np.ma.masked_where(instance_matrix == 0, instance_matrix)

axes[1].imshow(np.zeros(image_array.shape[:2]), cmap='gray')  # Background image
axes[1].imshow(masked_instance, cmap='nipy_spectral', alpha=0.6, interpolation='nearest')  # Overlay
axes[1].set_title('Rasterized PanopTILs CSV Polygons')
axes[1].axis('off')

plt.tight_layout()
plt.show()
