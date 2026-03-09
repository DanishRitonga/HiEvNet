# %%
from pathlib import Path

from hievnet.data.etl import ETLConfig, ParquetIngestor

# %%
config_dir = Path(__file__).parent.joinpath('dataset.yaml')
config_manager = ETLConfig(config_dir)

print(f'Successfully loaded datasets: {config_manager.list_datasets()}')

# 2. Get PanopTILs config
panoptils_name = 'PanNuke'
panoptils_config = config_manager.get_dataset_config(panoptils_name)

# %%
ingestor = ParquetIngestor(
    root_dir=panoptils_config['resolved_root_dir'],
    config=panoptils_config,
)

# %%
