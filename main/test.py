# %%
from pathlib import Path

from hievnet.data.ETL.config import ETLConfig
from hievnet.data.ETL.ingestors import BaseDataIngestor


# 1. Create a dummy subclass for testing
class DummyCsvIngestor(BaseDataIngestor):  # noqa: D101
    def process_item(self, row: dict):  # noqa: D102
        pass


# %%
config_dir = Path(__file__).parent.joinpath('dataset.yaml')
config_manager = ETLConfig(config_dir)

print(f'Successfully loaded datasets: {config_manager.list_datasets()}')

# 2. Get PanopTILs config
panoptils_name = 'CoNSeP'
panoptils_config = config_manager.get_dataset_config(panoptils_name)

# %%
ingestor = DummyCsvIngestor(
    root_dir=panoptils_config['resolved_root_dir'],
    config=panoptils_config,
)

# %%
