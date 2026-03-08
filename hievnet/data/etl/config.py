from pathlib import Path
from typing import Any

import yaml


class ETLConfig:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.raw_config = self._load_yaml()

        self.global_settings = self.raw_config.get('global_settings', {})
        self.datasets = self.raw_config.get('datasets', {})

        self._validate_schema()

    def _load_yaml(self) -> dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f'Configuration file not found: {self.config_path}')
        with open(self.config_path) as file:
            return yaml.safe_load(file)

    def _validate_schema(self):
        # --- TWEAK 1: Add root_dir to required globals ---
        required_globals = ['root_dir', 'output_image_size', 'output_mpp', 'patching_overlap_pct']
        for req in required_globals:
            if req not in self.global_settings:
                raise KeyError(f"Missing required global setting: '{req}'")

        if not self.datasets:
            raise ValueError("No datasets found in configuration under 'datasets:' key.")

        valid_split_seps = ['physical', 'filename_regex', 'none']
        valid_mod_seps = ['physical_parallel', 'physical_flat', 'bundled_archive']

        for dataset_name, d_conf in self.datasets.items():
            for req in ['root_dir', 'ingestion_method', 'split_separation', 'modality_separation']:
                if req not in d_conf:
                    raise KeyError(f"Dataset '{dataset_name}' is missing required key: '{req}'")

            if d_conf['split_separation'] not in valid_split_seps:
                raise ValueError(f"Dataset '{dataset_name}': Invalid split_separation")
            if d_conf['modality_separation'] not in valid_mod_seps:
                raise ValueError(f"Dataset '{dataset_name}': Invalid modality_separation")

            if d_conf['split_separation'] == 'physical':
                if 'split_dirs' not in d_conf:
                    raise KeyError(f"Dataset '{dataset_name}' missing 'split_dirs'")
                for key in d_conf['split_dirs']:
                    if not key.endswith('_dir'):
                        raise ValueError(f"Dataset '{dataset_name}': split_dirs key '{key}' must end with '_dir'")

            if (d_conf['split_separation'] == 'filename_regex') and (
                'split_args' not in d_conf or 'regex' not in d_conf['split_args']
            ):
                raise KeyError(f"Dataset '{dataset_name}' missing 'split_args.regex'")

            if d_conf['modality_separation'] == 'physical_parallel':
                if 'modality_dirs' not in d_conf:
                    raise KeyError(f"Dataset '{dataset_name}' missing 'modality_dirs'")
                if 'image_dir' not in d_conf['modality_dirs'] or 'mask_dir' not in d_conf['modality_dirs']:
                    raise KeyError(f"Dataset '{dataset_name}' modality_dirs must contain image_dir and mask_dir")

            if (d_conf['modality_separation'] != 'bundled_archive') and (
                'modality_pairing_rule' not in d_conf or 'match_extension' not in d_conf['modality_pairing_rule']
            ):
                raise KeyError(f"Dataset '{dataset_name}' missing 'modality_pairing_rule.match_extension'")

    def get_dataset_config(self, dataset_name: str) -> dict[str, Any]:
        """Returns the specific configuration block, with the root_dir fully resolved."""
        if dataset_name not in self.datasets:
            raise KeyError(f"Dataset '{dataset_name}' not found in configuration.")

        # --- TWEAK 2: Resolve the paths ---
        d_conf = self.datasets[dataset_name].copy()
        global_root = Path(self.global_settings['root_dir']).resolve()
        dataset_root = Path(d_conf['root_dir'])

        # pathlib magic: if dataset_root is absolute, it ignores global_root.
        resolved_root = global_root.joinpath(dataset_root)
        print(resolved_root)

        # Inject the fully resolved path back into the dictionary
        d_conf['resolved_root_dir'] = str(resolved_root)

        return d_conf

    def get_global_config(self) -> dict[str, Any]:
        """Returns the global settings."""
        return self.global_settings

    def list_datasets(self) -> list[str]:
        """Returns a list of available dataset."""
        return list(self.datasets.keys())
