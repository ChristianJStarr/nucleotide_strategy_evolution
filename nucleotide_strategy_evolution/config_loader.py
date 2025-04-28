"""Utility for loading configuration files."""

import yaml
from typing import Dict, Any
import os

# Consider adding Pydantic for schema validation later
# from pydantic import BaseModel
# class EvolutionParams(BaseModel): ...

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None:
            return {} # Handle empty file case
        return config_data
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {config_path}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred while loading config: {config_path}")
        raise e

# Example Usage (can be removed later):
# if __name__ == '__main__':
#     try:
#         # Adjust path relative to project root if needed
#         evo_params = load_config('../../config/evolution_params.yaml')
#         compliance_rules = load_config('../../config/compliance_rules.yaml')
#         print("Evolution Params:", evo_params)
#         print("Compliance Rules:", compliance_rules)
#     except Exception as e:
#         print(f"Failed to load config: {e}") 