import os
import yaml
import logging
from pprint import pformat

def load_yaml_config(config_path: str, logger) -> dict:
    """
    Load and parse a YAML configuration file safely,
    """

    # --- Resolve to absolute path ---
    abs_path = os.path.abspath(config_path)

    # --- Check if file exists ---
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Config file not found at path: {abs_path}")

    # --- Validate extension ---
    if not abs_path.lower().endswith((".yaml", ".yml")):
        raise ValueError(f"Invalid config file type: {abs_path}. Expected .yaml or .yml")

    # --- Try loading YAML safely ---
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in file: {abs_path}")
        raise ValueError(f"Failed to parse YAML file: {e}")

    if config is None:
        raise ValueError(f"Config file is empty: {abs_path}")

    # --- Log success ---
    logger.info(f"Successfully loaded YAML config: {abs_path}")

    # --- Pretty print parameters ---
    formatted_config = pformat(config, indent=2, width=100)
    logger.info("Configuration parameters:\n" + formatted_config)

    return config

if __name__ == "__main__":
    # Example usage for testing
    from args import *
    from logger import *
    import yaml
    args = get_args()
    logger = init_logger(args.logging)

    # Load the config
    cfg = load_yaml_config(args.config_file, logger)