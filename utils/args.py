import argparse
import os
import sys

def validate_config_file_path(config_file):
    """
    Validate the configuration file path.
    Args:
        config_file (str): Path to the configuration file.
    Returns:
        str: Validated absolute path to the configuration file.
    Raises:
        SystemExit: If the file does not exist or has an invalid extension.
    """
    if not os.path.isfile(config_file):
        print(f"❌ Error: Config file not found at path: {config_file}") #TODO: change to raisexception
        sys.exit(1)  # terminate the script gracefully
    else:
        # convert to full path
        config_file = os.path.abspath(config_file)

    # --- Check file extension ---
    valid_extensions = (".yaml", ".yml")
    if not config_file.lower().endswith(valid_extensions):
        print(f"❌ Error: Config file must be a YAML file ({valid_extensions}), got: {config_file}") #TODO: change to raisexception
        sys.exit(1)
    return config_file

def get_args():
    """
    Parse command line arguments for training script.
    Returns:
        argparse.Namespace: object with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a model using a YAML configuration file."
    )

    # Required config file path
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file (e.g., configs/resnet50.yaml)"
    )
    # Optional logging flag — default is False
    parser.add_argument(
        "--logging",
        type=str.upper,
        default=None,
        choices=["INFO", "DEBUG"],
        help="Enable logging with verbosity level (INFO or DEBUG). If not set, logging is disabled."
    )
    args = parser.parse_args()

    # ------------ VALIDATION ------------
    args.config_file = validate_config_file_path(args.config_file)
    return args

if __name__ == "__main__":
    args = get_args()

    print(f"📁 Config file: {args.config_file}")
    print(f"📝 Logging enabled: {args.logging}")