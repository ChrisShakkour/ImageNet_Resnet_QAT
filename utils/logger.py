import logging
import os
from datetime import datetime

# ANSI color codes for console output
LOG_COLORS = {
    "DEBUG": "\033[94m",    # Blue
    "INFO": "\033[92m",     # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",    # Red
    "CRITICAL": "\033[95m", # Magenta
}
RESET_COLOR = "\033[0m"

class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to console output based on log level."""
    def format(self, record):
        levelname = record.levelname
        if levelname in LOG_COLORS:
            record.levelname = f"{LOG_COLORS[levelname]}{levelname}{RESET_COLOR}"
        return super().format(record)


def init_logger(level: str = None, log_dir: str = "logs"):
    """
    Initialize and configure a logger with colorized console output.
    Args:
        level (str, optional): Logging level ('INFO', 'DEBUG', etc.).
                              If None, logging is disabled.
        log_dir (str, optional): Directory to save log files. Default: 'logs'
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.handlers.clear()  # Avoid duplicate handlers in notebooks / re-runs

    if not level:
        # Logging disabled
        logging.disable(logging.CRITICAL)
        print("🔇 Logging is disabled.")
        return logger

    # --- Ensure log directory exists ---
    os.makedirs(log_dir, exist_ok=True)

    # --- Log file name with timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    # --- Formatter ---
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    color_formatter = ColorFormatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- Console Handler ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)
    console_handler.setLevel(getattr(logging, level))

    # --- File Handler ---
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, level))

    # --- Attach handlers ---
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(getattr(logging, level))

    # --- Startup messages ---
    logger.info(f"Logger initialized at {level} level")
    logger.info(f"Log file saved at: {os.path.abspath(log_file)}")

    return logger


if __name__ == "__main__":
    from args import *
    args = get_args()

    # Initialize color logger
    logger = init_logger(args.logging)
    logger.info(f"Loaded config file: {args.config_file}")