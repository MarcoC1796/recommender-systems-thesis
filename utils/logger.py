import logging
from logging import Logger
import os
from datetime import datetime

BASE_DIR: str = "logs"
LOG_FILENAME: str = datetime.now().strftime("log_%Y_%m_%d_%H_%M_%S.log")
LOG_FILEPATH: str = os.path.join(BASE_DIR, LOG_FILENAME)


def get_logger(name: str = __name__, level: int = logging.INFO) -> Logger:
    """
    Create a logger with the specified name and level.

    Args:
    - name (str): Name of the logger to create.
    - level (int): Logging level, e.g., logging.INFO, logging.DEBUG, logging.ERROR.

    Returns:
    - Logger: A configured logger instance.
    """
    # Create a logger
    logger: Logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    c_handler: logging.StreamHandler = logging.StreamHandler()  # Console handler
    f_handler: logging.FileHandler = logging.FileHandler(LOG_FILEPATH)  # File handler

    # Create formatters and add them to handlers
    c_format: logging.Formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s"
    )
    f_format: logging.Formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger if they haven't been added already
    if not any(isinstance(h, type(c_handler)) for h in logger.handlers):
        logger.addHandler(c_handler)
    if not any(isinstance(h, type(f_handler)) for h in logger.handlers):
        logger.addHandler(f_handler)

    return logger


# Example usage
if __name__ == "__main__":
    logger: Logger = get_logger(level=logging.DEBUG)
    logger.info("This is an info message")
    logger.debug("This is a debug message")
    logger.error("This is an error message")
