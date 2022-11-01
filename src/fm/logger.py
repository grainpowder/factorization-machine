import logging
import sys


def make_logger() -> logging.Logger:
    """
    Return a logger that prints log in a predefined format

    Returns:
        logging.Logger instance
    """
    # Define log format to be used in each of handler
    formatter = logging.Formatter(
        fmt="%(asctime)s (%(funcName)s) : %(msg)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Define handler for console printouts
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # Make logger and attach each of handler to the logger
    logger = logging.getLogger("main")
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger
