"""Access function for the metatree package logger.

"""
import logging
from logging.handlers import RotatingFileHandler

DEFAULT_LOGGER_LEVEL = logging.DEBUG


def logger(output_path: str,
           logger_level: int=DEFAULT_LOGGER_LEVEL,
           file_level: int=DEFAULT_LOGGER_LEVEL,
           stream_level: int=DEFAULT_LOGGER_LEVEL):
    """Get the logger for the metatree package, and set a save file for the
    logged session.

    Returns a logger for the metatree package with two handlers: a file
    handler to write logs to an output file, and a stream handler to write
    logs to the console.

    Logs are capped to 1MiB.

    Args:
        output_path (str): File path where the logger should save.
        logger_level (int): Minimum level of messages to log.
        file_level (int): Minimum level of messages to save to file.
        stream_level (int): Minimum level of messages to print to console.

    Returns:
        metatree_logger (logging.Logger): The metatree logger.

    """
    # Get or create the logger
    metatree_logger = logging.getLogger('metatree')
    metatree_logger.setLevel(logger_level)
    # Log format
    formatter = logging.Formatter(
        fmt='%asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S')

    # Log output to file
    file_handler = RotatingFileHandler(output_path, maxBytes=1024*1024)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    metatree_logger.addHandler(file_handler)

    # Log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(formatter)
    metatree_logger.addHandler(stream_handler)

    # metatree_logger.debug('Created logger with log file {output_path}')

    return metatree_logger
