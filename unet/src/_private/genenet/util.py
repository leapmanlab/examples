"""Utility functions used by genenet.

"""
import logging
import os

import dill
import json
import metatree as mt
import matplotlib
import matplotlib.cm
import numpy as np
import tensorflow as tf

from tensorflow.python.client import device_lib

from typing import Dict, List, Optional, Union

DEFAULT_LOGGER_LEVEL = logging.INFO

logger = logging.getLogger('genenet.util')


def available_cpus() -> List[str]:
    """Return TensorFlow names for available CPUs.

    Returns:
        gpu_names (list of str): List of available CPU names.

    """
    return available_devices('CPU')


def available_devices(device_type: Optional[str]=None) -> List[str]:
    """Return TensorFlow names for available devices.

    Args:
        device_type (Optional[str]): 'CPU', 'GPU', or None for both.

    Returns:
        device_names (list of str): List of available device names.

    """
    devices = device_lib.list_local_devices()
    if device_type is not None:
        device_list = [d.name for d in devices if d.device_type == device_type]
    else:
        # Return all available devices
        device_list = [d.name for d in devices]

    # Reverse dictionary sort - if any GPUs are available, they'll come up first
    return sorted(device_list, reverse=True)


def available_gpus() -> List[str]:
    """Return TensorFlow names for available GPUs.

    Returns:
        gpu_names (list of str): List of available GPU names.

    """
    return available_devices('GPU')


def colorize(value: tf.Tensor,
             vmin: Optional[float]=None,
             vmax: Optional[float]=None,
             cmap: Optional[str]=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a
    matplotlib colormap for use with TensorBoard image summaries.

    Forked from @jimfleming:
        (https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b)

    By default it will normalize the input value to the range 0..1 before
    mapping to a grayscale colormap.

    Arguments:
        value (tf.Tensor): 2D Tensor of shape [height, width] or 3D Tensor of
            shape [height, width, 1].
        vmin (Optional[float]): the minimum value of the range used for
            normalization. (Default: value minimum).
        vmax (Optional[float]): the maximum value of the range used for
            normalization. (Default: value maximum).
        cmap (Optional[str]): a valid cmap named for use with matplotlib's
            `get_cmap`. (Default: 'gray').

    Returns:
        (tf.Tensor): A 3D tensor of shape [height, width, 3].

    Examples:
        ```
        output = tf.random_uniform(shape=[256, 256, 1])
        output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
        tf.summary.image('output', output_color)
        ```

    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    # noinspection PyTypeChecker
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    cm_colors = cm(np.arange(256))[:, :3]
    colors = tf.constant(cm_colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value


def get_logger(output_file: str=None,
               logger_level: int=DEFAULT_LOGGER_LEVEL,
               file_level: Optional[int]=None,
               stream_level: Optional[int]=None,
               output_dir: str=None):
    """Get logger for the genenet package.

    Returns a logger for the genenet package with two handlers:
    a file handler to write logs to an output file, and
    a stream handler to write logs to the console.

    Args:
        output_file (str): File where the logger should save to.
        logger_level (Optional[int]): Minimum level of messages to be logged.
        file_level (Optional[int]): Minimum level of messages to be saved to
            file.
        stream_level (Optional[int]): Minimum level of messages to be printed
            to console.
        output_dir (str): Directory for output file to be saved in. Defaults
            to current directory.

    Returns:
        (logging.logger): Logger for the genenet package.

    """
    new_logger = logging.getLogger('genenet')
    new_logger.setLevel(logger_level)
    # Use the same level as logger_level for everything else by default
    if file_level is None:
        file_level = logger_level
    if stream_level is None:
        stream_level = logger_level

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S')

    if output_file is not None:
        # Log to output to file
        if output_dir is not None:
            # Save logfile to output_dir
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            output_file = os.path.join(output_dir, output_file)

        filehandler = logging.FileHandler(output_file)
        filehandler.setLevel(file_level)
        filehandler.setFormatter(formatter)
        new_logger.addHandler(filehandler)

        # Log to console
        streamhandler = logging.StreamHandler()
        streamhandler.setLevel(stream_level)
        streamhandler.setFormatter(formatter)
        new_logger.addHandler(streamhandler)

        new_logger.debug('Created logger with log file {}'.format(
            output_file))

    return new_logger


def load_gene_graph(net_dir: str) -> mt.GeneGraph:
    """Load a GeneGraph from a saved GeneNet's save directory.

    Args:
        net_dir (str): A GeneNet save directory.

    Returns:
        (mt.GeneGraph): The saved GeneNet's GeneGraph.

    """
    logger.debug(f'Loading GeneGraph from {net_dir}')
    # Network architecture information serialized to this file
    arch_path = os.path.join(net_dir, 'architecture.info')
    # Load a GeneGraph from that file
    with open(arch_path, 'rb') as arch_file:
        graph: mt.GeneGraph = dill.load(arch_file)[0]

    return graph


def load_training_log(net_dir: str, ignore_error: bool=True) \
        -> Union[List[Dict[str, Dict]], None]:
    """Load most recent training log for the GeneNet saved in net_dir.

    Args:
        net_dir (str): Directory containing a saved GeneNet.
        ignore_error (bool): If True, ignore file not found errors.

    Returns:
        (Union[List[Dict[str, Dict]], None]): Training log of network info and
            performance metrics. Each list entry records training statistics
            at a validation instance. For information on log dict keys see
            TODO put a link here.

    """
    # Path containing training logs within the GeneNet save directory
    log_path = os.path.join(net_dir, 'log')

    try:
        # Get a list of all JSON log files in the log directory
        log_files = [f for f in os.listdir(log_path) if '.json' in f]

        # Load the most recently modified training log
        # Last-modified times
        log_mod_times = [os.path.getmtime(os.path.join(net_dir, f))
                         for f in log_files]
        # Index of the file with the most recent last-modified time
        latest_idx = int(np.argmax(log_mod_times))
        # The name of the most recent log file
        latest_log_file = log_files[latest_idx]

        # Get the training log from a JSON file
        with open(latest_log_file, 'r') as fl:
            train_log: List[Dict[str]] = json.load(fl)

        return train_log

    except FileNotFoundError as err:
        # Handle file not found errors
        if ignore_error:
            # Ignore the error if specified, just toss it into the logger as
            # a warning
            logger.warning(f'{err}')
            return None
        else:
            raise err
