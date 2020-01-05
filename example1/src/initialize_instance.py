"""Initialize an instance of the example - creating a folder for saved output
and archiving all source code at the time the instance is running.

"""
import logging
import os

# From _private
import genenet as gn
import leapmanlab as lab

from typing import Any, Dict, Tuple


def initialize_instance(
        save_dir: str,
        instance_settings: Dict[str, Any],
        logger_level: int) -> Tuple[Dict[str, Any], logging.Logger]:
    """Initialize an instance of the example. Create a folder for saved output
    within the experiment directory, and archive all source code at the time
    the instance is running.

    Args:
        save_dir (str): Directory where example output is saved. Each instance
            of the example that is run is saved in a subfolder inside
            `save_dir`, to simplify the use case where you run the example
            multiple times in short succession.
        instance_settings (Dict[str, Any]): Instance run instance_settings. Passed here for
            archival.
        logger_level (int): Logger level of detail.

    Returns:
        instance_settings (Dict[str, Any]): An updated version of `instance_settings` with an
            entry recording the instance save directory `instance_dir`.
        logger (logging.Logger): A logger for the example instance.
    """
    # Create the `save_dir` if it doesn't already exist
    os.makedirs(save_dir, exist_ok=True)
    # Create an instance save dir inside the main `save_dir`
    instance_dir = lab.create_output_dir(save_dir)
    instance_settings['instance_dir'] = instance_dir

    # Logger setup
    logger = gn.util.get_logger(
        os.path.join(instance_dir, 'log.log'),
        logger_level=logger_level)

    # Save the experiment instance_settings as a config file
    lab.save_config(os.path.join(instance_dir, 'instance_settings.cfg'), instance_settings)

    # Archive a copy of the example, saving all .py and .ipynb files
    example_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..'))
    lab.archive_experiment(example_dir, instance_dir, ['py', 'ipynb'])

    return instance_settings, logger
