"""Save a dictionary of experiment parameters as a text config file.

"""
from typing import Any, Dict


def save_config(filename: str, params: Dict[str, Any]):
    """Save a dictionary of experiment parameters as a text config file.

    Args:
        filename (str): Name for the save file.
        params (Dict[str, Any]): Experiment parameters.

    Returns: None

    """
    with open(filename, 'w') as f:
        for key in params:
            val = params[key]
            if isinstance(val, str):
                f.write(f"{key}='{val}'\n")
            else:
                f.write(f'{key}={val}\n')
    pass

