"""Parse python-style keyword args from a text config file, one key-value pair
per line.

"""
import ast

from typing import Any, Dict


def config_parser(fname: str) -> Dict[str, Any]:
    """

    Args:
        fname (str): Name of the config file

    Returns:
        (Dict[str, Any]): Dictionary of parsed config args.

    """
    parsed_kwargs = {}
    with open(fname, 'r') as file:
        lines = [line for line in file.read().split('\n') if len(line) > 0]

    for line in lines:
        if '=' in line:
            split = line.split('=')
            parsed_kwargs[split[0]] = ast.literal_eval(
                split[1].replace('\n', ''))
        else:
            print(f'Warning: Ignoring config line {line}')

    return parsed_kwargs
