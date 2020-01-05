"""Parse python-style keyword args from the command-line.

"""
import ast


def kwarg_parser(args):
    """Parse python-style keyword args from the command-line.
    Args:
        args: CLI arguments.
    Returns:
        parsed_kwargs: Dict of key-value pairs.
    """
    parsed_kwargs = {}
    for arg in args:
        if '=' in arg:
            split = arg.split('=')
            parsed_kwargs[split[0]] = ast.literal_eval(split[1])
    return parsed_kwargs
