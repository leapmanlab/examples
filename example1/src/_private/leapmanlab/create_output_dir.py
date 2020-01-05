"""Create an output directory for an experiment.

"""
import os


def create_output_dir(base_dir: str) -> str:
    """Create an output dir for a GeneNet in an experiment.

    Args:
        base_dir (str): Base directory containing the eventual output
            directory.

    Returns:
        (str): Name of the output directory.

    """
    os.makedirs(base_dir, exist_ok=True)
    while True:
        try:
            n_dirs = len([d for d in os.listdir(base_dir)
                          if os.path.isdir(os.path.join(base_dir, d))])
            output_dir = os.path.join(base_dir, f'{n_dirs}')
            os.makedirs(output_dir, exist_ok=False)
            break
        except FileExistsError:
            pass

    return output_dir

