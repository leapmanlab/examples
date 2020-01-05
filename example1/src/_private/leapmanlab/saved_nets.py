"""Return a list of all dirs containing saved nets within an input
dir.

"""
import os

from typing import List

GENEGRAPH = 'gene_graph.pkl'
HISTORY = 'file_history.json'


def saved_nets(base_dir: str) -> List[str]:
    """Return a list of all dirs within `base_dir` that are save folders for
    GeneNets.

    Args:
        base_dir (str): Base directory to check.

    Returns:
        (List[str]): All subdirectories of `base_dir` that are save folders for
            GeneNets.

    """
    # Walk through `base_dir` subdirectories recursively
    dirs_to_check = [base_dir]
    save_dirs = []
    while len(dirs_to_check) > 0:
        dtc = dirs_to_check.pop()
        sub_dirs = [os.path.join(dtc, d) for d in os.listdir(dtc)
                    if os.path.isdir(os.path.join(dtc, d))]
        for d in sub_dirs:
            if _is_a_save_dir(d):
                save_dirs.append(d)
            else:
                dirs_to_check.append(d)
    return save_dirs


def _is_a_save_dir(d: str) -> bool:
    """A directory is a GeneNet save dir if it contains an eval history file
    and a pickled GeneGraph. Return True if so, else False

    Args:
        d (str): The directory to check.

    Returns:
        (bool): True if d is a save dir, else False.

    """
    files = os.listdir(d)
    return HISTORY in files and GENEGRAPH in files