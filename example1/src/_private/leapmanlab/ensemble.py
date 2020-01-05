"""Create an ensemble image from a collection of saved networks.

"""
import os

import genenet as gn
import numpy as np

from typing import Optional, Sequence


def ensemble(net_dirs: Sequence[str],
             data: np.ndarray,
             filename: Optional[str]=None,
             mode: str='average') -> np.ndarray:
    """Return an ensemble segmentation of some image data, possibly saving
    it to disk.

    Args:
        net_dirs (Sequence[str]): Save directories for one or more GeneNets
            to use for the ensemble.
        data (np.ndarray): Image data to segment.
        filename (Optional[str]): If supplied, save the output segmentation to
            a TIF image with this filename.
        mode (str): Ensemble mode. Either 
            'average': Average each network's probability maps together 
                before producing a segmentation.
            'majority': Produce a segmentation from each network, then select
                the majority vote label for each voxel. In the event of a tie,
                select the label with smallest numerical value.

    Returns:
        (np.ndarray): The ensemble segmentation.

    """
