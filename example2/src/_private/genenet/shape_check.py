"""Check to see if the network encoded by a GeneGraph has a valid shape.

A shape can be invalid if (1) the final shape is <= 0 along any axis, or (2)
if the end of any encoder convblock has an odd shape along any axis.

"""
import logging

import metatree as mt

from typing import Sequence, Tuple, Union

logger = logging.getLogger('genenet.shape_check')


def shape_check(graph: mt.GeneGraph,
                min_size: Union[int, Sequence[int]]=0) -> Tuple[bool, bool]:
    """Check to see if the network encoded by the input GeneGraph has a valid
    shape.

    A shape can be invalid if (1) the final shape is <= 0 along any axis, or
    (2) if the end of any encoder convblock has an odd shape along any axis.

    Args:
        graph (GeneGraph): The net to check.
        min_size (Union[int, Sequence[int]]): If a nonnegative number,
            the minimum acceptable size for the graph's output window along
            each axis. If a sequence of numbers, the sequence should have the
            same length as the graph's output window's shape, and sizes are
            compared elementwise.

    Returns:
        (Tuple[bool, bool]): (too_small, odd_output). `too_small` is True if
            the network output shape becomes <= 0. `odd_output` is True if
            any encoder blocks have odd output shape.

    """
    # True if the shape goes <= 0
    too_small = False
    # True if any encoder blocks have odd output shape
    odd_output = False
    # If the min acceptable size is an int, make into a list of ints
    output_shape = graph.output_shape()
    if isinstance(min_size, int):
        min_size = [min_size] * len(output_shape)

    if any([s <= m for s, m in zip(output_shape, min_size)]):
        logger.info(f'Output shape {output_shape} has components '
                    f'<= {min_size}.')
        too_small = True
    # Check the output shape of each convolution block in the network encoder
    # path
    encoder_path = graph.genes['encoderdecoder'].children[0]
    for i, block in enumerate(encoder_path.children):
        if any([s % 2 == 1 for s in block.data_shape_out]):
            logger.info(f'Encoder block {i} has shape {block.data_shape_out}'
                        f'with odd components.')
            odd_output = True
    return too_small, odd_output
