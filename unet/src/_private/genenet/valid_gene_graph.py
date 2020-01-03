"""Creates a GeneGraph with a valid input shape from a sequence of Genes,
the first of which must have key 'encoderdecoder' and correspond to an
EncoderDecoderGene.

"""
import logging

import metatree as mt

from .shape_check import shape_check

logger = logging.getLogger('genenet.valid_gene_graph')


def valid_gene_graph(genes, input_shape, min_size=0) -> mt.GeneGraph:
    """

    Args:
        genes:
        input_shape:
        min_size:

    Returns:

    """
    original_input_shape = input_shape[:]
    graph = None
    valid_shape = False
    while not valid_shape:
        graph = mt.GeneGraph(input_shape, genes)
        too_small, odd_output = shape_check(graph, min_size)
        if too_small:
            raise ValueError('Desired input shape results in invalid output '
                             'shape')
        if odd_output:
            # Keep decreasing shape by 1 until we get a valid output
            if len(input_shape) == 2:
                input_shape = [s - 1 for s in input_shape]
            else:
                # Don't mess with the z dimension
                input_shape = [input_shape[0]] + \
                              [s - 1 for s in input_shape[1:]]
        else:
            # We're good to go
            valid_shape = True

    if original_input_shape != input_shape:
        logger.info(f'Changed input shape from {original_input_shape}'
                    f'to {input_shape} to make a valid network.')

    return graph
