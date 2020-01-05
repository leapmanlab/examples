"""Create a GeneGraph which builds an encoder-decoder GeneNet with residual
blocks.

"""
import logging

import metatree as mt

from . import hyperparameter_configs as configs
from .genes import EncoderDecoderGene, IdentityGene, \
    PredictorGene, ForwardEdge, ResidualEdge
from .shape_check import shape_check

from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger('genenet.gene_graph')


def residual_gene_graph(input_shape: Sequence[int],
                        n_classes: int,
                        net_settings: Optional[Dict[str, Any]] = None,
                        predictor_settings: Optional[Dict[str, Any]] = None,
                        optim_settings: Optional[Dict[str, Any]] = None,
                        gene_dict: Optional[Dict[str, mt.Gene]] = None) -> \
        mt.GeneGraph:
    """Create a GeneGraph which builds an encoder-decoder
    segmentation network with residual blocks.

    Args:
        input_shape (Sequence[int]): Desired network input window shape.
            Actual shape may be smaller if the specified shape is
            incompatible with the encoder-decoder network architecture.
        n_classes (int): Number of segmentation classes.
        net_settings (Optional[Dict[str, Any]]): A dictionary of
            EncoderDecoderGene hyperparameter instance_settings. Keys are names of
            hyperparameters, values are new values to which those
            hyperparameters are set. See
            genenet.hyperparameter_configs.encoder_decoder for
            hyperparameter information.
        predictor_settings (Optional[Dict[str, Any]]): A dictionary of
            PredictorGene hyperparameter instance_settings. Keys are names of
            hyperparameters, values are new values to which those
            hyperparameters are set. See
            genenet.hyperparameter_configs.predictor for
            hyperparameter information.
        optim_settings (Optional[Dict[str, Any]]): A dictionary of
            optimizer hyperparameter instance_settings. Keys are names of
            hyperparameters, values are new values to which those
            hyperparameters are set. See
            genenet.hyperparameter_configs.adam_optimizer for
            hyperparameter information.
        gene_dict (Optional[Dict[str, mt.Gene]]): If supplied, use
            already-created root Genes instead of new ones. Valid keys are
            'encoderdecoder' for an EncoderDecoderGene, or 'predictor' for a
            PredictorGene

    Returns:
        (mt.GeneGraph): The newly-created "default" GeneGraph.

    """
    if gene_dict is None:
        gene_dict = {}

    # Encoder-decoder network setup

    if 'encoderdecoder' in gene_dict:
        # Use a supplied EncoderDecoderGene if present
        net_gene: EncoderDecoderGene = gene_dict['encoderdecoder']
    else:
        # Set up a new EncoderDecoderGene with hyperparameters set by the
        # input `net_settings`

        # Encoder-decoder hyperparam config
        ed_hyperparams = configs.encoder_decoder()
        # Create a Gene to build an encoder-decoder network
        net_gene = EncoderDecoderGene('net',
                                      ed_hyperparams)
        # Apply hyperparameter setting changes if supplied
        if net_settings is not None:
            net_gene.set(**net_settings)

    # Turn the convolution blocks into residual blocks
    for blockset in net_gene.children:
        for block in blockset.children:
            if len(block.children) > 1:
                # Add an identity layer that adds the final convolution layer
                # and the first convolution layer
                identity = IdentityGene('identity', block)
                block.children.append(identity)
                # Identity layer will get the last conv as an input by
                # default, now also add in a residual connection from the
                # first conv
                residual_edge = ResidualEdge(block.children[0])
                identity.add_input(residual_edge)

    # Set up Encoder-decoder gene tree edges
    net_gene.setup_edges()

    # Class predictor module setup

    if 'predictor' in gene_dict:
        # Use a supplied PredictorGene if present
        predictor_gene: PredictorGene = gene_dict['predictor']

    else:
        # Class predictor hyperparameters
        pred_hyperparams = configs.predictor()

        predictor_gene = PredictorGene(n_classes,
                                       'predictor',
                                       pred_hyperparams)
        # Set the PredictorGene to use the same n_kernels value as the last
        # descendant of the EncoderDecoderGene
        n_kernels = net_gene.last_descendant().hyperparam('n_kernels')
        predictor_gene.set(n_kernels=n_kernels)
        # By default, take the regularization instance_settings from net_settings
        if predictor_settings is None and net_settings is not None:
            predictor_settings = {}
            for key in net_settings:
                if 'log_' in key:
                    predictor_settings[key] = net_settings[key]
        # Apply hyperparameter setting changes if supplied
        if predictor_settings is not None:
            predictor_gene.set(**predictor_settings)

    # Set up class predictor gene edges
    predictor_gene.setup_edges(ForwardEdge(net_gene.last_descendant()))

    # ADAM optimizer hyperparameters
    adam_hyperparams = configs.adam_optimizer()
    # Convert to a dict to pass to a GeneNet.model_fn()
    optim_dict = adam_hyperparams.values()
    # Directly modify that dict because it's easier
    for key in optim_settings:
        optim_dict[key] = optim_settings[key]

    # Create the GeneGraph
    genes = OrderedDict()
    genes['encoderdecoder'] = net_gene
    genes['predictor'] = predictor_gene
    # Make sure we have a valid input shape
    valid_shape = False
    # Track the original input shape for later reference
    original_input_shape = input_shape[:]
    graph = None
    while not valid_shape:
        graph = mt.GeneGraph(input_shape, genes)
        too_small, odd_output = shape_check(graph)
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
    # Add the optimizer hyperparameter dict to the GeneGraph
    graph.add_hyperparameter_config('optim', optim_dict)

    if original_input_shape != input_shape:
        logger.info(f'Changed input shape from {original_input_shape}'
                    f'to {input_shape} to make a valid network.')

    return graph
