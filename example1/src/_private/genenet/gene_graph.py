"""Create a "default" GeneGraph which builds an encoder-decoder GeneNet.

"""
import logging

import metatree as mt

from . import hyperparameter_configs as configs
from .genes import EncoderDecoderGene, PredictorGene, ForwardEdge
from .valid_gene_graph import valid_gene_graph

from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Union

logger = logging.getLogger('genenet.gene_graph')


def gene_graph(input_shape: Sequence[int],
               n_classes: int,
               net_settings: Optional[Dict[str, Any]]=None,
               predictor_settings: Optional[Dict[str, Any]]=None,
               optim_settings: Optional[Dict[str, Any]]=None,
               gene_dict: Optional[Dict[str, mt.Gene]]=None,
               min_size: Union[int, Sequence[int]]=0) -> \
        mt.GeneGraph:
    """Create a "default" GeneGraph which builds an encoder-decoder
    segmentation network as a GeneNet.

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
        min_size (Union[int, Sequence[int]]): If a nonnegative number,
            the minimum acceptable size for the graph's output window along
            each axis. If a sequence of numbers, the sequence should have the
            same length as the graph's output window's shape, and sizes are
            compared elementwise.

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
    if optim_settings is not None:
        for key in optim_settings:
            optim_dict[key] = optim_settings[key]

    # Create the GeneGraph
    genes = OrderedDict()
    genes['encoderdecoder'] = net_gene
    genes['predictor'] = predictor_gene
    graph = mt.GeneGraph(input_shape, genes)
    graph = valid_gene_graph(genes, input_shape, min_size)

    # Add the optimizer hyperparameter dict to the GeneGraph
    graph.add_hyperparameter_config('optim', optim_dict)

    return graph
