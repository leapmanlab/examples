"""Create a "default" GeneGraph which builds an encoder-decoder GeneNet.

"""
import logging

import genenet as gn
import metatree as mt

from Mixed3DEncoderDecoderGene import Mixed3DEncoderDecoderGene
from SpatialPyramidGene import SpatialPyramidGene

from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple

logger = logging.getLogger('genenet.gene_graph')


def hybrid_gene_graph(
        input_shape: Sequence[int],
        n_classes: int,
        comps_list: Tuple[int],
        net_3d_combine_mode: str = 'add',
        net_2d_settings: Optional[Dict[str, Any]] = None,
        predictor_2d_settings: Optional[Dict[str, Any]] = None,
        net_3d_settings: Optional[Dict[str, Any]] = None,
        predictor_3d_settings: Optional[Dict[str, Any]] = None,
        optim_settings: Optional[Dict[str, Any]] = None,
        gene_dict: Optional[Dict[str, mt.Gene]] = None) -> \
        mt.GeneGraph:
    """Create a "default" GeneGraph which builds an encoder-decoder
    segmentation network as a GeneNet.

    Args:
        input_shape (Sequence[int]): Desired network input window shape.
            Actual shape may be smaller if the specified shape is
            incompatible with the encoder-decoder network architecture.
        n_classes (int): Number of segmentation classes.
        comps_list (Tuple[int]): List of spatial pyramid computation sizes
        net_3d_combine_mode (str): Either 'add' or 'merge'.
        net_2d_settings (Optional[Dict[str, Any]]): A dictionary of
            EncoderDecoderGene hyperparameter settings. Keys are names of
            hyperparameters, values are new values to which those
            hyperparameters are set. See
            genenet.hyperparameter_configs.encoder_decoder for
            hyperparameter information.
        predictor_2d_settings (Optional[Dict[str, Any]]): A dictionary of
            PredictorGene hyperparameter settings. Keys are names of
            hyperparameters, values are new values to which those
            hyperparameters are set. See
            genenet.hyperparameter_configs.predictor for
            hyperparameter information.
        net_3d_settings (Optional[Dict[str, Any]]): A dictionary of
            SpatialPyramidGene hyperparameter settings. Keys are names of
            hyperparameters, values are new values to which those
            hyperparameters are set. See
            genenet.hyperparameter_configs.encoder_decoder for
            hyperparameter information.
        predictor_3d_settings (Optional[Dict[str, Any]]): A dictionary of
            PredictorGene hyperparameter settings. Keys are names of
            hyperparameters, values are new values to which those
            hyperparameters are set. See
            genenet.hyperparameter_configs.predictor for
            hyperparameter information.
        optim_settings (Optional[Dict[str, Any]]): A dictionary of
            optimizer hyperparameter settings. Keys are names of
            hyperparameters, values are new values to which those
            hyperparameters are set. See
            genenet.hyperparameter_configs.adam_optimizer for
            hyperparameter information.
        gene_dict (Optional[Dict[str, mt.Gene]]): If supplied, use
            already-created root Genes instead of new ones. Valid keys are
            'encoderdecoder' for an EncoderDecoderGene, 'spatialpyramid' for a
            SpatialPyramidGene, or 'predictor_3d' for a PredictorGene.

    Returns:
        (mt.GeneGraph): The newly-created "default" GeneGraph.

    """
    if gene_dict is None:
        gene_dict = {}

    ''' 2D Encoder-decoder module setup '''

    # Encoder-decoder hyperparam config
    ed_hyperparams = gn.hyperparameter_configs.encoder_decoder()
    if 'net_2d' in gene_dict:
        # Use a supplied EncoderDecoderGene if present
        net_2d_gene: mt.Gene = gene_dict['net_2d']
    else:
        # Set up a new EncoderDecoderGene with hyperparameters set by the
        # input `net_2d_settings`
        # Create a Gene to build an encoder-decoder module
        net_2d_gene = Mixed3DEncoderDecoderGene(
            'net',
            conv3d_k_width=3,
            hyperparameter_config=ed_hyperparams)
        # Apply hyperparameter setting changes if supplied
        if net_2d_settings is not None:
            net_2d_gene.set(**net_2d_settings)

    # Set up Encoder-decoder gene tree edges
    # noinspection PyArgumentList
    net_2d_gene.setup_edges()

    ''' 2D predictor setup'''

    if 'predictor_2d' in gene_dict:
        # Use a supplied PredictorGene if present
        predictor_2d_gene: mt.Gene = gene_dict['predictor_2d']

    else:
        # Class predictor_2d hyperparameters
        pred_hyperparams = gn.hyperparameter_configs.predictor()

        predictor_2d_gene = gn.genes.PredictorGene(
            n_classes,
            'predictor_2d',
            pred_hyperparams)
        # Set the PredictorGene to use the same n_kernels value as the last
        # descendant of the EncoderDecoderGene
        n_kernels = net_2d_gene.last_descendant().hyperparam('n_kernels')
        predictor_2d_gene.set(n_kernels=n_kernels)
        # By default, take the regularization settings from net_2d_settings
        if not predictor_2d_settings and net_2d_settings:
            predictor_2d_settings = {}
            for key in net_2d_settings:
                if 'log_' in key:
                    predictor_2d_settings[key] = net_2d_settings[key]
        # Apply hyperparameter setting changes if supplied
        if predictor_2d_settings:
            predictor_2d_gene.set(**predictor_2d_settings)

    # Set up class predictor_3d gene edges
    predictor_2d_gene.setup_edges(
        gn.genes.ForwardEdge(net_2d_gene.last_descendant()))

    ''' 3D Spatial pyramid module setup '''

    if 'net_3d' in gene_dict:
        net_3d_gene: mt.Gene = gene_dict['net_3d']

    else:
        # Spatial pyramid hyperparam config
        sp_hyperparams = gn.hyperparameter_configs.encoder_decoder()
        # Create a Gene to build a spatial pyramid module
        net_3d_gene = SpatialPyramidGene(
            combine_mode=net_3d_combine_mode,
            name='spatialpyramid',
            hyperparameter_config=sp_hyperparams)
        # By default, take the regularization settings from net_2d_settings
        if net_3d_settings is None and net_2d_settings is not None:
            net_3d_settings = {}
            for key in net_2d_settings:
                if 'log_' in key:
                    net_3d_settings[key] = net_2d_settings[key]
        # Additional default settings
        net_3d_settings = {'padding_type': 'same',
                           'spatial_mode': 1,
                           'n_blocks': len(comps_list)}
        # Apply hyperparameter setting changes
        net_3d_gene.set(**net_3d_settings)
        # Set number of convs in each child block
        for child, n_comps in zip(net_3d_gene.children, comps_list):
            child.set(n_comps=n_comps)

    net_3d_gene.setup_edges([
        gn.genes.ForwardEdge(predictor_2d_gene.last_descendant()),
        gn.genes.MergeEdge(net_2d_gene.last_descendant())])

    # Class predictor_3d module setup

    if 'predictor_3d' in gene_dict:
        # Use a supplied PredictorGene if present
        predictor_3d_gene = gene_dict['predictor_3d']

    else:
        # Class predictor_3d hyperparameters
        pred_hyperparams = gn.hyperparameter_configs.predictor()

        predictor_3d_gene = gn.genes.PredictorGene(
            n_classes,
            'predictor_3d',
            pred_hyperparams)
        # Set the PredictorGene to use the same n_kernels value as the last
        # descendant of the EncoderDecoderGene
        n_kernels = net_3d_gene.last_descendant().hyperparam('n_kernels')
        predictor_3d_gene.set(n_kernels=n_kernels)
        # By default, take the regularization settings from net_2d_settings
        if not predictor_3d_settings and net_2d_settings:
            predictor_3d_settings = {}
            for key in net_2d_settings:
                if 'log_' in key:
                    predictor_3d_settings[key] = net_2d_settings[key]
        # Apply hyperparameter setting changes if supplied
        if predictor_3d_settings:
            predictor_3d_gene.set(**predictor_3d_settings)

    # Set up class predictor_3d gene edges
    predictor_3d_gene.setup_edges(
        gn.genes.ForwardEdge(net_3d_gene.last_descendant()))

    # ADAM optimizer hyperparameters
    adam_hyperparams = gn.hyperparameter_configs.adam_optimizer()
    # Convert to a dict to pass to a GeneNet.model_fn()
    optim_dict = adam_hyperparams.values()
    # Directly modify that dict because it's easier
    if optim_settings is not None:
        for key in optim_settings:
            optim_dict[key] = optim_settings[key]

    # Create the GeneGraph
    genes = OrderedDict()
    genes['net_2d'] = net_2d_gene
    genes['predictor_2d'] = predictor_2d_gene
    genes['net_3d'] = net_3d_gene
    genes['predictor_3d'] = predictor_3d_gene
    graph = mt.GeneGraph(input_shape, genes)

    # Add the optimizer hyperparameter dict to the GeneGraph
    graph.add_hyperparameter_config('optim', optim_dict)

    return graph
