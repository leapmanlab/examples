"""EncoderDecoderGene class controls the creation of an encoder-decoder network.

"""
import metatree as mt
import tensorflow as tf

from typing import Optional

import genenet.hyperparameter_configs as configs

from .DecoderGene import DecoderGene
from .EncoderGene import EncoderGene
from .ScaleEdge import ForwardEdge
from .ScaleGene import ScaleGene


class EncoderDecoderGene(ScaleGene):
    """Controls the creation of an encoder-decoder network (u-net).

    TODO
    Attributes:

    """

    def __init__(self,
                 name: str,
                 hyperparameter_config: Optional[
                     mt.HyperparameterConfig] = None,
                 spatial_scale: int = 0):
        """Constructor.

        Args:
            name (str): This gene's name.
            hyperparameter_config (Optional[mt.HyperparameterConfig]): The
                HyperparameterConfig governing this Gene's hyperparameters.
                If none is supplied, use
                genenet.hyperparameter_config.encoder_decoder().
            spatial_scale (int): The spatial scale of the data that
                this gene processes. A spatial_scale == i means the data has
                been downsampled by a factor of 2**i relative to the input
                data used by the computation graph this ScaleGene builds.
        """
        if hyperparameter_config is None:
            hyperparameter_config = configs.encoder_decoder()
        super().__init__(name,
                         parent=None,
                         hyperparameter_config=hyperparameter_config,
                         spatial_scale=spatial_scale)
        self.setup_children()

        pass

    def build(self,
              mode: tf.estimator.ModeKeys,
              outputs: mt.BuildMap):
        """Build this Gene's module into a TensorFlow computation graph.

        Args:
            mode (tf.estimator.ModeKeys): TensorFlow Estimator mode.
            outputs (BuildMap): Computation graph build process' mapping from
                Tensor source to Tensor.

        Returns:
            None

        """
        with tf.variable_scope(self.name.replace(' ', '_')):
            # Build child block sets
            for blockset in self.children:
                blockset.build(mode, outputs)

            outputs[self] = self.children[-1]

        pass

    def setup_children(self):
        """Set up child encoder and decoder modules.

        Returns: None

        """
        # Only generate new children if there are none
        if len(self.children) == 0:
            # Create the encoder and decoder genes
            encoder = EncoderGene(name='encoder',
                                  parent=self,
                                  spatial_scale=self.hyperparam(
                                      'spatial_scale'))
            self.children = [encoder]

            decoder = DecoderGene(name='decoder',
                                  parent=self,
                                  spatial_scale=self.hyperparam(
                                      'spatial_scale'))

            self.children.append(decoder)

        pass

    def setup_edges(self, *, _unused=None):
        """Set up the Edges that target this Gene and its children.

        Args:
            _unused: Unused.

        Returns: None

        """
        # Edge from input data
        encoder_edge = ForwardEdge('input',
                                   self.hyperparameter_config)

        # Encoder setup
        self.children[0].setup_edges(encoder_edge)

        # Decoder setup
        decoder_edge = ForwardEdge(self.children[0].last_descendant())
        self.children[1].setup_edges(decoder_edge)

        # Gene setup is now complete
        self.setup_complete = True

        pass
