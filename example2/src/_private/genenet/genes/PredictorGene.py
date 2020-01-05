"""ClassPredictionGene produces a class prediction for each spatial element
in a convolutional feature map.

"""
import logging

import metatree as mt
import tensorflow as tf

from typing import Callable, Optional, Sequence, Union

import genenet.hyperparameter_configs as configs

from ._get_regularizers import get_regularizers
from .ScaleGene import ScaleGene
from ..summarize import variable_summary

logger = logging.getLogger('genenet.PredictorGene')


class PredictorGene(ScaleGene):
    """PredictorGene creates dense class predictions for images from
    one or more convolution feature maps.

    Hyperparameters:

    Attributes:

    """
    def __init__(self,
                 n_classes: int,
                 name: str = 'predictor',
                 hyperparameter_config: Optional[
                     mt.HyperparameterConfig] = None):
        """Constructor.

        Args:
            n_classes (int): Number of prediction classes. Class labels are
                assumed to be integers in [0, n_classes).
            name (str): This gene's name.
            hyperparameter_config (Optional[mt.HyperparameterConfig]): The
                HyperparameterConfig governing this Gene's hyperparameters.
                If none is supplied, use
                genenet.hyperparameter_config.predictor().
        """
        self.n_classes = n_classes

        if hyperparameter_config is None:
            hyperparameter_config = configs.predictor()

        super().__init__(name,
                         parent=None,
                         hyperparameter_config=hyperparameter_config,
                         spatial_scale=0)

        pass

    def build(self,
              mode: tf.estimator.ModeKeys,
              outputs: mt.BuildMap):
        """Build a portion of a TensorFlow computation graph.

        Args:
            mode (tf.estimator.ModeKeys): TensorFlow Estimator mode.
            outputs (mt.BuildMap): Computation graph build process' mapping
                from Tensor source to Tensor.

        Returns: None

        """
        # Get hyperparameter values
        f_activate: Callable = self.hyperparam('f_activate')
        spatial_mode: bool = self.hyperparam('spatial_mode')

        with tf.variable_scope(self.name.replace(' ', '_')):
            # Determine whether to use 2D or 3D convolution
            if spatial_mode == 0:
                conv = tf.layers.conv2d
            else:
                conv = tf.layers.conv3d

            # Build an activity regularizer
            kernel_regularizer, activity_regularizer = get_regularizers(self)

            # Build the input tensor
            inputs = self.input_tensor(mode, outputs)

            # Create a 1x1(x1) convolution op
            conv_op = conv(
                inputs=inputs,
                filters=self.n_classes,
                kernel_size=1,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='logits',
                use_bias=True,
                padding='VALID',
                data_format='channels_first',
                activation=f_activate,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer
            )

            # Create summary ops
            variable_summary(conv_op)

        logger.debug(f'Created conv layer with shape {conv_op.get_shape()}')

        outputs[self] = conv_op

        pass

    def setup_children(self):
        pass

    def setup_edges(self, edges: Union[mt.Edge, Sequence[mt.Edge]]):
        """Set up the Edges that target this Gene.

        Args:
            edges (Union[mt.Edge, Sequence[mt.Edge]]): One or more input
                Edges to this Gene.

        Returns: None

        """
        if not isinstance(edges, Sequence):
            edges = [edges]
        for edge in edges:
            self.add_input(edge)

        # Gene setup is now complete
        self.setup_complete = True

        pass
