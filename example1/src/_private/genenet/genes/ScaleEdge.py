"""ScaleEdge class defines an Edge which tracks a spatial scale attribute,
and processes input tensors to match output tensor spatial scale.

"""
import logging

import metatree as mt
import tensorflow as tf

from .ScaleGene import ScaleGene
from ._get_regularizers import get_regularizers
from ..summarize import variable_summary

from typing import List, Optional, Sequence, Tuple, Union


logger = logging.getLogger('genenet.ScaleEdge')


class ScaleEdge(mt.Edge):
    """An Edge which tracks a spatial scale attribute for its inputs and
    outputs.

    Hyperparameters:
        TODO: Fix this documentation!
        spatial_mode (bool): If True, Edge is operating on Tensors with 3 spatial
            dimensions, else 2.
        pooling_type (str): What kind of pooling to use for rescaling.

    TODO
    Attributes:

    """
    def __init__(self,
                 name: str,
                 source: Union[ScaleGene, str],
                 hyperparameter_config: Optional[mt.HyperparameterConfig]=None,
                 edge_type: str='forward'):
        """Constructor.

        Args:
            name (str): This edge's name.
            source (Union[ScaleGene, str]): This edge's source.
            hyperparameter_config (Optional[mt.HyperparameterConfig]): This
                edge's hyperparameter configuration object. If the edge
                source is a Gene and no hyperparameter_config is supplied,
                the constructor uses the source's hyperparameter_config.
            edge_type (str): Currently, ScaleEdges support three types:
                'residual', 'merge', and 'forward'. TODO: More info on this.
        """
        super().__init__(name, source, hyperparameter_config)

        # This information can be used by this ScaleEdge, as well as its
        # target ScaleGene, to process the Tensor passing through this
        # ScaleEdge.
        self.edge_type = edge_type

        self.data_shapes_in: List[int] = []
        self.data_shape_out: List[int] = []

        pass

    def build(self,
              mode: tf.estimator.ModeKeys,
              outputs: mt.BuildMap):
        """Edge building operation that scales source data to the correct
        spatial resolution.

        Adds ops to the active TensorFlow compute graph to change the spatial
        resolution of the source tf.Tensor to convert it to the scale of the
        parent/target ScaleGene.

        For downsampling, use downsampling type determined from the
        hyperparameter config. For upsampling, used transposed convolutions.

        Args:
            mode (tf.estimator.ModeKeys): TensorFlow Estimator mode.
            outputs (mt.BuildMap): Computation graph build process' mapping
                from Tensor source to Tensor.

        Returns: None

        """
        # Get spatial scales of the input and output data sources
        scale_in, scale_out = self.scales()

        # Get the edge input
        edge_input = self.source_tensor(outputs)

        if scale_in == scale_out:
            # No up/downsampling necessary
            outputs[self] = edge_input

        else:
            # Some resizing is necessary
            # Are we working in 2D or 3D?
            spatial_mode: int = self.hyperparam('spatial_mode')
            # What pooling type to use?
            pooling_type: str = self.hyperparam('pooling_type')
            # Output number of convolution kernels
            n_kernels = self.parent.hyperparam('n_kernels')

            # This edge's output's spatial scale, divided by its input's
            # spatial scale
            scaling_factor = self._scaling_factor()

            # When downsampling, strides is the inverse of scaling_factor.
            # When upsampling strides and scaling_factor are equal
            if scale_in < scale_out:
                strides = [int(round(1. / s)) for s in scaling_factor]
            else:
                strides = scaling_factor[:]

            # Kernel shape and strides are identical for sampling ops
            kernel_size = tuple(strides)

            # Start building!
            with tf.variable_scope(self.name.replace(' ', '_')):
                # Get regularizers for convolution ops
                kernel_regularizer, activity_regularizer = \
                    get_regularizers(self)

                # Kwargs for all conv ops are the same
                # TODO: Activations?
                conv_args = {
                    'inputs': edge_input,
                    'strides': strides,
                    'filters': n_kernels,
                    'kernel_size': kernel_size,
                    'kernel_initializer':
                        tf.contrib.layers.xavier_initializer_conv2d(),
                    'use_bias': True,
                    'kernel_regularizer': kernel_regularizer,
                    'bias_regularizer': kernel_regularizer,
                    'activity_regularizer': activity_regularizer,
                    'padding': 'SAME',
                    'data_format': 'channels_first',
                }
                # Kwargs for max pooling operations
                maxpool_args = {
                    'inputs': edge_input,
                    'pool_size': kernel_size,
                    'strides': strides,
                    'data_format': 'channels_first'
                }
                # Build the sampling layer
                sampling_layer = None
                if scale_out > scale_in:
                    # Downsample
                    if pooling_type == 'conv':
                        # Downsample with a strided convolution
                        if spatial_mode == 0:
                            downsampling_op = tf.layers.conv2d
                        else:
                            downsampling_op = tf.layers.conv3d

                        name = 'downsample_conv'
                        kwargs = conv_args
                    elif pooling_type == 'maxpool':
                        # Downsample using max pooling
                        if spatial_mode == 0:
                            downsampling_op = tf.layers.max_pooling2d
                        else:
                            downsampling_op = tf.layers.max_pooling3d

                        name = 'downsample_maxpool'
                        kwargs = maxpool_args

                    else:
                        # Invalid downsampling type
                        error = f'Invalid downsampling type {pooling_type}'
                        logger.exception(error)
                        raise ValueError(error)

                    # Build the downsampling op
                    sampling_layer = downsampling_op(name=name, **kwargs)

                else:
                    # Upsample with transposed convolutions
                    if spatial_mode == 0:
                        upsampling_op = tf.layers.conv2d_transpose
                    else:
                        upsampling_op = tf.layers.conv3d_transpose

                    sampling_layer = upsampling_op(name='upsample_conv',
                                                   **conv_args)

                # Create summary ops
                if pooling_type == 'conv':
                    variable_summary(sampling_layer)

            outputs[self] = sampling_layer

        pass

    def scales(self) -> Tuple[int, int]:
        """Determine the spatial scales of this ScaleEdge's source and parent.

        Returns:
            scale_in (int): Spatial scale of the source data.
            scale_out (int): Spatial scale of the parent ScaleGene data.

        """
        # Determine scale in
        if not isinstance(self.source, ScaleGene):
            # Assume the input is some actual input data, e.g. a TensorFlow
            # Tensor or NumPy ndarray. Input spatial scale is therefore 0,
            # i.e., 2^0 times the scale of the input data
            scale_in = 0
        else:
            # Use source ScaleGene's spatial_scale hyperparam
            scale_in: int = self.source.hyperparam('spatial_scale')

        # Determine scale out
        if self.parent is None:
            raise ValueError(
                'Cannot determine output spatial scale without a parent.')
        # Use parent's spatial_scale hyperparam
        scale_out: int = self.parent.hyperparam('spatial_scale')

        return scale_in, scale_out

    def shape_out(self, shape_in: Sequence[int]) -> List[int]:
        """Calculate the shape of the Tensor output by this Edge, if its
        input has shape `shape_in`. Update data shape attributes.

        Args:
            shape_in (Sequence[int]): Shape of the input Tensor for this Edge.

        Returns:
            (List[int]): Shape of the corresponding output Tensor from this
                Edge.

        """
        self.data_shapes_in = list(shape_in)
        # Size doubles each time you go up a spatial scale, halves each time
        # you go down. Maybe more or less than that depending on how much z
        # pooling is used
        scaling_factor = self._scaling_factor()
        self.data_shape_out = [int(f * s)
                               for f, s in zip(scaling_factor, shape_in)]
        return self.data_shape_out

    def _scaling_factor(self) -> List[int]:
        """Determine the scaling factor produced by this edge along each
        spatial dimension.

        Returns:
            (List[int]): Scaling factor produced by this edge along each
                spatial dimension.

        """
        # Get the spatial scales of this ScaleEdge's input and output data
        scale_in, scale_out = self.scales()
        scale_const = 2 ** (scale_in - scale_out)

        scaling_factor = [scale_const, scale_const]
        spatial_mode = self.hyperparam('spatial_mode')
        # Add another stride if we are working in 3D
        if not (spatial_mode == 0):
            # Figure out whether z-pooling is being used
            pool_z: bool
            if not isinstance(self.source, mt.Gene):
                # Input is a raw data array - use network level pool_z
                pool_z = self.root.hyperparam('pool_z')
            else:
                # Get the pool_z info from the source
                pool_z = self.source.hyperparam('pool_z')

            if pool_z:
                # Scale in the z direction, too
                first_dim = [scale_const]
            else:
                # Don't scale in the z direction
                first_dim = [1]
            # Concatenate the lists
            scaling_factor = first_dim + scaling_factor

        return scaling_factor


class ForwardEdge(ScaleEdge):
    """Wrapper around a ScaleEdge of 'forward' type.

    Represents a simple composition of functions - the Edge's source output
    is passed as an input to the Edge's target Gene. Logically the same as a
    residual connection, but it sounds weird to say that you need to put
    'residual' connections between simple successions of operations.

    """
    def __init__(self,
                 source: Union[mt.Gene, str],
                 hyperparameter_config: Optional[mt.HyperparameterConfig]=None,
                 name: str='forward'):
        """Constructor.

        Args:
            source (Union[mt.Gene, str]): This edge's source.
            hyperparameter_config (Optional[mt.HyperparameterConfig]): This
                edge's hyperparameter configuration object.
            name (str): Name of this edge.
        """
        super().__init__(name,
                         source,
                         hyperparameter_config,
                         edge_type='forward')


class ResidualEdge(ScaleEdge):
    """Wrapper around a ScaleEdge of 'residual' type.

    ResidualEdge inputs to a Gene are added together as input to the Gene.

    """
    def __init__(self,
                 source: Union[mt.Gene, str],
                 hyperparameter_config: Optional[mt.HyperparameterConfig]=None,
                 name: str='residual'):
        """Constructor.

        Args:
            source (Union[mt.Gene, str]): This edge's source.
            hyperparameter_config (Optional[mt.HyperparameterConfig]): This
                edge's hyperparameter configuration object.
            name (str): Name of this edge.
        """
        super().__init__(name,
                         source,
                         hyperparameter_config,
                         edge_type='residual')


class MergeEdge(ScaleEdge):
    """Wrapper around a ScaleEdge of 'merge' type.

    MergeEdge inputs to a Gene are concatenated together along their final
    axis, i.e. the axis along which convolution kernels are indexed for
    ConvolutionGenes.

    """

    def __init__(self,
                 source: Union[mt.Gene, str],
                 hyperparameter_config: Optional[mt.HyperparameterConfig]=None,
                 name: str='merge'):
        """Constructor.

        Args:
            source (Union[mt.Gene, str]): This edge's source.
            hyperparameter_config (Optional[mt.HyperparameterConfig]): This
                edge's hyperparameter configuration object.
            name (str): Name of this edge.
        """
        super().__init__(name,
                         source,
                         hyperparameter_config,
                         edge_type='merge')
