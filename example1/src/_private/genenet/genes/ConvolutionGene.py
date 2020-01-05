"""ConvolutionGene class defines a ScaleGene which performs a convolution
operation.

"""
import logging

import metatree as mt
import tensorflow as tf

from typing import Callable, Dict, List, Sequence, Union

from .ScaleEdge import ScaleEdge
from .ScaleGene import ScaleGene
from ._get_regularizers import get_regularizers

logger = logging.getLogger('genenet.ConvolutionGene')


class ConvolutionGene(ScaleGene):
    """A ScaleGene which builds a convolution operation into a computation
    graph.

    Hyperparameters:
        batch_norm (bool): If True, use batch normalization.
        f_activate (Callable): Choice of TensorFlow activation function.
        spatial_mode (int): Computation graph uses 2D data and 2D ops if 0,
            3D data and 3D ops if 1, 3D data and 2D ops if 2.
        k_width (int): Convolution kernel width (along each spatial dimension).
        n_kernels (int): Number of convolution kernels in this
            ConvolutionGene's convolution op.

    TODO
    Attributes:

    """
    def build(self,
              mode: tf.estimator.ModeKeys,
              outputs: mt.BuildMap):
        """Build this Gene's module into a TensorFlow computation graph.

        Args:
            mode (tf.estimator.ModeKeys): TensorFlow Estimator mode.
            outputs (mt.BuildMap): Computation graph build process' mapping from
                Tensor source to Tensor.

        Returns: None

        """
        # Convolution kernels have this size in each dimension
        k_width: int = self.hyperparam('k_width')
        # Convolution dilation rate
        dilation_rate: int = self.hyperparam('dilation_rate')
        # Convolution padding type
        padding_type: str = self.hyperparam('padding_type')
        # Choice of activation function
        f_activate: Callable = self.hyperparam('f_activate')
        # Number of convolution kernels
        n_kernels: int = self.hyperparam('n_kernels')
        # Will we use batch normalization?
        batch_norm: bool = self.hyperparam('batch_norm')

        spatial_mode: int = self.hyperparam('spatial_mode')

        # Determine the correct convolution operation
        if spatial_mode == 0:
            conv = tf.layers.conv2d
        else:
            conv = tf.layers.conv3d
            if spatial_mode == 2:
                k_width = [1, k_width, k_width]

        with tf.variable_scope(self.name.replace(' ', '_')):
            # Get the convolution input Tensor
            inputs = self.input_tensor(mode, outputs)

            # Get regularization functions
            kernel_regularizer, activity_regularizer = get_regularizers(self)
            # Kernel initializer
            kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d()
            # The convolution operation, before activation
            conv_op = conv(
                inputs=inputs,
                filters=n_kernels,
                kernel_size=k_width,
                kernel_initializer=kernel_initializer,
                name='conv',
                use_bias=True,
                padding=padding_type,
                dilation_rate=dilation_rate,
                data_format='channels_first',
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer
            )

            if batch_norm:
                # Apply batch normalization before the activation function
                batch_norm_op = tf.layers.batch_normalization(
                    inputs=conv_op,
                    axis=1,  # For 'channels_first' format
                    training=mode == tf.estimator.ModeKeys.TRAIN
                )
                final_op = f_activate(batch_norm_op)
            else:
                final_op = f_activate(conv_op)

        logger.debug(f'Created conv layer with shape {final_op.get_shape()}')

        # Add to the outputs dict
        outputs[self] = final_op

        pass

    def setup_children(self):
        """Setup for self.children, which in this case should always be empty.

        Returns: None

        """
        self.children = []

        pass

    def setup_edges(self, edges: Union[ScaleEdge, Sequence[ScaleEdge]]):
        """Set up the Edges that target this Gene and its children.

        Args:
            edges (Union[ScaleEdge, Sequence[ScaleEdge]]): One or more
                ScaleEdges that are inputs to this ConvolutionGene.

        Returns: None

        """
        # Wrap in a list for consistency
        edges = edges if isinstance(edges, Sequence) else [edges]

        for e in edges:
            self.add_input(e)

        # Gene setup is now complete
        self.setup_complete = True

        pass

    def shape_out(self,
                  shapes_in: Union[Sequence[int],
                                   Dict[mt.Edge, Sequence[int]]]) -> List[int]:
        """Calculate the spatial shape of the Tensor output by this Gene,
        given input spatial shapes specified in `shapes_in`.

        If there is a single input, the shape may be passed in as a single
        Sequence[int]. If there are multiple inputs, `shapes_in` must be a
        dict with `Edge` keys and spatial shape values: `shapes_in[e]` is the
        shape coming into this Gene from the source associated with input
        edge `e`.

        Args:
            shapes_in (Union[Sequence[int], Dict['Edge', Sequence[int]]]):
                The spatial shape of each source for this Gene's inputs. If
                there is a single input, the shape may be passed in as a single
                Sequence[int]. If there are multiple inputs, `shapes_in` must
                be a dict with `Edge` keys and spatial shape values:
                `shapes_in[e]` is the shape coming into this Gene from the
                source associated with input edge `e`.

        Returns:
            (List[int]): Shape of the corresponding output Tensor from this
                Gene.

        """
        self.data_shapes_in = shapes_in

        if isinstance(shapes_in, Sequence):
            # Single input shape, pass through the input forward edge

            # Find the edge
            forward_input = [e for e in self.inputs
                             if e.edge_type == 'forward']
            # Should be exactly one
            if len(forward_input) != 1:
                raise IndexError(f'Found {len(forward_input)} input '
                                 f'ForwardEdges. Exactly one needed.')
            forward_edge = forward_input[0]
            # Edge output shape
            edge_shapes = forward_edge.shape_out(shapes_in)
        else:
            # Multiple inputs. Pass each through its edge
            edge_shapes = {e: e.shape_out(shapes_in[e]) for e in self.inputs}

        # Get input tensor shape
        shape = self.input_tensor_shape(edge_shapes)

        if self.hyperparam('padding_type') == 'valid':
            # Pass the shape through the convolution operation
            k_width = self.hyperparam('k_width')
            dilation_rate = self.hyperparam('dilation_rate')

            spatial_mode = self.hyperparam('spatial_mode')
            # Effective kernel width
            k_width = k_width + (k_width-1)*(dilation_rate-1)
            if spatial_mode == 2:
                data_shape_out = [s - k_width + 1 if i != 0 else s
                                  for i, s in enumerate(shape)]
            else:
                data_shape_out = [s - k_width + 1 for s in shape]

        elif self.hyperparam('padding_type') == 'same':
            data_shape_out = shape[:]

        else:
            raise ValueError(f'Unrecognized padding type '
                             f'{self.hyperparam("padding_type")}')

        self.data_shape_out = data_shape_out

        return self.data_shape_out
