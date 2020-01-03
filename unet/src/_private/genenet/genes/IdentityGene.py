"""IdentityGene defines a ScaleGene which performs no operation (identity
operation). Useful for combining the outputs of other Genes before, e.g.,
a pooling operation

"""
import logging

import metatree as mt
import tensorflow as tf

from .ScaleEdge import ScaleEdge
from .ScaleGene import ScaleGene
from ._merge_tensors import merge_tensors
from ._sum_tensors import sum_tensors

from typing import Dict, List, Sequence, Union

logger = logging.getLogger('genenet.IdentityGene')


class IdentityGene(ScaleGene):
    """A ScaleGene which performs an identity operation on its inputs.

    TODO
    Hyperparameters: None

    TODO
    Attributes:

    """
    def build(self,
              mode: tf.estimator.ModeKeys,
              outputs: mt.BuildMap):
        """Build this Gene's computation module into a TensorFlow computation
        graph.

        Args:
            mode (tf.estimator.ModeKeys): TensorFlow Estimator mode.
            outputs (mt.BuildMap): Computation graph build process' mapping
                from Tensor source to Tensor.

        Returns: None

        """
        with tf.variable_scope(self.name.replace(' ', '_')):
            # Build the input Tensor
            inputs = self.input_tensor(mode, outputs)
            # Identity op (shouldn't do anything)
            identity_op = tf.identity(inputs, name='identity')
        # Add to the outputs dict
        outputs[self] = identity_op

        pass

    def input_tensor(self,
                     mode: tf.estimator.ModeKeys,
                     outputs: mt.BuildMap) -> tf.Tensor:
        """Assemble a TensorFlow Tensor from this Gene's input Edges.

        Right now, this handles two types of input Edge connections:
        residual connections and merge connections. First, all residual
        inputs are summed. Second, all remaining inputs are merged together
        by concatenation along the convolution kernel feature axis. A single
        input connection of any type is copied from the input Edge unchanged.

        Args:
            mode (tf.estimator.ModeKeys): TensorFlow Estimator mode.
            outputs (mt.BuildMap): Computation graph
                build process' mapping from Tensor source to Tensor.

        Returns:
            (tf.Tensor): A Tensor to use as input to the Gene's computations.

        """
        # Build input edges
        for e in self.inputs:
            e.build(mode, outputs)
        # Right now this handles two types of inputs: 'residual' and 'merge'
        with tf.variable_scope(self.name.replace(' ', '_')):
            # Handle residual connections
            residual_sources = [outputs[e] for e in self.inputs
                                if e.edge_type in ['residual', 'forward']]
            residual_sum = sum_tensors(residual_sources)

            # Merge the sum of the residual connections with the other merge
            # Edges
            merge_sources = [outputs[e] for e in self.inputs
                             if e.edge_type == 'merge'] + \
                [residual_sum]
            final_input = merge_tensors(merge_sources, name='input_merge')

        return final_input

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
        self.data_shape_out = self.input_tensor_shape(edge_shapes)

        return self.data_shape_out
