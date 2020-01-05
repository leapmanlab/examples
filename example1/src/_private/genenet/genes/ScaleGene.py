"""ScaleGene class defines a Gene which tracks a spatial scale attribute.

"""
import metatree as mt
import tensorflow as tf

from ._merge_tensors import merge_tensors
from ._sum_tensors import sum_tensors

from typing import Dict, List, Optional, Sequence, Union


class ScaleGene(mt.Gene):
    """A Gene used to build modules connected by ScaleEdges.

    Hyperparameters:
        spatial_scale (int): Spatial scale of data processed by this Gene,
            relative to computation graph input data.

    Attributes:
        See mt.Gene attributes.

    """
    def __init__(self,
                 name: str,
                 parent: Optional['ScaleGene']=None,
                 hyperparameter_config: Optional[mt.HyperparameterConfig]=None,
                 spatial_scale: Optional[int]=None):
        """Constructor.

        Args:
            name (str): Gene name.
            parent (ScaleGene): Parent Gene.
            hyperparameter_config (Optional[mt.HyperparameterConfig]): The
                HyperparameterConfig governing this Gene's hyperparameters.
                If none is supplied, inherit from parent.
            spatial_scale (Optional[int]): The spatial scale of the data that
                this ScaleGene processes. spatial_scale=i means the data has
                been downsampled by a factor of 2**i relative to the input
                data used by the computation graph this ScaleGene builds.
        """
        super().__init__(name, parent, hyperparameter_config)
        # Update spatial scale without triggering a setup_children() propagation
        if spatial_scale is not None:
            self.deltas['spatial_scale'] = spatial_scale - \
                                           self.hyperparam('spatial_scale')

        # Shape of the data coming into and out from this Gene, along its
        # spatial axes
        self.data_shapes_in: List[int] = []
        self.data_shape_out: List[int] = []

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

    @staticmethod
    def input_tensor_shape(edge_shapes: Union[Sequence[int],
                                              Dict[mt.Edge, Sequence[int]]]) \
            -> List[int]:
        """Return the spatial shape of this Gene's input Tensor, formed from
        its input Edge sources.

        For ScaleGene's, the spatial shape is the min along each dimension of
        the input edge spatial shapes.

        Args:
            edge_shapes (Union[Sequence[int], Dict['Edge', Sequence[int]]]):
                Spatial shapes of the Tensors that are output by each input
                edge.

        Returns:
            (List[int]): Spatial shape of the final input tensor that is
                passed through the `Gene.build()` method.

        """
        if isinstance(edge_shapes, Sequence):
            # Given a single input, just return it
            return list(edge_shapes)
        elif len(edge_shapes) == 1:
            return list(list(edge_shapes.values())[0])
        else:
            # Convert inputs to a list
            shape_list = list(edge_shapes.values())
            # This will hold the final shape out
            shape_out = []
            for i in range(len(shape_list[0])):
                # Iterate across spatial axes, taking the minimum size from
                # each one
                min_size = min(s[i] for s in shape_list)
                shape_out.append(min_size)
            return shape_out

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
        shape = self.input_tensor_shape(shapes_in)
        for child in self.children:
            shape = child.shape_out(shape)
        self.data_shape_out = shape
        return self.data_shape_out
