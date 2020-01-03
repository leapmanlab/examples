"""Gene class builds a module in a computation graph.

A tree of Genes constructs a computation graph in TensorFlow. Each Gene
builds a portion of the computation graph, assembling the constructions of
its child Genes and adding additional structure as needed.

Gene graph construction is determined by a collection of hyperparameters.
Each Gene tracks a hyperparameter delta value, and the final hyperparameter
value used by a Gene during construction is the sum of delta values at the
Gene and all of its ancestors. This strategy allows hyperparameter values to
be easily modifed at many network scopes, from single-layer modifications to
global modifications.

"""
import logging

import tensorflow as tf

from copy import copy

from typing import Dict, List, Optional, Sequence, Tuple, Union

from .HyperparameterConfig import HyperparameterConfig
from ._MutableTreeNode import MutableTreeNode
from .types import BuildMap

logger = logging.getLogger('metatree.Gene')


class Gene(MutableTreeNode):
    """Base class for all Genes.

    Genes and Edges are subclasses of the MutableTreeNode, which implements
    a graph-theoretic node1 in a tree with mutable *hyperparameters*. When
    a MutableTreeNode retrieves a hyperparameter value, the retrieved value is
    the sum of the values of the 'delta' attributes for the node1 and all of
    its ancestors.

    A Gene is a MutableTreeNode which receives Edge inputs and builds part of a
    TensorFlow Graph.

    Attributes:
        children (List[Gene]):
        inputs (List[Edge]):
        name (str):

    """
    def __init__(self,
                 name: str,
                 parent: Optional['Gene']=None,
                 hyperparameter_config: Optional[HyperparameterConfig]=None):
        """Constructor.

        Args:
            name (str): Gene name.
            parent (Optional[Gene]): Parent Gene.
            hyperparameter_config (Optional[HyperparameterConfig]): The
                HyperparameterConfig governing this Gene's hyperparameters.
                If none is supplied, inherit from parent.
        """
        # List of input Edges
        self.inputs: List['Edge'] = []
        # List of Gene children
        self.children: List['Gene'] = []

        # True once self.setup_edges() has run successfully.
        self.setup_complete: bool = False

        # Use a parent hyperparameter
        hyperparameter_config = hyperparameter_config \
            if hyperparameter_config \
            else parent.hyperparameter_config

        # Call superclass constructor
        super().__init__(name, hyperparameter_config, parent)

        pass

    def add_input(self, new_input: 'Edge'):
        """Add an input to this Gene.

        Args:
            new_input (Edge): An Edge that targets this Gene.

        Returns: None

        """
        if new_input.parent is not None:
            # Create a copy of the input Edge for this Gene
            new_input = copy(new_input)
            new_input.parent = None

        # Add the new_input to this Gene's inputs
        self.inputs.append(new_input)
        # Set the input Edge's parent to this Gene
        # noinspection PyUnresolvedReferences
        new_input.add_parent(self)

        pass

    def all_descendants(self) -> Tuple[List['Gene'], List['Edge']]:
        """Return all descendants of this Gene.

        Returns a list of all descendant Genes (including self), and a list
        of all Edges which are inputs to those Genes.

        Returns:
            (Tuple[List[Gene], List[Edge]]): A list of all descendant Genes (
                including self), and a list of all Edges which are inputs to
                those Genes.

        """
        # Iterate through the tree of Genes, adding children to the check
        # list, inputs to the edges list, and the iterated gene to the genes
        # list
        genes_to_check = [self]
        genes = []
        edges = []
        while len(genes_to_check) > 0:
            gene = genes_to_check.pop()
            genes.append(gene)
            genes_to_check += gene.children
            edges += gene.inputs

        return genes, edges

    def build(self,
              mode: tf.estimator.ModeKeys,
              outputs: BuildMap):
        """Build this Gene's module into a TensorFlow computation graph.

        Args:
            mode (tf.estimator.ModeKeys): TensorFlow Estimator mode.
            outputs (BuildMap): Computation graph build process' mapping from
                Tensor source to Tensor.

        Returns:
            None

        """
        raise NotImplementedError('Calling an abstract method.')

    def input_tensor(self,
                     mode: tf.estimator.ModeKeys,
                     outputs: BuildMap) -> \
            tf.Tensor:
        """Assemble a TensorFlow Tensor from this Gene's input Edges.

        Args:
            mode (tf.estimator.ModeKeys): TensorFlow Estimator mode.
            outputs (BuildMap): Computation graph build process' mapping from
                Tensor source to Tensor.

        Returns:
            (tf.Tensor): A Tensor to use as input to this Gene's computations.

        """
        raise NotImplementedError('Calling an abstract method.')

    def input_tensor_shape(self,
                           edge_shapes: Union[Sequence[int],
                                              Dict['Edge', Sequence[int]]]) \
        -> List[int]:
        """Return the spatial shape of this Gene's input Tensor, formed from
        its input Edge sources.

        Args:
            edge_shapes (Union[Sequence[int], Dict['Edge', Sequence[int]]]):
                Spatial shapes of the Tensors that are output by each input
                edge.

        Returns:
            (List[int]): Spatial shape of the final input tensor that is
                passed through the `Gene.build()` method.

        """

    def setup_edges(self, edges: Union['Edge', Sequence['Edge']]):
        """Set up the Edges that target this Gene and its children.

        Args:
            edges (Union[Edge, Sequence[Edge]]): One or more Edges that are
                inputs to this Gene.

        Returns: None

        """
        raise NotImplementedError('Calling an abstract method.')

    def shape_out(self,
                  shapes_in: Union[Sequence[int],
                                   Dict['Edge', Sequence[int]]]) -> List[int]:
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
        raise NotImplementedError('Calling an abstract method.')
