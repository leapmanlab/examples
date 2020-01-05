"""BlockSetGene class is the parent class of the EncoderGene and DecoderGene.

"""
import metatree as mt
import tensorflow as tf

from typing import Any, Dict, Optional, Sequence, Union

from .ScaleEdge import ScaleEdge, ForwardEdge
from .ScaleGene import ScaleGene


class BlockSetGene(ScaleGene):
    """Parent class for EncoderGene and DecoderGene.

    Basically a sequence of ConvBlockGenes.

    Hyperparameters:
        n_blocks (int): Number of computation block Genes descending from
            this BlockSetGene.

    TODO
    Attributes:

    """
    def __init__(self,
                 name: str,
                 parent: Optional[ScaleGene]=None,
                 hyperparameter_config: Optional[mt.HyperparameterConfig]=None,
                 spatial_scale: Optional[int] = None):
        """Constructor.

        Args:
            name (str): This gene's name.
            parent (Optional[ScaleGene]): Parent Gene.
            hyperparameter_config (Optional[mt.HyperparameterConfig]): The
                HyperparameterConfig governing this Gene's hyperparameters.
                If none is supplied, inherit from parent.
            spatial_scale (Optional[int]): The spatial scale of the data that
                this ScaleGene processes. spatial_scale=i means the data has
                been downsampled by a factor of 2**i relative to the input
                data used by the computation graph this ScaleGene builds.

        """
        super().__init__(name,
                         parent,
                         hyperparameter_config,
                         spatial_scale)
        self.setup_children()

        pass

    def add_child(self):
        """Add a child Gene. Overwritten by derived classes.

        """
        pass

    def build(self,
              mode: tf.estimator.ModeKeys,
              outputs: mt.BuildMap):
        """Build this Gene's module into a TensorFlow computation graph.

        Args:
            mode (tf.estimator.ModeKeys): TensorFlow Estimator mode.
            outputs (mt.BuildMap): Computation graph build process' mapping from
                Tensor source to Tensor.

        Returns:
            None

        """
        with tf.variable_scope(self.name.replace(' ', '_')):
            for block in self.children:
                block.build(mode, outputs)

        # BlockSetGene output is the output of the last child block
        outputs[self] = self.children[-1]

        pass

    def setup_children(self):
        """Set up child blocks.

        Returns: None

        """
        # In a BlockSetGene, children are blocks
        n_children = self.hyperparam('n_blocks')
        # How many children does this gene have already?
        n_children_now = len(self.children)
        # What change is needed to have n_children children?
        d_n_children = n_children - n_children_now

        if d_n_children > 0:
            # Add children
            for i in range(d_n_children):
                self.add_child()

        elif d_n_children < 0:
            # Remove children
            for i in range(-d_n_children):
                self.children.pop()

        # Deal with potential changes in spatial scales caused by the
        # addition or removal of blocks
        self._rescale_children()

        pass

    def setup_edges(self, edges: Union[ScaleEdge, Sequence[ScaleEdge]]):
        """Set up the Edges that target this Gene and its children.

        Args:
            edges (Union[ScaleEdge, Sequence[ScaleEdge]]): One or more
                ScaleEdges that are inputs to this ConvolutionGene.

        Returns: None

        """
        # First child receives the same edge inputs as this gene
        self.children[0].setup_edges(edges)
        if len(self.children) > 0:
            for i, child in enumerate(self.children[1:]):
                edge = ForwardEdge(self.children[i].last_descendant())
                child.setup_edges(edge)

        self.setup_complete = True

        pass

    def _rescale_children(self):
        """Rescale all child ScaleGenes. Implemented by child classes.

        Returns: None

        """
        raise NotImplementedError('Calling an abstract method.')

