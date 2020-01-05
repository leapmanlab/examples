"""ConvBlockGene class defines a sequence of convolution operations.

"""
import metatree as mt
import tensorflow as tf

from typing import Optional, Sequence, Union

from .ConvolutionGene import ConvolutionGene
from .ScaleEdge import ForwardEdge, ScaleEdge
from .ScaleGene import ScaleGene


class ConvBlockGene(ScaleGene):
    """A ScaleGene which builds a sequence of convolution operations into a
    computation graph.

    Hyperparameters:
        n_comps (int): Number of computation Genes descending from this
            ConvBlockGene.

    TODO
    Attributes:

    """
    def __init__(self,
                 name: str,
                 parent: Optional[ScaleGene]=None,
                 hyperparameter_config: Optional[mt.HyperparameterConfig]=None,
                 spatial_scale: Optional[int]=None):
        """Constructor.

        Args:
            name (str): ConvolutionGene name.
            parent (ScaleGene): Parent Gene.
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
        # Perform setup for child ConvolutionGenes
        self.setup_children()

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
            # Build each child operation's computation graph
            for computation in self.children:
                computation.build(mode, outputs)

        # ConvBlock output is the output of the last child computation
        outputs[self] = self.children[-1]

        pass

    def setup_children(self):
        """Set up child computations.

        Returns: None

        """
        # Number of ConvBlockGene children set by the 'n_comps' hyperparameter
        n_children = self.hyperparam('n_comps')
        # How many children does this ConvBlockGene have already?
        n_children_now = len(self.children)
        # What change is needed to have n_children children?
        d_n_children = n_children - n_children_now

        if d_n_children > 0:
            # Add children
            for i in range(d_n_children):
                name = f'comp {n_children_now + i}'

                # Create a new child ConvolutionGene
                self.children.append(ConvolutionGene(name,
                                                     parent=self))

        elif d_n_children < 0:
            # Remove children, starting at the end (LIFO)
            for i in range(-d_n_children):
                self.children.pop()

        # And if d_n_children == 0, do nothing!
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
                edge = ForwardEdge(self.children[i])
                child.setup_edges(edge)

        # Gene setup is now complete
        self.setup_complete = True

        pass
