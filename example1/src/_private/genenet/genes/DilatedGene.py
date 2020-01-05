"""DilatedGene builds a sequence of ConvBlockGenes all at the same spatial
scale, each block with a different dilation rate for its ConvGenes.

"""
import metatree as mt
import tensorflow as tf

from typing import Optional

import genenet.hyperparameter_configs as configs

from .ConvBlockGene import ConvBlockGene
from .ScaleEdge import ForwardEdge
from .ScaleGene import ScaleGene


class DilatedGene(ScaleGene):
    """A sequence of ConvBlockGenes with different dilation rates.

    Hyperparameters:
        n_blocks (int): Number of computation block Genes descending from
            this BlockSetGene.

    TODO
    Attributes:

    """
    def __init__(self,
                 name: str,
                 parent: Optional[ScaleGene]=None,
                 hyperparameter_config: Optional[mt.HyperparameterConfig]=None):
        """Constructor.

        Args:
            name (str): This gene's name.
            parent (Optional[ScaleGene]): Parent Gene.
            hyperparameter_config (Optional[mt.HyperparameterConfig]): The
                HyperparameterConfig governing this Gene's hyperparameters.
                If none is supplied, inherit from parent.

        """
        if hyperparameter_config is None:
            hyperparameter_config = configs.encoder_decoder()

        super().__init__(name,
                         parent=parent,
                         hyperparameter_config=hyperparameter_config,
                         spatial_scale=0)
        self.setup_children()

        pass

    def add_child(self):
        """Add a child ConvBlockGene to this gene's children.

        Returns: None

        """
        # The new child block
        child_number = len(self.children)
        new_block = ConvBlockGene(f'conv block {child_number}',
                                  parent=self,
                                  spatial_scale=0)
        # Add the new block to self.children
        self.children.append(new_block)

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

        # DilatedGene output is the output of the last child block
        outputs[self] = self.children[-1]

        pass

    def setup_children(self):
        """Set up child blocks, fixing the dilation_rate hyperparam for each.

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

        for i, block in enumerate(self.children):
            dilation_rate = self.hyperparam('dilation_rate') * 2**i
            block.setup_children()
            block.set(dilation_rate=dilation_rate)

        pass

    def setup_edges(self, *, _unused=None):
        """Set up the Edges that target this Gene and its children.

        Returns: None

        """
        # Edge from input data
        input_edge = ForwardEdge('input',
                                 self.hyperparameter_config)
        self.children[0].setup_edges(input_edge)

        if len(self.children) > 0:
            for i, child in enumerate(self.children[1:]):
                edge = ForwardEdge(self.children[i].last_descendant())
                child.setup_edges(edge)

        # Gene setup is now complete
        self.setup_complete = True

        pass
