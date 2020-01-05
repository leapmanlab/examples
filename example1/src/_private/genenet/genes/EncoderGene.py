"""EncoderGene class controls the creation of an encoder network module.

An encoder network feeds an input through a succession of convolution
blocks, downsampling the data between successive blocks.

"""
from .BlockSetGene import BlockSetGene
from .ConvBlockGene import ConvBlockGene
from .ScaleEdge import ScaleEdge

from typing import Sequence, Union


class EncoderGene(BlockSetGene):
    """Controls the creation of an encoder network module.

    An encoder network feeds an input through a succession of convolution
    blocks, downsampling the data between successive blocks.

    TODO
    Attributes:

    """
    def add_child(self):
        """Add a child ConvBlockGene to this gene's children, one spatial
        scale down from the last.

        Returns: None

        """
        # The new child block
        new_block = ConvBlockGene('encode block',
                                  parent=self)
        # Add the new block to self.children
        self.children.append(new_block)

        pass

    def setup_children(self):
        """Sets up children and if this is a mutation this function also
        triggers the decoder genes' setup_children() to apply mutations.

        Return: None

        """
        super().setup_children()

        if not len(self.root.children) == 0:
            decoder = self.root.children[1]
            decoder.setup_children()

        pass

    def setup_edges(self, edges: Union[ScaleEdge, Sequence[ScaleEdge]]):
        """Set up the Edges that target this Gene and its children.

        Args:
            edges (Union[ScaleEdge, Sequence[ScaleEdge]]): One or more
                ScaleEdges that are inputs to this ConvolutionGene.

        Returns: None
        """
        super().setup_edges(edges)

        # Gene setup is now complete
        self.setup_complete = True

        pass

    def _rescale_children(self):
        """Recalculate the spatial scales of child ScaleGenes.

        Used after adding or removing children. Operates on the assumption
        that the  first child's scale stays the same, and each subsequent
        child's scale increments by 1.

        Returns: None

        """
        self_scale = self.hyperparam('spatial_scale')

        for i, child in enumerate(self.children):
            # Update spatial scale
            new_scale = self_scale + i
            child.set(spatial_scale=new_scale)

            # Update n kernels
            d_kernels = 2**(new_scale - self_scale)
            self_kernels = self.hyperparam('n_kernels')
            child.set(n_kernels=d_kernels * self_kernels)

            # Update name
            child.name = f'encode block {new_scale}'

        pass

