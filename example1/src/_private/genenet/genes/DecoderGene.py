"""DecoderGene class controls the creation of a decoder network module.

A decoder network synthesizes an image (or something else?) from a collection of
encoded features created by an encoder network.

"""
from .BlockSetGene import BlockSetGene
from .ConvBlockGene import ConvBlockGene
from .ScaleEdge import MergeEdge, ScaleEdge

from typing import Sequence, Union


class DecoderGene(BlockSetGene):
    """Controls the creation of a decoder network module.

    A decoder network synthesizes an image (or something else?) from a
    collection of encoded features created by an encoder network.

    TODO
    Attributes:

    """
    def add_child(self):
        """Add a child ConvBlockGene to this gene's children.

        Returns: None

        """
        # The new child block
        new_block = ConvBlockGene('decode block',
                                  parent=self)
        # Add the new block to self.children
        self.children.append(new_block)

        pass

    def setup_children(self):
        """Set up child blocks.

        Returns: None

        """

        # Get the number of blocks of the encoder gene
        # (Note that the decoder part of the network will have an extra block)
        encoder = self.root.children[0]
        n_encoder_blocks = encoder.hyperparam('n_blocks')

        # In a BlockSetGene, children are blocks
        n_children = n_encoder_blocks + 1
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
        # Add forward edges
        super().setup_edges(edges)

        # Add merge edges
        encoder = self.root.children[0]
        for dblock in self.children:
            child_scale = dblock.hyperparam('spatial_scale')
            for eblock in encoder.children:
                if eblock.hyperparam('spatial_scale') == child_scale:
                    edge = MergeEdge(eblock.last_descendant())
                    dblock.children[0].add_input(edge)

        # Gene setup is now complete
        self.setup_complete = True
        pass

    def _rescale_children(self):
        """Recalculate the spatial scales of child ScaleGenes.

        Used after modifying the number of DecoderGene children. Operates on
        the assumption that the final child's scale stays the same, and each
        previous child's scale increments by 1.

        Returns: None

        """
        n_children = len(self.children)
        self_scale = self.hyperparam('spatial_scale')

        # Update children

        for i, child in enumerate(self.children):
            child: ConvBlockGene

            new_scale = self_scale + n_children - 1 - i

            # Update spatial scale
            self._update_scale(child, new_scale, self_scale)

        pass

    def _update_scale(self,
                      child: ConvBlockGene,
                      new_scale: int,
                      self_scale: int):
        """Update the spatial scale of a child ConvBlockGene.

        Args:
            child (ConvBlockGene): The child to update.
            new_scale (int): The new spatial scale.
            self_scale (int): Spatial scale of this DecoderGene.

        Returns: None

        """
        child.set(spatial_scale=new_scale)

        # Update n kernels
        d_kernels = 2 ** (new_scale - self_scale)
        self_kernels = self.hyperparam('n_kernels')
        child.set(n_kernels=d_kernels * self_kernels)

        # Update name
        child.name = f'decode block {new_scale}'
