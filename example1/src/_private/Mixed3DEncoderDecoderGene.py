"""Mixed3DEncoderDecoderGene is a typical 2D encoder-decoder with
conv block-initial conv3d's added in.

"""
import genenet as gn
import metatree as mt
import tensorflow as tf

from typing import Dict, List, Optional, Sequence, Union


class Mixed3DEncoderDecoderGene(gn.genes.EncoderDecoderGene):
    """

    """
    def __init__(self,
                 name: str,
                 conv3d_k_width: int = 3,
                 hyperparameter_config: Optional[
                     mt.HyperparameterConfig] = None,
                 spatial_scale: int = 0):
        """Constructor.

        Args:
            name (str): This gene's name.
            conv3d_k_width (Sequence[int]): Spatial width of the block-initial
                conv3d layer kernels.
            hyperparameter_config (Optional[mt.HyperparameterConfig]): The
                HyperparameterConfig governing this Gene's hyperparameters.
                If none is supplied, use
                genenet.hyperparameter_config.encoder_decoder().
            spatial_scale (int): The spatial scale of the data that
                this gene processes. A spatial_scale == i means the data has
                been downsampled by a factor of 2**i relative to the input
                data used by the computation graph this ScaleGene builds.
        """
        self.conv3d_k_width = conv3d_k_width
        super().__init__(name,
                         hyperparameter_config=hyperparameter_config,
                         spatial_scale=spatial_scale)
        self.setup_children()

        pass

    def setup_edges(self, *, _unused=None):
        """Set up the Edges that target this Gene and its children.

        Args:
            _unused: Unused.

        Returns: None

        """
        # TODO: Find better solution for this hack
        for blockset in self.children:
            for i, block in enumerate(blockset.children):
                if block.hyperparam('spatial_scale') < 2:
                    # Append block-initial conv3ds
                    conv3d_gene = gn.genes.ConvolutionGene(f'comp 3dinit {i}',
                                                           parent=block)
                    conv3d_gene.set(
                        False,
                        spatial_mode=1,
                        k_width=self.conv3d_k_width,
                        n_kernels=block.hyperparam('n_kernels') // 2)
                    block.children.insert(0, conv3d_gene)

        # Edge from input data
        encoder_edge = gn.genes.ForwardEdge(
            'input',
            self.hyperparameter_config)

        # Encoder setup
        self.children[0].setup_edges(encoder_edge)

        # Decoder setup
        decoder_edge = gn.genes.ForwardEdge(self.children[0].last_descendant())
        self.children[1].setup_edges(decoder_edge)

        # Gene setup is now complete
        self.setup_complete = True

        pass