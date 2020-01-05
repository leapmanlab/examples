"""SpatialPyramidGene generates a spatial pyramid from its input and
applies a block of convolutions to each pyramid element. The resulting
feature maps are upsampled to original resolution and recombined.

"""
import genenet as gn
import metatree as mt
import tensorflow as tf

from typing import Dict, List, Optional, Sequence, Union


class Mixed2D3DEncoderDecoderGene(gn.genes.EncoderDecoderGene):
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
                if block.spatial_scale < 2:
                    # Append block-initial conv3ds
                    conv3d_gene = gn.genes.ConvolutionGene(f'comp 3dinit {i}',
                                                           parent=block)
                    conv3d_gene.set(
                        False,
                        spatial_mode=1,
                        kwidth=self.conv3d_k_width,
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


class SpatialPyramidGene(gn.genes.BlockSetGene):
    """A Gene for a module with parallel convolution block paths at multiple
    spatial scales.

    """
    def __init__(self,
                 combine_mode: str = 'add',
                 name: str = 'spatial_pyramid',
                 parent: Optional[gn.genes.ScaleGene] = None,
                 hyperparameter_config: Optional[mt.HyperparameterConfig] = None,
                 spatial_scale: Optional[int] = None):
        """Constructor.

        Args:
            combine_mode (str): Combination mode for the output of the
                multiscale feature maps. Must be 'add', for a residual
                connection, or 'merge' for a merge connection.
            name (str): This gene's name.
            parent (Optional[gn.ScaleGene]): Parent gene.
            hyperparameter_config (Optional[mt.HyperparameterConfig]): The
                HyperparameterConfig governing this Gene's hyperparameters.
                If none is supplied, inherit from parent.
            spatial_scale (Optional[int]): The spatial scale of the data that
                this ScaleGene processes. spatial_scale=i means the data has
                been downsampled by a factor of 2**i relative to the input
                data used by the computation graph this ScaleGene builds.
        """
        self.combine_mode = combine_mode
        super().__init__(name, parent, hyperparameter_config, spatial_scale)
        pass

    def add_child(self):
        """Add a child ConvBlockGene to this gene's children, one spatial
        scale down from the last.

        Returns: None

        """
        # The new child block
        new_block = gn.genes.ConvBlockGene('conv block',
                                           parent=self)
        # Add the new block to self.children
        self.children.append(new_block)

        pass

    def setup_edges(self,
                    edges: Union[gn.genes.ScaleEdge,
                                 Sequence[gn.genes.ScaleEdge]]):
        """Set up the Edges that target this Gene and its children.

        Args:
            edges (Union[ScaleEdge, Sequence[ScaleEdge]]): One or more
                ScaleEdges that are inputs to this gene.

        Returns: None

        """
        # TODO: Remove
        if len(self.children) > 1 and self.hyperparam('spatial_mode') == 1:
            self.children[1].set(pool_z=True)
        for child in self.children:
            # Each child gets a copy of the input edge
            child.setup_edges(edges)

        # Create 1x1 conv Genes that match number of output features in each
        # path to the number in the first path
        path_outputs = []
        n_kernels = self.children[0].hyperparam('n_kernels')
        if len(self.children) > 1:
            for i, child in enumerate(self.children[1:]):
                path_end_gene = gn.genes.ConvolutionGene(f'path end {i + 1}',
                                                         parent=child)
                path_end_gene.set(n_kernels=n_kernels, k_width=1)
                path_end_gene.setup_edges(gn.genes.ForwardEdge(child))
                child.children.append(path_end_gene)
                path_outputs.append(path_end_gene)

            # Create a Gene that merges all of the path outputs
            end_gene = gn.genes.IdentityGene('path combine', self)
            # Create edges to combine path outputs
            edges = [gn.genes.ForwardEdge(self.children[0])]
            for i, child in enumerate(path_outputs):
                if self.combine_mode == 'add':
                    edges.append(gn.genes.ResidualEdge(child,
                                                       name=f'residual {i+1}'))
                elif self.combine_mode == 'merge':
                    edges.append(gn.genes.MergeEdge(child,
                                                    name=f'merge {i+1}'))
                else:
                    raise ValueError(f"Combine mode '{self.combine_mode}' "
                                     f"not recognized")

            self.children.append(end_gene)
            end_gene.setup_edges(edges)

        self.setup_complete = True

        pass

    @staticmethod
    def shape_out(shapes_in: Union[Sequence[int],
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
        if isinstance(shapes_in, Dict):
            return list(shapes_in.values())[0]
        else:
            return list(shapes_in)

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
            child.name = f'conv block {new_scale}'

        pass
