"""Manages a graph of Genes

"""
import logging

import tensorflow as tf

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Union

from .Edge import Edge
from .types import BuildMap
from .Gene import Gene
from .HyperparameterConfig import HyperparameterConfig

logger = logging.getLogger('metatree.GeneGraph')


class GeneGraph:
    """
    Attributes:
        input_shape (List[int]): Input data shape for the computation graph
            built by this Gene graph.
        genes (List[Gene]): List of Genes in this GeneGraph.
        hyperparameter_configs (List[HyperparameterConfig]): List of all
            HyperparameterConfigs used by this GeneGraph.
        name (str): Name of this GeneGraph.
        output_shape (List[int]): Output data shape for the computation graph
            built by this Gene graph.

    """
    def __init__(self,
                 input_shape: Sequence[int],
                 genes: Optional[Dict[str, Gene]]=None,
                 name: str='Gene graph'):
        """Constructor.

        **NOTE**: Input `genes` should be an OrderedDict if supplied,
        not a Dict. But Python throws an error if you try to use OrderedDict
        as a type hint. This will be fixed when Python gets fixed.
        TODO: Fix this

        Args:
            input_shape (Sequence[int]): Input data shape for the computation
                graph built by this Gene graph.
            genes (Optional[OrderedDict[str, Gene]]): Supply one or more
                Genes to the constructor in order to immediately add them to
                this GeneGraph. Use an OrderedDict (1) instead of a list,
                so that Genes can be tracked by their name and not just
                index, and (2) instead of a dict, so that genes can be
                iterated through in their order of insertion, e.g. during
                computation graph building.
            name (str): The name of this GeneGraph.

        """
        # Input data shape for the computation graph built by this Gene graph
        self.input_shape = list(input_shape)

        # The name of this GeneGraph
        self.name = name

        # Dict of all HyperparameterConfigs used by this GeneGraph
        self.hyperparameter_configs: Dict[str, Union[HyperparameterConfig,
                                                     Dict[str, Any]]] = {}

        # Genes in this GeneGraph.
        self.genes: OrderedDict[str, Gene] = OrderedDict()
        if genes is not None:
            for key in genes:
                self.add_gene(key, genes[key])

        # Memoization variable for the shape_out() function
        self._memo = {}

        pass

    def add_gene(self,
                 key: str,
                 gene: Gene):
        """Add a Gene to this GeneGraph.

        Because of how Genes are structured, this implicitly adds all
        children of the input `gene` to the GeneGraph as well.

        Args:
            key (str): Name for this Gene in the `self.genes` dict.
            gene (Gene): New Gene for this GeneGraph.

        Returns: None

        """
        self.genes[key] = gene

        # Add the gene's hyperparameter config to this GeneGraph's list
        self.add_hyperparameter_config(key, gene.hyperparameter_config)

        pass

    def add_hyperparameter_config(self,
                                  key: str,
                                  config: Union[HyperparameterConfig,
                                                Dict[str, Any]]):
        """Add a hyperparameter configuration to this GeneGraph.

        Args:
            key (str): Dictionary key for the config.
            config (Union[HyperparameterConfig, Dict[str, Any]]): The
                hyperparameter config to add. Can be a HyperparameterConfig
                object, or a dictionary mapping hyperparameter names to values.

        Returns: None

        """
        self.hyperparameter_configs[key] = config
        pass

    def build(self,
              features: Dict[str, tf.Tensor],
              mode: tf.estimator.ModeKeys) -> BuildMap:
        """Build a computation graph from the Gene graph.

        Args:
            features (Dict[str, tf.Tensor]): Dictionary of input Tensors.
            mode (tf.estimator.ModeKeys): TensorFlow Estimator mode.

        Returns: None

        """
        # Make sure genes are set up before building
        not_setup_genes = [g for g in self.genes.values()
                           if not g.setup_complete]
        if len(not_setup_genes) > 0:
            raise ValueError(f'Genes {not_setup_genes} are not set up')
        
        # Dict mapping Genes to their output Tensors during the build process
        outputs = {}

        # Assume input is passed to build() as an element of features with
        # key 'input'
        inputs = features['input']
        # Add the input Tensor into the output map
        outputs['input'] = inputs

        # Build the Genes. Note that they share the outputs dict,
        # and therefore Gene tree n can reference the output Tensors associated
        # with Genes in trees 1, ..., n-1.
        for gene in self.genes.values():
            gene.build(mode, outputs)

        return outputs

    def output_shape(self) -> List[int]:
        """Calculate the output shape of the GeneGraph.

        Assumes the output Gene is the last entry in self.genes.

        Returns:
            (List[int]):

        """
        # Reset the memoization dict
        self._memo = {}
        # Get the last entry in self.genes
        last_gene = list(self.genes.values())[-1].last_descendant()
        return self._output_shape(last_gene)

    def _output_shape(self, gene: Union[Gene, str]) -> List[int]:
        """Calculate recursively the output spatial shape of input `gene`,
        by backtracking along the `gene`'s sources to 'input' edges.

        Args:
            gene (Union[Gene, str]): Gene to calculate the output
                shape for.

        Returns:
            (List[int]): The spatial shape of the output of the computation
            graph elements associated with `gene`.

        """
        if gene not in self._memo:
            if gene == 'input':
                self._memo[gene] = self.input_shape
            else:
                # Recursively compute input shapes from edge source output
                # shapes
                shapes_in = {e: self._output_shape(e.source)
                             for e in gene.inputs}
                self._memo[gene] = gene.shape_out(shapes_in)
        return self._memo[gene]

