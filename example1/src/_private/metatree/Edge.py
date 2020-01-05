"""Edge class manages connections between Genes.

Each Edge has a source and and a target. The source is a Gene or a TensorFlow
Tensor, while the target is a Gene. The Edge manages the transformation of an
input tf.Tensor into something appropriate as an input for the target Gene.

"""
import logging

import tensorflow as tf

from typing import List, Optional, Sequence, Union

from .Gene import Gene
from .HyperparameterConfig import HyperparameterConfig
from ._MutableTreeNode import MutableTreeNode
from .types import BuildMap

logger = logging.getLogger('metatree.Edge')


class Edge(MutableTreeNode):
    """Controls connections between Genes.

    An Edge represents a connection in a computation graph, linking the
    output of one Gene with the input of another Gene.

    TODO
    Attributes:

    """
    def __init__(self,
                 name: str,
                 source: Union[Gene, str],
                 hyperparameter_config: Optional[HyperparameterConfig]=None):
        """Constructor.

        Args:
            name (str): This Edge's name.
            source (Union[Gene, str]): This edge's source Gene or
                the name of a TensorFlow Tensor in some TensorFlow Graph.
            hyperparameter_config (Optional[HyperparameterConfig]): This
                edge's hyperparameter configuration object. If the edge
                source is a Gene and no hyperparameter_config is supplied,
                the constructor uses the source's hyperparameter_config.

        """
        # If it's a Gene, the actual source is the final output of the `source`
        # arg: its last descendant
        if isinstance(source, Gene) and len(source.children) > 0:
            source = source.last_descendant()
        self.source = source
        if isinstance(source, Gene) and hyperparameter_config is None:
            hyperparameter_config = source.hyperparameter_config

        # No parent for now. It gets added later
        super().__init__(name, hyperparameter_config, parent=None)

        pass

    # noinspection PyUnusedLocal
    def build(self,
              mode: tf.estimator.ModeKeys,
              outputs: BuildMap):
        """Transforms this Edge's source tf.Tensor into a tf.Tensor to feed
        into the target Gene, using the input graph as the active computation
        graph. Probably overwritten by subclasses.

        Args:
            mode (tf.estimator.ModeKeys): TensorFlow Estimator mode.
            outputs (BuildMap): Computation graph build process' mapping from
                Tensor source to Tensor.

        Returns: None

        """
        # Just pass the source Tensor to the output
        outputs[self] = self.source_tensor(outputs)
        pass

    def setup_children(self):
        """Setup for self.children, which in this case should always be empty.

        Returns: None

        """
        self.children = []

    def shape_out(self, shape_in: Sequence[int]) -> List[int]:
        """Calculate the shape of the Tensor output by this Edge, if its
        input has shape `shape_in`. Update data shape attributes.

        Args:
            shape_in (Sequence[int]): Shape of the input Tensor for this Edge.

        Returns:
            (List[int]): Shape of the corresponding output Tensor from this
                Edge.

        """
        self.data_shape_in = list(shape_in)
        self.data_shape_out = self.data_shape_in
        return self.data_shape_out

    def source_tensor(self,
                      outputs: BuildMap) -> tf.Tensor:
        """Get the tf.Tensor at this Edge's source.

        Args:
            outputs (BuildMap): Computation graph build process' mapping from
                Tensor source to Tensor.

        Returns:
            tensor (tf.Tensor): The source Tensor.

        """
        # If the output map value associated with the key self.source is a
        # Gene, recursively return the output map value associated with that
        # Gene until we find something that is not a Gene
        current_value = outputs[self.source]
        while isinstance(current_value, Gene):
            current_value = outputs[current_value]
        return current_value
